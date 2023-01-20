//! A setup for training an agent to copy strokes from an image.
//! The agent reads in the current canvas and reference image, and outputs
//! brush strokes.
//! The environment stops once a difference threshold is reached, or after the time limit is hit.
//! The difference at each time step is given as the reward.
use clap::Parser;
use indicatif::ProgressIterator;

use paint_gym::gym::{PaintAction, PaintGym, PaintStepResult, Pixel};
use paint_gym::model_utils::{cleanup_py, export_model, prep_py, TrainableModel};
use paint_gym::monitor::Monitor;
use paint_gym::rollout_buffer::RolloutBuffer;

use pyo3::prelude::*;
use tch::nn::OptimizerConfig;

#[derive(Parser, Debug)]
struct Args {
    /// Number of environments to use. This many samples can be collected in one step.
    #[arg(long, default_value_t = 8)]
    env_count: u64,
    /// Number of steps to run through.
    #[arg(long, default_value_t = 10000)]
    steps: u32,
    /// Number of steps that comprise a single rollout.
    #[arg(long, default_value_t = 32)]
    rollout_steps: u32,
    /// Number of rollouts to run through before evaluating the model.
    #[arg(long, default_value_t = 100)]
    rollouts_before_eval: u32,
    /// Maximum number of steps to take in the environment before failure.
    #[arg(long, default_value_t = 30)]
    max_env_steps: u32,
    /// Reward given if the maximum number of steps is reached. Should be negative.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values(true))]
    failure_reward: f32,
    /// Number of iterations during training.
    #[arg(long, default_value_t = 10)]
    train_iterations: u32,
    /// Minibatch size during training.
    #[arg(long, default_value_t = 64)]
    train_batch_size: usize,
    /// Lambda used for GAE .
    #[arg(long, default_value_t = 0.95)]
    lambda: f32,
    /// Learning rate for value network.
    #[arg(long, default_value_t = 0.001)]
    v_lr: f64,
    /// Learning rate for policy network.
    #[arg(long, default_value_t = 0.0003)]
    p_lr: f64,
    /// Discount factor when propagating rewards.
    #[arg(long, default_value_t = 0.95)]
    discount: f32,
    /// Clip objective epsilon.
    #[arg(long, default_value_t = 0.2)]
    epsilon: f64,
    /// Whether render should occur when evaluating.
    #[arg(long, default_value_t = false)]
    render: bool,
    /// If true, saved models are loaded instead of training from scratch.
    #[arg(long, default_value_t = false)]
    cont: bool,
    /// Whether CUDA should be used. If false, training occurs on the CPU.
    #[arg(long, default_value_t = false)]
    cuda: bool,
}

// Canvas and reference pixels
const IMG_SIZE: i64 = 8;
const IMG_CHANNELS: i64 = 6;
// Start (x, y), end (x, y)
const ACTION_DIM: i64 = 4;
const AVG_REWARD_OUTPUT: &str = "temp/avg_rewards.png";

#[pyclass]
struct VNetParams {
    #[pyo3(get)]
    img_size: i64,
}

#[pyclass]
struct PNetParams {
    #[pyo3(get)]
    img_size: i64,
    #[pyo3(get)]
    action_dim: i64,
}

/// Converts a Vec of pixel tuples into a tensor.
pub fn pixels_to_tensor(pixels: &[(u8, u8, u8)], img_size: i64) -> tch::Tensor {
    tch::Tensor::stack(
        &pixels
            .iter()
            .fold(vec![Vec::new(); 3], |mut v, p| {
                v[0].push(p.0 as f32);
                v[1].push(p.1 as f32);
                v[2].push(p.2 as f32);
                v
            })
            .iter()
            .map(|channel| tch::Tensor::of_slice(channel))
            .collect::<Vec<tch::Tensor>>(),
        0,
    )
    .view([3, img_size, img_size])
}

/// Converts results to tensor of states.
pub fn results_to_state(results: &PaintStepResult, device: tch::Device) -> tch::Tensor {
    let mut obs_vec = Vec::new();
    for (state, _, _) in &results.results {
        let canvas_channels = pixels_to_tensor(&state.canvas, IMG_SIZE);
        let ref_channels = pixels_to_tensor(&state.reference, IMG_SIZE);
        obs_vec.push(tch::Tensor::cat(&[canvas_channels, ref_channels], 0));
    }
    tch::Tensor::stack(&obs_vec, 0).to(device)
}

/// Converts tensors to actions.
pub fn tensor_to_actions(tensor: &tch::Tensor) -> Vec<PaintAction> {
    let mut actions = Vec::new();
    let canvas_max = (IMG_SIZE - 1) as f64;
    for i in 0..(tensor.size2().unwrap().0) {
        let start_x = (tensor.double_value(&[i, 0]) * canvas_max) as u32;
        let start_y = (tensor.double_value(&[i, 1]) * canvas_max) as u32;
        let end_x = (tensor.double_value(&[i, 2]) * canvas_max) as u32;
        let end_y = (tensor.double_value(&[i, 3]) * canvas_max) as u32;
        actions.push(PaintAction {
            start: Pixel::new(start_x, start_y),
            end: Pixel::new(end_x, end_y),
        });
    }
    actions
}

/// Copies parameters from one network to another.
fn copy_params(src: &tch::nn::VarStore, dest: &mut tch::nn::VarStore) {
    tch::no_grad(|| {
        for (dest, src) in dest
            .trainable_variables()
            .iter_mut()
            .zip(src.trainable_variables().iter())
        {
            dest.copy_(src);
        }
    })
}

/// Samples actions from means and scales.
fn sample(means: &tch::Tensor, scales: &tch::Tensor, device: tch::Device) -> tch::Tensor {
    let normals = tch::Tensor::ones(&means.size(), (tch::Kind::Float, device)).normal_(0.0, 1.0);
    normals * scales + means
}

/// Returns the log probabilities of actions being performed.
/// For each item in the batch, a single probability is computed by summing the individual log probabilities of a single action.
fn action_log_probs_cont(
    values: &tch::Tensor,
    means: &tch::Tensor,
    scales: &tch::Tensor,
    device: tch::Device,
) -> tch::Tensor {
    let shape = means.size();
    let values = values.flatten(0, 1);
    let means = means.flatten(0, 1);
    let scales = scales.flatten(0, 1);
    let diffs = &values - &means;
    let log_probs: tch::Tensor = -(&diffs * &diffs) / (2.0 * &scales * &scales)
        - scales.log()
        - std::f32::consts::TAU.sqrt().ln()
            * tch::Tensor::ones(&scales.size(), (tch::Kind::Float, device));
    let log_probs = log_probs.reshape(&shape);
    let log_probs = log_probs.sum_dim_intlist(Some([1].as_slice()), false, tch::Kind::Float);
    log_probs
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let device = if args.cuda {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };

    // Plotting stuff
    let mut monitor = Monitor::new(500, AVG_REWARD_OUTPUT);
    let eval_reward_metric = monitor.add_metric("eval_reward");
    let v_loss_metric = monitor.add_metric("v_loss");
    let p_loss_metric = monitor.add_metric("p_loss");
    let avg_reward_metric = monitor.add_metric("avg_reward");
    let pred_value_metric = monitor.add_metric("pred_value");
    monitor.init();

    // Load models
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        prep_py(py);
        let batch_size = 10;
        export_model(
            py,
            "copy_stroke_models",
            "VNet",
            VNetParams { img_size: IMG_SIZE },
            &[&[
                batch_size,
                IMG_CHANNELS as u32,
                IMG_SIZE as u32,
                IMG_SIZE as u32,
            ]],
            args.cuda,
        );
        export_model(
            py,
            "copy_stroke_models",
            "PNet",
            PNetParams {
                img_size: IMG_SIZE,
                action_dim: ACTION_DIM,
            },
            &[&[
                batch_size,
                IMG_CHANNELS as u32,
                IMG_SIZE as u32,
                IMG_SIZE as u32,
            ]],
            args.cuda,
        );
    });
    cleanup_py();

    let (v_path, p_path) = if args.cont {
        ("temp/VNet.pt", "temp/PNet.pt")
    } else {
        ("temp/VNet.ptc", "temp/PNet.ptc")
    };
    let mut v_net = TrainableModel::load(v_path, device);
    let (mut p_net, mut p_net_old) = TrainableModel::load2(p_path, device);
    let mut v_opt = tch::nn::Adam::default().build(&v_net.vs, args.v_lr)?;
    let mut p_opt = tch::nn::Adam::default().build(&p_net.vs, args.p_lr)?;

    let (mut envs, mut results) = PaintGym::init(
        args.env_count as u32,
        IMG_SIZE as u32,
        args.failure_reward,
        args.max_env_steps,
        args.render,
    );
    let mut rollout_buffer = RolloutBuffer::new(
        &[IMG_CHANNELS, IMG_SIZE, IMG_SIZE],
        &[ACTION_DIM],
        tch::Kind::Float,
        args.env_count as usize,
        args.rollout_steps as usize,
        device,
    );
    for step in (0..args.steps).progress() {
        monitor.add_step();

        let obs = results_to_state(&results, device);
        let actions_tensor = tch::no_grad(|| -> Result<tch::Tensor, anyhow::Error> {
            let actions_tensor = p_net.module.forward_ts(&[&obs])?.to(device);
            let means = actions_tensor.get(0);
            let scales = actions_tensor.get(1);
            let actions = sample(&means, &scales, device);
            Ok(actions)
        })?;
        let actions = tensor_to_actions(&actions_tensor.clip(0.0, 1.0));

        let prev_state = obs.copy();
        results = envs.step(&actions, false);
        rollout_buffer.insert_step(
            &prev_state,
            &results_to_state(&results, device),
            &actions_tensor,
            &results.results.iter().map(|r| r.1).collect::<Vec<_>>(),
            &results.results.iter().map(|r| r.2).collect::<Vec<_>>(),
        );
        monitor.log_metric(
            avg_reward_metric,
            results.results.iter().map(|r| r.1).sum::<f32>() / args.env_count as f32,
        );

        if step % args.rollout_steps == 0 {
            envs.do_bg_work();
            // Train networks
            p_net.module.set_train();
            v_net.module.set_train();
            p_net_old.module.set_train();
            copy_params(&p_net.vs, &mut p_net_old.vs);
            let mut last_v_loss = 0.0;
            let mut last_p_loss = 0.0;

            for _ in 0..args.train_iterations {
                let batches = rollout_buffer.samples(
                    args.train_batch_size as u32,
                    args.discount,
                    args.lambda,
                    &v_net,
                );

                for (prev_states, _, actions, _, _, advantages, _) in &batches {
                    // Train policy network
                    let old_output = p_net_old.module.forward_ts(&[prev_states]).unwrap();
                    let old_log_probs = action_log_probs_cont(
                        actions,
                        &old_output.get(0),
                        &old_output.get(1),
                        device,
                    );
                    p_opt.zero_grad();
                    let new_output = p_net.module.forward_ts(&[prev_states]).unwrap();
                    let new_log_probs = action_log_probs_cont(
                        actions,
                        &new_output.get(0),
                        &new_output.get(1),
                        device,
                    );
                    let term1 = (&new_log_probs - &old_log_probs).exp() * advantages;
                    let term2: tch::Tensor = (1.0 + args.epsilon * advantages.sign()) * advantages;
                    let p_loss = -(term1.min_other(&term2).mean(tch::Kind::Float));
                    last_p_loss = p_loss.double_value(&[]) as f32;
                    p_loss.backward();
                    p_opt.step();
                }
            }

            for _ in 0..args.train_iterations {
                let batches = rollout_buffer.samples(
                    args.train_batch_size as u32,
                    args.discount,
                    args.lambda,
                    &v_net,
                );

                for (prev_states, _, _, _, rewards_to_go, _, _) in &batches {
                    // Train value network
                    v_opt.zero_grad();
                    let diff = v_net.module.forward_ts(&[prev_states]).unwrap()
                        - rewards_to_go.unsqueeze(1);
                    let v_loss = (&diff * &diff).mean(tch::Kind::Float);
                    last_v_loss = v_loss.double_value(&[]) as f32;
                    v_loss.backward();
                    v_opt.step();
                }
            }

            monitor.log_metric(v_loss_metric, last_v_loss);
            monitor.log_metric(p_loss_metric, last_p_loss);

            p_net.module.set_eval();
            v_net.module.set_eval();

            rollout_buffer.clear();

            // Eval after a certain number of rollouts
            if (step * args.rollout_steps) % args.rollouts_before_eval == 0 {
                envs.eval_mode();
                tch::no_grad(|| -> Result<(), anyhow::Error> {
                    // Evaluate the model's performance
                    let mut avg_reward = 0.0;
                    let mut pred_value = 0.0;
                    for _ in 0..args.max_env_steps {
                        let obs = results_to_state(&results, device);
                        let actions_tensor = &p_net.module.forward_ts(&[&obs])?;
                        let actions_tensor =
                            sample(&actions_tensor.get(0), &actions_tensor.get(1), device);
                        let actions = tensor_to_actions(&actions_tensor.clip(0.0, 1.0));
                        results = envs.step(&actions, true);
                        avg_reward += results.results.iter().map(|x| x.1).sum::<f32>();
                        pred_value += v_net
                            .module
                            .forward_ts(&[&obs])
                            .unwrap()
                            .sum(tch::Kind::Float)
                            .double_value(&[]) as f32;
                    }
                    monitor.log_metric(
                        eval_reward_metric,
                        avg_reward / ((args.max_env_steps * args.env_count as u32) as f32),
                    );
                    monitor.log_metric(
                        pred_value_metric,
                        pred_value / ((args.max_env_steps * args.env_count as u32) as f32),
                    );
                    Ok(())
                })?;
                envs.train_mode();
            }
        }
    }

    // envs.cleanup();
    v_net.module.save("temp/VNet.pt")?;
    p_net.module.save("temp/PNet.pt")?;
    monitor.stop();
    Ok(())
}
