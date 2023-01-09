use clap::Parser;
use indicatif::ProgressIterator;
use paint_gym::gym::{PaintAction, PaintGym, PaintStepResult, Pixel};
use paint_gym::model_utils::{cleanup_py, export_model, prep_py, TrainableModel};
use paint_gym::monitor::Monitor;
use paint_gym::replay_buffer::ReplayBuffer;
use pyo3::prelude::*;
use tch::nn::OptimizerConfig;

//
// A setup for training an agent to copy strokes from an image.
// The agent reads in the current canvas and reference image, and outputs
// brush strokes.
// The environment stops once a difference threshold is reached, or after the time limit is hit.
// The difference at each time step is given as the reward.
//

#[derive(Parser, Debug)]
struct Args {
    /// Number of environments to use. This many samples can be collected in one step.
    #[arg(long, default_value_t = 8)]
    env_count: u64,
    /// Number of steps to run through.
    #[arg(long, default_value_t = 10000)]
    steps: u32,
    /// Number of samples to collect before training, after the replay buffer is full.
    #[arg(long, default_value_t = 512)]
    samples_before_training: u32,
    /// Number of samples to collect before using a non-random policy.
    #[arg(long, default_value_t = 1024)]
    samples_before_policy: u32,
    /// Number of samples to collect before evaluating the model.
    #[arg(long, default_value_t = 2048)]
    samples_before_eval: u32,
    /// Maximum number of steps to take in the environment before failure.
    #[arg(long, default_value_t = 50)]
    max_env_steps: u32,
    /// Reward given if the maximum number of steps is reached. Should be negative.
    #[arg(long, default_value_t = -1.0, allow_hyphen_values(true))]
    failure_reward: f32,
    /// Number of iterations during training.
    #[arg(long, default_value_t = 10)]
    train_iterations: u32,
    /// Minibatch size during training.
    #[arg(long, default_value_t = 64)]
    train_batch_size: usize,
    /// Polyak factor for updating target networks.
    #[arg(long, default_value_t = 0.95)]
    polyak: f32,
    /// Discount factor when propagating rewards.
    #[arg(long, default_value_t = 0.95)]
    discount: f32,
    /// Maximum number of transitions that can be stored in the replay buffer.
    #[arg(long, default_value_t = 1000)]
    buffer_capacity: usize,
    /// Standard deviation of noise applied to policy. This decays as time goes on.
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
struct QNetParams {
    #[pyo3(get)]
    img_size: i64,
    #[pyo3(get)]
    action_dim: i64,
}

#[pyclass]
struct PNetParams {
    #[pyo3(get)]
    img_size: i64,
    #[pyo3(get)]
    action_dim: i64,
}

/// Performs polyak averaging between two networks.
/// When `p` is 0, `dest`'s weights are used. When `p` is 1, `src`'s weights are used.
/// This modifies `dest`.
fn polyak_avg(src: &tch::nn::VarStore, dest: &mut tch::nn::VarStore, p: f32) {
    tch::no_grad(|| {
        for (dest, src) in dest
            .trainable_variables()
            .iter_mut()
            .zip(src.trainable_variables().iter())
        {
            dest.copy_(&(p * src + (1.0 - p) * &*dest));
        }
    })
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

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let device = if args.cuda {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };
    let mut epsilon = args.epsilon;

    // Plotting stuff
    let avg_reward_output_path = std::path::Path::new(AVG_REWARD_OUTPUT);
    if avg_reward_output_path.exists() {
        std::fs::remove_file(AVG_REWARD_OUTPUT)?;
    }
    let mut monitor = Monitor::new(100, AVG_REWARD_OUTPUT);
    let eval_reward_metric = monitor.add_metric("Eval Reward");
    let q_loss_metric = monitor.add_metric("Q Loss");
    let p_loss_metric = monitor.add_metric("P Loss");
    let avg_reward_metric = monitor.add_metric("Avg. Reward");
    monitor.init();

    // Load models
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        prep_py(py);
        let batch_size = 10;
        export_model(
            py,
            "copy_stroke_ddpg",
            "QNet",
            QNetParams {
                img_size: IMG_SIZE,
                action_dim: ACTION_DIM,
            },
            &[
                &[
                    batch_size,
                    IMG_CHANNELS as u32,
                    IMG_SIZE as u32,
                    IMG_SIZE as u32,
                ],
                &[batch_size, ACTION_DIM as u32],
            ],
            args.cuda,
        );
        export_model(
            py,
            "copy_stroke_ddpg",
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

    let (q_path, p_path) = if args.cont {
        ("temp/QNet.pt", "temp/PNet.pt")
    } else {
        ("temp/QNet.ptc", "temp/PNet.ptc")
    };
    let mut q_net = TrainableModel::load(q_path, device);
    let mut p_net = TrainableModel::load(p_path, device);
    q_net.module.set_eval();
    p_net.module.set_eval();
    let mut q_target = TrainableModel::load(q_path, device);
    let mut p_target = TrainableModel::load(p_path, device);
    q_target.module.set_eval();
    p_target.module.set_eval();
    let mut q_opt = tch::nn::Adam::default().build(&q_net.vs, 0.001)?;
    let mut p_opt = tch::nn::Adam::default().build(&p_net.vs, 0.003)?;

    let (mut envs, mut results) = PaintGym::init(
        args.env_count as u32,
        IMG_SIZE as u32,
        args.failure_reward,
        args.max_env_steps,
        args.render,
    );
    let mut replay_buffer = ReplayBuffer::new(
        &[IMG_CHANNELS, IMG_SIZE, IMG_SIZE],
        &[ACTION_DIM],
        args.buffer_capacity,
        device,
    );
    for step in (0..args.steps).progress() {
        monitor.add_step();
        let sample = step * args.env_count as u32;

        let actions_tensor = tch::no_grad(|| -> Result<tch::Tensor, anyhow::Error> {
            // Execute random policy for first couple steps
            if sample < args.samples_before_policy {
                Ok(tch::Tensor::rand(
                    &[envs.num_envs as i64, 4],
                    (tch::Kind::Float, device),
                ))
            }
            // Use learned policy afterwards
            else {
                let obs = results_to_state(&results, device);
                let mut actions_tensor = p_net.module.forward_ts(&[obs])?;
                if epsilon > 0.001 {
                    actions_tensor += tch::Tensor::ones(
                        &[envs.num_envs as i64, ACTION_DIM],
                        (tch::Kind::Float, device),
                    )
                    .normal_(0.0, epsilon);
                }
                Ok((actions_tensor).clamp(0.0, 1.0))
            }
        })?
        .to(device);
        let actions = tensor_to_actions(&actions_tensor);
        epsilon -= 1.0 / args.steps as f64;

        let prev_state = results_to_state(&results, device);
        results = envs.step(&actions, false);
        replay_buffer.insert_batch(
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

        if replay_buffer.is_full() {
            if sample % args.samples_before_training == 0 {
                envs.do_bg_work();
                q_net.module.set_train();
                p_net.module.set_train();

                for _ in 0..(args.train_iterations) {
                    let (prev_states, states, actions, rewards, dones) =
                        replay_buffer.sample(args.train_batch_size);
                    let prev_states = prev_states;
                    let states = states;
                    let actions = actions;
                    let rewards = rewards;
                    let dones = dones;

                    // Perform value optimization
                    let mut targets: tch::Tensor = rewards
                        + args.discount
                            * (1.0 - dones)
                            * q_target
                                .module
                                .forward_ts(&[&states, &p_target.module.forward_ts(&[&states])?])?;
                    targets = targets.detach();
                    let diff = &targets - q_net.module.forward_ts(&[&prev_states, &actions])?;
                    let q_loss = (&diff * &diff).mean(tch::Kind::Float);
                    monitor.log_metric(q_loss_metric, q_loss.double_value(&[]) as f32);

                    q_opt.zero_grad();
                    q_loss.backward();
                    q_opt.step();

                    // Perform policy optimization
                    let p_loss = -q_net
                        .module
                        .forward_ts(&[&prev_states, &p_net.module.forward_ts(&[&prev_states])?])?
                        .mean(tch::Kind::Float);
                    monitor.log_metric(p_loss_metric, p_loss.double_value(&[]) as f32);

                    p_opt.zero_grad();
                    p_loss.backward();
                    p_opt.step();

                    // Move targets
                    polyak_avg(&q_net.vs, &mut q_target.vs, args.polyak);
                    polyak_avg(&p_net.vs, &mut p_target.vs, args.polyak);
                }

                q_net.module.set_eval();
                p_net.module.set_eval();
            }

            if sample % args.samples_before_eval == 0 {
                envs.eval_mode();
                tch::no_grad(|| -> Result<(), anyhow::Error> {
                    // Evaluate the model's performance
                    let mut avg_reward = 0.0;
                    for _ in 0..args.max_env_steps {
                        let obs = results_to_state(&results, device);
                        let actions_tensor = p_net.module.forward_ts(&[obs])?;
                        let actions = tensor_to_actions(&actions_tensor);
                        results = envs.step(&actions, true);
                        avg_reward += results.results.iter().map(|x| x.1).sum::<f32>();
                    }
                    monitor.log_metric(
                        eval_reward_metric,
                        avg_reward / ((args.max_env_steps * args.env_count as u32) as f32),
                    );
                    Ok(())
                })?;
                envs.train_mode();
            }
        }
    }

    envs.cleanup();
    q_net.module.save("temp/QNet.pt")?;
    p_net.module.save("temp/PNet.pt")?;
    monitor.stop();
    Ok(())
}
