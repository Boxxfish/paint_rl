//! A setup for training an agent on the cartpole environment.
use std::collections::HashMap;

use clap::Parser;
use indicatif::ProgressIterator;

use paint_gym::distributions::{self, Distribution};
use paint_gym::gym::{PaintAction, PaintGym, PaintStepResult, Pixel};
use paint_gym::model_utils::{cleanup_py, export_model, prep_py, TrainableModel};
use paint_gym::monitor::{BackendParams, Monitor, PlottersParams, WandbParams};
use paint_gym::rollout_buffer::RolloutBuffer;

use pyo3::prelude::*;
use rsrl::domains::Domain;
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
    #[arg(long, default_value_t = 16)]
    rollout_steps: u32,
    /// Number of rollouts to run through before evaluating the model.
    #[arg(long, default_value_t = 100)]
    rollouts_before_eval: u32,
    /// Maximum number of steps to take in the environment before failure.
    #[arg(long, default_value_t = 15)]
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
    /// Standard deviation of actions at start, decreases to 0 over time.
    #[arg(long, default_value_t = 0.5)]
    action_scale: f32,
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

#[derive(Copy, Clone)]
#[pyclass]
struct ModelParams {
    #[pyo3(get)]
    obs_dim: i64,
    #[pyo3(get)]
    act_dim: i64,
}

/// Converts results to tensor of states and dones.
pub fn results_to_state(
    results: &[rsrl::domains::Observation<Vec<f64>>],
    device: tch::Device,
) -> (tch::Tensor, Vec<bool>) {
    let mut obs_vec = Vec::new();
    let mut dones = Vec::new();
    for obs in results.iter() {
        let (obs, done) = match &obs {
            rsrl::domains::Observation::Full(s) => (
                tch::Tensor::of_slice(s).to_dtype(tch::Kind::Float, true, true),
                false,
            ),
            rsrl::domains::Observation::Partial(s) => (
                tch::Tensor::of_slice(s).to_dtype(tch::Kind::Float, true, true),
                false,
            ),
            rsrl::domains::Observation::Terminal(s) => (
                tch::Tensor::of_slice(s).to_dtype(tch::Kind::Float, true, true),
                true,
            ),
        };
        obs_vec.push(obs);
        dones.push(done);
    }
    (tch::Tensor::stack(&obs_vec, 0).to(device), dones)
}

/// Converts tensors to actions.
pub fn tensor_to_actions(
    tensor: &tch::Tensor,
) -> Vec<rsrl::domains::Action<rsrl::domains::CartPole>> {
    let mut actions = Vec::new();
    for idx in 0..tensor.size2().unwrap().1 {
        actions.push(tensor.get(0).int64_value(&[idx]) as usize);
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

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let device = if args.cuda {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };

    // Load models
    pyo3::prepare_freethreaded_python();
    cleanup_py();
    let model_file = "cartpole_models";
    let obs_dim = 4;
    let act_dim = 2;
    let model_params = ModelParams { obs_dim, act_dim };
    Python::with_gil(|py| {
        prep_py(py);
        let batch_size = 10;
        export_model(
            py,
            model_file,
            "VNet",
            model_params,
            &[&[batch_size, obs_dim as u32]],
            args.cuda,
        );
        export_model(
            py,
            model_file,
            "PNet",
            model_params,
            &[&[batch_size, obs_dim as u32]],
            args.cuda,
        );
    });

    let (v_path, p_path) = if args.cont {
        ("temp/VNet.pt", "temp/PNet.pt")
    } else {
        ("temp/VNet.ptc", "temp/PNet.ptc")
    };
    let mut v_net = TrainableModel::load(v_path, device);
    let (mut p_net, mut p_net_old) = TrainableModel::load2(p_path, device);
    let mut v_opt = tch::nn::Adam::default().build(&v_net.vs, args.v_lr)?;
    let mut p_opt = tch::nn::Adam::default().build(&p_net.vs, args.p_lr)?;

    // Plotting stuff
    let mut monitor = Monitor::new(
        500,
        BackendParams::Plotters(PlottersParams {
            output_path: "temp/avg_rewards.png".into(),
        }),
    );
    let eval_reward_metric = monitor.add_metric("eval_reward");
    let v_loss_metric = monitor.add_metric("v_loss");
    let p_loss_metric = monitor.add_metric("p_loss");
    let avg_reward_metric = monitor.add_metric("avg_reward");
    let pred_value_metric = monitor.add_metric("pred_value");
    monitor.init();

    let mut envs: Vec<rsrl::domains::CartPole> = (0..args.env_count)
        .map(|_| rsrl::domains::CartPole::default())
        .collect();
    let mut results: (Vec<_>, Vec<_>) = (
        envs.iter().map(|e| e.emit()).collect(),
        (0..args.env_count).map(|_| 0.0).collect(),
    );
    let mut rollout_buffer = RolloutBuffer::new(
        &[obs_dim],
        &[],
        tch::Kind::Int,
        args.env_count as usize,
        args.rollout_steps as usize,
        device,
    );
    for step in (0..args.steps).progress() {
        monitor.add_step();

        let (obs, dones) = results_to_state(&results.0, device);
        for (env, &done) in envs.iter_mut().zip(&dones) {
            if done {
                *env = rsrl::domains::CartPole::default();
            }
        }
        let actions_tensor = tch::no_grad(|| -> Result<tch::Tensor, anyhow::Error> {
            let log_probs = p_net.module.forward_ts(&[&obs])?.to(device);
            let actions = distributions::Categorical::new(log_probs).sample(&[1]);
            Ok(actions)
        })?;
        let actions = tensor_to_actions(&actions_tensor);

        results = envs
            .iter_mut()
            .zip(&actions)
            .map(|(e, action)| e.step(action))
            .unzip();
        let rewards: Vec<_> = results.1.iter().map(|&x| x as f32).collect();
        rollout_buffer.insert_step(&obs, &actions_tensor.squeeze(), &rewards, &dones);
        monitor.log_metric(
            avg_reward_metric,
            rewards.iter().sum::<f32>() / args.env_count as f32,
        );

        if step % args.rollout_steps == 0 {
            // Add final observation
            rollout_buffer.insert_final_step(&results_to_state(&results.0, device).0);

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

                for (prev_states, _, actions, _, rewards_to_go, advantages, _) in &batches {
                    // Train policy network
                    let old_log_probs = p_net_old.module.forward_ts(&[prev_states]).unwrap();
                    let old_action_log_probs =
                        distributions::Categorical::new(old_log_probs).log_prob(actions.copy());
                    p_opt.zero_grad();
                    let new_log_probs = p_net.module.forward_ts(&[prev_states]).unwrap();
                    let new_action_log_probs =
                        distributions::Categorical::new(new_log_probs).log_prob(actions.copy());
                    let term1 = (&new_action_log_probs - &old_action_log_probs).exp() * advantages;
                    let term2: tch::Tensor = (1.0 + args.epsilon * advantages.sign()) * advantages;
                    let p_loss = -(term1.min_other(&term2).mean(tch::Kind::Float));
                    last_p_loss = p_loss.double_value(&[]) as f32;
                    p_loss.backward();
                    p_opt.step();

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
                tch::no_grad(|| -> Result<(), anyhow::Error> {
                    // Evaluate the model's performance
                    let mut avg_reward = 0.0_f32;
                    let mut pred_value = 0.0;
                    for _ in 0..args.max_env_steps {
                        let obs = results_to_state(&results.0, device).0;
                        let log_probs = p_net.module.forward_ts(&[&obs])?;
                        let actions_tensor =
                            distributions::Categorical::new(log_probs).sample(&[1]);
                        let actions = tensor_to_actions(&actions_tensor);
                        results = envs
                            .iter_mut()
                            .zip(&actions)
                            .map(|(e, action)| e.step(action))
                            .unzip();
                        avg_reward += results.1.iter().copied().sum::<f64>() as f32;
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
            }
        }
    }

    // envs.cleanup();
    v_net.module.save("temp/VNet.pt")?;
    p_net.module.save("temp/PNet.pt")?;
    monitor.stop();
    Ok(())
}
