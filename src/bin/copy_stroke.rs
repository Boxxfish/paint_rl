use indicatif::ProgressIterator;
use paint_gym::gym::{PaintAction, PaintGym, PaintStepResult, Pixel};
use paint_gym::model_utils::{cleanup_py, export_model, prep_py, TrainableModel};
use plotters::prelude::*;

use pyo3::prelude::*;
use rand::Rng;
use tch::nn::OptimizerConfig;

//
// A setup for training an agent to copy strokes from an image.
// The agent reads in the current canvas and reference image, and outputs
// brush strokes.
// The environment stops once a difference threshold is reached, or after the time limit is hit.
// The difference at each time step is given as the reward.
//

const STEPS: u32 = 100000;
const STEPS_BEFORE_TRAINING: u32 = 500;
const STEPS_BEFORE_POLICY: u32 = 1000;
const MAX_ENV_STEPS: u32 = 50;
const EVAL_RUNS: u32 = 4;
const TRAIN_ITERATIONS: u32 = 10;
const TRAIN_BATCH_SIZE: usize = 64;
const STEPS_BEFORE_EVAL: u32 = STEPS_BEFORE_TRAINING * 10;
const STEPS_BEFORE_PLOTTING: u32 = STEPS_BEFORE_EVAL;
const AVG_REWARD_OUTPUT: &str = "temp/avg_rewards.png";
const POLYAK: f32 = 0.95;
const DISCOUNT: f32 = 0.95;
const BUFFER_CAPACITY: usize = 1000;
const FAILURE_REWARD: i32 = -10;
const EPSILON: f32 = 0.1;

// Canvas and reference pixels
const IMG_SIZE: u32 = 8;
const IMG_CHANNELS: u32 = 6;
// Start (x, y), end (x, y)
const ACTION_DIM: u32 = 4;

#[pyclass]
struct QNetParams {
    #[pyo3(get)]
    img_size: u32,
    #[pyo3(get)]
    action_dim: u32,
}

#[pyclass]
struct PNetParams {
    #[pyo3(get)]
    img_size: u32,
    #[pyo3(get)]
    action_dim: u32,
}

/// Stores transitions and generates mini batches.
struct ReplayBuffer {
    /// Previous state, state, action, reward, done
    pub prev_states: tch::Tensor,
    pub states: tch::Tensor,
    pub actions: tch::Tensor,
    pub rewards: tch::Tensor,
    pub dones: tch::Tensor,
    pub capacity: usize,
    pub next: usize,
    pub filled: bool,
}

impl ReplayBuffer {
    /// Creates a new instance of a replay buffer.
    pub fn new(capacity: usize) -> Self {
        let k = tch::Kind::Float;
        let d = tch::Device::Cpu;
        Self {
            capacity,
            next: 0,
            prev_states: tch::Tensor::zeros(
                &[
                    capacity as i64,
                    IMG_CHANNELS as i64,
                    IMG_SIZE as i64,
                    IMG_SIZE as i64,
                ],
                (k, d),
            )
            .requires_grad_(false),
            states: tch::Tensor::zeros(
                &[
                    capacity as i64,
                    IMG_CHANNELS as i64,
                    IMG_SIZE as i64,
                    IMG_SIZE as i64,
                ],
                (k, d),
            )
            .requires_grad_(false),
            actions: tch::Tensor::zeros(&[capacity as i64, ACTION_DIM as i64], (k, d))
                .requires_grad_(false),
            rewards: tch::Tensor::zeros(&[capacity as i64, 1], (k, d)).requires_grad_(false),
            dones: tch::Tensor::zeros(&[capacity as i64, 1], (k, d)).requires_grad_(false),
            filled: false,
        }
    }

    /// Inserts an element into the buffer.
    /// If the max capacity has been reached, the buffer wraps around.
    pub fn insert(
        &mut self,
        prev_state: &tch::Tensor,
        state: &tch::Tensor,
        action: &tch::Tensor,
        reward: f32,
        done: bool,
    ) {
        self.prev_states.get(self.next as _).copy_(prev_state);
        self.states.get(self.next as _).copy_(state);
        self.actions.get(self.next as _).copy_(action);
        self.rewards
            .get(self.next as _)
            .copy_(&tch::Tensor::of_slice(&[reward]));
        self.dones
            .get(self.next as _)
            .copy_(&tch::Tensor::of_slice(&[done as u8 as f32]));
        if self.next == self.capacity - 1 {
            self.filled = true;
        }
        self.next = (self.next + 1) % self.capacity;
    }

    /// Generates a mini batch of experience.
    pub fn sample(
        &self,
        batch_size: usize,
    ) -> (
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
    ) {
        let indices = tch::Tensor::randint(
            self.capacity as i64,
            &[batch_size as i64],
            (tch::Kind::Int, tch::Device::Cpu),
        );
        (
            self.prev_states.index_select(0, &indices),
            self.states.index_select(0, &indices),
            self.actions.index_select(0, &indices),
            self.rewards.index_select(0, &indices),
            self.dones.index_select(0, &indices),
        )
    }

    /// Returns true if the buffer is full.
    pub fn is_full(&self) -> bool {
        self.filled
    }
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
pub fn results_to_state(results: &PaintStepResult) -> tch::Tensor {
    let mut obs_vec = Vec::new();
    for (state, _, _) in &results.results {
        let canvas_channels = pixels_to_tensor(&state.canvas, IMG_SIZE as i64);
        let ref_channels = pixels_to_tensor(&state.reference, IMG_SIZE as i64);
        obs_vec.push(tch::Tensor::cat(&[canvas_channels, ref_channels], 0));
    }
    tch::Tensor::stack(&obs_vec, 0)
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
    // Plotting stuff
    let avg_reward_output_path = std::path::Path::new(AVG_REWARD_OUTPUT);
    if avg_reward_output_path.exists() {
        std::fs::remove_file(AVG_REWARD_OUTPUT)?;
    }
    let root = BitMapBackend::new(AVG_REWARD_OUTPUT, (640, 480)).into_drawing_area();
    let mut avg_rewards: Vec<f32> = Vec::with_capacity((STEPS / STEPS_BEFORE_PLOTTING) as usize);

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
                &[batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE],
                &[batch_size, ACTION_DIM],
            ],
        );
        export_model(
            py,
            "copy_stroke_ddpg",
            "PNet",
            PNetParams {
                img_size: IMG_SIZE,
                action_dim: ACTION_DIM,
            },
            &[&[batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE]],
        );
    });
    cleanup_py();

    let mut q_net = TrainableModel::load("temp/QNet.ptc");
    let mut p_net = TrainableModel::load("temp/PNet.ptc");
    q_net.module.set_eval();
    p_net.module.set_eval();
    let mut q_target = TrainableModel::load("temp/QNet.ptc");
    let mut p_target = TrainableModel::load("temp/PNet.ptc");
    q_target.module.set_eval();
    p_target.module.set_eval();
    let mut q_opt = tch::nn::Adam::default().build(&q_net.vs, 0.001)?;
    let mut p_opt = tch::nn::Adam::default().build(&p_net.vs, 0.003)?;

    let mut rng = rand::thread_rng();
    let (mut envs, mut results) = PaintGym::init(1, IMG_SIZE, FAILURE_REWARD, MAX_ENV_STEPS, false);
    let mut replay_buffer = ReplayBuffer::new(BUFFER_CAPACITY);
    for step in (0..STEPS).progress() {
        let actions_tensor = tch::no_grad(|| -> Result<tch::Tensor, anyhow::Error> {
            // Execute random policy for first couple steps
            if step < STEPS_BEFORE_POLICY {
                let mut action_tensors = Vec::new();
                for _ in 0..envs.num_envs {
                    let action_tensor = (1.0 / envs.canvas_size as f32)
                        * tch::Tensor::of_slice(&[
                            rng.gen_range(0..envs.canvas_size) as f32,
                            rng.gen_range(0..envs.canvas_size) as f32,
                            rng.gen_range(0..envs.canvas_size) as f32,
                            rng.gen_range(0..envs.canvas_size) as f32,
                        ])
                        .unsqueeze(0);
                    action_tensors.push(action_tensor);
                }
                Ok(tch::Tensor::cat(&action_tensors, 0))
            }
            // Use learned policy afterwards
            else {
                let obs = results_to_state(&results);
                let noise = tch::Tensor::ones(
                    &[envs.num_envs as i64, ACTION_DIM as i64],
                    (tch::Kind::Float, tch::Device::Cpu),
                )
                .normal_(0.0, EPSILON as f64);
                Ok((p_net.module.forward_ts(&[obs])? + noise).clamp(0.0, 1.0))
            }
        })?;
        let actions = tensor_to_actions(&actions_tensor);

        let prev_state = results_to_state(&results);
        results = envs.step(&actions, false);
        replay_buffer.insert(
            &prev_state.squeeze(),
            &results_to_state(&results).squeeze(),
            &actions_tensor.squeeze(),
            results.results[0].1,
            results.results[0].2,
        );

        if replay_buffer.is_full() {
            if step % STEPS_BEFORE_TRAINING == 0 {
                envs.do_bg_work();
                q_net.module.set_train();
                p_net.module.set_train();

                for _ in 0..TRAIN_ITERATIONS {
                    let (prev_states, states, actions, rewards, dones) =
                        replay_buffer.sample(TRAIN_BATCH_SIZE);

                    // Perform value optimization
                    let mut targets: tch::Tensor = rewards
                        + DISCOUNT
                            * (1.0 - dones)
                            * q_target
                                .module
                                .forward_ts(&[&states, &p_target.module.forward_ts(&[&states])?])?;
                    targets = targets.detach();
                    let diff = &targets - q_net.module.forward_ts(&[&prev_states, &actions])?;
                    let q_loss =
                        (1.0 / TRAIN_BATCH_SIZE as f32) * (&diff * &diff).sum(tch::Kind::Float);

                    q_opt.zero_grad();
                    q_loss.backward();
                    q_opt.step();

                    // Perform policy optimization
                    let p_loss = -(1.0 / TRAIN_BATCH_SIZE as f32)
                        * q_net
                            .module
                            .forward_ts(&[
                                &prev_states,
                                &p_net.module.forward_ts(&[&prev_states])?,
                            ])?
                            .sum(tch::Kind::Float);

                    p_opt.zero_grad();
                    p_loss.backward();
                    p_opt.step();

                    // Move targets
                    polyak_avg(&q_net.vs, &mut q_target.vs, POLYAK);
                    polyak_avg(&p_net.vs, &mut p_target.vs, POLYAK);
                }

                q_net.module.set_eval();
                p_net.module.set_eval();
            }

            if step % STEPS_BEFORE_EVAL == 0 {
                tch::no_grad(|| -> Result<(), anyhow::Error> {
                    // Evaluate the model's performance
                    envs.eval_mode();
                    let mut avg_rewards_now = Vec::new();
                    for _ in 0..EVAL_RUNS {
                        let mut avg_reward = 0.0;
                        for _ in 0..MAX_ENV_STEPS {
                            let obs = results_to_state(&results);
                            let actions_tensor = p_net.module.forward_ts(&[obs])?;
                            let actions = tensor_to_actions(&actions_tensor);
                            results = envs.step(&actions, true);
                            avg_reward += results.results[0].1;
                            if results.results[0].2 {
                                break;
                            }
                        }
                        avg_rewards_now.push(avg_reward);
                    }
                    avg_rewards.push(
                        avg_rewards_now.iter().sum::<f32>() / ((MAX_ENV_STEPS * EVAL_RUNS) as f32),
                    );
                    envs.train_mode();
                    Ok(())
                })?;
            }

            if step % STEPS_BEFORE_PLOTTING == 0 {
                // Plot results
                let min_reward = *avg_rewards.iter().min_by(|&&x, &y| x.total_cmp(y)).unwrap();
                let max_reward = *avg_rewards.iter().max_by(|&&x, &y| x.total_cmp(y)).unwrap();
                root.fill(&WHITE)?;
                let mut chart = ChartBuilder::on(&root)
                    .margin(100)
                    .x_label_area_size(30)
                    .y_label_area_size(30)
                    .build_cartesian_2d(0..avg_rewards.len(), min_reward..max_reward)?;
                chart.configure_mesh().draw()?;
                chart.draw_series(LineSeries::new(
                    avg_rewards.iter().enumerate().map(|(x, y)| (x, *y)),
                    RED,
                ))?;
                root.present()?;
            }
        }
    }

    envs.cleanup();
    q_net.module.save("temp/QNet.pt")?;
    p_net.module.save("temp/PNet.pt")?;
    Ok(())
}
