use indicatif::ProgressIterator;
use paint_gym::gym::{PaintAction, PaintGym, Pixel};
use plotters::prelude::*;
use rand::Rng;

///
/// A setup for training an agent to copy strokes from an image.
///

const STEPS: u32 = 10000;
const STEPS_BEFORE_TRAINING: u32 = 200;
const EVAL_STEPS: u32 = 100;
const EVAL_RUNS: u32 = 4;
const STEPS_BEFORE_EVAL: u32 = STEPS_BEFORE_TRAINING;
const STEPS_BEFORE_PLOTTING: u32 = STEPS_BEFORE_EVAL * 10;
const AVG_REWARD_OUTPUT: &str = "avg_rewards.png";

fn main() {
    // Plotting stuff
    let avg_reward_output_path = std::path::Path::new(AVG_REWARD_OUTPUT);
    if avg_reward_output_path.exists() {
        std::fs::remove_file(AVG_REWARD_OUTPUT).unwrap();
    }
    let root = BitMapBackend::new(AVG_REWARD_OUTPUT, (640, 480)).into_drawing_area();
    let mut avg_rewards = Vec::with_capacity((STEPS / STEPS_BEFORE_PLOTTING) as usize);

    let mut envs = PaintGym::init(4, 256, false);
    for step in (0..STEPS).progress() {
        // Perform random policy
        let mut rng = rand::thread_rng();
        let mut actions = Vec::new();
        for _ in 0..envs.num_envs {
            let start = Pixel::new(
                rng.gen_range(0..envs.canvas_size),
                rng.gen_range(0..envs.canvas_size),
            );
            let end = Pixel::new(
                rng.gen_range(0..envs.canvas_size),
                rng.gen_range(0..envs.canvas_size),
            );
            actions.push(PaintAction { start, end });
        }

        let _results = envs.step(&actions, false);

        if (step + 1) % STEPS_BEFORE_TRAINING == 0 {
            envs.do_bg_work();

            // TODO: Train the model
        }

        if (step + 1) % STEPS_BEFORE_PLOTTING == 0 {
            // Evaluate the model's performance
            envs.eval_mode();
            let mut avg_rewards_now = Vec::new();
            for _ in 0..EVAL_RUNS {
                let mut avg_reward = 0.0;
                for _ in 0..EVAL_STEPS {
                    let mut actions = Vec::new();
                    for _ in 0..envs.num_envs {
                        let start = Pixel::new(
                            rng.gen_range(0..envs.canvas_size),
                            rng.gen_range(0..envs.canvas_size),
                        );
                        let end = Pixel::new(
                            rng.gen_range(0..envs.canvas_size),
                            rng.gen_range(0..envs.canvas_size),
                        );
                        actions.push(PaintAction { start, end });
                    }
                    let results = envs.step(&actions, true);
                    avg_reward += results.results[0].2;
                }
                avg_rewards_now.push(avg_reward);
            }
            avg_rewards
                .push(avg_rewards_now.iter().sum::<f32>() / ((EVAL_STEPS * EVAL_RUNS) as f32));
            envs.train_mode();
        }

        if (step + 1) % STEPS_BEFORE_PLOTTING == 0 {
            // Plot results
            let min_reward = *avg_rewards.iter().min_by(|&&x, &y| x.total_cmp(y)).unwrap();
            let max_reward = *avg_rewards.iter().max_by(|&&x, &y| x.total_cmp(y)).unwrap();
            root.fill(&WHITE).unwrap();
            let mut chart = ChartBuilder::on(&root)
                .margin(100)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(0..avg_rewards.len(), min_reward..max_reward)
                .unwrap();
            chart.configure_mesh().draw().unwrap();
            chart
                .draw_series(LineSeries::new(
                    avg_rewards.iter().enumerate().map(|(x, y)| (x, *y)),
                    RED,
                ))
                .unwrap();
            root.present().unwrap();
        }
    }

    envs.cleanup();
}
