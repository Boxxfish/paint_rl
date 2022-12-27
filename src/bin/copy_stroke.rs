/// A setup for training an agent to copy strokes from an image.
use paint_gym::gym::{PaintAction, PaintGym, Pixel};

use rand::Rng;

const STEPS_BEFORE_TRAINING: u32 = 200;

fn main() {
    let mut envs = PaintGym::init(4, 256, true);
    for step in 0..1000 {
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

        let _results = envs.step(&actions, true);

        if (step + 1) % STEPS_BEFORE_TRAINING == 0 {
            envs.do_bg_work();
        }
    }

    envs.cleanup();
}
