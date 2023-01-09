use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use paint_gym::gym::*;

const TOTAL_STEPS: u32 = 100;
const CANVAS_SIZE: u32 = 256;
const STEPS_BEFORE_TRAINING: u32 = 10;

fn run_copy_stroke(env_count: u32, inf_time_millis: u64, train_time_millis: u64) {
    let (mut envs, _) = PaintGym::init(env_count, CANVAS_SIZE, 0.0, 50, false);
    for step in 0..TOTAL_STEPS {
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
        std::thread::sleep(std::time::Duration::from_millis(inf_time_millis));

        envs.step(&actions, true);

        if (step + 1) % STEPS_BEFORE_TRAINING == 0 {
            envs.do_bg_work();
            std::thread::sleep(std::time::Duration::from_millis(train_time_millis));
        }
    }
}

fn copy_stroke_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_stroke");
    group.sample_size(10);
    group.bench_function("envs: 1, inference time: 0.02s, train time: 0.1s", |b| {
        b.iter(|| run_copy_stroke(1, 20, 100))
    });
    group.bench_function("envs: 8, inference time: 0.02s, train time: 0.1s", |b| {
        b.iter(|| run_copy_stroke(8, 20, 100))
    });
    group.bench_function("envs: 16, inference time: 0.02s, train time: 0.1s", |b| {
        b.iter(|| run_copy_stroke(16, 20, 100))
    });
    group.bench_function("envs: 32, inference time: 0.02s, train time: 0.1s", |b| {
        b.iter(|| run_copy_stroke(32, 20, 100))
    });
}

criterion_group!(benches, copy_stroke_bench);
criterion_main!(benches);
