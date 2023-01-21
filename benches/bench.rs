use criterion::{criterion_group, criterion_main, Criterion};
use pyo3::{pyclass, Python};
use rand::prelude::*;

use paint_gym::{
    gym::*,
    model_utils::{cleanup_py, export_model, prep_py, TrainableModel},
    rollout_buffer::RolloutBuffer,
};

const OBS_SIZE: i64 = 256 * 256 * 3;
const ACTION_SIZE: i64 = 4;
const ROLLOUT_STEPS: usize = 16;

fn run_rollout_buffer(env_count: usize, v_net: &TrainableModel) {
    let mut rng = rand::thread_rng();
    let device = tch::Device::Cpu;
    #[allow(unused_variables)]
    let guard = tch::no_grad_guard();
    let mut buffer = RolloutBuffer::new(
        &[OBS_SIZE],
        &[ACTION_SIZE],
        tch::Kind::Float,
        env_count,
        ROLLOUT_STEPS,
        device,
    );

    for _ in 0..ROLLOUT_STEPS {
        let states = tch::Tensor::rand(&[env_count as i64, OBS_SIZE], (tch::Kind::Float, device));
        let actions =
            tch::Tensor::rand(&[env_count as i64, ACTION_SIZE], (tch::Kind::Float, device));
        let rewards: Vec<_> = (0..env_count).map(|_| rng.gen()).collect();
        let dones: Vec<_> = (0..env_count).map(|_| rng.gen()).collect();
        buffer.insert_step(&states, &actions, &rewards, &dones);
    }

    #[allow(unused_variables)]
    let batches = buffer.samples(((ROLLOUT_STEPS * env_count) / 4) as u32, 0.9, 0.9, v_net);
}

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

#[pyclass]
struct VNetParams {
    #[pyo3(get)]
    obs_size: i64,
}

fn rollout_buffer_bench(c: &mut Criterion) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        prep_py(py);
        let batch_size = 10;
        export_model(
            py,
            "bench_models",
            "VNet",
            VNetParams { obs_size: OBS_SIZE },
            &[&[batch_size, OBS_SIZE as u32]],
            false,
        );
    });
    cleanup_py();

    let v_net = TrainableModel::load("temp/VNet.ptc", tch::Device::Cpu);

    let mut group = c.benchmark_group("rollout_buffer");
    group.sample_size(30);
    group.bench_function("envs: 32", |b| b.iter(|| run_rollout_buffer(32, &v_net)));
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

criterion_group!(benches, rollout_buffer_bench, copy_stroke_bench);
criterion_main!(benches);
