#![allow(dead_code)]

/// A gym interface for a painting program.
/// Vectorized by default.
struct PaintGym {
    num_envs: u32,
    /// The width and height of each canvas.
    canvas_size: u32,
    /// All canvases, in one contiguous array.
    canvases: Vec<(u8, u8, u8)>,
}

impl PaintGym {
    /// Loads all environments and data.
    /// This may take a while, at least a couple seconds.
    pub fn init(num_envs: u32, canvas_size: u32) -> Self {
        let canvases = (0..(canvas_size * canvas_size * num_envs))
            .map(|_| (255, 255, 255))
            .collect();
        Self {
            num_envs,
            canvas_size,
            canvases,
        }
    }

    /// Performs one step through all environments and performs
    /// the actions given. No per-environment `done` flag exists,
    /// since the environment will auto reset itself if it detects some
    /// end criteria.
    pub fn step(&mut self, _actions: &[PaintAction]) -> PaintStepResult {
        PaintStepResult {
            results: (0..self.num_envs)
                .map(|_| (PaintState {}, PaintState {}, 0.0))
                .collect(),
        }
    }

    /// Performs some work as the agent is trained.
    /// Since this may potentially take a long time, this function
    /// ensures that period doesn't go to waste by doing things like
    /// perform IO bound operations.
    pub fn do_bg_work(&mut self) {}
}

/// A representation of the canvas.
struct PaintState {}

/// An action that can be performed on the canvas.
struct PaintAction {}

/// The result of performing a step.
struct PaintStepResult {
    /// (Previous state, current state, reward)
    pub results: Vec<(PaintState, PaintState, f32)>,
}

const STEPS_BEFORE_TRAINING: u32 = 200;

fn main() {
    let mut envs = PaintGym::init(1, 32);
    for step in 0..1000 {
        let _results = envs.step(&[PaintAction {}]);

        if (step + 1) % STEPS_BEFORE_TRAINING == 0 {
            envs.do_bg_work();
        }
    }
}
