#![allow(dead_code)]

use minifb::{Window, WindowOptions};

const BRUSH_RADIUS: u32 = 2;

/// A gym interface for a painting program.
/// Vectorized by default.
struct PaintGym {
    num_envs: u32,
    /// The width and height of each canvas.
    canvas_size: u32,
    /// All canvases, in one contiguous array.
    canvases: Vec<(u8, u8, u8)>,
    window: Window,
}

impl PaintGym {
    /// Loads all environments and data.
    /// This may take a while, at least a couple seconds.
    pub fn init(num_envs: u32, canvas_size: u32) -> Self {
        let canvases = (0..(canvas_size * canvas_size * num_envs))
            .map(|_| (255, 255, 255))
            .collect();
        let scale = match canvas_size {
            0 => panic!("Cannot use a canvas size of 0!"),
            1..=8 => minifb::Scale::X32,
            9..=16 => minifb::Scale::X16,
            17..=32 => minifb::Scale::X8,
            33..=64 => minifb::Scale::X4,
            65..=128 => minifb::Scale::X2,
            _ => minifb::Scale::X1,
        };
        let mut window = Window::new(
            "PaintGym",
            canvas_size as usize,
            canvas_size as usize,
            WindowOptions {
                resize: false,
                scale,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));
        Self {
            num_envs,
            canvas_size,
            canvases,
            window,
        }
    }

    /// Performs one step through all environments and performs
    /// the actions given. No per-environment `done` flag exists,
    /// since the environment will auto reset itself if it detects some
    /// end criteria.
    pub fn step(&mut self, actions: &[PaintAction], render: bool) -> PaintStepResult {
        // Render the stroke
        for (env_id, action) in actions.iter().enumerate() {
            let canvas_offset = env_id * (self.canvas_size * self.canvas_size) as usize;

            // Draw line
            let dx = action.end.x as i32 - action.start.x as i32;
            let dy = action.end.y as i32 - action.start.y as i32;

            let (step_x, step_y, steps) = if dx.abs() > dy.abs() {
                (1.0, dy as f32 / dx as f32, dx.abs())
            } else {
                (dx as f32 / dy as f32, 1.0, dy.abs())
            };
            let mut start_x = action.start.x as f32;
            let mut start_y = action.start.y as f32;
            for _ in 0..steps {
                self.canvases[canvas_offset
                    + start_y.floor() as usize * self.canvas_size as usize
                    + start_x.floor() as usize] = (0, 0, 0);
                start_x += step_x;
                start_y += step_y;
            }
        }

        // If we're rendering to the screen, update the window
        if render {
            // Render the first env
            let pixel_buffer: Vec<u32> = self.canvases.as_slice()
                [..(self.canvas_size * self.canvas_size) as usize]
                .iter()
                .map(|(r, g, b)| ((*r as u32) << 16) + ((*g as u32) << 8) + (*b as u32))
                .collect();
            self.window
                .update_with_buffer(
                    &pixel_buffer,
                    self.canvas_size as usize,
                    self.canvas_size as usize,
                )
                .unwrap();
        }

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

    /// Performs cleanup work.
    pub fn cleanup(&mut self) {}
}

/// A representation of the canvas.
struct PaintState {}

/// Represents a pixel on screen.
/// (0, 0) is the top left.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Pixel {
    pub x: u32,
    pub y: u32,
}

impl Pixel {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

/// An action that can be performed on the canvas.
struct PaintAction {
    pub start: Pixel,
    pub end: Pixel,
}

/// The result of performing a step.
struct PaintStepResult {
    /// (Previous state, current state, reward)
    pub results: Vec<(PaintState, PaintState, f32)>,
}

const STEPS_BEFORE_TRAINING: u32 = 200;

fn main() {
    let mut envs = PaintGym::init(2, 256);
    for step in 0..1000 {
        let _results = envs.step(
            &[PaintAction {
                start: Pixel::new(10, 4),
                end: Pixel::new(22, 28),
            }],
            true,
        );

        if (step + 1) % STEPS_BEFORE_TRAINING == 0 {
            envs.do_bg_work();
        }
    }

    envs.cleanup();
}
