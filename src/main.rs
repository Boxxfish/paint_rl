#![allow(dead_code)]

use minifb::{Window, WindowOptions};
use rand::Rng;

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

fn plot_line_low(
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    canvases: &mut [(u8, u8, u8)],
    canvas_offset: usize,
    canvas_size: u32,
) {
    let dx = x1 - x0;
    let mut dy = y1 - y0;
    let mut y = y0;
    let mut yi = 1;
    if dy < 0 {
        yi = -1;
        dy = -dy;
    }
    let mut d = 2 * dy - dx;

    for x in x0..x1 {
        canvases[canvas_offset + y as usize * canvas_size as usize + x as usize] = (0, 0, 0);
        if d > 0 {
            y += yi;
            d += 2 * (dy - dx);
        } else {
            d += 2 * dy;
        }
    }
}

fn plot_line_high(
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    canvases: &mut [(u8, u8, u8)],
    canvas_offset: usize,
    canvas_size: u32,
) {
    let mut dx = x1 - x0;
    let dy = y1 - y0;
    let mut x = x0;
    let mut xi = 1;
    if dx < 0 {
        xi = -1;
        dx = -dx;
    }
    let mut d = 2 * dx - dy;

    for y in y0..y1 {
        canvases[canvas_offset + y as usize * canvas_size as usize + x as usize] = (0, 0, 0);
        if d > 0 {
            x += xi;
            d += 2 * (dx - dy);
        } else {
            d += 2 * dx;
        }
    }
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
        window.limit_update_rate(Some(std::time::Duration::from_millis(16)));
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

            // Handle straight lines
            if action.start.x == action.end.x {
                let start_y = action.start.y.min(action.end.y);
                let end_y = action.start.y.max(action.end.y);
                for y in start_y..end_y {
                    self.canvases[canvas_offset
                        + y as usize * self.canvas_size as usize
                        + action.start.x as usize] = (0, 0, 0);
                }
            } else if action.start.y == action.end.y {
                let start_x = action.start.x.min(action.end.x);
                let end_x = action.start.x.max(action.end.x);
                for x in start_x..end_x {
                    self.canvases[canvas_offset
                        + action.start.y as usize * self.canvas_size as usize
                        + x as usize] = (0, 0, 0);
                }
            }
            // Otherwise, handle sloped lines
            else {
                let x0 = action.start.x as i32;
                let x1 = action.end.x as i32;
                let y0 = action.start.y as i32;
                let y1 = action.end.y as i32;

                if (y1 - y0).abs() < (x1 - x0).abs() {
                    if x0 > x1 {
                        plot_line_low(
                            x1,
                            x0,
                            y1,
                            y0,
                            &mut self.canvases,
                            canvas_offset,
                            self.canvas_size,
                        );
                    } else {
                        plot_line_low(
                            x0,
                            x1,
                            y0,
                            y1,
                            &mut self.canvases,
                            canvas_offset,
                            self.canvas_size,
                        );
                    }
                } else if y0 > y1 {
                    plot_line_high(
                        x1,
                        x0,
                        y1,
                        y0,
                        &mut self.canvases,
                        canvas_offset,
                        self.canvas_size,
                    );
                } else {
                    plot_line_high(
                        x0,
                        x1,
                        y0,
                        y1,
                        &mut self.canvases,
                        canvas_offset,
                        self.canvas_size,
                    );
                }
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
    let mut envs = PaintGym::init(4, 256);
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
