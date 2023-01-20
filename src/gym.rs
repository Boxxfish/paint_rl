use minifb::{Window, WindowOptions};
use rand::Rng;

/// Minimum L1 distance before env is considered done.
const DONE_CUTOFF: u64 = 10 * u8::MAX as u64;

/// A gym interface for a painting program.
/// Vectorized by default.
pub struct PaintGym {
    pub num_envs: u32,
    /// The width and height of each canvas.
    pub canvas_size: u32,
    /// All canvases, in one contiguous array.
    canvases: Vec<(u8, u8, u8)>,
    references: Vec<(u8, u8, u8)>,
    /// Number of steps performed in each environment.
    steps: Vec<u32>,
    /// Reward if the time limit is hit. Should be negative.
    failure_reward: f32,
    /// Time limit for each environment.
    max_steps: u32,
    window: Option<Window>,
}

fn draw_reference(
    rng: &mut rand::rngs::ThreadRng,
    references: &mut [(u8, u8, u8)],
    canvas_offset: usize,
    canvas_size: u32,
) {
    let stroke_count = rng.gen_range(20..30);
    for _ in 0..stroke_count {
        let x0 = rng.gen_range(0..canvas_size) as i32;
        let x1 = rng.gen_range(0..canvas_size) as i32;
        let y0 = rng.gen_range(0..canvas_size) as i32;
        let y1 = rng.gen_range(0..canvas_size) as i32;
        plot_line(x0, x1, y0, y1, references, canvas_offset, canvas_size);
    }
}

fn plot_line(
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    canvases: &mut [(u8, u8, u8)],
    canvas_offset: usize,
    canvas_size: u32,
) {
    // Handle straight lines
    if x0 == x1 {
        let start_y = y0.min(y1);
        let end_y = y0.max(y1);
        for y in start_y..end_y {
            canvases[canvas_offset + y as usize * canvas_size as usize + x0 as usize] = (0, 0, 0);
        }
    } else if y0 == y1 {
        let start_x = x0.min(x1);
        let end_x = x0.max(x1);
        for x in start_x..end_x {
            canvases[canvas_offset + y0 as usize * canvas_size as usize + x as usize] = (0, 0, 0);
        }
    }
    // Otherwise, handle sloped lines
    else {
        #[allow(clippy::collapsible_else_if)]
        if (y1 - y0).abs() < (x1 - x0).abs() {
            if x0 > x1 {
                plot_line_low(x1, x0, y1, y0, canvases, canvas_offset, canvas_size);
            } else {
                plot_line_low(x0, x1, y0, y1, canvases, canvas_offset, canvas_size);
            }
        } else if y0 > y1 {
            plot_line_high(x1, x0, y1, y0, canvases, canvas_offset, canvas_size);
        } else {
            plot_line_high(x0, x1, y0, y1, canvases, canvas_offset, canvas_size);
        }
    }
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
    pub fn init(
        num_envs: u32,
        canvas_size: u32,
        failure_reward: f32,
        max_steps: u32,
        render: bool,
    ) -> (Self, PaintStepResult) {
        let canvases: Vec<(u8, u8, u8)> = (0..(canvas_size * canvas_size * num_envs))
            .map(|_| (255, 255, 255))
            .collect();
        let window = if render {
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
                canvas_size as usize * 2,
                WindowOptions {
                    resize: false,
                    scale,
                    borderless: true,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("{e}"));
            window.limit_update_rate(Some(std::time::Duration::from_millis(16)));
            Some(window)
        } else {
            None
        };

        // Generate references
        let mut rng = rand::thread_rng();
        let mut references: Vec<(u8, u8, u8)> = (0..(canvas_size * canvas_size * num_envs))
            .map(|_| (255, 255, 255))
            .collect();

        for env_id in 0..(num_envs as usize) {
            let canvas_offset = env_id * (canvas_size * canvas_size) as usize;
            draw_reference(&mut rng, &mut references, canvas_offset, canvas_size);
        }

        // Generate first step observations
        let mut results = Vec::new();
        for env_id in 0..(num_envs as usize) {
            let canvas_offset = env_id * (canvas_size * canvas_size) as usize;
            results.push((
                PaintState {
                    canvas: canvases
                        [canvas_offset..(canvas_offset + (canvas_size * canvas_size) as usize)]
                        .to_vec(),
                    reference: references
                        [canvas_offset..(canvas_offset + (canvas_size * canvas_size) as usize)]
                        .to_vec(),
                },
                0.0,
                false,
            ))
        }
        let result = PaintStepResult { results };

        (
            Self {
                num_envs,
                canvas_size,
                canvases,
                references,
                window,
                steps: vec![0; num_envs as usize],
                failure_reward,
                max_steps,
            },
            result,
        )
    }

    /// Performs one step through all environments and performs
    /// the actions given. No per-environment `done` flag exists,
    /// since the environment will auto reset itself if it detects some
    /// end criteria.
    pub fn step(&mut self, actions: &[PaintAction], render: bool) -> PaintStepResult {
        // Render the stroke
        for (env_id, action) in actions.iter().enumerate() {
            let canvas_offset = env_id * (self.canvas_size * self.canvas_size) as usize;

            plot_line(
                action.start.x as i32,
                action.end.x as i32,
                action.start.y as i32,
                action.end.y as i32,
                &mut self.canvases,
                canvas_offset,
                self.canvas_size,
            );
        }

        // If we're rendering to the screen, update the window
        if let Some(window) = &mut self.window {
            if render {
                // Render the first env
                let mut pixel_buffer: Vec<u32> = self.canvases.as_slice()
                    [..(self.canvas_size * self.canvas_size) as usize]
                    .iter()
                    .map(|(r, g, b)| ((*r as u32) << 16) + ((*g as u32) << 8) + (*b as u32))
                    .collect();
                let ref_pixel_buffer: Vec<u32> = self.references.as_slice()
                    [..(self.canvas_size * self.canvas_size) as usize]
                    .iter()
                    .map(|(r, g, b)| ((*r as u32) << 16) + ((*g as u32) << 8) + (*b as u32))
                    .collect();
                pixel_buffer.extend(ref_pixel_buffer.iter());
                window
                    .update_with_buffer(
                        &pixel_buffer,
                        self.canvas_size as usize,
                        self.canvas_size as usize * 2,
                    )
                    .unwrap();
            }
        }

        // Compute the L1 distance between the environment images and the reference images.
        // We divide by the maximum L1 distance to keep the reward from -1 to 0.
        let mut rewards = Vec::with_capacity(self.num_envs as usize);
        let mut l1s = Vec::with_capacity(self.num_envs as usize);
        let r_divisor = (self.canvas_size * self.canvas_size * 3 * u8::MAX as u32) as f32;
        for env_id in 0..(self.num_envs as usize) {
            let canvas_offset = env_id * (self.canvas_size * self.canvas_size) as usize;
            let mut l1 = 0;
            for i in 0..(self.canvas_size * self.canvas_size) as usize {
                let canvas_pix = self.canvases[canvas_offset + i];
                let ref_pix = self.references[canvas_offset + i];
                l1 += canvas_pix.0.abs_diff(ref_pix.0) as u64
                    + canvas_pix.1.abs_diff(ref_pix.1) as u64
                    + canvas_pix.2.abs_diff(ref_pix.2) as u64;
            }
            rewards.push(-(l1 as f32) / r_divisor);
            l1s.push(l1);
        }

        // If any environments have a lower L1 distance than the cutoff, mark as done
        let mut dones = vec![false; self.num_envs as usize];
        for (env_id, &l1) in l1s.iter().enumerate() {
            if l1 < DONE_CUTOFF {
                dones[env_id] = true;
            }
        }

        // If any environments hit the time limit, mark as done and apply penalty
        for env_id in 0..(self.num_envs as usize) {
            self.steps[env_id] += 1;
            if self.steps[env_id] > self.max_steps {
                dones[env_id] = true;
                rewards[env_id] += self.failure_reward;
            }
        }

        // Obtains results to return
        let mut results = Vec::new();
        for env_id in 0..(self.num_envs as usize) {
            let canvas_offset = env_id * (self.canvas_size * self.canvas_size) as usize;
            results.push((
                PaintState {
                    canvas: self.canvases[canvas_offset
                        ..(canvas_offset + (self.canvas_size * self.canvas_size) as usize)]
                        .to_vec(),
                    reference: self.references[canvas_offset
                        ..(canvas_offset + (self.canvas_size * self.canvas_size) as usize)]
                        .to_vec(),
                },
                rewards[env_id],
                dones[env_id],
            ))
        }

        // Reset finished environments
        let mut rng = rand::thread_rng();
        for (env_id, &done) in dones.iter().enumerate() {
            if done {
                let canvas_total = (self.canvas_size * self.canvas_size) as usize;
                let canvas_offset = env_id * canvas_total;

                // Reset the canvas
                self.canvases.splice(
                    canvas_offset..(canvas_offset + canvas_total),
                    std::iter::repeat((255, 255, 255)).take(canvas_total),
                );

                // Regenerate reference
                self.references.splice(
                    canvas_offset..(canvas_offset + canvas_total),
                    std::iter::repeat((255, 255, 255)).take(canvas_total),
                );
                draw_reference(
                    &mut rng,
                    &mut self.references,
                    canvas_offset,
                    self.canvas_size,
                );

                self.steps[env_id] = 0;
            }
        }

        PaintStepResult { results }
    }

    /// Switches the environments to use eval mode.
    pub fn eval_mode(&mut self) {
        self.canvases = (0..(self.canvas_size * self.canvas_size * self.num_envs))
            .map(|_| (255, 255, 255))
            .collect();

        let mut rng = rand::thread_rng();
        let mut references: Vec<(u8, u8, u8)> =
            (0..(self.canvas_size * self.canvas_size * self.num_envs))
                .map(|_| (255, 255, 255))
                .collect();

        for env_id in 0..(self.num_envs as usize) {
            let canvas_offset = env_id * (self.canvas_size * self.canvas_size) as usize;
            draw_reference(&mut rng, &mut references, canvas_offset, self.canvas_size);
        }

        self.steps = vec![0; self.num_envs as usize];
    }

    /// Switches the environments to use train mode.
    pub fn train_mode(&mut self) {
        self.canvases = (0..(self.canvas_size * self.canvas_size * self.num_envs))
            .map(|_| (255, 255, 255))
            .collect();

        let mut rng = rand::thread_rng();
        let mut references: Vec<(u8, u8, u8)> =
            (0..(self.canvas_size * self.canvas_size * self.num_envs))
                .map(|_| (255, 255, 255))
                .collect();

        for env_id in 0..(self.num_envs as usize) {
            let canvas_offset = env_id * (self.canvas_size * self.canvas_size) as usize;
            draw_reference(&mut rng, &mut references, canvas_offset, self.canvas_size);
        }

        self.steps = vec![0; self.num_envs as usize];
    }

    /// Performs some work as the agent is trained.
    /// Since this may take a long time, this function ensures that period
    /// doesn't go to waste by doing things like IO bound operations.
    pub fn do_bg_work(&mut self) {}

    /// Performs cleanup.
    pub fn cleanup(&mut self) {}
}

/// A representation of the canvas.
#[derive(Clone)]
pub struct PaintState {
    pub canvas: Vec<(u8, u8, u8)>,
    pub reference: Vec<(u8, u8, u8)>,
}

/// Represents a pixel on screen.
/// (0, 0) is the top left.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Pixel {
    pub x: u32,
    pub y: u32,
}

impl Pixel {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

/// An action that can be performed on the canvas.
#[derive(Debug)]
pub struct PaintAction {
    pub start: Pixel,
    pub end: Pixel,
}

/// The result of performing a step.
pub struct PaintStepResult {
    /// (current state, reward, done)
    pub results: Vec<(PaintState, f32, bool)>,
}
