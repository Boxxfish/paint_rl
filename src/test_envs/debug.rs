//! Very simple environments for debugging specific elements of an algorithm.
//! Based off https://andyljones.com/posts/rl-debugging.html.
//! If an algorithm isn't working and you don't know why, run it on each of
//! these environments and the moment something stops working, you'll know
//! exactly what's wrong.
//!
//! All environments have the same interface so you can swap them out easily.
//! More specifically, they take a float between -1 and 1 as an observation,
//! an integer of either 0 or 1 as an action, and a float between -1 and 1 as
//! a reward. They are also all vectorized.

use rand::Rng;

fn panic_on_bad_action_len(env_count: usize, action_count: usize) {
    if action_count != env_count {
        panic!(
            "Number of actions ({action_count}) is different from environment count ({env_count})!"
        );
    }
}

/// Returns an observation of 0 and a reward of 1 on each timestep. Runs for a
/// single timestep before terminating. If the agent thinks the reward is anything
/// but 1, there's an issue with the value calculation.
pub struct Obs0Reward1Env {
    pub env_count: usize,
}

impl Obs0Reward1Env {
    pub fn new(env_count: usize) -> Self {
        Self { env_count }
    }

    /// Returns the first observation.
    pub fn reset(&self) -> Vec<f32> {
        vec![0.0; self.env_count]
    }

    /// Returns the next observations, rewards, and done flags.
    pub fn step(&self, actions: &[i32]) -> Vec<(f32, f32, bool)> {
        panic_on_bad_action_len(self.env_count, actions.len());
        vec![(0.0, 1.0, true); self.env_count]
    }
}

/// Returns an observation of -1/1 and a reward of -1/1. Runs for a single timestep
/// before terminating. The reward is based on the observation. If the agent can't
/// associate the observation with the reward, something's wrong with the backpropagation.
pub struct ObsDependentRewardEnv {
    pub env_count: usize,
    pub last_obs: Vec<f32>,
}

impl ObsDependentRewardEnv {
    pub fn new(env_count: usize) -> Self {
        Self {
            env_count,
            last_obs: vec![0.0; env_count],
        }
    }

    /// Returns the first observation.
    pub fn reset(&mut self) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        for i in 0..self.env_count {
            self.last_obs[i] = (rng.gen_bool(0.5) as i32 * 2 - 1) as f32;
        }
        self.last_obs.clone()
    }

    /// Returns the next observations, rewards, and done flags.
    pub fn step(&mut self, actions: &[i32]) -> Vec<(f32, f32, bool)> {
        panic_on_bad_action_len(self.env_count, actions.len());
        let mut rng = rand::thread_rng();
        let mut results = Vec::with_capacity(self.env_count);
        for i in 0..self.env_count {
            let obs = self.last_obs[i];
            let reward = obs * -1.0;
            self.last_obs[i] = (rng.gen_bool(0.5) as i32 * 2 - 1) as f32;
            results.push((obs, reward, true))
        }
        results
    }
}

/// Returns an observation and reward of 0 on timestep 1, then 1 of both on timestep 2.
/// If the agent can't learn the delayed reward, there's something wrong with the discounting.
pub struct DelayedRewardEnv {
    pub env_count: usize,
    pub timestep: u32,
}

impl DelayedRewardEnv {
    pub fn new(env_count: usize) -> Self {
        Self {
            env_count,
            timestep: 0,
        }
    }

    /// Returns the first observation.
    pub fn reset(&mut self) -> Vec<f32> {
        self.timestep = 1;
        vec![0.0; self.env_count]
    }

    /// Returns the next observations, rewards, and done flags.
    pub fn step(&mut self, actions: &[i32]) -> Vec<(f32, f32, bool)> {
        panic_on_bad_action_len(self.env_count, actions.len());
        let results = vec![
            (
                self.timestep as f32,
                self.timestep as f32,
                self.timestep == 1
            );
            self.env_count
        ];
        self.timestep = (self.timestep + 1) % 2;
        results
    }
}

/// Returns an observation of 0 and a reward of -1/1, based on whether the agent chooses
/// the correct action. The correct action is the same every time. If the agent fails
/// to choose the correct action, there's something wrong with the advantage calculations
/// or policy updates.
pub struct ConstCorrectActionEnv {
    pub env_count: usize,
}

impl ConstCorrectActionEnv {
    pub fn new(env_count: usize) -> Self {
        Self { env_count }
    }

    /// Returns the first observation.
    pub fn reset(&mut self) -> Vec<f32> {
        vec![0.0; self.env_count]
    }

    /// Returns the next observations, rewards, and done flags.
    pub fn step(&mut self, actions: &[i32]) -> Vec<(f32, f32, bool)> {
        panic_on_bad_action_len(self.env_count, actions.len());
        let mut results = Vec::with_capacity(self.env_count);
        for &action in actions {
            let reward = if action == 1 { 1.0 } else { -1.0 };
            results.push((0.0, reward, true));
        }
        results
    }
}

/// Returns an observation of -1/1 and a reward of -1/1. The reward is based on both
/// the observation and action taken.
pub struct ObsActionDependentRewardEnv {
    pub env_count: usize,
    pub last_obs: Vec<bool>,
}

impl ObsActionDependentRewardEnv {
    pub fn new(env_count: usize) -> Self {
        Self {
            env_count,
            last_obs: Vec::new(),
        }
    }

    /// Returns the first observation.
    pub fn reset(&mut self) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut obs = Vec::new();
        for _ in 0..self.env_count {
            obs.push(rng.gen_bool(0.5));
        }
        self.last_obs = obs;
        self.last_obs
            .iter()
            .map(|b| if *b { 1.0 } else { -1.0 })
            .collect()
    }

    /// Returns the next observations, rewards, and done flags.
    pub fn step(&mut self, actions: &[i32]) -> Vec<(f32, f32, bool)> {
        panic_on_bad_action_len(self.env_count, actions.len());
        let mut rng = rand::thread_rng();
        let mut results = Vec::with_capacity(self.env_count);
        let mut last_obs = Vec::with_capacity(self.env_count);
        for (&obs, &action) in self.last_obs.iter().zip(actions) {
            let reward = if (action == 0) == obs { 1.0 } else { -1.0 };
            let new_obs = rng.gen_bool(0.5);
            last_obs.push(new_obs);
            results.push((if new_obs { 1.0 } else { -1.0 }, reward, true));
        }
        self.last_obs = last_obs;
        results
    }
}
