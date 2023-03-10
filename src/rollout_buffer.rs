use crate::model_utils::TrainableModel;

/// Stores transitions and generates mini batches from the latest policy.
/// Also computes advantage estimates.
pub struct RolloutBuffer {
    // The shape of each element is [num_steps, num_envs, element_shape].
    pub states: tch::Tensor,
    pub actions: tch::Tensor,
    pub rewards: tch::Tensor,
    pub dones: tch::Tensor,
    pub num_envs: usize,
    pub num_steps: usize,
    pub next: usize,
    pub device: tch::Device,
}

impl RolloutBuffer {
    /// Creates a new instance of a rollout buffer.
    pub fn new(
        state_shape: &[i64],
        action_shape: &[i64],
        action_dtype: tch::Kind,
        num_envs: usize,
        num_steps: usize,
        device: tch::Device,
    ) -> Self {
        let k = tch::Kind::Float;
        let num_envs_i64 = num_envs as i64;
        let num_steps_i64 = num_steps as i64;
        let mut state_shape_vec = vec![num_steps_i64 + 1, num_envs_i64];
        state_shape_vec.extend_from_slice(state_shape);
        let mut action_shape_vec = vec![num_steps_i64, num_envs_i64];
        action_shape_vec.extend_from_slice(action_shape);
        Self {
            num_envs,
            num_steps,
            next: 0,
            states: tch::Tensor::zeros(&state_shape_vec, (k, device)).requires_grad_(false),
            actions: tch::Tensor::zeros(&action_shape_vec, (action_dtype, device))
                .requires_grad_(false),
            rewards: tch::Tensor::zeros(&[num_steps_i64, num_envs_i64], (k, device))
                .requires_grad_(false),
            dones: tch::Tensor::zeros(&[num_steps_i64, num_envs_i64], (k, device))
                .requires_grad_(false),
            device,
        }
    }

    /// Inserts a transition from each environment into the buffer.
    /// Make sure more data than steps aren't inserted.
    /// Insert the state that was observed PRIOR to performing the action.
    /// The final state returned will be inserted using `insert_final_step`.
    pub fn insert_step(
        &mut self,
        states: &tch::Tensor,
        actions: &tch::Tensor,
        rewards: &[f32],
        dones: &[bool],
    ) {
        tch::no_grad(|| {
            self.states.get(self.next as _).copy_(states);
            self.actions.get(self.next as _).copy_(actions);
            self.rewards
                .get(self.next as _)
                .copy_(&tch::Tensor::of_slice(rewards).to(self.device));
            self.dones.get(self.next as _).copy_(
                &tch::Tensor::of_slice(dones)
                    .to_dtype(tch::Kind::Float, true, false)
                    .to(self.device),
            );
        });
        self.next += 1;
    }

    /// Inserts the final observation observed.
    pub fn insert_final_step(&mut self, states: &tch::Tensor) {
        tch::no_grad(|| {
            self.states.get(self.next as _).copy_(states);
        });
    }

    /// Generates minibatches of experience, incorporating advantage estimates.
    /// Returns previous states, states, actions, rewards, rewards to go, advantages, and dones.
    pub fn samples(
        &self,
        batch_size: u32,
        discount: f32,
        lambda: f32,
        v_net: &TrainableModel,
    ) -> Vec<(
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
    )> {
        // `guard` must be written in this way, or the compiler might remove it
        #[allow(unused_variables)]
        let guard = tch::no_grad_guard();
        // Calculate advantage estimates and rewards to go
        let tensor_opts = (tch::Kind::Float, self.device);
        let rewards_to_go =
            tch::Tensor::zeros(&[self.num_steps as i64, self.num_envs as i64], tensor_opts)
                .requires_grad_(false);
        let advantages =
            tch::Tensor::zeros(&[self.num_steps as i64, self.num_envs as i64], tensor_opts)
                .requires_grad_(false);
        let mut step_rewards_to_go = v_net
            .module
            .forward_ts(&[&self.states.get(self.next as _)])
            .unwrap()
            .squeeze();
        let mut state_values = step_rewards_to_go.copy();
        let mut step_advantages = tch::Tensor::zeros(&[self.num_envs as i64], tensor_opts);
        for i in (0..self.num_steps).rev() {
            let prev_states = self.states.get(i as _);
            let rewards = self.rewards.get(i as _);
            let inv_dones = 1.0 - self.dones.get(i as _);
            let prev_state_values = v_net.module.forward_ts(&[prev_states]).unwrap().squeeze();
            let delta = &rewards + discount * &inv_dones * state_values - &prev_state_values;
            step_rewards_to_go = rewards + discount * step_rewards_to_go * &inv_dones;
            rewards_to_go.get(i as _).copy_(&step_rewards_to_go);
            step_advantages = delta + discount * lambda * step_advantages * inv_dones;
            advantages.get(i as _).copy_(&step_advantages);
            state_values = prev_state_values;
        }
        let exp_count = self.num_envs * self.num_steps;
        let indices = tch::Tensor::randperm(exp_count as i64, (tch::Kind::Int, self.device));
        let rand_prev_states = self.states.flatten(0, 1).index_select(0, &indices);
        let rand_actions = self.actions.flatten(0, 1).index_select(0, &indices);
        let rand_rewards = self.rewards.flatten(0, 1).index_select(0, &indices);
        let rand_rewards_to_go = rewards_to_go.flatten(0, 1).index_select(0, &indices);
        let rand_advantages = advantages.flatten(0, 1).index_select(0, &indices);
        let rand_dones = self.dones.flatten(0, 1).index_select(0, &indices);
        let rand_states = self.states.flatten(0, 1).index_select(0, &(indices + 1));
        let batch_count = exp_count as u32 / batch_size;
        let mut batches = Vec::new();
        for i in 0..batch_count {
            let start = (i * batch_size) as i64;
            let end = ((i + 1) * batch_size) as i64;
            batches.push((
                rand_prev_states.slice(0, start, end, 1),
                rand_states.slice(0, start, end, 1),
                rand_actions.slice(0, start, end, 1),
                rand_rewards.slice(0, start, end, 1),
                rand_rewards_to_go.slice(0, start, end, 1),
                rand_advantages.slice(0, start, end, 1),
                rand_dones.slice(0, start, end, 1),
            ))
        }
        batches
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.next = 0;
    }
}
