/// Stores transitions and generates mini batches.
/// When new elements are inserted, they overwrite old data.
pub struct ReplayBuffer {
    pub prev_states: tch::Tensor,
    pub states: tch::Tensor,
    pub actions: tch::Tensor,
    pub rewards: tch::Tensor,
    pub dones: tch::Tensor,
    pub capacity: usize,
    pub next: usize,
    pub filled: bool,
    pub device: tch::Device,
}

impl ReplayBuffer {
    /// Creates a new instance of a replay buffer.
    pub fn new(
        state_shape: &[i64],
        action_shape: &[i64],
        capacity: usize,
        device: tch::Device,
    ) -> Self {
        let k = tch::Kind::Float;
        let mut state_shape_vec = vec![capacity as i64];
        state_shape_vec.extend_from_slice(state_shape);
        let mut action_shape_vec = vec![capacity as i64];
        action_shape_vec.extend_from_slice(action_shape);
        Self {
            capacity,
            next: 0,
            prev_states: tch::Tensor::zeros(&state_shape_vec, (k, device)).requires_grad_(false),
            states: tch::Tensor::zeros(&state_shape_vec, (k, device)).requires_grad_(false),
            actions: tch::Tensor::zeros(&action_shape_vec, (k, device)).requires_grad_(false),
            rewards: tch::Tensor::zeros(&[capacity as i64], (k, device)).requires_grad_(false),
            dones: tch::Tensor::zeros(&[capacity as i64], (k, device)).requires_grad_(false),
            filled: false,
            device,
        }
    }

    /// Inserts a batch of elements into the buffer.
    /// If the max capacity has been reached, the buffer wraps around.
    /// Do not attempt to insert more elements than the buffer's capacity.
    pub fn insert_batch(
        &mut self,
        prev_states: &tch::Tensor,
        states: &tch::Tensor,
        actions: &tch::Tensor,
        rewards: &[f32],
        dones: &[bool],
    ) {
        let batch_size = dones.len();
        let d = self.device;
        tch::no_grad(|| {
            let indices = tch::Tensor::arange_start(
                self.next as i64,
                (self.next + batch_size) as i64,
                (tch::Kind::Int64, d),
            )
            .remainder(batch_size as i64);
            self.prev_states = self.prev_states.index_copy(0, &indices, prev_states);
            self.states = self.states.index_copy(0, &indices, states);
            self.actions = self.actions.index_copy(0, &indices, actions);
            self.rewards = self.rewards.index_copy(
                0,
                &indices,
                &tch::Tensor::of_slice(rewards).to(self.device),
            );
            self.dones = self.dones.index_copy(
                0,
                &indices,
                &tch::Tensor::of_slice(dones)
                    .to_dtype(tch::Kind::Float, true, false)
                    .to(self.device),
            );
        });
        self.next = (self.next + batch_size) % self.capacity;
        if self.next == 0 {
            self.filled = true;
        }
    }

    /// Generates a mini batch of experience.
    pub fn sample(
        &self,
        batch_size: usize,
    ) -> (
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
        tch::Tensor,
    ) {
        let indices = tch::Tensor::randint(
            self.capacity as i64,
            &[batch_size as i64],
            (tch::Kind::Int, self.device),
        );
        (
            self.prev_states.index_select(0, &indices),
            self.states.index_select(0, &indices),
            self.actions.index_select(0, &indices),
            self.rewards.index_select(0, &indices),
            self.dones.index_select(0, &indices),
        )
    }

    /// Returns true if the buffer is full.
    pub fn is_full(&self) -> bool {
        self.filled
    }
}
