use crate::{
    model_utils::TrainableModel,
    monitor::{MetricID, Monitor},
    replay_buffer::ReplayBuffer,
};

/// Performs polyak averaging between two networks.
/// When `p` is 0, `dest`'s weights are used. When `p` is 1, `src`'s weights are used.
/// This modifies `dest`.
pub fn polyak_avg(src: &tch::nn::VarStore, dest: &mut tch::nn::VarStore, p: f32) {
    tch::no_grad(|| {
        for (dest, src) in dest
            .trainable_variables()
            .iter_mut()
            .zip(src.trainable_variables().iter())
        {
            dest.copy_(&(p * src + (1.0 - p) * &*dest));
        }
    })
}

/// Trains DDPG models.
#[allow(clippy::too_many_arguments)]
pub fn train(
    q_net: &mut TrainableModel,
    p_net: &mut TrainableModel,
    q_target: &mut TrainableModel,
    p_target: &mut TrainableModel,
    q_opt: &mut tch::nn::Optimizer,
    p_opt: &mut tch::nn::Optimizer,
    q_loss_metric: MetricID,
    p_loss_metric: MetricID,
    monitor: &mut Monitor,
    replay_buffer: &ReplayBuffer,
    train_iterations: u32,
    train_batch_size: usize,
    discount: f32,
    polyak: f32,
) -> Result<(), anyhow::Error> {
    q_net.module.set_train();
    p_net.module.set_train();

    for _ in 0..(train_iterations) {
        let (prev_states, states, actions, rewards, dones) = replay_buffer.sample(train_batch_size);

        // Perform value optimization
        let mut targets: tch::Tensor = rewards
            + discount
                * (1.0 - dones)
                * q_target
                    .module
                    .forward_ts(&[&states, &p_target.module.forward_ts(&[&states])?])?;
        targets = targets.detach();
        let diff = &targets - q_net.module.forward_ts(&[&prev_states, &actions])?;
        let q_loss = (&diff * &diff).mean(tch::Kind::Float);
        monitor.log_metric(q_loss_metric, q_loss.double_value(&[]) as f32);

        q_opt.zero_grad();
        q_loss.backward();
        q_opt.step();

        // Perform policy optimization
        let p_loss = -q_net
            .module
            .forward_ts(&[&prev_states, &p_net.module.forward_ts(&[&prev_states])?])?
            .mean(tch::Kind::Float);
        monitor.log_metric(p_loss_metric, p_loss.double_value(&[]) as f32);

        p_opt.zero_grad();
        p_loss.backward();
        p_opt.step();

        // Move targets
        polyak_avg(&q_net.vs, &mut q_target.vs, polyak);
        polyak_avg(&p_net.vs, &mut p_target.vs, polyak);
    }

    q_net.module.set_eval();
    p_net.module.set_eval();

    Ok(())
}
