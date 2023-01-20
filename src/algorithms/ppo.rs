/// Samples actions from log logits.
pub fn sample_logits(logits: &tch::Tensor, device: tch::Device) -> tch::Tensor {
    // We're cheating a bit here since we know we only have 2 actions
    let batch_size = logits.size()[0];
    let logits = logits.exp();
    let samples = tch::Tensor::rand(&[batch_size], (tch::Kind::Float, device));
    let mut sample_idxs = Vec::new();
    for i in 0..batch_size {
        let sample_idx = if samples.double_value(&[i]) < logits.double_value(&[i, 0]) { 0 } else {1};
        sample_idxs.push(sample_idx as i64);
    }
    tch::Tensor::of_slice(&sample_idxs)
}

/// Returns the log probabilities of discrete actions being performed.
pub fn action_log_probs_discrete(
    actions: &tch::Tensor,
    log_probs: &tch::Tensor,
    _device: tch::Device,
) -> tch::Tensor {
    log_probs.take(&actions.squeeze().unsqueeze(0).to_dtype(tch::Kind::Int64, false, true)).squeeze()
}