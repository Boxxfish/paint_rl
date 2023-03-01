//! Torch distributions in Rust.

use tch::IndexOp;

trait Distribution {
    fn sample(&self, sample_shape: &[i64]) -> tch::Tensor;
    fn log_prob(&self, value: tch::Tensor) -> tch::Tensor;
}

fn extended_shape(sample_shape: &[i64], batch_shape: &[i64], event_shape: &[i64]) -> Vec<i64> {
    let mut shape = Vec::new();
    shape.extend_from_slice(sample_shape);
    shape.extend_from_slice(batch_shape);
    shape.extend_from_slice(event_shape);
    shape
}

pub struct Multinomial {}

pub struct Categorical {
    pub logits: tch::Tensor,
    pub probs: tch::Tensor,
    _num_events: i64,
    _batch_shape: Vec<i64>,
    _event_shape: Vec<i64>,
}

impl Distribution for Categorical {
    fn sample(&self, sample_shape: &[i64]) -> tch::Tensor {
        let probs_2d = self.probs.reshape(&[-1, self._num_events]);
        let samples_2d = probs_2d.multinomial(sample_shape.len() as i64, true).t_();
        samples_2d.reshape(&extended_shape(
            sample_shape,
            &self._batch_shape,
            &self._event_shape,
        ))
    }

    fn log_prob(&self, value: tch::Tensor) -> tch::Tensor {
        let value = value.internal_cast_long(true).unsqueeze(-1);
        let value_log_pmf = tch::Tensor::broadcast_tensors(&[value, self.logits.copy()]);
        let value = value_log_pmf[0].copy();
        let log_pmf = value_log_pmf[1].copy();
        let value = value.i((.., 1..));
        log_pmf.gather(-1, &value, false).squeeze()
    }
}
