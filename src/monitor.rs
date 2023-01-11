use std::{collections::HashMap, sync::mpsc::Sender};

use plotters::{
    prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea},
    series::LineSeries,
    style::{RED, WHITE},
};

pub type MetricID = usize;

/// Type of message sent to the BG thread.
#[derive(Clone)]
pub enum ThreadMsg {
    Data(Vec<Vec<Option<f32>>>),
    Stop,
}

/// Provides lightweight monitoring during an experiment.
/// Whenever `plot_every` number of steps have been hit,
pub struct Monitor {
    pub plot_every: u32,
    pub unsent_data: Vec<Vec<Option<f32>>>,
    pub data_tx: Option<Sender<ThreadMsg>>,
    pub bg_work_handle: Option<std::thread::JoinHandle<Vec<Vec<Option<f32>>>>>,
    pub current_step: u32,
    pub metrics: HashMap<usize, String>,
    pub metric_count: usize,
    pub output_path: String,
}

impl Monitor {
    /// Constructs a new monitor.
    pub fn new(plot_every: u32, output_path: &str) -> Self {
        Self {
            plot_every,
            bg_work_handle: None,
            unsent_data: Vec::new(),
            data_tx: None,
            current_step: 0,
            metrics: HashMap::new(),
            metric_count: 0,
            output_path: output_path.to_string(),
        }
    }

    /// Adds a new metric to the monitor and returns its ID.
    /// This should only be called prior to initializing the bg thread.
    pub fn add_metric(&mut self, name: &str) -> MetricID {
        self.metrics.insert(self.metric_count, name.to_string());
        self.metric_count += 1;
        self.metric_count - 1
    }

    /// Creates a new row of data for this step.
    /// Subsequent calls to `log_metric` will use this new step.
    /// If plotting should occur on this step, stored data is flushed into the channel.
    pub fn add_step(&mut self) {
        self.current_step += 1;
        if self.current_step % self.plot_every == 0 {
            self.data_tx
                .as_ref()
                .unwrap()
                .send(ThreadMsg::Data(self.unsent_data.clone()))
                .unwrap();
            self.unsent_data.clear();
        }
        self.unsent_data
            .push((0..self.metric_count).map(|_| None).collect());
    }

    /// Logs a metric for this step.
    /// Try to only use this once a step, since only one value per step
    /// is recorded. If another value with the same metric ID is used
    /// for the same step, the old value gets overwritten.
    pub fn log_metric(&mut self, metric_id: MetricID, value: f32) {
        self.unsent_data.last_mut().unwrap()[metric_id] = Some(value);
    }

    /// Starts the background work thread.
    /// The background thread performs plotting and saving.
    pub fn init(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        let output_path = self.output_path.clone();
        let graph_width = 600;
        let graph_height = 480;
        let graph_count = self.metric_count as u32;
        let metrics = self.metrics.clone();
        let handle = std::thread::spawn(move || {
            let mut rows: Vec<Vec<Option<f32>>> = Vec::new();
            loop {
                let incoming: ThreadMsg = rx.recv().unwrap();
                match incoming {
                    ThreadMsg::Data(data) => rows.extend(data),
                    ThreadMsg::Stop => break rows,
                }

                if rows.is_empty() {
                    continue;
                }

                // Plot data.
                // Steps are used as the X axis, with various metrics on the Y axis.
                let root =
                    BitMapBackend::new(&output_path, (graph_width, graph_height * graph_count))
                        .into_drawing_area();
                root.fill(&WHITE).unwrap();
                let areas = root.split_evenly((graph_count as usize, 1));
                for (metric_index, area) in areas.iter().enumerate() {
                    let metric_name = metrics.get(&metric_index).unwrap();
                    let metric_values = rows
                        .iter()
                        .map(|row| row[metric_index])
                        .enumerate()
                        .filter(|(_, val)| val.is_some())
                        .map(|(i, val)| (i, val.unwrap()))
                        .collect::<Vec<(usize, f32)>>();
                    if !metric_values.is_empty() {
                        let min_range = metric_values
                            .iter()
                            .min_by(|x, y| x.1.total_cmp(&y.1))
                            .unwrap()
                            .1;
                        let max_range = metric_values
                            .iter()
                            .max_by(|x, y| x.1.total_cmp(&y.1))
                            .unwrap()
                            .1;
                        let mut chart = ChartBuilder::on(area)
                            .margin(20)
                            .x_label_area_size(30)
                            .y_label_area_size(30)
                            .caption(metric_name, ("sans-serif", 32))
                            .build_cartesian_2d(0..rows.len(), min_range..max_range)
                            .unwrap();
                        chart.configure_mesh().draw().unwrap();
                        chart
                            .draw_series(LineSeries::new(metric_values, RED))
                            .unwrap();
                    }
                }
                root.present().unwrap();
            }
        });

        self.bg_work_handle = Some(handle);
        self.data_tx = Some(tx);
    }

    /// Stops the monitor and outputs latest metrics to console.
    pub fn stop(self) {
        self.data_tx.unwrap().send(ThreadMsg::Stop).unwrap();
        let rows = self.bg_work_handle.unwrap().join().unwrap();
        for (&index, name) in &self.metrics {
            let val = rows.iter().filter_map(|row| row[index]).last();
            let val = match val {
                Some(val) => val.to_string(),
                None => "None".to_string(),
            };
            println!("{name}: {val}");
        }
    }
}
