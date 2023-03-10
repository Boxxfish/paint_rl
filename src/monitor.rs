use std::{collections::HashMap, sync::mpsc::Sender};

use plotters::{
    prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea},
    series::LineSeries,
    style::{RED, WHITE},
};
use pyo3::{types::IntoPyDict, Python};

pub type MetricID = usize;

/// Parameters for the plotters backend.
pub struct PlottersParams {
    pub output_path: String,
}

/// Parameters for the Weights and Biases backend.
pub struct WandbParams {
    pub project: String,
    pub config: HashMap<String, f32>,
}

/// Parameters for a given backend.
pub enum BackendParams {
    Plotters(PlottersParams),
    Wandb(WandbParams),
}

/// State for the plotters backend.
#[derive(Clone)]
pub struct PlottersData {
    output_path: String,
}

/// State for the Weights and Biases backend.
#[derive(Clone)]
pub struct WandbData {
    pub project: String,
    config: HashMap<String, f32>,
    run: Option<pyo3::PyObject>,
}

/// Data for a backend.
#[derive(Clone)]
pub enum BackendData {
    Plotters(PlottersData),
    Wandb(WandbData),
}

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
    pub backend_data: BackendData,
}

impl Monitor {
    /// Constructs a new monitor.
    pub fn new(plot_every: u32, backend_params: BackendParams) -> Self {
        let backend_data = match backend_params {
            BackendParams::Plotters(params) => BackendData::Plotters(PlottersData {
                output_path: params.output_path,
            }),
            BackendParams::Wandb(params) => BackendData::Wandb(WandbData {
                project: params.project,
                run: None,
                config: params.config,
            }),
        };

        Self {
            plot_every,
            bg_work_handle: None,
            unsent_data: Vec::new(),
            data_tx: None,
            current_step: 0,
            metrics: HashMap::new(),
            metric_count: 0,
            backend_data,
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
        let graph_width = 600;
        let graph_height = 480;
        let graph_count = self.metric_count as u32;
        let metrics = self.metrics.clone();
        let backend_data = match &self.backend_data {
            BackendData::Wandb(params) => {
                let run = Python::with_gil(|py| {
                    let wandb = py.import("python.wandb_utils").unwrap();
                    Some(
                        wandb
                            .call_method1(
                                "start_run",
                                (&params.project, params.config.clone().into_py_dict(py)),
                            )
                            .unwrap()
                            .into(),
                    )
                });
                BackendData::Wandb(WandbData {
                    project: params.project.clone(),
                    run,
                    config: params.config.clone(),
                })
            }
            _ => self.backend_data.clone(),
        };
        self.backend_data = backend_data.clone();
        let handle = std::thread::spawn(move || {
            let mut rows: Vec<Vec<Option<f32>>> = Vec::new();
            loop {
                let incoming: ThreadMsg = rx.recv().unwrap();
                let new_data = match incoming {
                    ThreadMsg::Data(data) => {
                        rows.extend(data.clone());
                        data
                    }
                    ThreadMsg::Stop => break rows,
                };

                if rows.is_empty() {
                    continue;
                }

                // Use backend to display data
                match &backend_data {
                    BackendData::Plotters(data) => {
                        // Steps are used as the X axis, with various metrics on the Y axis.
                        let root = BitMapBackend::new(
                            &data.output_path,
                            (graph_width, graph_height * graph_count),
                        )
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
                    BackendData::Wandb(data) => {
                        Python::with_gil(|py| {
                            let wandb = py.import("python.wandb_utils").unwrap();
                            let mut values = HashMap::new();
                            for (&metric_id, metric_name) in &metrics {
                                let mut metric_values = Vec::new();
                                for new_data_row in &new_data {
                                    metric_values.push(new_data_row[metric_id]);
                                }
                                values.insert(metric_name.to_owned(), metric_values);
                            }
                            wandb
                                .call_method1(
                                    "log",
                                    (data.run.as_ref().unwrap(), values, new_data.len()),
                                )
                                .unwrap();
                        });
                    }
                }
            }
        });

        self.bg_work_handle = Some(handle);
        self.data_tx = Some(tx);
    }

    /// Stops the monitor and outputs latest metrics to console.
    pub fn stop(self) {
        self.data_tx.unwrap().send(ThreadMsg::Stop).unwrap();
        let rows = self.bg_work_handle.unwrap().join().unwrap();

        if let BackendData::Wandb(data) = self.backend_data {
            Python::with_gil(|py| {
                let wandb = py.import("python.wandb_utils").unwrap();
                wandb
                    .call_method1("finish_run", (data.run.as_ref().unwrap(),))
                    .unwrap();
            });
        }

        for (&index, name) in &self.metrics {
            let mut vals: Vec<f32> = rows.iter().filter_map(|row| row[index]).collect();
            // Use average of the last 5 values
            let avg_samples = 5;
            let mut avg_val_vec = Vec::new();
            for _ in 0..avg_samples {
                avg_val_vec.push(vals.pop().unwrap());
            }
            let val = avg_val_vec.iter().sum::<f32>() / avg_samples as f32;
            println!("{name}: {val}");
        }
    }
}
