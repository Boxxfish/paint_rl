use pyo3::prelude::*;

/// Stores both trainable model's architecture and its variables.
pub struct TrainableModel {
    pub module: tch::TrainableCModule,
    pub vs: tch::nn::VarStore,
    pub path: String,
}

impl TrainableModel {
    /// Loads a model from a path.
    pub fn load(path: &str, device: tch::Device) -> Self {
        let vs = tch::nn::VarStore::new(device);
        let module = tch::jit::TrainableCModule::load(path, vs.root()).unwrap();
        Self {
            module,
            vs,
            path: path.to_string(),
        }
    }
}

/// Prepares an interpreter for usage.
/// More specifically, it:
/// - Adds the cwd to PATH for importing local modules
pub fn prep_py(py: Python) {
    py.run(
        r#"
import os, sys
sys.path.append(os.getcwd())
"#,
        None,
        None,
    )
    .unwrap();
}

/// Cleans up Python.
/// More specifically, it:
/// - Reinstates the SIGINT handler
pub fn cleanup_py() {
    ctrlc::set_handler(|| {
        std::process::exit(1);
    })
    .unwrap();
}

/// Prints a stack trace and panics if an exception has occurred.
pub fn py_unwrap<T>(py: Python, result: Result<T, PyErr>) -> T {
    match result {
        Ok(val) => val,
        Err(err) => {
            err.print(py);
            panic!("Exception raised from Python");
        }
    }
}

/// Exports a model from a module.
pub fn export_model(
    py: Python,
    file_name: &str,
    model_name: &str,
    params: impl IntoPy<PyObject>,
    input_shapes: &[&[u32]],
    cuda: bool,
) {
    let model_utils = py_unwrap(py, py.import("models.model_utils"));
    py_unwrap(
        py,
        model_utils.call_method1(
            "export_model",
            (
                file_name,
                model_name,
                params.into_py(py),
                input_shapes
                    .iter()
                    .map(|x| x.to_vec())
                    .collect::<Vec<Vec<_>>>(),
                cuda,
            ),
        ),
    );
}
