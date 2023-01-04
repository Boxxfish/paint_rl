use pyo3::prelude::*;
use tch::nn::VarStore;

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

/// Exports a model from a module and loads it.
pub fn load_model(
    py: Python,
    vs: &VarStore,
    file_name: &str,
    model_name: &str,
    params: impl IntoPy<PyObject>,
    input_shape: &[u32],
) -> tch::TrainableCModule {
    let model_utils = py_unwrap(py, py.import("models.model_utils"));
    py_unwrap(
        py,
        model_utils.call_method1(
            "export_model",
            (
                file_name,
                model_name,
                params.into_py(py),
                input_shape.to_vec(),
            ),
        ),
    );
    tch::jit::TrainableCModule::load(format!("temp/{model_name}.ptc"), vs.root()).unwrap()
}
