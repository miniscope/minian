//! Python extension `minian.minian_rs` (Rust-backed helpers).

mod filter;
mod workers;

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
fn minian_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::filter::filt_fft_f64, m)?)?;
    m.add_function(wrap_pyfunction!(crate::filter::filt_fft_vec_f64, m)?)?;
    m.add_class::<crate::workers::ThreadAllocation>()?;
    m.add_function(wrap_pyfunction!(crate::workers::logical_parallelism, m)?)?;
    m.add_function(wrap_pyfunction!(crate::workers::thread_allocation, m)?)?;
    m.add_function(wrap_pyfunction!(crate::workers::default_cluster_workers, m)?)?;
    Ok(())
}
