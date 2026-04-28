//! Dask/local cluster worker count (best-effort CPU detection).

use pyo3::prelude::*;

/// Logical CPUs usable by this process (via [`std::thread::available_parallelism`]),
/// minus `reserve`, floored at 1. Intended as a starting point for
/// ``LocalCluster(..., n_workers=...)``.
#[pyfunction]
#[pyo3(signature = (reserve=1))]
pub fn default_cluster_workers(reserve: usize) -> usize {
    let n = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    n.saturating_sub(reserve).max(1)
}
