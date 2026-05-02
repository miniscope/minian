//! Host thread / CPU view and Dask ``LocalCluster`` worker defaults.

use pyo3::prelude::*;

/// Default fraction of ``(logical CPUs − reserve)`` used as ``cluster_workers`` (floored, ≥ 1).
pub const DEFAULT_WORKER_CPU_RATIO: f64 = 2.0 / 3.0;

fn logical_parallelism_count() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

#[derive(Clone, Copy)]
struct Allocation {
    logical_cpus: usize,
    after_reserve_cpus: usize,
    cluster_workers: usize,
    worker_cpu_ratio: f64,
}

fn sanitize_worker_cpu_ratio(worker_cpu_ratio: f64) -> f64 {
    if !worker_cpu_ratio.is_finite() || worker_cpu_ratio <= 0.0 {
        return DEFAULT_WORKER_CPU_RATIO;
    }
    worker_cpu_ratio.clamp(f64::MIN_POSITIVE, 1.0)
}

fn compute_allocation(reserve: usize, worker_cpu_ratio: f64) -> Allocation {
    let logical_cpus = logical_parallelism_count();
    let after_reserve_cpus = logical_cpus.saturating_sub(reserve);
    let r = sanitize_worker_cpu_ratio(worker_cpu_ratio);
    let raw = (after_reserve_cpus as f64 * r).floor();
    let cluster_workers = (raw as usize).max(1);
    Allocation {
        logical_cpus,
        after_reserve_cpus,
        cluster_workers,
        worker_cpu_ratio: r,
    }
}

/// Logical CPUs visible to this process (via [`std::thread::available_parallelism`]).
#[pyfunction]
pub fn logical_parallelism() -> usize {
    logical_parallelism_count()
}

/// ``logical_parallelism()`` minus ``reserve``, times ``worker_cpu_ratio`` (floored), at least **1**.
///
/// Default ``worker_cpu_ratio`` is **2/3**. Use a smaller ratio for a more conservative
/// ``LocalCluster(..., n_workers=...)`` so the OS / Jupyter keep headroom.
#[pyfunction]
#[pyo3(signature = (reserve=1, worker_cpu_ratio=None))]
pub fn default_cluster_workers(reserve: usize, worker_cpu_ratio: Option<f64>) -> usize {
    let r = worker_cpu_ratio.unwrap_or(DEFAULT_WORKER_CPU_RATIO);
    compute_allocation(reserve, r).cluster_workers
}

/// CPU-based defaults in one place: raw parallelism, headroom after ``reserve``, and Dask worker count.
#[pyclass(name = "ThreadAllocation", frozen)]
pub struct ThreadAllocation {
    #[pyo3(get)]
    pub logical_cpus: usize,
    #[pyo3(get)]
    pub after_reserve_cpus: usize,
    #[pyo3(get)]
    pub cluster_workers: usize,
    #[pyo3(get)]
    pub worker_cpu_ratio: f64,
}

#[pyfunction]
#[pyo3(signature = (reserve=1, worker_cpu_ratio=None))]
pub fn thread_allocation(reserve: usize, worker_cpu_ratio: Option<f64>) -> ThreadAllocation {
    let r = worker_cpu_ratio.unwrap_or(DEFAULT_WORKER_CPU_RATIO);
    let a = compute_allocation(reserve, r);
    ThreadAllocation {
        logical_cpus: a.logical_cpus,
        after_reserve_cpus: a.after_reserve_cpus,
        cluster_workers: a.cluster_workers,
        worker_cpu_ratio: a.worker_cpu_ratio,
    }
}
