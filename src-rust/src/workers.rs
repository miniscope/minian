//! Host thread / CPU view and Dask ``LocalCluster`` worker defaults.

use pyo3::prelude::*;

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
}

fn compute_allocation(reserve: usize) -> Allocation {
    let logical_cpus = logical_parallelism_count();
    let after_reserve_cpus = logical_cpus.saturating_sub(reserve);
    let cluster_workers = ((after_reserve_cpus.saturating_mul(2)) / 3).max(1);
    Allocation {
        logical_cpus,
        after_reserve_cpus,
        cluster_workers,
    }
}

/// Logical CPUs visible to this process (via [`std::thread::available_parallelism`]).
#[pyfunction]
pub fn logical_parallelism() -> usize {
    logical_parallelism_count()
}

/// ``logical_parallelism()`` minus ``reserve``, scaled by **two thirds** (integer), at least **1**.
///
/// Conservative default for ``LocalCluster(..., n_workers=...)`` so the OS / Jupyter keep headroom.
#[pyfunction]
#[pyo3(signature = (reserve=1))]
pub fn default_cluster_workers(reserve: usize) -> usize {
    compute_allocation(reserve).cluster_workers
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
}

#[pyfunction]
#[pyo3(signature = (reserve=1))]
pub fn thread_allocation(reserve: usize) -> ThreadAllocation {
    let a = compute_allocation(reserve);
    ThreadAllocation {
        logical_cpus: a.logical_cpus,
        after_reserve_cpus: a.after_reserve_cpus,
        cluster_workers: a.cluster_workers,
    }
}
