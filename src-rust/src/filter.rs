//! Band-limited FFT filtering along rows (mirror of `cnmf.filt_fft` / `filt_fft_vec`).

use ndarray::{Array1, Array2};
use num_complex::Complex;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// One 1-D series: FFT → zero band → inverse FFT (`cnmf.filt_fft`). Returns a new vec.
///
/// **Scaling:** `realfft`’s inverse is unnormalized vs NumPy/pyFFTW’s `irfft`; we divide by `t`
/// so results match ``numpy_fft.{rfft, irfft}(..., n=len(x))`` / Minian legacy.
fn filt_fft_1d(
    mut row: Vec<f64>,
    freq: f64,
    btype: &str,
    r2c: &Arc<dyn RealToComplex<f64>>,
    c2r: &Arc<dyn ComplexToReal<f64>>,
) -> Vec<f64> {
    let t = row.len();
    if t == 0 {
        return row;
    }
    let mut spectrum = forward_real_fft(r2c, &mut row);

    let k = (freq * t as f64) as usize;
    match btype {
        "low" => {
            spectrum[k..]
                .iter_mut()
                .for_each(|c| *c = Complex::new(0.0, 0.0));
        }
        "high" => {
            spectrum[..k]
                .iter_mut()
                .for_each(|c| *c = Complex::new(0.0, 0.0));
        }
        _ => unreachable!(),
    }

    let mut output = c2r.make_output_vec();
    c2r.process(&mut spectrum, &mut output)
        .expect("realfft inverse");
    output.truncate(t);
    // Match NumPy / PyFFTW `numpy_fft.irfft(_, n=len(x))` amplitude (realfft uses no 1/n here).
    let inv_t = 1.0 / (t as f64);
    for v in &mut output {
        *v *= inv_t;
    }
    output
}

#[inline]
fn forward_real_fft(r2c: &Arc<dyn RealToComplex<f64>>, row: &mut [f64]) -> Vec<Complex<f64>> {
    let mut spectrum = r2c.make_output_vec();
    r2c.process(row, &mut spectrum).expect("realfft forward");
    spectrum
}

fn fft_planners(t: usize) -> (Arc<dyn RealToComplex<f64>>, Arc<dyn ComplexToReal<f64>>) {
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(t);
    let c2r = planner.plan_fft_inverse(t);
    (r2c, c2r)
}

/// Row-wise filtering; same semantics as [`cnmf.filt_fft_vec`] (along last dimension).
///
/// `parallel=true` uses rayon over rows (still holds GIL; use gil-free FFI later if needed).
#[pyfunction]
#[pyo3(signature = (x, freq, btype, parallel = false))]
pub fn filt_fft_vec_f64<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    freq: f64,
    btype: &str,
    parallel: bool,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    if !matches!(btype, "low" | "high") {
        return Err(PyValueError::new_err("btype must be 'low' or 'high'"));
    }

    let view = x.as_array();
    let (n_units, n_frames) = view.dim();

    let (r2c, c2r) = fft_planners(n_frames);

    let out_flat: Vec<f64> = if parallel && n_units > 1 {
        (0..n_units)
            .into_par_iter()
            .map(|iu| {
                let row_owned = view.row(iu).iter().cloned().collect::<Vec<_>>();
                filt_fft_1d(row_owned, freq, btype, &r2c, &c2r)
            })
            .collect::<Vec<Vec<_>>>()
            .into_iter()
            .flatten()
            .collect()
    } else {
        let mut flat = Vec::with_capacity(n_units * n_frames);
        for row in view.outer_iter() {
            let row_owned = row.iter().cloned().collect::<Vec<_>>();
            flat.extend_from_slice(&filt_fft_1d(row_owned, freq, btype, &r2c, &c2r));
        }
        flat
    };

    let out_nd = Array2::from_shape_vec((n_units, n_frames), out_flat)
        .map_err(|_| PyValueError::new_err("filt_fft_vec: reshape failed"))?;

    Ok(out_nd.into_pyarray(py))
}

/// 1-D band-limited FFT filter; same semantics as [`cnmf.filt_fft`] (PyFFTW fallback).
#[pyfunction]
#[pyo3(signature = (x, freq, btype))]
pub fn filt_fft_f64<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'_, f64>,
    freq: f64,
    btype: &str,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    if !matches!(btype, "low" | "high") {
        return Err(PyValueError::new_err("btype must be 'low' or 'high'"));
    }

    let view = x.as_array();
    let row_owned = view.iter().cloned().collect::<Vec<_>>();
    let t = row_owned.len();

    if t == 0 {
        return Ok(Array1::<f64>::zeros((0,)).into_pyarray(py));
    }

    let (r2c, c2r) = fft_planners(t);
    let out = filt_fft_1d(row_owned, freq, btype, &r2c, &c2r);

    Ok(Array1::from_vec(out).into_pyarray(py))
}
