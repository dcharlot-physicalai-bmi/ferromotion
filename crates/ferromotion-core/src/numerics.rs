//! **Guarded dense linear algebra** — the one place this workspace is allowed to call a dynamically
//! sized SVD.
//!
//! nalgebra's `DMatrix` SVD **does not terminate** on a matrix holding a `NaN`. Measured with a 3 s
//! watchdog per call (`tests/nalgebra_nan_behaviour.rs`):
//!
//! | routine | matrix holding one `NaN` |
//! |---|---|
//! | `DMatrix::singular_values()`, 3×3 and 6×6 | **hangs** |
//! | `DMatrix::svd(true, true)` | **hangs** |
//! | `Matrix3::svd(true, true)` (fixed size) | returns |
//! | `symmetric_eigen`, `cholesky`, `try_inverse`, `lu().solve`, `qr` | return |
//!
//! So the hazard is exactly the dynamically sized SVD, and every other decomposition degrades to `NaN`
//! or `None` on its own. A hang is worse than a panic because nothing reports it: in a control loop it
//! is a robot that stops responding. These wrappers check finiteness first and report instead.
//!
//! `tests/no_unguarded_dynamic_svd.rs` fails if any library file calls the raw routines on a `DMatrix`
//! outside this module, so the hazard cannot be reintroduced quietly.

use nalgebra::{DMatrix, SVD, Dyn};

/// Singular values of `m`, largest first, or `None` if any entry is non-finite.
pub fn finite_singular_values(m: &DMatrix<f64>) -> Option<Vec<f64>> {
    if !m.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut s: Vec<f64> = m.clone().singular_values().iter().copied().collect();
    s.sort_by(|a, b| b.total_cmp(a));
    Some(s)
}

/// Full SVD of `m`, or `None` if any entry is non-finite.
pub fn finite_svd(m: &DMatrix<f64>, compute_u: bool, compute_v: bool) -> Option<SVD<f64, Dyn, Dyn>> {
    if !m.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(m.clone().svd(compute_u, compute_v))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both wrappers refuse a non-finite matrix, and that refusal is what stops the hang. Each call
    /// runs behind a watchdog so a regression FAILS in seconds rather than hanging the whole suite.
    #[test]
    fn the_wrappers_refuse_a_non_finite_matrix_instead_of_spinning() {
        /// Run `f` on its own thread and return its value, failing rather than hanging if it does not
        /// come back. A guard regression here is non-termination, not a panic.
        fn within_10s<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(f());
            });
            rx.recv_timeout(std::time::Duration::from_secs(10))
                .unwrap_or_else(|_| panic!("{what} must RETURN on a non-finite matrix, not spin forever"))
        }

        let good = DMatrix::<f64>::identity(3, 4);
        assert_eq!(finite_singular_values(&good).expect("control returns values").len(), 3);
        assert!(finite_svd(&good, true, true).is_some(), "control returns a factorization");

        let mut nan = DMatrix::<f64>::identity(3, 4);
        nan[(1, 1)] = f64::NAN;
        let nan2 = nan.clone();
        assert!(within_10s("finite_singular_values", move || finite_singular_values(&nan).is_none()), "must report None for a NaN");
        assert!(within_10s("finite_svd", move || finite_svd(&nan2, true, true).is_none()), "must report None for a NaN");

        let mut inf = DMatrix::<f64>::identity(3, 4);
        inf[(0, 2)] = f64::NEG_INFINITY;
        let inf2 = inf.clone();
        assert!(within_10s("finite_singular_values", move || finite_singular_values(&inf).is_none()), "and for an infinity");
        assert!(within_10s("finite_svd", move || finite_svd(&inf2, true, true).is_none()), "and for an infinity");
    }
}
