//! **Savitzky–Golay filtering** (Savitzky & Golay, 1964) — smooth a noisy signal, or estimate its
//! derivatives, by fitting a low-order polynomial to a sliding window in the **least-squares** sense and
//! reading the fitted value (or its derivative) at the window centre. Unlike a moving average, it preserves
//! the height and width of peaks and the shape of trends — because a polynomial of order `d` passes through
//! them undistorted — while still averaging away noise. It is the workhorse smoother/differentiator for
//! encoder, IMU, force-torque, and other robot sensor streams, where a clean numerical derivative
//! (velocity from position, acceleration from velocity) is needed without amplifying noise.
//!
//! The key exactness: a Savitzky–Golay filter of order `d` **reproduces any polynomial of degree ≤ d
//! exactly** (zero bias), and its derivative filter returns those polynomials' exact derivatives — including
//! at the signal's edges, handled here by one-sided fits. Verified against both. Pure `nalgebra` →
//! WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A Savitzky–Golay filter with half-window `half_window` (window size `2·half_window + 1`) fitting
/// polynomials of degree `order` (`order ≤ 2·half_window`).
#[derive(Clone, Copy, Debug)]
pub struct SavGol {
    pub half_window: usize,
    pub order: usize,
}

impl SavGol {
    /// Smooth (`deriv = 0`) or differentiate (`deriv = 1` for velocity, `2` for acceleration, …) the
    /// `signal` sampled uniformly at spacing `dt`. Edge samples use one-sided polynomial fits. Returns a
    /// signal of the same length.
    pub fn apply(&self, signal: &[f64], dt: f64, deriv: usize) -> Vec<f64> {
        let n = signal.len();
        let m = self.half_window as isize;
        let d = self.order;
        // deriv! factor for reading the derivative off the polynomial coefficients
        let fact: f64 = (1..=deriv).map(|k| k as f64).product::<f64>().max(1.0);
        let scale = fact / dt.powi(deriv as i32);

        (0..n)
            .map(|i| {
                // window indices clamped to the signal, offsets measured from i
                let lo = (i as isize - m).max(0);
                let hi = (i as isize + m).min(n as isize - 1);
                let w = (hi - lo + 1) as usize;
                let cols = (d + 1).min(w);
                let a = DMatrix::from_fn(w, cols, |r, k| {
                    let x = (lo + r as isize - i as isize) as f64;
                    x.powi(k as i32)
                });
                let y = DVector::from_fn(w, |r, _| signal[(lo + r as isize) as usize]);
                // normal-equations least squares: c = (AᵀA)⁻¹ Aᵀ y
                let ata = a.transpose() * &a;
                let coeffs = ata.try_inverse().map(|inv| inv * a.transpose() * &y);
                match coeffs {
                    Some(c) if deriv < cols => c[deriv] * scale,
                    // derivative higher than the fitted order ⇒ identically zero
                    Some(_) => 0.0,
                    None => signal[i],
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_reproduces_a_low_order_polynomial_exactly() {
        // THE ORACLE. An order-3 filter applied to samples of a cubic returns the cubic unchanged (zero
        // bias), edges included — the defining Savitzky–Golay property.
        let dt = 0.1;
        let poly = |t: f64| 2.0 - 1.5 * t + 0.7 * t * t - 0.3 * t * t * t;
        let sig: Vec<f64> = (0..40).map(|i| poly(i as f64 * dt)).collect();
        let sg = SavGol { half_window: 4, order: 3 };
        let smoothed = sg.apply(&sig, dt, 0);
        for (i, (&s, orig)) in smoothed.iter().zip(&sig).enumerate() {
            assert!((s - orig).abs() < 1e-9, "sample {i}: {s} vs {orig}");
        }
    }

    #[test]
    fn its_derivative_filter_is_exact_on_polynomials() {
        // The order-3 derivative filter returns the analytic derivative of a cubic exactly.
        let dt = 0.05;
        let poly = |t: f64| 1.0 + 0.5 * t - 0.4 * t * t + 0.2 * t * t * t;
        let dpoly = |t: f64| 0.5 - 0.8 * t + 0.6 * t * t;
        let sig: Vec<f64> = (0..50).map(|i| poly(i as f64 * dt)).collect();
        let sg = SavGol { half_window: 5, order: 3 };
        let vel = sg.apply(&sig, dt, 1);
        for (i, &v) in vel.iter().enumerate() {
            let t = i as f64 * dt;
            assert!((v - dpoly(t)).abs() < 1e-7, "derivative at {t}: {v} vs {}", dpoly(t));
        }
    }

    #[test]
    fn it_reduces_noise_while_preserving_the_signal() {
        // THE APPLICATION. On a sine corrupted by deterministic noise, the smoothed signal is closer to the
        // clean sine than the raw noisy one.
        let dt = 0.02;
        let clean: Vec<f64> = (0..200).map(|i| (i as f64 * dt * 3.0).sin()).collect();
        let mut seed = 1u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.4
        };
        let noisy: Vec<f64> = clean.iter().map(|&c| c + noise()).collect();
        let sg = SavGol { half_window: 8, order: 3 };
        let smoothed = sg.apply(&noisy, dt, 0);
        let err = |a: &[f64]| a.iter().zip(&clean).map(|(x, c)| (x - c).powi(2)).sum::<f64>().sqrt();
        assert!(err(&smoothed) < 0.5 * err(&noisy), "smoothing should more than halve the error: {} vs {}", err(&smoothed), err(&noisy));
    }

    #[test]
    fn a_constant_signal_passes_through_unchanged() {
        let sg = SavGol { half_window: 3, order: 2 };
        let sig = vec![5.0; 20];
        let out = sg.apply(&sig, 1.0, 0);
        assert!(out.iter().all(|&x| (x - 5.0).abs() < 1e-12), "constant should be preserved");
        // and its derivative is zero
        let vel = sg.apply(&sig, 1.0, 1);
        assert!(vel.iter().all(|&x| x.abs() < 1e-9), "derivative of a constant is zero");
    }
}
