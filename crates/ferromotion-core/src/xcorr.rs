//! **Fast convolution & cross-correlation** (via the [`crate::fft`]). Two `O(n log n)` operations built on
//! the FFT: **linear convolution** (apply an FIR filter / smoothing kernel to a signal, or compute a running
//! response) and **cross-correlation with time-delay estimation** (find the integer lag that best aligns two
//! signals). In physical AI these power **sensor time-synchronization** (align an IMU stream to a camera or
//! encoder), **matched filtering** (detect a known template/chirp in a noisy stream), sound-source /
//! time-of-arrival lag estimation, and fast filtering — all far cheaper than the direct `O(n²)` sums for
//! long signals.
//!
//! Verified: FFT convolution matches the direct convolution sum (and convolving with an impulse is the
//! identity); autocorrelation peaks at zero lag; and a signal delayed by a known shift is recovered by
//! `time_delay`. Pure `nalgebra` → WASM-clean.

use crate::fft::{fft, ifft};
use nalgebra::Complex;

fn next_pow2(x: usize) -> usize {
    x.next_power_of_two().max(1)
}

/// Linear convolution `a ∗ b` (length `a.len() + b.len() − 1`) via zero-padded FFTs.
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;
    let n = next_pow2(out_len);
    let mut ca = vec![Complex::new(0.0, 0.0); n];
    let mut cb = vec![Complex::new(0.0, 0.0); n];
    for (i, &v) in a.iter().enumerate() {
        ca[i] = Complex::new(v, 0.0);
    }
    for (i, &v) in b.iter().enumerate() {
        cb[i] = Complex::new(v, 0.0);
    }
    let fa = fft(&ca);
    let fb = fft(&cb);
    let prod: Vec<Complex<f64>> = fa.iter().zip(&fb).map(|(x, y)| x * y).collect();
    ifft(&prod).iter().take(out_len).map(|z| z.re).collect()
}

/// **Circular cross-correlation** `r[k] = Σₙ a[n]·b[(n+k) mod N]` (both signals padded to a common power-of-
/// two length `N`), via `ifft(conj(fft(a))·fft(b))`. The lag maximizing `r` is the shift of `b` relative to
/// `a`.
pub fn cross_correlation(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = next_pow2(a.len().max(b.len()));
    let mut ca = vec![Complex::new(0.0, 0.0); n];
    let mut cb = vec![Complex::new(0.0, 0.0); n];
    for (i, &v) in a.iter().enumerate() {
        ca[i] = Complex::new(v, 0.0);
    }
    for (i, &v) in b.iter().enumerate() {
        cb[i] = Complex::new(v, 0.0);
    }
    let fa = fft(&ca);
    let fb = fft(&cb);
    let prod: Vec<Complex<f64>> = fa.iter().zip(&fb).map(|(x, y)| x.conj() * y).collect();
    ifft(&prod).iter().map(|z| z.re).collect()
}

/// The integer time delay of `b` relative to `a` (the circular shift `s` with `b[n] ≈ a[n − s]`), as the
/// lag maximizing the cross-correlation, mapped to the signed range `[−N/2, N/2)`.
pub fn time_delay(a: &[f64], b: &[f64]) -> isize {
    let r = cross_correlation(a, b);
    let n = r.len();
    let k = (0..n).max_by(|&i, &j| r[i].partial_cmp(&r[j]).unwrap()).unwrap();
    if k > n / 2 {
        k as isize - n as isize
    } else {
        k as isize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                out[i + j] += ai * bj;
            }
        }
        out
    }

    #[test]
    fn fft_convolution_matches_the_direct_sum() {
        // THE ORACLE. FFT convolution equals the O(n²) direct convolution.
        let a = [1.0, 2.0, 3.0, -1.0, 0.5];
        let b = [0.5, -1.0, 2.0];
        let fast = convolve(&a, &b);
        let direct = direct_convolve(&a, &b);
        assert_eq!(fast.len(), direct.len());
        for (f, d) in fast.iter().zip(&direct) {
            assert!((f - d).abs() < 1e-10, "conv {f} vs {d}");
        }
        // convolving with a unit impulse returns the signal (padded)
        let id = convolve(&a, &[1.0]);
        for (i, &v) in a.iter().enumerate() {
            assert!((id[i] - v).abs() < 1e-10, "impulse convolution is identity");
        }
    }

    #[test]
    fn autocorrelation_peaks_at_zero_lag() {
        let x: Vec<f64> = (0..32).map(|t| (t as f64 * 0.4).sin() + 0.3).collect();
        let r = cross_correlation(&x, &x);
        let peak = (0..r.len()).max_by(|&i, &j| r[i].partial_cmp(&r[j]).unwrap()).unwrap();
        assert_eq!(peak, 0, "autocorrelation should peak at lag 0");
    }

    #[test]
    fn it_recovers_a_known_time_delay() {
        // THE APPLICATION. b is a circularly delayed by 7; time_delay recovers +7.
        let n = 64;
        let a: Vec<f64> = (0..n).map(|t| (t as f64 * 0.7).sin() + 0.5 * (t as f64 * 0.13).cos()).collect();
        let shift = 7usize;
        let b: Vec<f64> = (0..n).map(|t| a[(t + n - shift) % n]).collect(); // b[t] = a[t − shift]
        assert_eq!(time_delay(&a, &b), shift as isize, "should recover the delay");
        // a negative delay too
        let c: Vec<f64> = (0..n).map(|t| a[(t + 5) % n]).collect(); // c[t] = a[t + 5] ⇒ delay −5
        assert_eq!(time_delay(&a, &c), -5, "should recover a negative delay");
    }
}
