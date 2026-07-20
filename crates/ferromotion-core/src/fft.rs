//! **Fast Fourier Transform** (iterative radix-2 Cooley–Tukey). An `O(n log n)` discrete Fourier transform
//! (and its inverse) for power-of-two lengths — the workhorse of frequency-domain signal processing that the
//! crate needed beyond the naive `O(n²)` DFT tucked inside [`crate::Icem`]. Its uses in physical AI: spectral
//! analysis of vibration / IMU / contact-force signals (find resonances, detect gait frequency), fast
//! convolution and correlation, spectral derivatives, and any frequency-domain filter.
//!
//! In-place iterative implementation (bit-reversal permutation + butterfly stages), so no recursion and one
//! allocation for the output. Verified: the transform of a unit impulse is flat, a pure sinusoid lands in a
//! single frequency bin, `ifft(fft(x)) = x`, Parseval's energy identity holds, and it matches a direct DFT.
//! Pure `nalgebra` (its re-exported `Complex`) → WASM-clean.

use nalgebra::Complex;
use std::f64::consts::PI;

// in-place iterative radix-2 FFT; `inverse` selects the sign of the exponent (no 1/n scaling here)
fn transform(x: &mut [Complex<f64>], inverse: bool) {
    let n = x.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two");
    if n <= 1 {
        return;
    }
    // bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    // butterfly stages
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * PI / len as f64;
        let wlen = Complex::new(ang.cos(), ang.sin());
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = x[i + k];
                let v = x[i + k + half] * w;
                x[i + k] = u + v;
                x[i + k + half] = u - v;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Forward FFT of a complex signal (length a power of two).
pub fn fft(x: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut out = x.to_vec();
    transform(&mut out, false);
    out
}

/// Inverse FFT (with the `1/n` normalization, so `ifft(fft(x)) = x`).
pub fn ifft(x: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut out = x.to_vec();
    transform(&mut out, true);
    let inv_n = 1.0 / x.len() as f64;
    for v in &mut out {
        *v *= inv_n;
    }
    out
}

/// Forward FFT of a real signal (convenience wrapper).
pub fn fft_real(x: &[f64]) -> Vec<Complex<f64>> {
    let c: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(x: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let n = x.len();
        (0..n)
            .map(|k| {
                (0..n).map(|t| { let a = -2.0 * PI * (k * t) as f64 / n as f64; x[t] * Complex::new(a.cos(), a.sin()) }).sum()
            })
            .collect()
    }

    #[test]
    fn the_impulse_transforms_to_a_flat_spectrum() {
        // THE ORACLE. FFT of a unit impulse [1,0,…] is all ones.
        let mut x = vec![Complex::new(0.0, 0.0); 8];
        x[0] = Complex::new(1.0, 0.0);
        let f = fft(&x);
        for v in f {
            assert!((v - Complex::new(1.0, 0.0)).norm() < 1e-12, "impulse spectrum should be flat");
        }
    }

    #[test]
    fn a_pure_sinusoid_lands_in_one_bin() {
        // A cosine at frequency bin 3 (n=16) has energy only at bins 3 and n−3.
        let n = 16;
        let k0 = 3;
        let x: Vec<Complex<f64>> = (0..n).map(|t| Complex::new((2.0 * PI * (k0 * t) as f64 / n as f64).cos(), 0.0)).collect();
        let f = fft(&x);
        for (k, v) in f.iter().enumerate() {
            if k == k0 || k == n - k0 {
                assert!((v.norm() - n as f64 / 2.0).abs() < 1e-9, "bin {k} should hold the sinusoid: {}", v.norm());
            } else {
                assert!(v.norm() < 1e-9, "bin {k} should be empty: {}", v.norm());
            }
        }
    }

    #[test]
    fn ifft_inverts_fft() {
        let x: Vec<Complex<f64>> = (0..32).map(|t| Complex::new((t as f64 * 0.3).sin(), (t as f64 * 0.1).cos())).collect();
        let back = ifft(&fft(&x));
        for (a, b) in x.iter().zip(&back) {
            assert!((a - b).norm() < 1e-12, "ifft∘fft should be identity");
        }
    }

    #[test]
    fn parsevals_identity_holds_and_it_matches_the_naive_dft() {
        let x: Vec<Complex<f64>> = (0..64).map(|t| Complex::new((t as f64 * 0.7).sin() + 0.5, 0.3 * (t as f64 * 0.2).cos())).collect();
        let f = fft(&x);
        // Parseval: Σ|x|² = (1/n) Σ|X|²
        let time_energy: f64 = x.iter().map(|v| v.norm_sqr()).sum();
        let freq_energy: f64 = f.iter().map(|v| v.norm_sqr()).sum::<f64>() / x.len() as f64;
        assert!((time_energy - freq_energy).abs() < 1e-8, "Parseval: {time_energy} vs {freq_energy}");
        // matches a direct DFT
        let dft = naive_dft(&x);
        for (a, b) in f.iter().zip(&dft) {
            assert!((a - b).norm() < 1e-8, "FFT should match the naive DFT");
        }
    }
}
