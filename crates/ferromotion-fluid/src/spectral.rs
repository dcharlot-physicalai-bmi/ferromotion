//! **Fourier pseudo-spectral operators** (Honest Fluids — stage 7, the spectral paradigm). On a
//! periodic domain, differentiation is exact in Fourier space (`∂ₓ ↔ ik`) and the Poisson solve is a
//! division — the pressure projection of a spectral Navier–Stokes code. The identity property is
//! *spectral accuracy*: for a smooth periodic field the error is not polynomial in the grid but
//! **exponentially small**, hitting machine precision at modest resolution. Verified as exactly
//! that, against finite differences which only ever converge polynomially.
//!
//! A self-contained radix-2 Cooley–Tukey FFT (power-of-two lengths) backs a spectral derivative,
//! Laplacian, and periodic Poisson solve.

use std::f64::consts::PI;

/// A minimal complex number for the transform.
#[derive(Clone, Copy)]
pub struct Cplx {
    pub re: f64,
    pub im: f64,
}
impl Cplx {
    pub fn new(re: f64, im: f64) -> Self {
        Cplx { re, im }
    }
    fn add(self, o: Cplx) -> Cplx {
        Cplx::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Cplx) -> Cplx {
        Cplx::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: Cplx) -> Cplx {
        Cplx::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

/// In-place iterative radix-2 FFT (`len` must be a power of two). `inverse` applies the conjugate
/// transform and the `1/N` scaling.
pub fn fft(a: &mut [Cplx], inverse: bool) {
    let n = a.len();
    debug_assert!(n.is_power_of_two(), "FFT length must be a power of two");
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
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = (if inverse { 2.0 } else { -2.0 }) * PI / len as f64;
        let wl = Cplx::new(ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut w = Cplx::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = a[i + k];
                let v = a[i + k + len / 2].mul(w);
                a[i + k] = u.add(v);
                a[i + k + len / 2] = u.sub(v);
                w = w.mul(wl);
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        for x in a.iter_mut() {
            x.re /= n as f64;
            x.im /= n as f64;
        }
    }
}

/// Angular wavenumbers for an `n`-point periodic signal on `[0, L)`: `k_m = 2π m / L` for the first
/// half, negative for the second (standard FFT ordering).
pub fn wavenumbers(n: usize, l: f64) -> Vec<f64> {
    (0..n).map(|m| if m <= n / 2 { 2.0 * PI * m as f64 / l } else { 2.0 * PI * (m as f64 - n as f64) / l }).collect()
}

/// Spectral first derivative of a real periodic signal on `[0, L)`.
pub fn derivative(f: &[f64], l: f64) -> Vec<f64> {
    let n = f.len();
    let mut a: Vec<Cplx> = f.iter().map(|&x| Cplx::new(x, 0.0)).collect();
    fft(&mut a, false);
    let k = wavenumbers(n, l);
    for m in 0..n {
        // multiply by i·k (drop the Nyquist mode's imaginary derivative to stay real for even n)
        let ik = if m == n / 2 && n.is_multiple_of(2) { Cplx::new(0.0, 0.0) } else { Cplx::new(0.0, k[m]) };
        a[m] = a[m].mul(ik);
    }
    fft(&mut a, true);
    a.iter().map(|c| c.re).collect()
}

/// Solve the periodic Poisson equation `∇²φ = f` on the `n × n` square `[0, L)²` spectrally
/// (`φ̂ = −f̂/|k|²`, mean-zero). `f` is row-major `n×n`.
pub fn poisson_2d(f: &[f64], n: usize, l: f64) -> Vec<f64> {
    let mut a: Vec<Cplx> = f.iter().map(|&x| Cplx::new(x, 0.0)).collect();
    fft2(&mut a, n, false);
    let k = wavenumbers(n, l);
    for i in 0..n {
        for j in 0..n {
            let k2 = k[i] * k[i] + k[j] * k[j];
            let idx = i * n + j;
            if k2 == 0.0 {
                a[idx] = Cplx::new(0.0, 0.0); // mean-zero gauge
            } else {
                a[idx] = a[idx].mul(Cplx::new(-1.0 / k2, 0.0));
            }
        }
    }
    fft2(&mut a, n, true);
    a.iter().map(|c| c.re).collect()
}

/// 2-D FFT of a row-major `n×n` array: transform each row, then each column.
fn fft2(a: &mut [Cplx], n: usize, inverse: bool) {
    let mut row = vec![Cplx::new(0.0, 0.0); n];
    for i in 0..n {
        row.copy_from_slice(&a[i * n..i * n + n]);
        fft(&mut row, inverse);
        a[i * n..i * n + n].copy_from_slice(&row);
    }
    let mut col = vec![Cplx::new(0.0, 0.0); n];
    for j in 0..n {
        for i in 0..n {
            col[i] = a[i * n + j];
        }
        fft(&mut col, inverse);
        for i in 0..n {
            a[i * n + j] = col[i];
        }
    }
}

#[cfg(test)]
mod verification {
    use super::*;

    /// The spectral derivative of a trigonometric polynomial is EXACT — machine precision, not a
    /// convergence rate. `d/dx sin(2πx) = 2π cos(2πx)` recovered to round-off.
    #[test]
    fn spectral_derivative_is_exact_for_band_limited() {
        let n = 32;
        let l = 1.0;
        let f: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin() + 0.5 * (4.0 * PI * i as f64 / n as f64).cos()).collect();
        let d = derivative(&f, l);
        let mut err = 0.0f64;
        for i in 0..n {
            let x = i as f64 / n as f64;
            let exact = 2.0 * PI * (2.0 * PI * x).cos() - 0.5 * 4.0 * PI * (4.0 * PI * x).sin();
            err = err.max((d[i] - exact).abs());
        }
        eprintln!("spectral derivative max error (band-limited): {err:.2e}");
        assert!(err < 1e-11, "spectral derivative not exact: {err}");
    }

    /// Spectral accuracy: for a smooth (analytic) periodic field the derivative error falls
    /// EXPONENTIALLY with resolution — orders of magnitude per grid doubling, faster than any
    /// polynomial. (A finite difference would gain a fixed factor of 4 per doubling.)
    #[test]
    fn convergence_is_spectral_not_polynomial() {
        let l = 2.0 * PI;
        // f = exp(sin x): smooth, periodic, NOT band-limited. f' = cos x · exp(sin x).
        let err_at = |n: usize| -> f64 {
            let f: Vec<f64> = (0..n).map(|i| (l * i as f64 / n as f64).sin().exp()).collect();
            let d = derivative(&f, l);
            let mut e = 0.0f64;
            for i in 0..n {
                let x = l * i as f64 / n as f64;
                e = e.max((d[i] - x.cos() * x.sin().exp()).abs());
            }
            e
        };
        let e8 = err_at(8);
        let e16 = err_at(16);
        let e32 = err_at(32);
        eprintln!("spectral convergence: e8 {e8:.2e}  e16 {e16:.2e}  e32 {e32:.2e}");
        // Each doubling cuts the error by FAR more than the factor-4 of a 2nd-order FD scheme.
        assert!(e16 < e8 / 50.0, "not super-algebraic (8→16): {e8} → {e16}");
        assert!(e32 < 1e-12, "did not reach machine precision by n=32: {e32}");
    }

    /// The 2-D periodic Poisson solve — the spectral pressure projection — recovers a manufactured
    /// solution to machine precision.
    #[test]
    fn poisson_2d_recovers_manufactured_solution() {
        let n = 32;
        let l = 1.0;
        let k = 2.0 * PI;
        // φ = sin(2πx)cos(4πy) (mean zero, periodic); f = ∇²φ = −(k² + (2k)²)φ.
        let phi_exact = |x: f64, y: f64| (k * x).sin() * (2.0 * k * y).cos();
        let lam = -(k * k + (2.0 * k) * (2.0 * k));
        let mut f = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                f[i * n + j] = lam * phi_exact(i as f64 / n as f64, j as f64 / n as f64);
            }
        }
        let phi = poisson_2d(&f, n, l);
        let mut err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                err = err.max((phi[i * n + j] - phi_exact(i as f64 / n as f64, j as f64 / n as f64)).abs());
            }
        }
        eprintln!("spectral 2-D Poisson max error: {err:.2e}");
        assert!(err < 1e-11, "spectral Poisson not exact: {err}");
    }
}
