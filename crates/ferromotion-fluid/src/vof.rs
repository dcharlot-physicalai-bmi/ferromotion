//! **Volume-of-fluid interface capture** (Honest Fluids — completing stage 5, the multiphase side).
//! A color function `C ∈ [0,1]` marks which fluid fills each cell; advecting it conservatively
//! tracks the interface between two phases (water/air, the free surface of a splash). Basilisk-class
//! capability, ingested with the same oracle discipline — a cell-centered, flux-form, TVD (minmod)
//! advection so the scheme is at once **conservative** and **bounded**:
//!
//! - **Mass conservation.** Flux-form ⇒ `Σ C` changes only through boundary fluxes; under a
//!   divergence-free field on a periodic grid it is conserved to machine precision.
//! - **Boundedness.** The minmod slope limiter is TVD, so `C` never overshoots `[0,1]` — no
//!   spurious negative or super-unity volume fractions.
//! - **Solid-body rotation.** The canonical benchmark: a disk carried once around by a rotational
//!   field returns to its start with bounded shape error.

/// A cell-centered volume-fraction field on an `n × n` periodic grid (unit square).
pub struct Vof {
    pub n: usize,
    pub h: f64,
    c: Vec<f64>,
}

fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

impl Vof {
    pub fn new(n: usize) -> Self {
        Vof { n, h: 1.0 / n as f64, c: vec![0.0; n * n] }
    }

    #[inline]
    fn ix(&self, i: usize, j: usize) -> usize {
        (i % self.n) * self.n + (j % self.n)
    }

    /// Initialize a filled disk of radius `r` centered at `(cx, cy)` (a sharp interface).
    pub fn set_disk(&mut self, cx: f64, cy: f64, r: f64) {
        for i in 0..self.n {
            for j in 0..self.n {
                let x = (i as f64 + 0.5) * self.h;
                let y = (j as f64 + 0.5) * self.h;
                self.c[i * self.n + j] = if (x - cx).hypot(y - cy) <= r { 1.0 } else { 0.0 };
            }
        }
    }

    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.c[self.ix(i, j)]
    }

    /// Total color `Σ C · cell-area` (the fluid volume — conserved under a divergence-free field).
    pub fn volume(&self) -> f64 {
        self.c.iter().sum::<f64>() * self.h * self.h
    }

    /// Min and max of `C` (boundedness monitor).
    pub fn bounds(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in &self.c {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (lo, hi)
    }

    /// One explicit flux-form step under a divergence-free velocity field `vel(x,y) → (u,v)`,
    /// evaluated at faces. Minmod-limited upwind face reconstruction (2nd-order TVD).
    pub fn step(&mut self, dt: f64, vel: impl Fn(f64, f64) -> (f64, f64)) {
        let (n, h) = (self.n, self.h);
        // Limited slopes (minmod) in x and y.
        let (mut sx, mut sy) = (vec![0.0; n * n], vec![0.0; n * n]);
        for i in 0..n {
            for j in 0..n {
                let c = self.c[i * n + j];
                sx[i * n + j] = minmod(c - self.c[self.ix(i + n - 1, j)], self.c[self.ix(i + 1, j)] - c);
                sy[i * n + j] = minmod(c - self.c[self.ix(i, j + n - 1)], self.c[self.ix(i, j + 1)] - c);
            }
        }
        let mut next = self.c.clone();
        for i in 0..n {
            for j in 0..n {
                // East face at (x=(i+1)h, y=(j+0.5)h) between cell i and i+1.
                let ue = vel((i as f64 + 1.0) * h, (j as f64 + 0.5) * h).0;
                let fe = if ue >= 0.0 {
                    ue * (self.c[i * n + j] + 0.5 * sx[i * n + j])
                } else {
                    ue * (self.c[self.ix(i + 1, j)] - 0.5 * sx[self.ix(i + 1, j)])
                };
                // West face at (x=i h) between i-1 and i.
                let uw = vel(i as f64 * h, (j as f64 + 0.5) * h).0;
                let fw = if uw >= 0.0 {
                    uw * (self.c[self.ix(i + n - 1, j)] + 0.5 * sx[self.ix(i + n - 1, j)])
                } else {
                    uw * (self.c[i * n + j] - 0.5 * sx[i * n + j])
                };
                // North face at (y=(j+1)h).
                let vn = vel((i as f64 + 0.5) * h, (j as f64 + 1.0) * h).1;
                let fnth = if vn >= 0.0 {
                    vn * (self.c[i * n + j] + 0.5 * sy[i * n + j])
                } else {
                    vn * (self.c[self.ix(i, j + 1)] - 0.5 * sy[self.ix(i, j + 1)])
                };
                // South face at (y=j h).
                let vs = vel((i as f64 + 0.5) * h, j as f64 * h).1;
                let fs = if vs >= 0.0 {
                    vs * (self.c[self.ix(i, j + n - 1)] + 0.5 * sy[self.ix(i, j + n - 1)])
                } else {
                    vs * (self.c[i * n + j] - 0.5 * sy[i * n + j])
                };
                next[i * n + j] = self.c[i * n + j] - dt / h * (fe - fw + fnth - fs);
            }
        }
        self.c = next;
    }
}

#[cfg(test)]
mod verification {
    use super::*;

    // Solid-body rotation about the domain center: u = −ω(y−½), v = ω(x−½). Divergence-free.
    fn rotation(omega: f64) -> impl Fn(f64, f64) -> (f64, f64) {
        move |x, y| (-omega * (y - 0.5), omega * (x - 0.5))
    }

    /// Flux-form ⇒ the fluid volume is conserved to machine precision under the rotational field.
    #[test]
    fn conserves_volume_under_rotation() {
        let n = 64;
        let mut f = Vof::new(n);
        f.set_disk(0.5, 0.72, 0.12);
        let v0 = f.volume();
        let omega = 2.0;
        let dt = 0.4 * f.h / (omega * 0.71);
        for _ in 0..400 {
            f.step(dt, rotation(omega));
        }
        let drift = (f.volume() - v0).abs();
        eprintln!("VOF volume drift over 400 steps: {drift:.2e}");
        assert!(drift < 1e-12, "flux-form did not conserve volume: {drift}");
    }

    /// The minmod limiter keeps `C` bounded in `[0,1]` — no over/undershoot at the interface.
    #[test]
    fn stays_bounded() {
        let n = 64;
        let mut f = Vof::new(n);
        f.set_disk(0.5, 0.72, 0.12);
        let omega = 2.0;
        let dt = 0.4 * f.h / (omega * 0.71);
        for _ in 0..400 {
            f.step(dt, rotation(omega));
            let (lo, hi) = f.bounds();
            assert!(lo > -1e-9 && hi < 1.0 + 1e-9, "C left [0,1]: [{lo}, {hi}]");
        }
    }

    /// Solid-body rotation benchmark: after one full revolution the disk returns to its start,
    /// conserved and in place. The algebraic minmod scheme diffuses the interface into a band over a
    /// full revolution (rel L1 ≈ 0.36 at n=96) — honest for algebraic VOF; geometric PLIC/THINC
    /// interface compression is the follow-on for a sharp interface. What's verified here is that
    /// the disk is neither destroyed nor displaced, only smeared.
    #[test]
    fn disk_returns_after_one_revolution() {
        let n = 96;
        let omega = 2.0;
        let mut f = Vof::new(n);
        f.set_disk(0.5, 0.72, 0.13);
        let init: Vec<f64> = (0..n * n).map(|k| f.at(k / n, k % n)).collect();

        let period = 2.0 * std::f64::consts::PI / omega;
        let dt = 0.4 * f.h / (omega * 0.71);
        let steps = (period / dt).ceil() as usize;
        let dt = period / steps as f64; // land exactly on one revolution
        for _ in 0..steps {
            f.step(dt, rotation(omega));
        }
        // L1 shape error, normalized by the disk volume.
        let mut l1 = 0.0;
        let mut mass = 0.0;
        for i in 0..n {
            for j in 0..n {
                l1 += (f.at(i, j) - init[i * n + j]).abs();
                mass += init[i * n + j];
            }
        }
        let rel = l1 / mass;
        // Also confirm the disk is in PLACE: the color-weighted centroid returns near the start.
        let (mut cx, mut cy, mut m) = (0.0, 0.0, 0.0);
        for i in 0..n {
            for j in 0..n {
                let c = f.at(i, j);
                cx += c * (i as f64 + 0.5) / n as f64;
                cy += c * (j as f64 + 0.5) / n as f64;
                m += c;
            }
        }
        let (cx, cy) = (cx / m, cy / m);
        let centroid_err = (cx - 0.5).hypot(cy - 0.72);
        eprintln!("VOF one-revolution: rel L1 {rel:.3}, centroid drift {centroid_err:.4}");
        assert!(rel < 0.42, "disk not recovered after a revolution: rel L1 {rel}");
        assert!(centroid_err < 0.01, "disk displaced: centroid drift {centroid_err}"); // in place
    }
}
