//! **Volume-of-fluid interface capture** (Honest Fluids — completing stage 5, the multiphase side).
//! A color function `C ∈ [0,1]` marks which fluid fills each cell; advecting it conservatively
//! tracks the interface between two phases (water/air, the free surface of a splash). Basilisk-class
//! capability, ingested with the same oracle discipline — a cell-centered, flux-form, TVD (minmod)
//! advection so the scheme is at once **conservative** and **bounded**:
//!
//! - **Mass conservation.** Flux-form ⇒ `Σ C` changes only through boundary fluxes; under a
//!   divergence-free field on a periodic grid it is conserved to machine precision.
//! - **Boundedness, under a CFL condition.** The minmod limiter is TVD, but TVD for a MUSCL scheme
//!   marched with **forward Euler** holds only below a Courant limit — and [`Vof::step`] accepts any
//!   `dt` the caller passes. This doc previously asserted boundedness *unconditionally* ("`C` never
//!   overshoots `[0,1]`"), which is false (2026-08-14). Measured on the solid-body-rotation benchmark
//!   below, over 200 steps. The tests there scale `dt = factor·h/(ω·0.71)`; note that `factor` is *not*
//!   the cell CFL, because `ω·0.71` bounds the speed **magnitude** while the per-component Courant
//!   number governing a dimension-summed flux update is smaller by about √2. Measured
//!   `CFL ≈ 0.69·factor`:
//!
//!   | `factor` | per-component CFL ([`Vof::max_cfl`]) | worst `C` |
//!   |---|---|---|
//!   | 0.4 | 0.28 | exactly `[0, 1]`, volume drift 1.7e-16 |
//!   | 0.6 | 0.41 | exactly `[0, 1]` |
//!   | **0.8** | **0.55** | **−1.34e-2 — a negative volume fraction** |
//!   | 0.9 | 0.62 | −3.1e11 (blow-up) |
//!   | 1.0 | 0.69 | −8.1e25 |
//!
//!   Keep the per-component cell CFL at or below **0.5**, the classical MUSCL/forward-Euler bound. The
//!   measured onset straddles exactly that: bounded at 0.41, negative fractions by 0.55.
//!   [`Vof::max_cfl`] computes it for your own `dt` and field, and [`Vof::bounds`] reports what actually
//!   happened. Note that flux-form conservation **outlives** boundedness — at `factor` 0.8 the volume
//!   drift was still 0 — so a conservation check cannot stand in for a boundedness check.
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
    ///
    /// **Propagates `NaN` deliberately (2026-08-14).** `f64::min`/`f64::max` return the *non-`NaN`*
    /// operand by IEEE-754, so every `NaN` cell was simply skipped and the interval described only the
    /// finite ones — from the single function whose job is to report that `C` left `[0,1]`. The sharp
    /// case is a field with *some* `NaN` cells among finite neighbours, which reported a clean, plausible
    /// sub-interval of `[0,1]`. (A wholly non-finite field happened to come back as `[-inf, inf]`, which
    /// a caller testing `is_finite` would catch — but one testing `lo >= 0 && hi <= 1` would too, and
    /// neither catches the mixed case.) Past the CFL limit this field does go non-finite, so the monitor
    /// has to survive the case it exists to catch.
    pub fn bounds(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in &self.c {
            if v.is_nan() {
                return (f64::NAN, f64::NAN);
            }
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (lo, hi)
    }

    /// The largest cell Courant number `|u|·dt/h` this `dt` and velocity field produce, taken over
    /// every face the step actually samples.
    ///
    /// [`Vof::step`] is a MUSCL scheme marched with forward Euler, so its TVD/boundedness property is
    /// **conditional** on this staying small — keep it at or below `0.5`. It is offered as a check
    /// rather than enforced inside `step`, because the caller owns the timestep and a solver that
    /// silently clamps `dt` is a worse surprise than one that lets you assert.
    pub fn max_cfl(&self, dt: f64, vel: impl Fn(f64, f64) -> (f64, f64)) -> f64 {
        let (n, h) = (self.n, self.h);
        let mut worst = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let ue = vel((i as f64 + 1.0) * h, (j as f64 + 0.5) * h).0;
                let vn = vel((i as f64 + 0.5) * h, (j as f64 + 1.0) * h).1;
                for s in [ue, vn] {
                    let c = s.abs() * dt / h;
                    worst = if worst.is_nan() || c.is_nan() { f64::NAN } else { worst.max(c) };
                }
            }
        }
        worst
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

    /// Boundedness is **conditional on the CFL number**, which the module doc used to assert
    /// unconditionally. `step` marches a MUSCL reconstruction with forward Euler and accepts any `dt`.
    #[test]
    fn boundedness_is_conditional_on_the_cfl_number() {
        let omega = 2.0;
        let worst_over = |factor: f64| {
            let mut f = Vof::new(48);
            f.set_disk(0.5, 0.75, 0.15);
            let dt = factor * f.h / (omega * 0.71);
            let cfl = f.max_cfl(dt, rotation(omega));
            let (mut wlo, mut whi) = (0.0f64, 1.0f64);
            for _ in 0..200 {
                f.step(dt, rotation(omega));
                let (lo, hi) = f.bounds();
                if !(lo.is_finite() && hi.is_finite()) {
                    return (cfl, f64::NAN, f64::NAN);
                }
                wlo = wlo.min(lo);
                whi = whi.max(hi);
            }
            (cfl, wlo, whi)
        };

        // `factor` is NOT the cell CFL. The shipped scaling divides by ω·0.71, which bounds the speed
        // MAGNITUDE, while the Courant number that governs a dimension-summed flux update is
        // per-component and smaller by about √2. Asserting factor == CFL is how a first version of this
        // test failed — measured, CFL ≈ 0.69·factor. Pin that relationship so the doc's table stays
        // honest if the field or the face sampling ever changes.
        let (cfl04, lo04, hi04) = worst_over(0.4);
        assert!(
            (cfl04 - 0.276).abs() < 0.02,
            "per-component CFL at factor 0.4 should be ≈0.276, got {cfl04}; the doc's factor→CFL table \
             needs re-measuring"
        );

        // At and below the shipped step, C stays in [0,1] exactly. CFL here is 0.28 and 0.41.
        assert!(lo04 >= 0.0 && hi04 <= 1.0, "factor 0.4 (CFL 0.28) must stay bounded: [{lo04}, {hi04}]");
        let (cfl06, lo06, hi06) = worst_over(0.6);
        assert!(cfl06 < 0.5, "factor 0.6 should still be inside the 0.5 bound, got CFL {cfl06}");
        assert!(lo06 >= 0.0 && hi06 <= 1.0, "factor 0.6 (CFL 0.41) must stay bounded: [{lo06}, {hi06}]");

        // Past 0.5 the limiter does NOT save the scheme: measured −1.34e-2 at CFL 0.55, then blow-up.
        let (cfl08, lo08, _) = worst_over(0.8);
        assert!(cfl08 > 0.5, "factor 0.8 should be past the 0.5 bound, got CFL {cfl08}");
        assert!(
            lo08 < -1e-4 || lo08.is_nan(),
            "factor 0.8 (CFL {cfl08}) produced a bounded field ({lo08}); if the scheme or the limiter \
             changed, the CFL table in the module doc needs re-measuring rather than trusting"
        );
        let (_, lo09, _) = worst_over(0.9);
        assert!(lo09.is_nan() || lo09 < -1e6, "factor 0.9 should be a blow-up, got {lo09}");
    }

    /// The boundedness monitor must survive the case it exists to report.
    #[test]
    fn the_bounds_monitor_does_not_hide_a_non_finite_field() {
        let omega = 2.0;
        let mut f = Vof::new(32);
        f.set_disk(0.5, 0.75, 0.15);
        // Far past the CFL limit: this diverges to non-finite within a few hundred steps.
        let dt = 8.0 * f.h / (omega * 0.71);
        // Detect the NaN field INDEPENDENTLY of bounds(), via the raw accessor. The whole defect is that
        // bounds() disagrees with the field, so using bounds() to decide when to check bounds() makes the
        // test fail on the wrong assertion with a misleading message.
        let n = f.n;
        let field_has_nan = |g: &Vof| (0..n * n).any(|k| g.at(k / n, k % n).is_nan());
        let mut steps = 0;
        while steps < 400 && !field_has_nan(&f) {
            f.step(dt, rotation(omega));
            steps += 1;
        }
        assert!(field_has_nan(&f), "expected this CFL to drive C to NaN within 400 steps so the monitor can be checked");
        let (lo, hi) = f.bounds();
        assert!(
            lo.is_nan() && hi.is_nan(),
            "bounds() reported the finite interval [{lo}, {hi}] for a NaN field — f64::min/max return \
             the non-NaN operand, so the boundedness monitor was the one thing that could not report \
             unboundedness"
        );
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
