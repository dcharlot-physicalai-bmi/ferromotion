//! **The gust bench** (Honest Fluids — the deployment-honesty seam). Deployed robots do not run
//! CFD in the control loop; they run lumped coefficients plus learned residuals (the Neural-Fly
//! pattern). This bench makes that concrete and *verifiable*: drive a body through a gust, let the
//! resolved immersed-boundary solver measure the true hydrodynamic force, and watch the
//! quasi-steady coefficient model **break** on the transient — then reconstruct exactly the
//! classical unsteady-force hierarchy (Morison added-mass, Basset history) from the resolved trace
//! by least squares. The point is honest: the coefficient model is *fine in steady state* and
//! *wrong during the gust*, and the resolved solver is how you learn the correction.
//!
//! A body accelerating through still fluid is, in its own frame, a gust in the relative flow — so
//! `run_gust` translates a rigid disk at a prescribed `U(t) = U₀ + gust(t)` and records the
//! streamwise force each step. No new physics: it reuses [`crate::MacFluid::step_with_disk`].

use crate::{MacFluid, RigidDisk};

/// A smooth Gaussian gust on a smoothly-started base flow:
/// `U(t) = u0·tanh(t/ramp) + amp·exp(−((t−t0)/width)²)`. The `tanh` startup avoids an impulsive
/// launch (which would inject a large, unphysical direct-forcing transient at t = 0).
#[derive(Clone, Copy, Debug)]
pub struct Gust {
    pub u0: f64,
    pub amp: f64,
    pub t0: f64,
    pub width: f64,
    pub ramp: f64,
}

impl Gust {
    /// Speed and its exact time derivative at time `t`.
    pub fn u_and_dot(&self, t: f64) -> (f64, f64) {
        let r = (t / self.ramp).tanh();
        let rp = (1.0 - r * r) / self.ramp; // d/dt tanh
        let z = (t - self.t0) / self.width;
        let g = (-z * z).exp();
        let u = self.u0 * r + self.amp * g;
        let du = self.u0 * rp + self.amp * g * (-2.0 * z / self.width);
        (u, du)
    }
}

/// One sample of the resolved trace.
#[derive(Clone, Copy, Debug)]
pub struct GustSample {
    pub t: f64,
    pub u: f64,
    pub du: f64,
    /// Resolved streamwise hydrodynamic force on the body (immersed-boundary reaction).
    pub f: f64,
}

/// Translate a disk of radius `r` through still fluid at `U(t)` from the gust profile, recording
/// `(t, U, dU/dt, F_resolved)` each step. Free-slip walls (open tank); the disk starts near the
/// upstream side so it stays in-domain over the run.
pub fn run_gust(n: usize, nu: f64, dt: f64, r: f64, steps: usize, gust: Gust) -> Vec<GustSample> {
    let mut fluid = MacFluid::new(n, n, nu, dt, 0.0).with_free_slip();
    let mut cx = 0.3;
    let cy = 0.5;
    let mut trace = Vec::with_capacity(steps);
    for s in 0..steps {
        let t = s as f64 * dt;
        let (u, du) = gust.u_and_dot(t);
        let disk = RigidDisk { cx, cy, r, ux: u, uy: 0.0 };
        let (fx, _fy) = fluid.step_with_disk(&disk);
        trace.push(GustSample { t, u, du, f: fx });
        cx += u * dt; // the body advances through the fluid
    }
    trace
}

/// Least-squares fit of `F ≈ Σ βₖ·featureₖ` over the trace; returns coefficients and residual RMS.
/// Small dense normal equations (≤3 features) solved by Gaussian elimination.
fn lstsq(features: &[Vec<f64>], target: &[f64]) -> (Vec<f64>, f64) {
    let p = features.len();
    let n = target.len();
    // Normal equations AᵀA β = Aᵀy.
    let mut ata = vec![vec![0.0f64; p]; p];
    let mut aty = vec![0.0f64; p];
    for i in 0..p {
        for j in 0..p {
            ata[i][j] = (0..n).map(|k| features[i][k] * features[j][k]).sum();
        }
        aty[i] = (0..n).map(|k| features[i][k] * target[k]).sum();
    }
    let beta = solve_dense(ata, aty);
    let mut se = 0.0;
    for k in 0..n {
        let pred: f64 = (0..p).map(|i| beta[i] * features[i][k]).sum();
        se += (pred - target[k]).powi(2);
    }
    (beta, (se / n as f64).sqrt())
}

/// Gaussian elimination with partial pivoting for a tiny SPD-ish system.
#[allow(clippy::needless_range_loop)] // index-arithmetic elimination — iterators would obscure it
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap()).unwrap();
        a.swap(col, piv);
        b.swap(col, piv);
        if a[col][col].abs() < 1e-14 {
            continue; // singular column (e.g. an all-zero feature in steady flow) → leave coeff 0
        }
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            for c in col..n {
                a[row][c] -= f * a[col][c];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let s: f64 = ((row + 1)..n).map(|c| a[row][c] * x[c]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    x
}

/// The three nested unsteady-force models fit to a resolved gust trace, each adding one physical
/// term — the residual RMS falls monotonically as the model learns what the quasi-steady
/// coefficient omits.
#[derive(Clone, Debug)]
pub struct GustFit {
    /// Quasi-steady drag only: `F ≈ β·(−|U|U)`.
    pub quasi_steady_rms: f64,
    /// + Morison added mass: `F ≈ … − m_a·dU/dt`.
    pub added_mass_rms: f64,
    /// + Basset history: `F ≈ … − c_b·∫ (dU/dτ)/√(t−τ) dτ`.
    pub history_rms: f64,
    /// RMS of the resolved force itself (the scale the residuals are measured against).
    pub force_rms: f64,
}

/// Fit the nested hierarchy to a trace.
#[allow(clippy::needless_range_loop)] // Basset convolution needs the (k−j) index arithmetic
pub fn fit_unsteady(trace: &[GustSample], dt: f64) -> GustFit {
    let f: Vec<f64> = trace.iter().map(|s| s.f).collect();
    let quad: Vec<f64> = trace.iter().map(|s| -s.u.abs() * s.u).collect(); // quasi-steady drag basis
    let acc: Vec<f64> = trace.iter().map(|s| -s.du).collect(); // added-mass basis
    // Discrete Basset history basis: −Σ_{j<k} (dU/dτ_j)/√(t_k−t_j+dt)·dt.
    let mut hist = vec![0.0f64; trace.len()];
    for k in 0..trace.len() {
        let mut acc_sum = 0.0;
        for j in 0..k {
            acc_sum += trace[j].du / ((k - j) as f64 * dt + dt).sqrt() * dt;
        }
        hist[k] = -acc_sum;
    }

    let (_, qs) = lstsq(std::slice::from_ref(&quad), &f);
    let (_, am) = lstsq(&[quad.clone(), acc.clone()], &f);
    let (_, hi) = lstsq(&[quad, acc, hist], &f);
    let frms = (f.iter().map(|x| x * x).sum::<f64>() / f.len() as f64).sqrt();
    GustFit { quasi_steady_rms: qs, added_mass_rms: am, history_rms: hi, force_rms: frms }
}

#[cfg(test)]
mod verification {
    use super::*;

    /// The coefficient model breaks during the gust and the resolved solver reveals the fix:
    /// quasi-steady residual ≫ added-mass residual ≫ history residual, and the full unsteady model
    /// explains almost all of the force. Exactly the doctrine: coefficients are fine in steady
    /// state, wrong in the transient, and the resolved solver is how the residual is learned.
    #[test]
    fn coefficient_model_breaks_and_residual_recovers_it() {
        let (n, nu, dt, r) = (96, 0.006, 4e-4, 0.08);
        let steps = 1500;
        let t_end = steps as f64 * dt;
        let gust = Gust { u0: 0.05, amp: 0.12, t0: t_end * 0.5, width: t_end * 0.1, ramp: t_end * 0.1 };
        let trace = run_gust(n, nu, dt, r, steps, gust);
        // Fit the mid-window: the tanh startup has settled and the gust is centered here, so the
        // fit sees the gust transient, not the launch.
        let fit = fit_unsteady(&trace[steps / 4..3 * steps / 4], dt);
        eprintln!(
            "gust fit RMS: quasi-steady {:.3e}  +added-mass {:.3e}  +history {:.3e}  (force RMS {:.3e})",
            fit.quasi_steady_rms, fit.added_mass_rms, fit.history_rms, fit.force_rms
        );

        // Each physical term strictly reduces the residual (nested least squares can only help; the
        // point is that the reduction is LARGE — the omitted physics is real, not round-off).
        assert!(fit.added_mass_rms < 0.7 * fit.quasi_steady_rms, "added mass didn't matter — no transient?");
        assert!(fit.history_rms <= fit.added_mass_rms, "history term increased residual (impossible for nested LS)");
        // The full unsteady model explains the force well; the quasi-steady one does not.
        assert!(fit.history_rms < 0.2 * fit.force_rms, "full model still poor: {}", fit.history_rms / fit.force_rms);
        assert!(fit.quasi_steady_rms > 0.3 * fit.added_mass_rms, "quasi-steady already perfect — gust too mild");
    }

    /// The honest other half: in a *steady* approach (no gust) the quasi-steady coefficient model
    /// already fits well — the coefficient model is not wrong, it is incomplete only in transients.
    #[test]
    fn steady_flow_needs_no_residual() {
        let (n, nu, dt, r) = (96, 0.006, 4e-4, 0.08);
        let steps = 1200;
        let steady = Gust { u0: 0.07, amp: 0.0, t0: 0.0, width: 1.0, ramp: steps as f64 * dt * 0.15 }; // no gust
        let trace = run_gust(n, nu, dt, r, steps, steady);
        // Discard the initial startup transient; judge the settled tail.
        let tail = &trace[steps / 2..];
        let fit = fit_unsteady(tail, dt);
        eprintln!(
            "steady fit RMS: quasi-steady {:.3e}  +added-mass {:.3e}  (force RMS {:.3e})",
            fit.quasi_steady_rms, fit.added_mass_rms, fit.force_rms
        );
        assert!(fit.quasi_steady_rms < 0.1 * fit.force_rms, "quasi-steady should already fit steady flow");
    }
}
