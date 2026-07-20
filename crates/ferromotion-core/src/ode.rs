//! **Adaptive ODE integration — Dormand–Prince RK45** (DOPRI5), the method behind MATLAB's `ode45` and
//! SciPy's `RK45`. A single step evaluates the right-hand side seven times to produce a **fifth-order**
//! estimate together with an **embedded fourth-order** estimate; their difference is a local error estimate
//! that a controller uses to grow or shrink the step size — big steps through smooth stretches, small steps
//! through fast transients — hitting a requested tolerance at near-minimal cost. This is the workhorse
//! integrator for simulating the dynamics/control models in the crate to a guaranteed accuracy, where the
//! fixed-step RK4 dotted through the other modules cannot certify its error.
//!
//! Verified: on `y' = −k y` it matches the analytic `e^{−kt}` to the requested tolerance; on the harmonic
//! oscillator it tracks `cos ωt` and nearly conserves energy; a tighter tolerance yields a smaller error;
//! and the step size genuinely adapts (a fast region takes more steps than a slow one). Pure `nalgebra` →
//! WASM-clean.

use nalgebra::DVector;

// Dormand–Prince (DOPRI5) Butcher tableau.
const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;

/// One DOPRI5 step from `(t, y)` over `h`. Returns the 5th-order solution and the (5th−4th) error estimate.
pub fn dopri5_step(f: &impl Fn(f64, &DVector<f64>) -> DVector<f64>, t: f64, y: &DVector<f64>, h: f64) -> (DVector<f64>, DVector<f64>) {
    let k1 = f(t, y);
    let k2 = f(t + C2 * h, &(y + &k1 * (h / 5.0)));
    let k3 = f(t + C3 * h, &(y + &k1 * (h * 3.0 / 40.0) + &k2 * (h * 9.0 / 40.0)));
    let k4 = f(t + C4 * h, &(y + &k1 * (h * 44.0 / 45.0) - &k2 * (h * 56.0 / 15.0) + &k3 * (h * 32.0 / 9.0)));
    let k5 = f(t + C5 * h, &(y + &k1 * (h * 19372.0 / 6561.0) - &k2 * (h * 25360.0 / 2187.0) + &k3 * (h * 64448.0 / 6561.0) - &k4 * (h * 212.0 / 729.0)));
    let k6 = f(t + h, &(y + &k1 * (h * 9017.0 / 3168.0) - &k2 * (h * 355.0 / 33.0) + &k3 * (h * 46732.0 / 5247.0) + &k4 * (h * 49.0 / 176.0) - &k5 * (h * 5103.0 / 18656.0)));
    // 5th-order solution (b == the k7 row, FSAL)
    let y5 = y + &k1 * (h * 35.0 / 384.0) + &k3 * (h * 500.0 / 1113.0) + &k4 * (h * 125.0 / 192.0) - &k5 * (h * 2187.0 / 6784.0) + &k6 * (h * 11.0 / 84.0);
    let k7 = f(t + h, &y5);
    // error = h · Σ (b_i − b*_i) k_i
    let e1 = 35.0 / 384.0 - 5179.0 / 57600.0;
    let e3 = 500.0 / 1113.0 - 7571.0 / 16695.0;
    let e4 = 125.0 / 192.0 - 393.0 / 640.0;
    let e5 = -2187.0 / 6784.0 - -92097.0 / 339200.0;
    let e6 = 11.0 / 84.0 - 187.0 / 2100.0;
    let e7 = -1.0 / 40.0;
    let err = (&k1 * e1 + &k3 * e3 + &k4 * e4 + &k5 * e5 + &k6 * e6 + &k7 * e7) * h;
    (y5, err)
}

/// Result of an adaptive integration: the accepted `(t, y)` samples and the number of rejected steps.
#[derive(Clone, Debug)]
pub struct OdeSolution {
    pub t: Vec<f64>,
    pub y: Vec<DVector<f64>>,
    pub rejected: usize,
}

impl OdeSolution {
    /// The final state.
    pub fn last(&self) -> &DVector<f64> {
        self.y.last().unwrap()
    }
}

/// Integrate `y' = f(t, y)` from `t0` to `t_end` with adaptive DOPRI5 step control to relative/absolute
/// tolerances `rtol`/`atol`. Returns every accepted step.
pub fn integrate(f: impl Fn(f64, &DVector<f64>) -> DVector<f64>, t0: f64, y0: DVector<f64>, t_end: f64, rtol: f64, atol: f64) -> OdeSolution {
    let mut t = t0;
    let mut y = y0.clone();
    let mut ts = vec![t0];
    let mut ys = vec![y0];
    let mut rejected = 0;
    // initial step guess
    let mut h = (t_end - t0).abs() * 1e-3;
    let (facmin, facmax, safety) = (0.2, 5.0, 0.9);

    while t < t_end - 1e-15 {
        if t + h > t_end {
            h = t_end - t;
        }
        let (y_new, err) = dopri5_step(&f, t, &y, h);
        // scaled error norm (RMS over components)
        let mut sq = 0.0;
        for i in 0..y.len() {
            let sc = atol + rtol * y[i].abs().max(y_new[i].abs());
            sq += (err[i] / sc).powi(2);
        }
        let err_norm = (sq / y.len() as f64).sqrt();

        if err_norm <= 1.0 {
            // accept
            t += h;
            y = y_new;
            ts.push(t);
            ys.push(y.clone());
        } else {
            rejected += 1;
        }
        // step-size update (5th-order ⇒ exponent 1/5)
        let factor = if err_norm > 0.0 { safety * err_norm.powf(-0.2) } else { facmax };
        h *= factor.clamp(facmin, facmax);
    }
    OdeSolution { t: ts, y: ys, rejected }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }

    #[test]
    fn it_matches_exponential_decay_to_tolerance() {
        // THE ORACLE. y' = −k y ⇒ y(t) = y0 e^{−kt}. Integrate to t=5 and compare to the closed form.
        let k = 0.8;
        let sol = integrate(move |_t, y| -k * y, 0.0, dv(&[1.0]), 5.0, 1e-9, 1e-12);
        let exact = (-k * 5.0f64).exp();
        assert!((sol.last()[0] - exact).abs() < 1e-7, "got {} vs exact {exact}", sol.last()[0]);
    }

    #[test]
    fn it_tracks_the_harmonic_oscillator_and_conserves_energy() {
        // y'' = −ω²y as [y, v]; solution y = cos ωt, and energy ½v² + ½ω²y² is invariant.
        let w = 2.0;
        let sol = integrate(move |_t, s| dv(&[s[1], -w * w * s[0]]), 0.0, dv(&[1.0, 0.0]), 6.0, 1e-10, 1e-12);
        let last = sol.last();
        let exact = (w * 6.0f64).cos();
        assert!((last[0] - exact).abs() < 1e-6, "position {} vs cos {exact}", last[0]);
        let e0 = 0.5 * w * w;
        let e_end = 0.5 * last[1] * last[1] + 0.5 * w * w * last[0] * last[0];
        assert!((e_end - e0).abs() < 1e-5, "energy drift {}", (e_end - e0).abs());
    }

    #[test]
    fn a_tighter_tolerance_gives_a_smaller_error() {
        let k = 1.3;
        let exact = (-k * 4.0f64).exp();
        let loose = integrate(move |_t, y| -k * y, 0.0, dv(&[1.0]), 4.0, 1e-4, 1e-7);
        let tight = integrate(move |_t, y| -k * y, 0.0, dv(&[1.0]), 4.0, 1e-10, 1e-13);
        let e_loose = (loose.last()[0] - exact).abs();
        let e_tight = (tight.last()[0] - exact).abs();
        assert!(e_tight < e_loose, "tighter tol should be more accurate: {e_tight} vs {e_loose}");
    }

    #[test]
    fn the_step_size_adapts_to_the_dynamics() {
        // A signal that is nearly flat then swings fast should force more/smaller steps than a slow one over
        // the same interval — the adaptivity that fixed-step RK4 lacks.
        let fast = integrate(|t: f64, _y: &DVector<f64>| dv(&[10.0 * (10.0 * t).cos()]), 0.0, dv(&[0.0]), 6.0, 1e-8, 1e-10);
        let slow = integrate(|t: f64, _y: &DVector<f64>| dv(&[(0.2 * t).cos()]), 0.0, dv(&[0.0]), 6.0, 1e-8, 1e-10);
        assert!(fast.t.len() > slow.t.len(), "fast dynamics should need more steps: {} vs {}", fast.t.len(), slow.t.len());
    }
}
