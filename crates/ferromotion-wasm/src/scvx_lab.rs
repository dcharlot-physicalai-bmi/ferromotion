//! **Successive-convexification lab** — the rig behind the textbook chapter on planning a rocket landing.
//! Drives the real [`ferromotion_control::ScvxProblem`] on a 2-D powered-descent problem with genuinely
//! non-convex dynamics (thrust over a *depleting* mass), captures the trajectory after every SCvx
//! iteration, and lets the reader scrub from the dynamically-infeasible straight-line guess to the
//! converged landing while watching the dynamics defect fall superlinearly.

use ferromotion_control::{ScvxOpts, ScvxProblem};
use nalgebra::DVector;
use wasm_bindgen::prelude::*;

const G: f64 = 1.0;
const ALPHA: f64 = 0.05;

fn rocket_f(x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
    let m = x[4].max(0.2);
    let tn = (u[0] * u[0] + u[1] * u[1] + 1e-9).sqrt();
    DVector::from_vec(vec![x[2], x[3], u[0] / m, u[1] / m - G, -ALPHA * tn])
}

fn rocket_jac(x: &DVector<f64>, u: &DVector<f64>) -> (nalgebra::DMatrix<f64>, nalgebra::DMatrix<f64>) {
    let m = x[4].max(0.2);
    let tn = (u[0] * u[0] + u[1] * u[1] + 1e-9).sqrt();
    let mut a = nalgebra::DMatrix::zeros(5, 5);
    a[(0, 2)] = 1.0;
    a[(1, 3)] = 1.0;
    a[(2, 4)] = -u[0] / (m * m);
    a[(3, 4)] = -u[1] / (m * m);
    let mut b = nalgebra::DMatrix::zeros(5, 2);
    b[(2, 0)] = 1.0 / m;
    b[(3, 1)] = 1.0 / m;
    b[(4, 0)] = -ALPHA * u[0] / tn;
    b[(4, 1)] = -ALPHA * u[1] / tn;
    (a, b)
}

#[wasm_bindgen]
pub struct ScvxLab {
    n: usize,
    xs_history: Vec<Vec<DVector<f64>>>,
    defect_history: Vec<f64>,
    radius_history: Vec<f64>,
    x0: DVector<f64>,
    converged: bool,
    final_fuel: f64,
}

#[wasm_bindgen]
impl ScvxLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ScvxLab {
        let n = 30;
        let x0 = DVector::from_vec(vec![-3.0, 4.0, 0.3, 0.0, 1.0]);
        let xg = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let prob = ScvxProblem {
            nx: 5, nu: 2, n, dt: 0.15,
            x_init: x0.clone(), x_goal: xg.clone(), terminal_free_tail: 1,
            u_min: DVector::from_vec(vec![-6.0, 0.0]), u_max: DVector::from_vec(vec![6.0, 10.0]),
            fuel_weight: 1.0, lambda: 5e3,
        };
        // dynamically-infeasible straight-line guess, hover-ish control
        let xs: Vec<DVector<f64>> = (0..=n).map(|i| &x0 + (&xg - &x0) * (i as f64 / n as f64)).collect();
        let us = vec![DVector::from_vec(vec![0.0, G]); n];
        let opts = ScvxOpts { r0: 1.5, max_iter: 150, capture_history: true, ..Default::default() };
        let rep = prob.solve(&rocket_f, &rocket_jac, xs, us, opts);
        ScvxLab {
            n,
            x0,
            xs_history: rep.xs_history,
            defect_history: rep.defect_history,
            radius_history: rep.radius_history,
            converged: rep.converged,
            final_fuel: rep.final_fuel,
        }
    }

    /// Number of iteration snapshots (index 0 = the initial straight-line guess).
    pub fn frames(&self) -> usize {
        self.xs_history.len()
    }
    /// Number of trajectory waypoints.
    pub fn waypoints(&self) -> usize {
        self.n + 1
    }
    /// Horizontal position of waypoint `i` at iteration snapshot `k`.
    pub fn px(&self, k: usize, i: usize) -> f64 {
        self.xs_history[k.min(self.frames() - 1)][i][0]
    }
    /// Vertical (altitude) position of waypoint `i` at iteration snapshot `k`.
    pub fn py(&self, k: usize, i: usize) -> f64 {
        self.xs_history[k.min(self.frames() - 1)][i][1]
    }
    /// The dynamics defect at snapshot `k` (→ 0 ⇒ dynamically feasible).
    pub fn defect(&self, k: usize) -> f64 {
        self.defect_history[k.min(self.defect_history.len() - 1)]
    }
    /// The trust radius at snapshot `k`.
    pub fn radius(&self, k: usize) -> f64 {
        self.radius_history[k.min(self.radius_history.len() - 1)]
    }
    pub fn initial_defect(&self) -> f64 {
        self.defect_history[0]
    }
    pub fn final_defect(&self) -> f64 {
        *self.defect_history.last().unwrap()
    }
    pub fn converged(&self) -> bool {
        self.converged
    }
    pub fn final_fuel(&self) -> f64 {
        self.final_fuel
    }
    pub fn start_x(&self) -> f64 {
        self.x0[0]
    }
    pub fn start_y(&self) -> f64 {
        self.x0[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lab_solves_and_lands() {
        let lab = ScvxLab::new();
        assert!(lab.frames() > 3, "should capture several iteration frames");
        assert!(lab.initial_defect() > 0.1, "guess is infeasible");
        assert!(lab.final_defect() < 1e-3, "converges to a feasible landing: {}", lab.final_defect());
        // final frame touches the pad
        let last = lab.frames() - 1;
        let tip = lab.waypoints() - 1;
        assert!(lab.px(last, tip).abs() < 1e-2 && lab.py(last, tip).abs() < 1e-2, "lands at the pad");
    }
}
