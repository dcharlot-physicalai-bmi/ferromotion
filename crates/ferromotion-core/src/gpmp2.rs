//! **GPMP2 — Gaussian-Process Motion Planning** (Mukadam, Dong, Yan, Dellaert & Boots, IJRR 2018): plan a
//! trajectory as *probabilistic inference* on a factor graph. The trajectory is a sparse Gaussian process
//! with a **white-noise-on-acceleration (WNOA)** prior — a constant-velocity model over states `θ = [x;
//! v]` — and planning is MAP inference: a Gauss–Newton solve balancing a **GP prior** factor (keep the
//! trajectory smooth/dynamically consistent) against **obstacle** factors read from an [`crate::Esdf`]. The
//! GP structure also gives *continuous-time* querying: interpolate the state at any time between support
//! nodes in closed form.
//!
//! The clean anchor: with **no obstacles**, the WNOA GP MAP between fixed boundary states is exactly the
//! analytic **cubic Hermite** interpolation (the minimum-∫‖acceleration‖² trajectory of a double
//! integrator) — verified to machine precision. With an obstacle it bows around it, like CHOMP but from the
//! Bayesian-smoothing side. Complements [`crate::Chomp`] (covariant gradient) with the factor-graph view.
//! Pure `nalgebra` → WASM-clean.

use crate::esdf::Esdf;
use nalgebra::{DMatrix, DVector};

/// A GPMP2 planner for a point robot in `d` dimensions over `n` intervals (`n+1` support nodes) of
/// duration `dt`, WNOA process power `qc`, obstacle weight `obs_w`, and margin `epsilon`.
#[derive(Clone, Debug)]
pub struct Gpmp2 {
    pub d: usize,
    pub n: usize,
    pub dt: f64,
    pub qc: f64,
    pub obs_w: f64,
    pub epsilon: f64,
}

impl Gpmp2 {
    fn sd(&self) -> usize {
        2 * self.d
    }

    /// The WNOA transition `Φ = [[I, Δt·I], [0, I]]`.
    fn phi(&self) -> DMatrix<f64> {
        let d = self.d;
        let mut p = DMatrix::identity(2 * d, 2 * d);
        p.view_mut((0, d), (d, d)).copy_from(&(DMatrix::identity(d, d) * self.dt));
        p
    }

    /// The inverse GP process covariance `Q⁻¹` for the WNOA model (the standard closed form).
    fn q_inv(&self) -> DMatrix<f64> {
        let d = self.d;
        let (dt, qc) = (self.dt, self.qc);
        let i = DMatrix::<f64>::identity(d, d) / qc;
        let mut qi = DMatrix::zeros(2 * d, 2 * d);
        qi.view_mut((0, 0), (d, d)).copy_from(&(&i * (12.0 / dt.powi(3))));
        qi.view_mut((0, d), (d, d)).copy_from(&(&i * (-6.0 / dt.powi(2))));
        qi.view_mut((d, 0), (d, d)).copy_from(&(&i * (-6.0 / dt.powi(2))));
        qi.view_mut((d, d), (d, d)).copy_from(&(&i * (4.0 / dt)));
        qi
    }

    /// Plan from boundary state `start = [x0; v0]` to `goal = [xN; vN]` (each length `2d`), optionally
    /// avoiding obstacles in `esdf` for a robot of the given `radius`. Returns the `n+1` support states.
    pub fn plan(&self, start: &DVector<f64>, goal: &DVector<f64>, esdf: Option<&Esdf>, radius: f64, iters: usize) -> Vec<DVector<f64>> {
        let (d, n, sd) = (self.d, self.n, self.sd());
        // initialize on the straight line (position) with a constant velocity
        let mut theta: Vec<DVector<f64>> = (0..=n)
            .map(|i| {
                let s = i as f64 / n as f64;
                let mut v = DVector::zeros(sd);
                for k in 0..d {
                    v[k] = start[k] * (1.0 - s) + goal[k] * s;
                    v[d + k] = start[d + k] * (1.0 - s) + goal[d + k] * s;
                }
                v
            })
            .collect();
        theta[0] = start.clone();
        theta[n] = goal.clone();

        let phi = self.phi();
        let qi = self.q_inv();
        let nv = (n - 1) * sd; // interior nodes 1..n-1
        let slot = |i: usize| if i == 0 || i == n { None } else { Some((i - 1) * sd) };

        for _ in 0..iters.max(1) {
            let mut h = DMatrix::<f64>::zeros(nv, nv);
            let mut g = DVector::<f64>::zeros(nv);

            // GP prior factors: e_i = θ_{i+1} − Φ θ_i, weighted by Q⁻¹
            for i in 0..n {
                let e = &theta[i + 1] - &phi * &theta[i];
                // Jacobians: ∂e/∂θ_i = −Φ, ∂e/∂θ_{i+1} = I
                let js = [(i, -&phi), (i + 1, DMatrix::identity(sd, sd))];
                for (a, ja) in &js {
                    if let Some(sa) = slot(*a) {
                        let newg = g.rows(sa, sd) + ja.transpose() * &qi * &e;
                        g.rows_mut(sa, sd).copy_from(&newg);
                        for (b, jb) in &js {
                            if let Some(sb) = slot(*b) {
                                let block = ja.transpose() * &qi * jb;
                                let cur = h.view((sa, sb), (sd, sd)).into_owned();
                                h.view_mut((sa, sb), (sd, sd)).copy_from(&(cur + block));
                            }
                        }
                    }
                }
            }

            // obstacle factors at interior nodes (position sub-state), from the ESDF hinge
            if let Some(e) = esdf {
                for (idx, ti) in theta[1..n].iter().enumerate() {
                    let x = ti.rows(0, d).into_owned();
                    let xv = nalgebra::Vector3::new(x[0], x.get(1).copied().unwrap_or(0.0), x.get(2).copied().unwrap_or(0.0));
                    let clr = e.distance(&xv) - radius;
                    if clr < self.epsilon {
                        let res = self.obs_w * (self.epsilon - clr); // scalar residual
                        let grad = e.gradient(&xv); // ∇d
                        // ∂res/∂x = −obs_w·∇d (on the position sub-block)
                        let sa = idx * sd;
                        let mut jrow = DVector::<f64>::zeros(sd);
                        for k in 0..d {
                            jrow[k] = -self.obs_w * grad[k];
                        }
                        let cur_h = h.view((sa, sa), (sd, sd)).into_owned();
                        h.view_mut((sa, sa), (sd, sd)).copy_from(&(cur_h + &jrow * jrow.transpose()));
                        let newg = g.rows(sa, sd) + &jrow * res;
                        g.rows_mut(sa, sd).copy_from(&newg);
                    }
                }
            }

            let Some(delta) = h.clone().lu().solve(&(-&g)) else { break };
            for (idx, ti) in theta[1..n].iter_mut().enumerate() {
                let sa = idx * sd;
                *ti += delta.rows(sa, sd);
            }
            if esdf.is_none() {
                break; // the prior-only problem is linear ⇒ one solve is exact
            }
        }
        theta
    }

    /// GP interpolation of the state at time `t = i·Δt + tau` (`0 ≤ tau ≤ Δt`) between support nodes `i`,
    /// `i+1` — the WNOA posterior mean, which reduces to the cubic Hermite of the two boundary states.
    pub fn interpolate(&self, theta: &[DVector<f64>], i: usize, tau: f64) -> DVector<f64> {
        let (d, dt) = (self.d, self.dt);
        let s = tau / dt;
        let (h00, h10, h01, h11) = (2.0 * s * s * s - 3.0 * s * s + 1.0, s * s * s - 2.0 * s * s + s, -2.0 * s * s * s + 3.0 * s * s, s * s * s - s * s);
        let (dh00, dh10, dh01, dh11) = ((6.0 * s * s - 6.0 * s) / dt, (3.0 * s * s - 4.0 * s + 1.0), (-6.0 * s * s + 6.0 * s) / dt, (3.0 * s * s - 2.0 * s));
        let mut out = DVector::zeros(2 * d);
        for k in 0..d {
            let (x0, v0, x1, v1) = (theta[i][k], theta[i][d + k], theta[i + 1][k], theta[i + 1][d + k]);
            out[k] = h00 * x0 + h10 * dt * v0 + h01 * x1 + h11 * dt * v1;
            out[d + k] = dh00 * x0 + dh10 * v0 + dh01 * x1 + dh11 * v1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }

    // The analytic cubic Hermite position for the 1-D boundary (x0,v0)→(xN,vN) over total time T.
    fn hermite(x0: f64, v0: f64, x1: f64, v1: f64, t: f64, tt: f64) -> f64 {
        let s = t / tt;
        (2.0 * s * s * s - 3.0 * s * s + 1.0) * x0 + (s * s * s - 2.0 * s * s + s) * tt * v0 + (-2.0 * s * s * s + 3.0 * s * s) * x1 + (s * s * s - s * s) * tt * v1
    }

    #[test]
    fn the_obstacle_free_map_is_the_cubic_hermite() {
        // THE ORACLE. With no obstacles the WNOA GP MAP is exactly the minimum-acceleration cubic Hermite
        // between the boundary states.
        let g = Gpmp2 { d: 1, n: 10, dt: 0.2, qc: 1.0, obs_w: 0.0, epsilon: 0.1 };
        let (x0, v0, x1, v1) = (0.0, 0.3, 2.0, -0.1);
        let theta = g.plan(&dv(&[x0, v0]), &dv(&[x1, v1]), None, 0.0, 1);
        let tt = g.n as f64 * g.dt;
        for (i, node) in theta.iter().enumerate() {
            let t = i as f64 * g.dt;
            assert!((node[0] - hermite(x0, v0, x1, v1, t, tt)).abs() < 1e-8, "node {i} pos {} vs Hermite {}", node[0], hermite(x0, v0, x1, v1, t, tt));
        }
    }

    #[test]
    fn interpolation_is_exact_at_the_support_nodes() {
        let g = Gpmp2 { d: 1, n: 6, dt: 0.25, qc: 1.0, obs_w: 0.0, epsilon: 0.1 };
        let theta = g.plan(&dv(&[0.0, 0.5]), &dv(&[1.5, 0.0]), None, 0.0, 1);
        for i in 0..g.n {
            assert!((g.interpolate(&theta, i, 0.0) - &theta[i]).norm() < 1e-9, "interp at τ=0 should be node {i}");
            assert!((g.interpolate(&theta, i, g.dt) - &theta[i + 1]).norm() < 1e-9, "interp at τ=Δt should be node {}", i + 1);
        }
    }

    #[test]
    fn an_obstacle_bows_the_trajectory_around_it() {
        // THE HEADLINE. An obstacle on the straight path pushes the trajectory to keep clearance — planning
        // by inference against the ESDF.
        let g = Gpmp2 { d: 2, n: 14, dt: 0.15, qc: 4.0, obs_w: 30.0, epsilon: 0.25 };
        // a wall of occupied points straddling the straight line at the midpoint (just off-axis)
        let occ: Vec<Vector3<f64>> = (0..9).flat_map(|i| (0..3).map(move |j| Vector3::new(1.0, 0.12 + (i as f64 - 4.0) * 0.06, (j as f64 - 1.0) * 0.06))).collect();
        let esdf = Esdf::from_occupied(occ, f64::INFINITY);
        let start = dv(&[0.0, 0.0, 0.0, 0.0]);
        let goal = dv(&[2.0, 0.0, 0.0, 0.0]);
        let straight_clr = esdf.distance(&Vector3::new(1.0, 0.0, 0.0)) - 0.05;
        assert!(straight_clr < 0.2, "the straight line should pass close to the obstacle: {straight_clr}");
        let theta = g.plan(&start, &goal, Some(&esdf), 0.05, 30);
        let min_clr = theta.iter().map(|n| esdf.distance(&Vector3::new(n[0], n[1], n[2])) - 0.05).fold(f64::INFINITY, f64::min);
        assert!(min_clr > straight_clr + 0.05, "GPMP2 should increase clearance: {straight_clr} → {min_clr}");
        // endpoints preserved
        assert!((theta.first().unwrap() - &start).norm() < 1e-9 && (theta.last().unwrap() - &goal).norm() < 1e-9);
    }
}
