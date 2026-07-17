//! **C3 — Consensus Complementarity Control** (Aydinoglu & Posa, T-RO 2024): real-time contact-implicit
//! MPC for a Linear Complementarity System, which *discovers the contact schedule online* — no a-priori
//! mode sequence, unlike a fixed-schedule collocation. The plant is
//!
//! ```text
//!   x_{k+1} = A x_k + B u_k + D λ_k + d
//!   0 ≤ λ_k ⊥ (E x_k + F λ_k + H u_k + c) ≥ 0        (the complementarity: force ⊥ gap)
//! ```
//!
//! C3 splits the trajectory problem by ADMM: a **consensus QP** over `(u, λ)` with the linear cost and
//! dynamics (the smooth part), a **per-step projection** of each `(λ_k, φ_k)` onto the (nonconvex)
//! complementarity set, and a dual update binding them. Contact turns on and off through the projection,
//! so the optimizer decides *when to touch* rather than being told. Verified on a pusher–slider, where
//! the object has no actuator and can only be moved *through contact*. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A linear complementarity system.
#[derive(Clone, Debug)]
pub struct Lcs {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub d: DMatrix<f64>,
    pub d_aff: DVector<f64>,
    pub e: DMatrix<f64>,
    pub f: DMatrix<f64>,
    pub h: DMatrix<f64>,
    pub c: DVector<f64>,
}

impl Lcs {
    fn nx(&self) -> usize { self.a.nrows() }
    fn nl(&self) -> usize { self.f.nrows() }

    /// Solve the per-step LCP `0 ≤ λ ⊥ (Fλ + q) ≥ 0` with `q = E x + H u + c`, by projected
    /// Gauss–Seidel (exact for a P-matrix `F`; converges for the diagonally-dominant contact `F`).
    pub fn solve_lambda(&self, x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        let q = &self.e * x + &self.h * u + &self.c;
        let n = self.nl();
        let mut lam = DVector::<f64>::zeros(n);
        for _ in 0..500 {
            let mut delta = 0.0f64;
            for i in 0..n {
                let mut ax = q[i];
                for j in 0..n {
                    if j != i {
                        ax += self.f[(i, j)] * lam[j];
                    }
                }
                let fii = self.f[(i, i)].max(1e-12);
                let new = (-ax / fii).max(0.0);
                delta = delta.max((new - lam[i]).abs());
                lam[i] = new;
            }
            if delta < 1e-12 {
                break;
            }
        }
        lam
    }

    /// Advance the LCS one step (solving the contact LCP first).
    pub fn step(&self, x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        let lam = self.solve_lambda(x, u);
        &self.a * x + &self.b * u + &self.d * &lam + &self.d_aff
    }

    /// The complementarity residual `Σ |λ_i · φ_i| + |min(0,λ)| + |min(0,φ)|` at `(x,u,λ)` — zero iff
    /// the pair is a valid LCP solution.
    pub fn comp_residual(&self, x: &DVector<f64>, u: &DVector<f64>, lam: &DVector<f64>) -> f64 {
        let phi = &self.e * x + &self.f * lam + &self.h * u + &self.c;
        let mut r = 0.0;
        for i in 0..self.nl() {
            r += (lam[i] * phi[i]).abs() + lam[i].min(0.0).abs() + phi[i].min(0.0).abs();
        }
        r
    }
}

/// Project a scalar pair `(a, b)` onto the complementarity set `{λ≥0, φ≥0, λφ=0}` (two rays); returns
/// the nearest point.
fn proj_comp_pair(a: f64, b: f64) -> (f64, f64) {
    let on_lambda = (a.max(0.0), 0.0); // φ = 0 branch
    let on_phi = (0.0, b.max(0.0)); // λ = 0 branch
    let d1 = (a - on_lambda.0).powi(2) + (b - on_lambda.1).powi(2);
    let d2 = (a - on_phi.0).powi(2) + (b - on_phi.1).powi(2);
    if d1 <= d2 { on_lambda } else { on_phi }
}

/// A C3 contact-implicit MPC controller.
#[derive(Clone, Debug)]
pub struct C3 {
    pub lcs: Lcs,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
    pub horizon: usize,
    pub rho: f64,
    pub admm_iters: usize,
}

impl C3 {
    /// One receding-horizon solve from `x0` toward `x_ref`, returning the first control `u_0`.
    pub fn control(&self, x0: &DVector<f64>, x_ref: &DVector<f64>) -> DVector<f64> {
        let (nx, nu, nl, n) = (self.lcs.nx(), self.lcs.b.ncols(), self.lcs.nl(), self.horizon);
        // Decision vector w = [u_0..u_{N-1}, λ_0..λ_{N-1}]; dim = N(nu+nl).
        let du = n * nu;
        let dw = n * (nu + nl);
        // Condense the rollout: x_k affine in w. Build S so that x_stack (N·nx) = Sx0·x0 + Sw·w + Sd.
        // x_{k+1} = A x_k + B u_k + D λ_k + d.
        let mut sx0 = DMatrix::<f64>::zeros(n * nx, nx);
        let mut sw = DMatrix::<f64>::zeros(n * nx, dw);
        let mut sd = DVector::<f64>::zeros(n * nx);
        // running maps for x_k (before writing x_{k+1})
        let mut cur_x0 = DMatrix::<f64>::identity(nx, nx); // x_0 = I·x0
        let mut cur_w = DMatrix::<f64>::zeros(nx, dw);
        let mut cur_d = DVector::<f64>::zeros(nx);
        for k in 0..n {
            // x_{k+1} = A·x_k + B·u_k + D·λ_k + d
            let nx0 = &self.lcs.a * &cur_x0;
            let mut nw = &self.lcs.a * &cur_w;
            for i in 0..nx {
                for j in 0..nu {
                    nw[(i, k * nu + j)] += self.lcs.b[(i, j)];
                }
                for j in 0..nl {
                    nw[(i, du + k * nl + j)] += self.lcs.d[(i, j)];
                }
            }
            let nd = &self.lcs.a * &cur_d + &self.lcs.d_aff;
            sx0.view_mut((k * nx, 0), (nx, nx)).copy_from(&nx0);
            sw.view_mut((k * nx, 0), (nx, dw)).copy_from(&nw);
            sd.rows_mut(k * nx, nx).copy_from(&nd);
            cur_x0 = nx0;
            cur_w = nw;
            cur_d = nd;
        }

        // Cost J = Σ (x_k − r)ᵀ Q (x_k − r) + Σ uᵀ R u, quadratic in w: ½ wᵀ H0 w + f0ᵀ w (+const).
        let mut h0 = DMatrix::<f64>::zeros(dw, dw);
        let mut f0 = DVector::<f64>::zeros(dw);
        // state cost via Sw
        for k in 0..n {
            let swk = sw.view((k * nx, 0), (nx, dw));
            let bias = sx0.view((k * nx, 0), (nx, nx)) * x0 + sd.rows(k * nx, nx) - x_ref;
            let qsw = &self.q * swk;
            h0 += swk.transpose() * &qsw * 2.0;
            f0 += swk.transpose() * (&self.q * &bias) * 2.0;
        }
        // control cost
        for k in 0..n {
            for i in 0..nu {
                for j in 0..nu {
                    h0[(k * nu + i, k * nu + j)] += 2.0 * self.r[(i, j)];
                }
            }
        }

        // ADMM over the complementarity copies δ_k = (λ_k, φ_k), duals y_k.
        let mut w = DVector::<f64>::zeros(dw);
        let mut delta = vec![(DVector::<f64>::zeros(nl), DVector::<f64>::zeros(nl)); n];
        let mut yl = vec![DVector::<f64>::zeros(nl); n];
        let mut yp = vec![DVector::<f64>::zeros(nl); n];
        let rho = self.rho;

        for _ in 0..self.admm_iters {
            // φ_k(w) = E x_k + F λ_k + H u_k + c, affine in w. Build once per iter as needed.
            // Assemble the penalized QP: minimize ½ wᵀH w + fᵀw with
            //   H = H0 + ρ Σ (Jλ_kᵀ Jλ_k + Jφ_kᵀ Jφ_k),  f = f0 − ρ Σ (Jλ_kᵀ(δλ−yλ) + Jφ_kᵀ(δφ−yφ)).
            let mut h = h0.clone();
            let mut f = f0.clone();
            for k in 0..n {
                // Jλ: selects λ_k from w
                let mut jl = DMatrix::<f64>::zeros(nl, dw);
                for i in 0..nl {
                    jl[(i, du + k * nl + i)] = 1.0;
                }
                // φ_k = E x_k + F λ_k + H u_k + c
                let swk = sw.view((k * nx, 0), (nx, dw));
                let xbias = sx0.view((k * nx, 0), (nx, nx)) * x0 + sd.rows(k * nx, nx);
                let mut jp = &self.lcs.e * swk; // E·(Sw) part
                for i in 0..nl {
                    for j in 0..nl {
                        jp[(i, du + k * nl + j)] += self.lcs.f[(i, j)];
                    }
                    for j in 0..nu {
                        jp[(i, k * nu + j)] += self.lcs.h[(i, j)];
                    }
                }
                let pbias = &self.lcs.e * &xbias + &self.lcs.c; // constant part of φ
                // penalties
                let (dl, dp) = &delta[k];
                h += (jl.transpose() * &jl + jp.transpose() * &jp) * rho;
                f -= jl.transpose() * (dl - &yl[k]) * rho;
                f += jp.transpose() * (&pbias - (dp - &yp[k])) * rho;
            }
            // solve H w = −f
            let hsym = 0.5 * (&h + h.transpose());
            w = hsym.clone().lu().solve(&(-&f)).unwrap_or(w);

            // projection + dual update per step
            for k in 0..n {
                let lam_k = w.rows(du + k * nl, nl).into_owned();
                // φ_k(w)
                let swk = sw.view((k * nx, 0), (nx, dw));
                let xk = sx0.view((k * nx, 0), (nx, nx)) * x0 + swk * &w + sd.rows(k * nx, nx);
                let uk = w.rows(k * nu, nu).into_owned();
                let phi_k = &self.lcs.e * &xk + &self.lcs.f * &lam_k + &self.lcs.h * &uk + &self.lcs.c;
                let mut ndl = DVector::zeros(nl);
                let mut ndp = DVector::zeros(nl);
                for i in 0..nl {
                    let (pl, pp) = proj_comp_pair(lam_k[i] + yl[k][i], phi_k[i] + yp[k][i]);
                    ndl[i] = pl;
                    ndp[i] = pp;
                }
                yl[k] += &lam_k - &ndl;
                yp[k] += &phi_k - &ndp;
                delta[k] = (ndl, ndp);
            }
        }
        w.rows(0, nu).into_owned()
    }

    /// Closed-loop rollout: apply the receding-horizon control on the true LCS for `steps`.
    pub fn simulate(&self, x0: &DVector<f64>, x_ref: &DVector<f64>, steps: usize) -> DVector<f64> {
        let mut x = x0.clone();
        for _ in 0..steps {
            let u = self.control(&x, x_ref);
            x = self.lcs.step(&x, &u);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1-D pusher–slider LCS. State [x_p, v_p, x_s, v_s]; control = pusher force u. The slider has no
    /// actuator — it moves only when the pusher contacts it (gap = x_s − x_p). Contact force
    /// λ ⊥ gap⁺ ≥ 0. This is the canonical "manipulation only through contact" system.
    fn pusher_slider(dt: f64) -> Lcs {
        let (mp, ms, b) = (1.0, 1.0, 3.0); // pusher mass, slider mass, slider damping
        // states: x_p, v_p, x_s, v_s
        let a = DMatrix::from_row_slice(4, 4, &[
            1.0, dt, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, dt,
            0.0, 0.0, 0.0, 1.0 - dt * b / ms,
        ]);
        let bmat = DMatrix::from_row_slice(4, 1, &[0.0, dt / mp, 0.0, 0.0]);
        // contact force λ ≥ 0 pushes pusher back (−) and slider forward (+), applied to velocities
        let d = DMatrix::from_row_slice(4, 1, &[0.0, -dt / mp, 0.0, dt / ms]);
        let d_aff = DVector::zeros(4);
        // gap⁺ = x_s⁺ − x_p⁺ ; expressed as E x + F λ + H u + c (semi-implicit, one-step lookahead)
        // x_p⁺ = x_p + dt(v_p + dt/mp(u − λ)); x_s⁺ = x_s + dt(v_s(1−dt b/ms) + dt/ms λ)
        let e = DMatrix::from_row_slice(1, 4, &[-1.0, -dt, 1.0, dt * (1.0 - dt * b / ms)]);
        let f = DMatrix::from_row_slice(1, 1, &[dt * dt / mp + dt * dt / ms]);
        let h = DMatrix::from_row_slice(1, 1, &[-dt * dt / mp]);
        let c = DVector::zeros(1);
        Lcs { a, b: bmat, d, d_aff, e, f, h, c }
    }

    #[test]
    fn the_lcp_step_satisfies_complementarity() {
        // The forward model's contact solve is a valid LCP solution: force ⊥ gap, both nonnegative.
        let lcs = pusher_slider(0.05);
        for &(gap, push) in &[(0.0, 5.0), (0.3, 5.0), (0.0, -2.0), (0.1, 0.0)] {
            let x = DVector::from_row_slice(&[0.0, 0.0, gap, 0.0]);
            let u = DVector::from_row_slice(&[push]);
            let lam = lcs.solve_lambda(&x, &u);
            let r = lcs.comp_residual(&x, &u, &lam);
            assert!(r < 1e-9, "complementarity violated (gap {gap}, push {push}): residual {r}, λ={lam:?}");
        }
    }

    #[test]
    fn contact_only_transmits_a_push_when_touching() {
        // In contact (gap 0) a forward push moves the slider; with a gap it does not (yet).
        let lcs = pusher_slider(0.05);
        let touching = lcs.step(&DVector::from_row_slice(&[0.0, 0.0, 0.0, 0.0]), &DVector::from_row_slice(&[5.0]));
        assert!(touching[3] > 1e-3, "a push in contact should accelerate the slider: v_s={}", touching[3]);
        let gapped = lcs.step(&DVector::from_row_slice(&[0.0, 0.0, 0.4, 0.0]), &DVector::from_row_slice(&[5.0]));
        assert!(gapped[3].abs() < 1e-9, "a push across a gap should not move the slider yet: v_s={}", gapped[3]);
    }

    #[test]
    fn the_projection_lands_on_the_complementarity_set() {
        for &(a, b) in &[(1.0, 1.0), (-0.5, 2.0), (3.0, -1.0), (-1.0, -1.0)] {
            let (l, p) = proj_comp_pair(a, b);
            assert!(l >= -1e-12 && p >= -1e-12 && (l * p).abs() < 1e-12, "proj({a},{b})=({l},{p}) not on the set");
        }
    }

    fn slider_task() -> (C3, DVector<f64>, DVector<f64>) {
        let dt = 0.05;
        let lcs = pusher_slider(dt);
        let mut q = DMatrix::<f64>::zeros(4, 4);
        q[(2, 2)] = 80.0; // slider position — the objective
        q[(3, 3)] = 2.0; // slider velocity — mild (damping already limits overshoot)
        let r = DMatrix::from_row_slice(1, 1, &[0.05]);
        let c3 = C3 { lcs, q, r, horizon: 25, rho: 30.0, admm_iters: 25 };
        let x0 = DVector::from_row_slice(&[-0.15, 0.0, 0.0, 0.0]); // pusher just behind the slider
        let x_ref = DVector::from_row_slice(&[0.0, 0.0, 0.4, 0.0]);
        (c3, x0, x_ref)
    }

    #[test]
    fn c3_drives_the_slider_to_the_target_through_contact() {
        // THE CHAPTER. The slider starts at 0 and must reach 0.4 — reachable only by making contact and
        // pushing (the slider has no actuator). C3 discovers the contact schedule online, pushes, and
        // lets damping arrest the slider near the target without a pull-back it does not have.
        let (c3, x0, x_ref) = slider_task();
        let xf = c3.simulate(&x0, &x_ref, 140);
        assert!((xf[2] - 0.4).abs() < 0.08, "slider did not settle near the target: x_s={}", xf[2]);
        assert!(xf[3].abs() < 0.15, "slider should arrive with low speed: v_s={}", xf[3]);
    }

    #[test]
    fn a_contact_blind_planner_never_moves_the_slider() {
        // Contact-awareness is necessary: a controller that plans as if there were no contact (λ≡0)
        // sees no way its force reaches the slider, so it never bothers to push — the slider stays put.
        // (Same cost/task; we simply zero the contact coupling D and F the planner reasons over.)
        let (c3, x0, x_ref) = slider_task();
        let mut blind = c3.clone();
        blind.lcs.d = DMatrix::zeros(4, 1); // planner believes contact transmits nothing
        // roll the blind planner's control out on the TRUE contact dynamics
        let mut x = x0.clone();
        for _ in 0..140 {
            let u = blind.control(&x, &x_ref);
            x = c3.lcs.step(&x, &u); // true plant (real contact)
        }
        assert!(x[2] < 0.1, "a contact-blind planner should fail to purposefully move the slider: x_s={}", x[2]);
    }
}
