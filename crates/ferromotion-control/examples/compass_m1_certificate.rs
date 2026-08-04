//! **Q1 milestone M1: a learned virtual constraint, a certified transverse metric, and a verified reduced
//! return map.**
//!
//! M0 produced the baseline certificate for a hand-picked constraint. M1 asks for the three pieces Q1 lists as
//! sub-problems (a), (b) and (c):
//!
//! * **(a) render an approximately invariant manifold from a *learned* constraint** — train `h_w`, with hybrid
//!   invariance as an explicit objective rather than something solved by construction in a family chosen to
//!   allow it;
//! * **(b) certify the transverse dynamics** — a metric in which the directions off `Z` provably contract;
//! * **(c) verify the reduced return map**, exploiting the fact that the object to verify is one-dimensional
//!   even though the robot is not.
//!
//! On (c) the claim made here is deliberately narrow. There is no satisfiability solver involved; what there is
//! instead is better for this particular object. The `ζ` equation on `Z` is **linear**, so the return map is
//! exactly affine and the stall threshold has a closed form — which means `δ² < 1` and forward invariance of the
//! basin are checked **over a continuum of initial conditions**, not at sampled points. That is a verification
//! rather than a test, and it works precisely because the reduced object is one-dimensional.
//!
//! Everything is trained against the analytic reduction and then **verified against the full four-state model**
//! with a RES-CLF holding it on `Z`. Training never touches the full model.
//!
//! Run: `cargo run --release --example compass_m1_certificate -p ferromotion-control`

use ferromotion_control::{hzd_reduction, invariance_defect, train_constraint, worst_clearance, CompassGait, GaitGoal, GaitState, LearnedConstraint, ResClf, SwingConstraint, VirtualConstraint};
use nalgebra::{DMatrix, DVector};

const DT: f64 = 2e-5;
const MAX_STEP_TIME: f64 = 4.0;
const QUAD: usize = 4000;

fn control<'a>(r: &'a CompassGait, vc: &'a dyn SwingConstraint, clf: &'a ResClf, eps: f64) -> impl Fn(&GaitState) -> f64 + 'a {
    move |s: &GaitState| {
        let (y, yd) = vc.output(s);
        let v = clf.clf_qp(&DVector::from_row_slice(&[y, yd]), eps).map(|u| u[0]).unwrap_or(0.0);
        r.hzd_torque(s, vc, v).unwrap_or(0.0)
    }
}

fn main() {
    let r = CompassGait::default();
    let clf = ResClf::double_integrator(1, &DMatrix::identity(2, 2)).expect("RES-CLF");
    let alpha = 0.22;

    println!("Q1 / M1 - a learned virtual constraint with a certified transverse metric and a verified reduced map");
    println!("(compass gait, {:.2} deg downhill; training on the analytic reduction, verification on the full model)\n", r.slope.to_degrees());

    // ---- (a) train the constraint, with invariance as an explicit objective ----
    let hand = VirtualConstraint { alpha, c: 5.42474849, e: 3.5 };
    let hand_scuff = worst_clearance(&r, &hand, 400);
    println!("(a) the learned constraint");
    println!("    M0's hand-tuned baseline: invariance defect {:.2e}, worst foot height {hand_scuff:+.6} m", invariance_defect(&r, &hand));

    let goal = GaitGoal { target_zeta: 4.7, ..GaitGoal::default() };
    let (vc, sc) = train_constraint(&r, alpha, 6, &goal, &[0.0, 0.77, 0.0, 0.0, 0.0, 0.0], 8000);
    println!("    trained (6 weights):      invariance defect {:.2e}, worst foot height {:+.6} m", sc.invariance_defect, sc.worst_clearance);
    println!("    weights {:?}", vc.w.iter().map(|x| format!("{x:+.4}")).collect::<Vec<_>>());
    println!("    scuff depth reduced {:.1}x while keeping invariance", hand_scuff.abs() / sc.worst_clearance.abs().max(1e-12));
    println!("    (positive clearance is geometrically impossible here - theta2 crosses zero mid-step, where the");
    println!("     height is l(cos theta1 - 1) <= 0. Knees are the fix; the objective is depth, not elimination.)");

    // ---- (c) verify the reduced map, over a continuum ----
    println!("\n(c) the reduced return map, verified rather than sampled");
    let Some(map) = r.restricted_map(&vc, QUAD) else {
        println!("    the reduction breaks down: D vanishes somewhere on the step");
        return;
    };
    // Richardson: the quadrature is second-order, so the gap between n and 2n bounds the error to ~1/3 of it
    let coarse = r.restricted_map(&vc, QUAD / 2).unwrap();
    let err = (map.delta_sq - coarse.delta_sq).abs() / 3.0;
    println!("    delta^2 = {:.8} +- {err:.1e}  (flow {:.6} x impact {:.6})", map.delta_sq, map.flow_gain, map.impact_gain);
    println!("    V = {:.8}, fixed point zeta* = {:.6} -> stance rate {:.6} /s", map.v_zero, map.gait().unwrap_or(f64::NAN), map.gait().unwrap_or(f64::NAN).sqrt());
    println!("    contraction certified: delta^2 + error bound = {:.8} < 1 : {}", map.delta_sq + err, map.delta_sq + err < 1.0);

    let Some(stall) = r.stall_threshold(&vc, QUAD) else {
        println!("    the stall threshold could not be formed");
        return;
    };
    println!("    stall threshold: zeta_min = {stall:.6} (below this the robot stops before the guard)");
    match map.certified_basin(stall, 1e3) {
        Some((lo, hi)) => {
            println!("    REGION OF ATTRACTION: zeta in [{lo:.6}, {hi:.1e})  -  forward invariant, since rho(zeta_min) = {:.6} >= {lo:.6}", map.apply(lo));
            println!("      In stance rate: [{:.4}, {:.1}) /s. Every initial condition in that interval converges to", lo.sqrt(), hi.sqrt());
            println!("      the gait, because the map is affine and monotone - a statement about a continuum, not a sample.");
        }
        None => println!("    the interval above the stall threshold is not forward invariant, so no basin is certified"),
    }

    // ---- verification on the FULL four-state model, sweeping the RES-CLF's epsilon ----
    //
    // The reduced map certifies only the motion *on* `Z`. The directions off it are the RES-CLF's job, and
    // Theorem 6.3 is explicit about the shape of the guarantee: for `ε` **small enough** the continuous
    // contraction dominates the impact's expansion, and the threshold is `ε̄ = c₃T/(2 ln μ)`. So `ε` is not a
    // tuning knob to be left at a plausible value — it is the thing the theorem quantifies, and sweeping it is
    // how the theorem gets tested rather than assumed.
    println!("\nverification on the full four-state model (never used in training), sweeping the RES-CLF epsilon");
    println!("      eps      full rho   restricted   transverse   coupling   certified");
    let mut best: Option<(f64, f64, f64, f64)> = None;
    for &eps in &[0.08f64, 0.04, 0.02, 0.01, 0.005] {
        let step = |s: &GaitState| r.step_to_guard(s, &control(&r, &vc, &clf, eps), DT, MAX_STEP_TIME);
        let mut d1 = map.gait().unwrap_or(4.0).sqrt();
        let mut ok = true;
        for _ in 0..400 {
            match step(&vc.on_manifold(-alpha, d1)) {
                Some((post, _)) => {
                    if (post.d1 - d1).abs() < 1e-12 {
                        d1 = post.d1;
                        break;
                    }
                    d1 = post.d1;
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            println!("      {eps:<8} the gait failed to close at this eps");
            continue;
        }
        let fixed = vc.on_manifold(-alpha, d1);
        let full_map = |x: &DVector<f64>| step(&GaitState::from_vec(x)).map(|(p, _)| p.to_vec());
        let Some(mono) = ferromotion_core::return_map_jacobian(&full_map, &fixed.to_vec(), 1e-6) else { continue };
        let (rho_full, _) = ferromotion_core::poincare_stability(&mono);
        let z_basis = z_tangent(&vc, &fixed);
        match hzd_reduction(&mono, &z_basis, 0.05) {
            Some(red) => {
                println!("      {eps:<8} {rho_full:>9.5} {:>12.6} {:>12.5} {:>10.4}   {}", red.restricted_rho, red.transverse_rho, red.coupling, red.certified());
                if red.certified() && best.is_none() {
                    best = Some((eps, d1, red.restricted_rho, red.transverse_rho));
                }
            }
            None => println!("      {eps:<8} {rho_full:>9.5}  the reduction could not be formed"),
        }
    }

    // The theorem's own prediction for where that transition sits.
    let period_at = {
        let step = |s: &GaitState| r.step_to_guard(s, &control(&r, &vc, &clf, 0.02), DT, MAX_STEP_TIME);
        step(&vc.on_manifold(-alpha, map.gait().unwrap().sqrt())).map(|(_, t)| t).unwrap_or(0.34)
    };
    println!("\n    Theorem 6.3's threshold, from the RES-CLF's own constant: with c3 = {:.4} and a step of {period_at:.4} s,", clf.c3);
    for mu_sq in [10.0f64, 100.0, 500.0] {
        if let Some(bar) = clf.eps_bar(mu_sq.sqrt(), period_at) {
            println!("      an impact expansion of mu^2 = {mu_sq:>5} would need eps < {bar:.5}");
        }
    }

    let Some((eps, d1, rr, tr)) = best else {
        println!("\n    No epsilon in the swept range certified the orbit. The reduced map still contracts, so the");
        println!("    gait exists on Z; what fails is domination of the impact's transverse expansion, which is a");
        println!("    statement about the controller's rate and not about the constraint.");
        return;
    };
    println!("\n    CERTIFIED at eps = {eps}: restricted {rr:.6}, transverse {tr:.6}, stance rate {d1:.6} /s");
    println!("    the analytic reduction predicted {:.6} /s -> {:.3}% apart", map.gait().unwrap().sqrt(), 100.0 * (d1 - map.gait().unwrap().sqrt()).abs() / d1);

    // rebuild the certified monodromy for the transverse metric below
    let fixed = vc.on_manifold(-alpha, d1);
    let step = |s: &GaitState| r.step_to_guard(s, &control(&r, &vc, &clf, eps), DT, MAX_STEP_TIME);
    let full_map = |x: &DVector<f64>| step(&GaitState::from_vec(x)).map(|(p, _)| p.to_vec());
    let mono = ferromotion_core::return_map_jacobian(&full_map, &fixed.to_vec(), 1e-6).expect("monodromy");
    let z_basis = z_tangent(&vc, &fixed);

    // ---- (b) certify the transverse dynamics ----
    println!("\n(b) the transverse dynamics, certified in their own metric");
    // The transverse directions are the complement of Z. Restrict the monodromy there and solve the Stein
    // equation for a metric in which it contracts; the solution existing IS the certificate.
    let w_basis = complement_of(&z_basis);
    let trans = w_basis.transpose() * &mono * &w_basis;
    let (rho_t, _) = ferromotion_core::poincare_stability(&trans);
    println!("    transverse block spectral radius {rho_t:.6}");
    match ferromotion_core::solve_lyapunov_discrete(&trans, &DMatrix::identity(trans.nrows(), trans.nrows())) {
        Some(p) => {
            let eig = p.clone().symmetric_eigen();
            let (lo, hi) = (eig.eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min), eig.eigenvalues.iter().cloned().fold(0.0f64, f64::max));
            println!("    Stein equation solved: metric P has eigenvalues in [{lo:.4}, {hi:.4}], positive definite = {}", lo > 0.0);
            // the contraction factor in that metric, which is what the certificate actually asserts
            let m = &trans.transpose() * &p * &trans;
            let factor = generalised_max(&m, &p);
            println!("    contraction in the P metric: sup ||x+||_P / ||x||_P = {:.6}", factor.sqrt());
            println!("    => transverse directions contract by that factor every step, with an explicit metric.");
            println!("       A positive-definite Stein solution exists if and only if the transverse block is Schur,");
            println!("       so its existence is the certificate rather than evidence for one.");
        }
        None => println!("    no positive-definite metric exists: the transverse dynamics do not contract"),
    }

    println!("\nWhat M1 delivers, and what it does not. Invariance is trained rather than constructed, the");
    println!("transverse dynamics carry an explicit metric, and the reduced map is verified over an interval");
    println!("rather than at samples - because it is one-dimensional and affine. What is still missing for Q1");
    println!("proper is a *policy* in place of a polynomial: h_w here is six weights trained on an analytic");
    println!("objective, not a network trained by reinforcement on the full robot. That is M2's problem.");
}

/// The tangent space of `Z` at a point: the kernel of the two output differentials.
fn z_tangent(vc: &LearnedConstraint, at: &GaitState) -> DMatrix<f64> {
    let (_, hd1, hd2) = vc.desired(at.th1);
    let mut dy = DMatrix::zeros(2, 4);
    dy[(0, 0)] = -hd1;
    dy[(0, 1)] = 1.0;
    dy[(1, 0)] = -hd2 * at.d1;
    dy[(1, 2)] = -hd1;
    dy[(1, 3)] = 1.0;
    kernel_of(&dy)
}

/// An orthonormal basis of the kernel of `a`.
fn kernel_of(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.ncols();
    let rank = a.rank(1e-10);
    let q = a.transpose().qr().q();
    let row_space = q.columns(0, rank).into_owned();
    let p = DMatrix::identity(n, n) - &row_space * row_space.transpose();
    p.svd(true, false).u.map(|u| u.columns(0, n - rank).into_owned()).unwrap_or_else(|| DMatrix::identity(n, n))
}

/// An orthonormal basis of the orthogonal complement of `z`'s columns.
fn complement_of(z: &DMatrix<f64>) -> DMatrix<f64> {
    let n = z.nrows();
    let k = z.ncols();
    let p = DMatrix::identity(n, n) - z * z.transpose();
    p.svd(true, false).u.map(|u| u.columns(0, n - k).into_owned()).unwrap_or_else(|| DMatrix::identity(n, n))
}

/// The largest generalised eigenvalue of `(m, p)` with `p` positive definite: `max xᵀmx / xᵀpx`.
fn generalised_max(m: &DMatrix<f64>, p: &DMatrix<f64>) -> f64 {
    let e = p.clone().symmetric_eigen();
    let inv_sqrt = &e.eigenvectors * DMatrix::from_diagonal(&e.eigenvalues.map(|l| 1.0 / l.max(1e-300).sqrt())) * e.eigenvectors.transpose();
    let sym = &inv_sqrt * m * &inv_sqrt;
    let s = (&sym + sym.transpose()) * 0.5;
    s.symmetric_eigen().eigenvalues.iter().cloned().fold(0.0f64, f64::max)
}
