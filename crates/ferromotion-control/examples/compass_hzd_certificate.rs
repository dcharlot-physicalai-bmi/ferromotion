//! **Q1 milestone M0: a verified hybrid-zero-dynamics certificate for a known controller on a simulated
//! biped.**
//!
//! The reconciliation-gap agenda's first problem asks for a machine-checkable orbital-stability certificate
//! attached to a *learned* gait. Its first milestone is to earn the right to attempt that, by producing the
//! certificate for a controller whose structure is known — reproducing the baseline the learned case has to
//! match. This is that baseline, end to end, on the compass-gait biped.
//!
//! The pipeline runs in the order the theory demands, and each stage can fail loudly:
//!
//! 1. **The model is verified first**, by physics rather than by inspection — energy conserved to 1e-13,
//!    angular momentum about the impacting foot conserved to 1e-16 (see the module's own tests).
//! 2. **A virtual constraint** `θ₂ = h_d(θ₁)` is imposed and feedback-linearised, with the stabilising term
//!    supplied by a **RES-CLF** so the convergence rate is a design parameter (Theorem 6.3).
//! 3. **Hybrid invariance is measured, not assumed.** `Δ(S ∩ Z) ⊆ Z` is what makes the reduction legal, and a
//!    generic constraint does not satisfy it. The one free shape parameter `c` is solved for.
//! 4. **The limit cycle** is found as a fixed point of the return map on the guard.
//! 5. **The restricted return map** is measured and checked against the affine form `ρ(ζ) = δ²ζ − V` that
//!    hybrid zero dynamics predicts in momentum coordinates — the contraction coming from the *impact*.
//! 6. **Morris-Grizzle** is applied: the full four-dimensional monodromy is tested for block-triangularity,
//!    and only if it holds does the one-dimensional restricted map certify the full-order orbit.
//!
//! Run: `cargo run --release --example compass_hzd_certificate -p ferromotion-control`

use ferromotion_control::{hzd_reduction, CompassGait, GaitState, ResClf, VirtualConstraint, ZeroDynamicsReturnMap};
use nalgebra::{DMatrix, DVector};

const DT: f64 = 2e-5;
const MAX_STEP_TIME: f64 = 4.0;

/// The controller: feedback-linearise the output and stabilise it with the RES-CLF's descent rate.
fn control<'a>(r: &'a CompassGait, vc: &'a VirtualConstraint, clf: &'a ResClf, eps: f64) -> impl Fn(&GaitState) -> f64 + 'a {
    move |s: &GaitState| {
        let (y, yd) = vc.output(s);
        let eta = DVector::from_row_slice(&[y, yd]);
        // the CLF-QP's minimum-norm descent input, applied as the desired output acceleration
        let v = clf.clf_qp(&eta, eps).map(|u| u[0]).unwrap_or(0.0);
        r.hzd_torque(s, vc, v).unwrap_or(0.0)
    }
}

/// One step of the closed loop, from a post-impact state back to the next post-impact state.
fn step(r: &CompassGait, vc: &VirtualConstraint, clf: &ResClf, eps: f64, s: &GaitState) -> Option<(GaitState, f64)> {
    r.step_to_guard(s, &control(r, vc, clf, eps), DT, MAX_STEP_TIME)
}

fn main() {
    let r = CompassGait::default();
    println!("Q1 / M0 - a verified HZD certificate for a known controller on a simulated biped");
    println!("(compass gait: 1 m legs, 5 kg each, 10 kg hip, {:.2} deg downhill; dt {DT} s)\n", r.slope.to_degrees());
    let clf = ResClf::double_integrator(1, &DMatrix::identity(2, 2)).expect("RES-CLF");
    let eps = 0.08;
    println!("RES-CLF: c1 {:.4}, c2 {:.4}, c3 {:.4}; at eps {eps} the guaranteed output rate is {:.2} /s", clf.c1, clf.c2, clf.c3, clf.rate(eps));

    // ---- stage 3: hybrid invariance, solved rather than searched ----
    //
    // The condition is one scalar equation. On `Z` the pre-impact velocity ratio is fixed by the constraint;
    // the impact is *linear in velocity* at a fixed configuration, so it maps that ratio to a definite
    // post-impact ratio; and landing back on `Z` requires that ratio to equal `h_d'(−α)`. One equation — so
    // for any `e` it is a root-find on `c`, not a search over simulations. (`y` is automatically zero after the
    // impact because `h_d(∓α) = ±α` by construction; only `ẏ` has to be bought.)
    let alpha = 0.22;
    println!("\nstage 3: hybrid invariance. Delta(S n Z) subset Z is what makes the reduction legal.");

    let ratio_defect = |c: f64, e: f64| -> f64 {
        let vc = vc_at(alpha, c, e);
        let post = r.impact(&vc.on_manifold(alpha, 1.0)); // linear in velocity, so the scale is irrelevant
        let (_, hd1_post, _) = vc.desired(-alpha);
        post.d2 / post.d1 - hd1_post
    };
    // For a given `e`, solve for the `c` that makes the impact land back on Z.
    let solve_c = |e: f64| -> Option<f64> {
        let mut prev = (-6.0f64, ratio_defect(-6.0, e));
        for k in 1..=800 {
            let c = -6.0 + k as f64 * 0.02;
            let d = ratio_defect(c, e);
            if prev.1 * d < 0.0 {
                let (mut lo, mut hi) = (prev.0, c);
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if ratio_defect(lo, e) * ratio_defect(mid, e) <= 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return Some(0.5 * (lo + hi));
            }
            prev = (c, d);
        }
        None
    };

    // ---- stage 4: with invariance enforced for every `e`, find the `e` that closes a periodic gait ----
    //
    // The compass gait dissipates energy at every impact, so on level ground a periodic gait exists only if the
    // constraint puts it back. That is what the second parameter is for: `c` is spent on invariance, `e` buys
    // the energy balance. A one-parameter family gives an invariant manifold with no gait on it, which is
    // exactly what the first attempt at this produced.
    println!("\nstage 4: the periodic gait. Sweeping the energy parameter e, with c re-solved for invariance at each:");
    let one_step = |vc: &VirtualConstraint, d1: f64| -> Option<f64> {
        step(&r, vc, &clf, eps, &vc.on_manifold(-alpha, d1)).map(|(post, _)| post.d1)
    };
    // For each `e`, iterate the return map and see whether it settles.
    let settle = |vc: &VirtualConstraint| -> Option<(f64, usize)> {
        let mut d1 = 1.05;
        for k in 0..300 {
            let next = one_step(vc, d1)?;
            if !next.is_finite() || next <= 0.05 {
                return None;
            }
            if (next - d1).abs() < 1e-11 {
                return Some((next, k + 1));
            }
            d1 = next;
        }
        Some((d1, 300))
    };

    let mut found: Option<(VirtualConstraint, f64, usize)> = None;
    for k in 0..=14 {
        let e = -3.5 + k as f64 * 0.5;
        let Some(c) = solve_c(e) else {
            println!("   e = {e:>5.2}: no c gives hybrid invariance");
            continue;
        };
        let vc = vc_at(alpha, c, e);
        match settle(&vc) {
            Some((d1, iters)) => println!("   e = {e:>5.2}: c = {c:>9.5}, gait settles at stance rate {d1:.6} /s after {iters} steps"),
            None => println!("   e = {e:>5.2}: c = {c:>9.5}, the gait dies out"),
        }
        if let Some((d1, iters)) = settle(&vc)
            && found.as_ref().is_none_or(|f| d1 > f.1)
        {
            found = Some((vc, d1, iters));
        }
    }

    let Some((vc, d1, iters)) = found else {
        println!("\n   No member of this constraint family carries a periodic gait on level ground.");
        println!("   That is a real negative result about the family, not a solver failure: hybrid invariance");
        println!("   fixes c, and if the remaining freedom cannot close the energy balance there is no orbit to");
        println!("   certify. A richer constraint (a Bezier with more control points) is the standard remedy.");
        return;
    };
    let c = vc.c;
    println!("\n   chosen: alpha = {alpha}, c = {c:.8}, e = {:.4}", vc.e);
    println!("   invariance check on the impact itself: distance to Z = {:.3e}", {
        let post = r.impact(&vc.on_manifold(alpha, 1.0));
        let (y, yd) = vc.output(&post);
        (y * y + yd * yd).sqrt()
    });
    println!("   for contrast, c off by 20%: distance to Z = {:.4} - invariance is a measure-zero condition", {
        let g = vc_at(alpha, c * 1.2, vc.e);
        let post = r.impact(&g.on_manifold(alpha, 1.0));
        let (y, yd) = g.output(&post);
        (y * y + yd * yd).sqrt()
    });
    println!("   fixed point: stance rate {d1:.8} /s, reached in {iters} steps");
    let fixed = vc.on_manifold(-alpha, d1);
    let Some((check, period)) = step(&r, &vc, &clf, eps, &fixed) else { return };
    println!("   |P(x) - x| on the section = {:.3e}, step period {period:.5} s", (check.d1 - d1).abs());
    println!("   energy at the fixed point {:.4} J", r.energy(&fixed));

    // ---- stage 5: the restricted return map, against the affine form HZD predicts ----
    println!("\nstage 5: the restricted return map, in momentum coordinates zeta = d1^2.");
    println!("   HZD predicts rho(zeta) = delta^2 zeta - V, with delta^2 set by the IMPACT, not by feedback.");
    let mut pts = Vec::new();
    for f in [0.90f64, 0.95, 1.0, 1.05, 1.10] {
        let z_in = (d1 * f) * (d1 * f);
        if let Some((post, _)) = step(&r, &vc, &clf, eps, &vc.on_manifold(-alpha, d1 * f)) {
            pts.push((z_in, post.d1 * post.d1));
        }
    }
    if pts.len() < 3 {
        println!("   not enough of the neighbourhood survives to fit the map");
        return;
    }
    // least squares for (delta^2, V)
    let n = pts.len() as f64;
    let (sx, sy) = (pts.iter().map(|p| p.0).sum::<f64>(), pts.iter().map(|p| p.1).sum::<f64>());
    let (sxx, sxy) = (pts.iter().map(|p| p.0 * p.0).sum::<f64>(), pts.iter().map(|p| p.0 * p.1).sum::<f64>());
    let delta_sq = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let v_zero = -(sy - delta_sq * sx) / n;
    let map = ZeroDynamicsReturnMap { delta_sq, v_zero };
    let residual = pts.iter().map(|(x, y)| (map.apply(*x) - y).abs()).fold(0.0f64, f64::max);
    println!("   fitted delta^2 = {delta_sq:.6}, V = {v_zero:.6}; worst deviation from affine {residual:.3e}");
    println!("   fixed point of the fitted map: zeta* = {:?}, measured zeta* = {:.6}", map.fixed_point().map(|z| format!("{z:.6}")), d1 * d1);
    println!("   exponentially stable by the reduced map: {}", map.exponentially_stable());

    // ---- stage 6: Morris-Grizzle. Does the reduced answer certify the full-order orbit? ----
    println!("\nstage 6: Morris-Grizzle. Is the FULL monodromy block-triangular, so the 1-D map suffices?");
    let full_map = |x: &DVector<f64>| -> Option<DVector<f64>> {
        let s = GaitState::from_vec(x);
        step(&r, &vc, &clf, eps, &s).map(|(post, _)| post.to_vec())
    };
    let x_star = fixed.to_vec();
    let Some(mono) = ferromotion_core::return_map_jacobian(&full_map, &x_star, 1e-6) else {
        println!("   the full return map could not be differenced");
        return;
    };
    let (rho_full, stable_full) = ferromotion_core::poincare_stability(&mono);
    println!("   full 4x4 monodromy: rho = {rho_full:.6}, contracting = {stable_full}");

    // Z's tangent space at the fixed point: the manifold is {y = 0, ydot = 0}, so its tangent is the kernel of
    // the two output differentials.
    let (_, hd1, hd2) = vc.desired(fixed.th1);
    let mut dy = DMatrix::zeros(2, 4);
    dy[(0, 0)] = -hd1; // dy/dth1
    dy[(0, 1)] = 1.0; // dy/dth2
    dy[(1, 0)] = -hd2 * fixed.d1; // d(ydot)/dth1
    dy[(1, 2)] = -hd1; // d(ydot)/dd1
    dy[(1, 3)] = 1.0; // d(ydot)/dd2
    let z_basis = kernel_of(&dy);
    println!("   Z has dimension {} of 4 (the reduced object the certificate rests on)", z_basis.ncols());

    match hzd_reduction(&mono, &z_basis, 0.05) {
        Some(red) => {
            println!("   restricted rho = {:.6}, transverse rho = {:.6}, coupling {:.4}", red.restricted_rho, red.transverse_rho, red.coupling);
            println!("   reduction valid = {}, CERTIFIED = {}", red.valid, red.certified());
            if red.valid {
                println!("\n   => the full-order orbit is exponentially stable, certified through a {}-dimensional map.", z_basis.ncols());
                println!("      That is the HZD guarantee class, on the full hybrid model, machine-checked.");
            } else {
                println!("\n   => the monodromy is NOT block-triangular here, so the reduced map does not certify");
                println!("      the full orbit. The full rho above is the answer that stands.");
            }
        }
        None => println!("   the reduction could not be formed"),
    }

    // Where the contraction comes from, measured on the orbit itself rather than on a guessed state: flow the
    // fixed point to the guard, then apply the impact and compare.
    let ctrl = control(&r, &vc, &clf, eps);
    let mut at_guard = fixed;
    let mut prev;
    let mut t = 0.0;
    while t < MAX_STEP_TIME {
        prev = at_guard;
        at_guard = r.flow_step(&at_guard, ctrl(&at_guard), DT);
        t += DT;
        if prev.th1 > 0.0 && r.guard(&prev) > 0.0 && r.guard(&at_guard) <= 0.0 {
            break;
        }
    }
    let post = r.impact(&at_guard);
    println!("\nwhere the contraction comes from, on the orbit:");
    println!("   just before the impact: stance rate {:.5} /s, kinetic energy {:.4} J", at_guard.d1, r.kinetic(&at_guard));
    println!("   just after:             stance rate {:.5} /s, kinetic energy {:.4} J ({:.1}% retained)", post.d1, r.kinetic(&post), 100.0 * r.kinetic(&post) / r.kinetic(&at_guard));
    println!("   Continuous feedback only holds the robot on Z. The collision is what supplies delta^2 < 1,");
    println!("   and gravity down the {:.2} deg slope is what pays for it.", r.slope.to_degrees());
}

fn vc_at(alpha: f64, c: f64, e: f64) -> VirtualConstraint {
    VirtualConstraint { alpha, c, e }
}

/// An orthonormal basis of the kernel of `a`, as the orthogonal complement of its row space.
fn kernel_of(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.ncols();
    let qr = a.transpose().qr();
    let q = qr.q();
    let rank = a.rank(1e-10);
    let row_space = q.columns(0, rank).into_owned();
    let p = DMatrix::identity(n, n) - &row_space * row_space.transpose();
    let svd = p.svd(true, false);
    svd.u.map(|u| u.columns(0, n - rank).into_owned()).unwrap_or_else(|| DMatrix::identity(n, n))
}
