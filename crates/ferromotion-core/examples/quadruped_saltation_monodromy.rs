//! **A monodromy for the quadruped's gait, and what it took to get one.**
//!
//! `quadruped_gait_certificate` found that finite-differencing a whole gait cycle gave a spectral radius
//! running 11 → 291 → 769 → 21088 as the probe went 1e-3 → 1e-6, and concluded that a rigid-contact return
//! map is simply not differentiable. The first half of that was a correct measurement. The second half was
//! a wrong inference, and this example is what found the four separate defects actually behind it — three
//! of them in the tooling rather than the physics.
//!
//! 1. **The events are contact-mode changes, not touchdowns.** `quadruped_mode_events` counted 205 mode
//!    changes per period against 4 touchdowns, and 160 of them were chatter — a resting foot's impulse
//!    toggling every single timestep. Stabilising the contact solver's gap feedback took that to 18.
//! 2. **The probe must preserve the mode sequence.** At 1e-4 or 1e-5, several of the 25 perturbation pairs
//!    cross a mode boundary. At 1e-8, none do.
//! 3. **The map must be a function of the state.** Warm-starting the contact impulses from the previous
//!    step is what a simulator wants, but it makes the impulses hidden state, and then the chain rule fails
//!    across a split. Every linearisation here runs cold.
//! 4. **The state must be the whole state.** Packing the base pose as `(z, roll, pitch)` silently discards
//!    `x`, `y` and yaw, and reinterprets the base velocity against a different orientation on the way back.
//!
//! With all four in place the gait linearises cleanly and **contracts**: `ρ = 0.727`, steady across three
//! probe decades, matching an independently saltation-composed monodromy to 0.04% and predicting measured
//! motion to 0.01%. Two caveats travel with it and neither is small. The worst one-period gain is 10.7, so a
//! perturbation is amplified tenfold on its way to decaying — contracting is not the same as safe. And once
//! a perturbation grows enough to cross a contact-mode boundary it is amplified by about a million, which is
//! what bounds the certificate to a basin.
//!
//! Run: `cargo run --release --example quadruped_saltation_monodromy -p ferromotion-core`

use ferromotion_core::{compose_monodromy, plastic_impact_jacobian, poincare_stability, quadruped, quadruped_trot_tau, tree_floating_mass_matrix, whole_body_contact_jacobian, whole_body_contact_step_pgs, whole_body_forward_kinematics, HybridEvent, LinkInertia, WholeBodyContactPoint};
use nalgebra::{DMatrix, DVector, Isometry3, Matrix3, Point3, Translation3, UnitQuaternion, Vector3, Vector6};

const DT: f64 = 5e-4;
const PERIOD: f64 = 1.0;
const STAND_Z: f64 = 0.60;
const MU: f64 = 0.9;
/// Sweeps when warm-starting, which is what the simulator and the browser bench run.
const ITERS: usize = 40;
/// Sweeps when cold-starting. Converging from zero impulses needs more of them, and the answer has to be
/// converged or the finite difference measures solver residual instead of dynamics.
const COLD_ITERS: usize = 400;
/// Analysis coordinates: `[z, roll, pitch, v0(6), q(8), qd(8)]`.
const NX: usize = 25;
/// The gap below which `whole_body_contact_step_pgs` admits a contact. This, not the floor, is the guard
/// surface the dynamics switch on.
const ACTIVATE: f64 = 1e-3;

fn base_inertia() -> LinkInertia {
    LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)) }
}

/// The **full** state, carried by the flow. Projecting the base pose to `(z, roll, pitch)` is lossy: it
/// discards `x`, `y` and yaw, and since the base velocity is referred to the base orientation, a round trip
/// through such a projection also reinterprets the velocity. That loss is silent, and it is enough on its
/// own to break the chain rule across a split.
#[derive(Clone, Debug)]
struct Full {
    base: Isometry3<f64>,
    v0: Vector6<f64>,
    q: Vec<f64>,
    qd: Vec<f64>,
}

/// The coordinates the analysis works in. The three left out — `x`, `y`, yaw — are exact symmetries here
/// (flat infinite floor, vertical gravity, a controller reading only joint angles), so this block is closed.
/// Excluding them also keeps three trivial unit eigenvalues out of the spectral radius.
fn sub(x: &Full) -> DVector<f64> {
    let (roll, pitch, _yaw) = x.base.rotation.euler_angles();
    let mut v = DVector::zeros(NX);
    v[0] = x.base.translation.z;
    v[1] = roll;
    v[2] = pitch;
    for i in 0..6 {
        v[3 + i] = x.v0[i];
    }
    for i in 0..8 {
        v[9 + i] = x.q[i];
        v[17 + i] = x.qd[i];
    }
    v
}

/// Put analysis coordinates back into a full state, keeping `reference`'s symmetry coordinates.
fn with_sub(reference: &Full, v: &DVector<f64>) -> Full {
    let (_r, _p, yaw) = reference.base.rotation.euler_angles();
    let t = reference.base.translation;
    Full {
        base: Isometry3::from_parts(Translation3::new(t.x, t.y, v[0]), UnitQuaternion::from_euler_angles(v[1], v[2], yaw)),
        v0: Vector6::from_iterator((0..6).map(|i| v[3 + i])),
        q: (0..8).map(|i| v[9 + i]).collect(),
        qd: (0..8).map(|i| v[17 + i]).collect(),
    }
}

/// Height of every foot above the floor: a smooth function of the state with no contact in it, so it may be
/// differenced freely.
fn foot_heights(x: &Full) -> Vec<f64> {
    let (joints, _i, parent, feet) = quadruped();
    let world = whole_body_forward_kinematics(&joints, &parent, x.base, &x.q);
    feet.iter().map(|&(b, off, _)| (world[b] * Point3::from(off)).coords.z).collect()
}

/// Advance `secs` of the clocked gait. `warm_start` decides whether each step hands its contact impulses to
/// the next: fast when simulating, but it makes the impulses hidden state.
fn flow_opt(x: &Full, t0: f64, secs: f64, warm_start: bool, iters: usize) -> Option<Full> {
    if secs <= 0.0 {
        return Some(x.clone());
    }
    let (joints, inertia, parent, feet) = quadruped();
    let bi = base_inertia();
    let pts: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, MU)).collect();
    let g = Vector3::new(0.0, 0.0, -9.81);
    let mut s = x.clone();
    let mut warm: Option<Vec<Vector3<f64>>> = None;
    for k in 0..(secs / DT).round().max(1.0) as usize {
        let t = t0 + k as f64 * DT;
        let tau = quadruped_trot_tau(&s.q, &s.qd, std::f64::consts::TAU * t);
        let r = whole_body_contact_step_pgs(&joints, &inertia, &parent, &bi, s.base, s.v0, &s.q, &s.qd, &tau, &pts, 0.0, DT, iters, g, warm.as_deref());
        s = Full { base: r.base, v0: r.v0, q: r.q, qd: r.qd };
        warm = if warm_start { Some(r.impulses) } else { None };
        if !s.base.translation.vector.iter().all(|v| v.is_finite()) {
            return None;
        }
    }
    Some(s)
}

/// The flow every linearisation here uses: cold-started, so it is a function of the state alone.
fn flow_cold(x: &Full, t0: f64, secs: f64) -> Option<Full> {
    flow_opt(x, t0, secs, false, COLD_ITERS)
}

/// Jacobian of a stretch in analysis coordinates, by central differences. Valid when the stretch holds a
/// single contact mode, which is what the probe size has to guarantee.
fn stretch_jacobian(x: &Full, t0: f64, secs: f64, eps: f64) -> Option<DMatrix<f64>> {
    let s0 = sub(x);
    let mut j = DMatrix::zeros(NX, NX);
    for c in 0..NX {
        let (mut sp, mut sm) = (s0.clone(), s0.clone());
        sp[c] += eps;
        sm[c] -= eps;
        let a = flow_cold(&with_sub(x, &sp), t0, secs)?;
        let b = flow_cold(&with_sub(x, &sm), t0, secs)?;
        j.set_column(c, &((sub(&a) - sub(&b)) / (2.0 * eps)));
    }
    Some(j)
}

/// Find touchdowns in `[t0, t0+horizon)`, resolved to a single timestep.
///
/// The guard is the surface the *simulator* switches on: `whole_body_contact_step_pgs` admits a contact once
/// its gap falls below [`ACTIVATE`], so that is where the vector field changes and where a saltation matrix
/// belongs. The correction term divides by `gᵀf⁻`, which leaves no tolerance for being evaluated off the
/// surface, so the bracket has to be one step and not one chunk.
fn find_touchdowns(x0: &Full, t0: f64, horizon: f64) -> Vec<(f64, usize)> {
    let mut events = Vec::new();
    let mut x = x0.clone();
    let mut prev = foot_heights(&x);
    for k in 0..(horizon / DT).round() as usize {
        let t = t0 + k as f64 * DT;
        let Some(nx) = flow_cold(&x, t, DT) else { break };
        let now = foot_heights(&nx);
        for f in 0..prev.len() {
            if prev[f] > ACTIVATE && now[f] <= ACTIVATE {
                events.push((t, f)); // the state at `t` is the last one before the switch
            }
        }
        x = nx;
        prev = now;
    }
    events
}

/// The reset Jacobian in analysis coordinates: positions pass through, velocities are projected by the
/// plastic impact of the landing foot.
fn packed_reset(x: &Full, foot: usize) -> DMatrix<f64> {
    let (joints, inertia, parent, feet) = quadruped();
    let bi = base_inertia();
    let world = whole_body_forward_kinematics(&joints, &parent, x.base, &x.q);
    let m = tree_floating_mass_matrix(&joints, &inertia, &parent, &bi, &x.q);
    let (body, off, _mu) = feet[foot];
    let jc = whole_body_contact_jacobian(&joints, &parent, &world, x.base, Some(body), off);
    let delta = plastic_impact_jacobian(&m, &jc); // 14x14 on [v0(6); qd(8)]

    let mut r = DMatrix::identity(NX, NX);
    let vel: Vec<usize> = (3..9).chain(17..25).collect(); // v0 -> 3..9, qd -> 17..25
    for (a, &ia) in vel.iter().enumerate() {
        for (b, &ib) in vel.iter().enumerate() {
            r[(ia, ib)] = delta[(a, b)];
        }
    }
    r
}

/// The vector field in analysis coordinates, by one short cold step on whichever side the caller places `x`.
fn vector_field(x: &Full, t: f64) -> Option<DVector<f64>> {
    let nx = flow_cold(x, t, DT)?;
    Some((sub(&nx) - sub(x)) / DT)
}

fn main() {
    println!("A monodromy for the quadruped's gait");
    println!("(dt {DT} s, period {PERIOD} s, mu {MU}, hard contact via Gauss-Seidel, {COLD_ITERS} cold sweeps)\n");

    let start = Full { base: Isometry3::translation(0.0, 0.0, STAND_Z), v0: Vector6::zeros(), q: vec![0.0; 8], qd: vec![0.0; 8] };
    // Settle on the cold trajectory, since that is the one every linearisation below is taken about.
    let Some(settled) = flow_cold(&start, 0.0, 6.0 * PERIOD) else {
        println!("the gait diverged while settling");
        return;
    };
    println!("after 6 cycles: torso z {:.4} m, foot heights {:?}", settled.base.translation.z, foot_heights(&settled).iter().map(|h| format!("{h:.3}")).collect::<Vec<_>>());

    // Prerequisite: is the map a function of the state? If splitting the flow changes where it ends up, the
    // chain rule does not hold across a split and no composition can match the whole. Check before building.
    println!("\nis the flow a function of the state alone? splitting one period at its midpoint:");
    let mut splits_cleanly = false;
    for (label, warm, iters) in [("warm-started (what the simulator runs)", true, ITERS), ("cold-started (what the linearisation uses)", false, COLD_ITERS)] {
        let Some(whole) = flow_opt(&settled, 0.0, PERIOD, warm, iters) else { continue };
        let Some(half) = flow_opt(&settled, 0.0, PERIOD * 0.5, warm, iters) else { continue };
        let Some(split) = flow_opt(&half, PERIOD * 0.5, PERIOD * 0.5, warm, iters) else { continue };
        let d = (sub(&whole) - sub(&split)).norm();
        let ok = d < 1e-12;
        println!("   {label}: one 1.0 s call vs two 0.5 s calls differ by {d:.3e}");
        println!("      {}", if ok { "identical, so the chain rule holds across a split" } else { "NOT identical: something outside the state is being carried, so a composition over the state cannot match the whole" });
        if !warm {
            splits_cleanly = ok;
        }
    }

    let events = find_touchdowns(&settled, 0.0, PERIOD);
    println!("\ntouchdowns found in one period: {} (guard at a gap of {ACTIVATE:.0e} m)", events.len());
    for &(t, f) in &events {
        // How far from the guard the linearisation point sits. The correction term divides by g'f-, so this
        // is the number that decides whether a saltation matrix means anything at this point.
        let off = flow_cold(&settled, 0.0, t.max(DT)).map(|x| foot_heights(&x)[f] - ACTIVATE).unwrap_or(f64::NAN);
        println!("   t = {t:.4} s   foot {f}   distance from the guard {off:+.2e} m");
    }

    // Route one: difference the whole period at a probe fine enough to preserve the mode sequence.
    println!("\nroute one, the whole period differenced directly:");
    let mut direct = None;
    for eps in [1e-7_f64, 1e-8, 1e-9] {
        if let Some(j) = stretch_jacobian(&settled, 0.0, PERIOD, eps) {
            let (rho, stable) = poincare_stability(&j);
            // The largest singular value, alongside the spectral radius. rho governs the asymptotic rate;
            // the singular value governs what a single period can do to the worst direction. When the two
            // are far apart the matrix is non-normal, and a contracting gait can still amplify a
            // perturbation several-fold on the way down. That gap is the difference between "settles
            // eventually" and "safe now".
            let gain = j.singular_values().iter().cloned().fold(0.0f64, f64::max);
            println!("   probe {eps:.0e}: rho = {rho:.4}, contracting = {stable}, worst one-period gain = {gain:.3}");
            if eps == 1e-8 {
                direct = Some((rho, j));
            }
        }
    }

    if events.is_empty() {
        println!("\nNo activation crossings in this period, so there is no event for a saltation matrix to");
        println!("correct and route one is already the linearisation.");
    } else {
        // Route two: split at the touchdowns and compose. Three variants, because two questions are open.
        // A time-stepping simulator resolves the landing inside the step after the crossing, so an explicit
        // plastic-impact reset may be counting the same impact twice; and the plain chain rule over the same
        // split is the control that says whether the split itself is sound.
        println!("\nroute two, split at the touchdowns and composed:");
        for (label, use_reset, use_saltation) in [("with the plastic-impact reset", true, true), ("no reset, saltation timing correction only", false, true), ("plain chain rule over the same split", false, false)] {
            println!("   {label}:");
            for eps in [1e-8_f64, 1e-9] {
                let mut chain: Vec<HybridEvent> = Vec::new();
                let mut x = settled.clone();
                let mut t = 0.0;
                let mut ok = true;

                for &(te, foot) in &events {
                    let span = te - t;
                    if span <= DT {
                        continue;
                    }
                    let Some(phi) = stretch_jacobian(&x, t, span, eps) else { ok = false; break };
                    let Some(x_minus) = flow_cold(&x, t, span) else { ok = false; break };

                    // guard normal: the gap gradient, differenced as the smooth function it is
                    let s_minus = sub(&x_minus);
                    let mut g = DVector::zeros(NX);
                    for c in 0..NX {
                        let (mut sp, mut sm) = (s_minus.clone(), s_minus.clone());
                        sp[c] += eps;
                        sm[c] -= eps;
                        g[c] = (foot_heights(&with_sub(&x_minus, &sp))[foot] - foot_heights(&with_sub(&x_minus, &sm))[foot]) / (2.0 * eps);
                    }
                    let reset = if use_reset { packed_reset(&x_minus, foot) } else { DMatrix::identity(NX, NX) };
                    let Some(f_minus) = vector_field(&x_minus, te) else { ok = false; break };
                    let x_plus = with_sub(&x_minus, &(&reset * &s_minus));
                    let Some(f_plus) = vector_field(&x_plus, te) else { ok = false; break };

                    chain.push(HybridEvent { flow_jacobian: phi, reset_jacobian: reset, guard_normal: g, f_minus, f_plus });
                    x = x_plus;
                    t = te;
                }
                if !ok {
                    println!("      probe {eps:.0e}: a stretch failed to integrate");
                    continue;
                }
                let Some(phi_end) = stretch_jacobian(&x, t, (PERIOD - t).max(DT), eps) else {
                    println!("      probe {eps:.0e}: the closing stretch failed");
                    continue;
                };
                let composed = if use_saltation {
                    compose_monodromy(&chain, &phi_end)
                } else {
                    Some(&phi_end * chain.iter().fold(DMatrix::identity(NX, NX), |acc, e| &e.reset_jacobian * &e.flow_jacobian * acc))
                };
                match composed {
                    Some(mono) => {
                        let (rho, stable) = poincare_stability(&mono);
                        let agree = direct.as_ref().map(|(d, _)| format!(", {:>8.1}% from route one", 100.0 * (rho - d).abs() / d.abs().max(1e-30))).unwrap_or_default();
                        println!("      probe {eps:.0e}: {} events composed, rho = {rho:.4}, contracting = {stable}{agree}", chain.len());
                    }
                    None => println!("      probe {eps:.0e}: a crossing was non-transversal, so no linearisation exists"),
                }
            }
        }
    }

    // The arbiter. A Jacobian's whole job is to predict where a perturbed gait goes, so measure that
    // directly. This uses no Jacobian, so it can contradict one, and it is the only thing here that can
    // settle a disagreement between two probe-stable answers.
    if let Some((rho, j)) = direct {
        println!("\nthe arbiter: measured motion against route one's prediction, no Jacobian in the measurement");
        println!("      |delta|    relative error of J*delta");
        let mut dir = DVector::from_fn(NX, |i, _| ((i * 37 % 17) as f64 - 8.0) / 8.0);
        dir /= dir.norm();
        let Some(f0) = flow_cold(&settled, 0.0, PERIOD) else { return };
        for mag in [1e-7_f64, 1e-8, 1e-9] {
            let d = &dir * mag;
            let Some(fp) = flow_cold(&with_sub(&settled, &(sub(&settled) + &d)), 0.0, PERIOD) else { continue };
            let measured = sub(&fp) - sub(&f0);
            let err = (&measured - &j * &d).norm() / measured.norm().max(1e-30);
            println!("      {mag:.0e}   {:>10.2}%", 100.0 * err);
        }

        println!("\n      and over several periods, does a perturbation grow by rho = {rho:.3} each time?");
        println!("      period   measured |delta|   growth this period   rho^n");
        let mag = 1e-9;
        let (mut xa, mut xb) = (settled.clone(), with_sub(&settled, &(sub(&settled) + &dir * mag)));
        let mut last = mag;
        for n in 1..=4 {
            let (Some(na), Some(nb)) = (flow_cold(&xa, 0.0, PERIOD), flow_cold(&xb, 0.0, PERIOD)) else { break };
            let dn = (sub(&nb) - sub(&na)).norm();
            println!("      {n:>6}   {dn:>15.3e}   {:>18.3}   {:>15.3e}", dn / last, mag * rho.powi(n));
            last = dn;
            xa = na;
            xb = nb;
        }
    }

    println!("\nWhat to take from it. The split-consistency line comes first: without it the chain rule does");
    println!("not hold and route two cannot match route one however good the saltation matrices are.");
    if splits_cleanly {
        println!("It holds here, so a disagreement between the routes is about the events, not the split.");
    } else {
        println!("It does not hold here, so route two is measuring a different trajectory and route one is the");
        println!("only linearisation of the simulator itself. That is a finding about the engine, not a bug in");
        println!("the composition, which reproduces the analytic bouncing ball exactly.");
    }
    println!("The arbiter settles the value: two probe-stable answers can disagree, and only measured motion");
    println!("says which one describes the robot.");
}
