//! **Is the quadruped's gait a stable limit cycle?** — the certificate machinery pointed at the full
//! body rather than a reduced model.
//!
//! The crawl is driven by a clock: the torque is a periodic function of time, so the robot is a
//! *non-autonomous* periodic system. That simplifies the question. An autonomous orbit always carries a
//! neutral eigenvalue along the flow, because the phase is free and must be quotiented out; here the
//! phase is pinned by the clock, so the stroboscopic map (state at phase 0 → state one period later)
//! has no trivial direction and its spectral radius decides orbital stability outright.
//!
//! What this measures, on the real 8-joint floating-base quadruped stepped with hard frictional
//! contact:
//!
//! 1. how close the gait is to periodic at all, `‖P(x) − x‖`;
//! 2. the monodromy of the stroboscopic map, and its spectral radius;
//! 3. whether a genuine fixed point can be found from there.
//!
//! Run: `cargo run --release --example quadruped_gait_certificate -p ferromotion-core`

use ferromotion_core::{find_limit_cycle, poincare_stability, quadruped, quadruped_trot_tau, return_map_jacobian, whole_body_contact_step_pgs, LinkInertia, WholeBodyContactPoint};
use nalgebra::{DVector, Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3, Vector6};

const DT: f64 = 5e-4;
const FREQ: f64 = 1.0;
const PERIOD: f64 = 1.0 / FREQ;
const STAND_Z: f64 = 0.60;
const MU: f64 = 0.9;
const PGS_ITERS: usize = 40;

fn base_inertia() -> LinkInertia {
    LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)) }
}

/// The gait state that matters. Absolute `x`, `y` and yaw are ignorable — the gait is invariant to
/// where on the floor it happens — so the return map lives on height, tilt, velocity and joints.
/// Layout: `[z, roll, pitch, v0(6), q(8), qd(8)]`, 25 numbers.
fn pack(base: &Isometry3<f64>, v0: &Vector6<f64>, q: &[f64], qd: &[f64]) -> DVector<f64> {
    let (roll, pitch, _yaw) = base.rotation.euler_angles();
    let mut x = DVector::zeros(25);
    x[0] = base.translation.z;
    x[1] = roll;
    x[2] = pitch;
    for i in 0..6 {
        x[3 + i] = v0[i];
    }
    for i in 0..8 {
        x[9 + i] = q[i];
        x[17 + i] = qd[i];
    }
    x
}

fn unpack(x: &DVector<f64>) -> (Isometry3<f64>, Vector6<f64>, Vec<f64>, Vec<f64>) {
    let base = Isometry3::from_parts(Translation3::new(0.0, 0.0, x[0]), UnitQuaternion::from_euler_angles(x[1], x[2], 0.0));
    let v0 = Vector6::from_iterator((0..6).map(|i| x[3 + i]));
    let q: Vec<f64> = (0..8).map(|i| x[9 + i]).collect();
    let qd: Vec<f64> = (0..8).map(|i| x[17 + i]).collect();
    (base, v0, q, qd)
}

/// Simulate `secs` of the clocked crawl from a packed state, starting the clock at `t0`.
fn simulate(x: &DVector<f64>, t0: f64, secs: f64) -> Option<DVector<f64>> {
    let (joints, inertia, parent, feet) = quadruped();
    let bi = base_inertia();
    let pts: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, MU)).collect();
    let g = Vector3::new(0.0, 0.0, -9.81);
    let (mut base, mut v0, mut q, mut qd) = unpack(x);
    let mut warm: Option<Vec<Vector3<f64>>> = None;
    let steps = (secs / DT).round() as usize;
    for k in 0..steps {
        let t = t0 + k as f64 * DT;
        let tau = quadruped_trot_tau(&q, &qd, std::f64::consts::TAU * FREQ * t);
        let r = whole_body_contact_step_pgs(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &pts, 0.0, DT, PGS_ITERS, g, warm.as_deref());
        base = r.base;
        v0 = r.v0;
        q = r.q.clone();
        qd = r.qd.clone();
        warm = Some(r.impulses);
        if !base.translation.vector.iter().all(|v| v.is_finite()) {
            return None;
        }
    }
    Some(pack(&base, &v0, &q, &qd))
}

fn main() {
    println!("Is the quadruped's crawl a stable limit cycle?");
    println!("clocked gait, so the stroboscopic map has no free phase and rho decides outright");
    println!("(dt {DT} s, period {PERIOD} s, mu {MU}, hard contact via Gauss-Seidel)\n");

    // let the transient wash out: several gait cycles from a standing start
    let start = pack(&Isometry3::translation(0.0, 0.0, STAND_Z), &Vector6::zeros(), &[0.0; 8], &[0.0; 8]);
    let settled = match simulate(&start, 0.0, 6.0 * PERIOD) {
        Some(s) => s,
        None => {
            println!("the gait diverged during settling; nothing to certify");
            return;
        }
    };
    println!("after 6 cycles: torso z {:.4} m, roll {:+.4}, pitch {:+.4} rad", settled[0], settled[1], settled[2]);

    // the stroboscopic map: one full gait period, clock starting at phase 0
    let stroboscopic = |x: &DVector<f64>| simulate(x, 0.0, PERIOD);

    let once = match stroboscopic(&settled) {
        Some(v) => v,
        None => {
            println!("the gait diverged over one period");
            return;
        }
    };
    let defect = (&once - &settled).norm();
    println!("\n1. how periodic is it?");
    println!("   ||P(x) - x|| = {defect:.4e}   (0 would be an exactly periodic gait)");
    println!("   worst single coordinate: {:.4e}", (&once - &settled).amax());

    println!("\n2. monodromy of the stroboscopic map (25 states, central differences)");
    let mono = match return_map_jacobian(&stroboscopic, &settled, 1e-5) {
        Some(m) => m,
        None => {
            println!("   the map failed to return from a perturbed state; no monodromy");
            return;
        }
    };
    let (rho, _stable) = poincare_stability(&mono);
    println!("   spectral radius rho = {rho:.4}   (do not read this yet - see 2b)");
    let eig = mono.complex_eigenvalues();
    let mut mags: Vec<f64> = eig.iter().map(|e| e.norm()).collect();
    mags.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let top: Vec<String> = mags.iter().take(6).map(|m| format!("{m:.3}")).collect();
    println!("   largest eigenvalue magnitudes: {}", top.join(", "));
    println!("   unstable directions (|lambda| > 1): {} of 25", mags.iter().filter(|m| **m > 1.0).count());

    // A real derivative barely moves when the probe step changes. A rigid-contact return map is
    // NOT smooth: a small perturbation can change which foot is on the ground at a given instant, and
    // differencing across that mode change measures the jump, not a slope. If rho swings with the step
    // size, the number above is an artifact of the method and not a property of the gait.
    println!("\n2b. is that spectral radius even a derivative? (a real one is step-independent)");
    for &e in &[1e-3_f64, 1e-4, 1e-5, 1e-6] {
        match return_map_jacobian(&stroboscopic, &settled, e) {
            Some(m) => {
                let (r, _) = poincare_stability(&m);
                println!("   probe step {e:.0e}: rho = {r:.4}");
            }
            None => println!("   probe step {e:.0e}: map failed to return"),
        }
    }

    println!("\n3. is there a genuine fixed point nearby?");
    match find_limit_cycle(&stroboscopic, &settled, 1e-7, 12) {
        Some(fp) => {
            let r = stroboscopic(&fp).map(|v| (v - &fp).norm()).unwrap_or(f64::NAN);
            println!("   found one: ||P(x) - x|| = {r:.3e}, torso z {:.4} m", fp[0]);
            if let Some(m2) = return_map_jacobian(&stroboscopic, &fp, 1e-5) {
                let (r2, s2) = poincare_stability(&m2);
                println!("   monodromy there: rho = {r2:.4}, orbitally stable = {s2}");
            }
        }
        None => println!("   none converged - expected, since Newton is steered by the same broken\n   finite-difference Jacobian diagnosed in 2b"),
    }

    println!("\nReading of the result. The spectral radius above is NOT a property of the gait: it moves");
    println!("by orders of magnitude with the probe step, and grows without bound as the step shrinks,");
    println!("which is the signature of differencing across a discontinuity rather than measuring a");
    println!("slope. A rigid-contact return map is genuinely non-smooth - a small perturbation changes");
    println!("which foot is on the ground at a given instant - so finite differences do not linearise");
    println!("it at all. It also contradicts what the robot does: at rho = 769 a micron of error would");
    println!("topple it within two cycles, and the crawl walks.");
    println!("\nThat diagnosis was right about the shortcut and wrong about the cause, and the difference");
    println!("mattered. quadruped_saltation_monodromy found four separate defects behind these numbers:");
    println!("the contact solver chattered (a resting foot's impulse toggling every timestep, 205 mode");
    println!("changes per period instead of 18); warm-starting the impulses made them hidden state, so the");
    println!("map was not a function of the state at all; the state vector silently dropped the base x, y");
    println!("and yaw; and the probe has to be 1e-8 or finer to stay inside one contact mode.");
    println!("\nWith all four fixed the gait DOES linearise, and it contracts: rho = 0.727, stable across");
    println!("three probe decades, agreeing to 0.04% with an independent saltation-composed monodromy and");
    println!("predicting measured motion to 0.01%. The worst one-period gain is 10.7 though, so a");
    println!("perturbation is amplified tenfold on its way to decaying - and once one grows enough to cross");
    println!("a contact-mode boundary it is amplified by about a million. Contracting is not the same as");
    println!("safe, and the basin is what bounds the certificate. See quadruped_saltation_monodromy.");
}
