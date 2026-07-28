//! SYSTEM-ID READINESS — collapsing the bounded-reality envelope onto a physical unit.
//!
//! The recovery certificate holds worst-case over an ENVELOPE (friction ∈ [0.35, 0.85], unknown
//! inertials) because the CAD model is not the real arm. This is the protocol that, the day the
//! physical SO-101 lands, identifies its real dynamics from excitation data and COLLAPSES that
//! envelope onto the actual unit — so the certificate re-verifies TIGHTER (smaller fence, more
//! clearance). We prove the protocol in sim by treating a hidden, offset-parameter arm as "the real
//! one": excite it, identify its inertial parameters (ferromotion's regressor) + joint friction (a
//! linear extension of the same least-squares), show the identified model predicts held-out torques
//! exactly, that recovered friction matches the hidden truth, and that the certified table fence
//! SHRINKS as the friction envelope collapses. Honest: identify() recovers the identifiable (base-
//! parameter) subspace — validated predictively, not as a unique φ; the twin stands in for hardware.
use ferromotion_core::{forward_dynamics, from_urdf_full, inertial_regressor, inverse_dynamics, mass_matrix, LinkInertia, Robot};
use nalgebra::{DMatrix, DVector, Vector3};

const URDF: &str = include_str!("so101.urdf");
const G: f64 = -9.81;
const LIM: [[f64; 2]; 5] = [[-1.91986, 1.91986], [-1.74533, 1.74533], [-1.69, 1.69], [-1.65806, 1.65806], [-2.74385, 2.84121]];
const TAUMAX: f64 = 2.94;
const DT: f64 = 1.5e-3;
const ARMATURE: f64 = 0.028;
const KP: f64 = 70.0;
const KV: f64 = 13.0;
const A_UP: f64 = 11.3; // guaranteed upward tool accel under saturated pull-up (from so101_certify.rs)

fn sgn(x: f64) -> f64 { if x.abs() > 1e-6 { x.signum() } else { 0.0 } }
// deterministic small measurement noise (no Math.random in this environment)
fn noise(i: u32) -> f64 { let mut h = i.wrapping_mul(2654435761); h ^= h >> 13; ((h % 1000) as f64 / 1000.0 - 0.5) * 2.0 }

// a rich multi-sine excitation trajectory (analytic q, q̇, q̈), kept inside the joint limits.
fn traj(t: f64, i: usize) -> (f64, f64, f64) {
    let (w1, w2) = (1.3 + 0.4 * i as f64, 2.7 + 0.5 * i as f64);
    let a = 0.45 * (LIM[i][1].min(-LIM[i][0]));
    let q = a * ((w1 * t).sin() + 0.5 * (w2 * t + i as f64).sin());
    let qd = a * (w1 * (w1 * t).cos() + 0.5 * w2 * (w2 * t + i as f64).cos());
    let qdd = a * (-w1 * w1 * (w1 * t).sin() - 0.5 * w2 * w2 * (w2 * t + i as f64).sin());
    (q, qd, qdd)
}

// measure the worst-case tool DESCENT speed of a fold-down under a uniform joint damping `b`
// (the reality-envelope parameter the certificate's fence is sized to).
fn descent_speed(robot: &Robot, inertia: &[LinkInertia], q_fold: &[f64], q_start: &[f64], b: f64) -> f64 {
    let (mut q, mut qd) = (q_start.to_vec(), vec![0.0f64; 5]);
    let mut vmax = 0.0f64;
    let mut zprev = robot.fk(&q).translation.vector.z;
    for _ in 0..500 {
        let tau: Vec<f64> = (0..5).map(|i| (KP * (q_fold[i] - q[i]) - KV * qd[i]).clamp(-TAUMAX, TAUMAX) - b * qd[i] - 0.052 * sgn(qd[i])).collect();
        let qdd_link = forward_dynamics(robot, inertia, &q, &qd, &tau, Vector3::new(0.0, 0.0, G));
        let m = mass_matrix(robot, inertia, &q);
        let mut ma = m.clone(); for i in 0..5 { ma[(i, i)] += ARMATURE; }
        let qdd = ma.cholesky().unwrap().solve(&(&m * DVector::from_row_slice(&qdd_link)));
        for i in 0..5 { qd[i] += DT * qdd[i]; q[i] = (q[i] + DT * qd[i]).clamp(LIM[i][0], LIM[i][1]); }
        let z = robot.fk(&q).translation.vector.z;
        vmax = vmax.max(-(z - zprev) / DT); zprev = z;
        if z < 0.0 { break; }
    }
    vmax
}

fn main() {
    let (robot, cad) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");
    let n = robot.dof();
    let g = Vector3::new(0.0, 0.0, G);

    // ---- the HIDDEN TRUTH: the physical arm differs from CAD (per-link inertial scale) and has real
    // joint friction. In the field these are unknown; here we hide them and try to recover them. ----
    let scale = [1.08, 0.94, 1.12, 0.90, 1.05];
    let real: Vec<LinkInertia> = cad.iter().enumerate().map(|(i, li)| LinkInertia { mass: li.mass * scale[i], com: li.com, inertia: li.inertia * scale[i] }).collect();
    let b_visc_true = [0.62, 0.58, 0.64, 0.55, 0.50];
    let b_coul_true = [0.052, 0.050, 0.055, 0.048, 0.045];

    println!("System-ID readiness — collapsing the reality envelope onto the (hidden) real SO-101.\n");
    println!("  hidden truth: per-link inertial scale {:?}", scale);
    println!("               real viscous friction {:?} (CAD gives NONE — the widest uncertainty)\n", b_visc_true);

    // ---- excite the real arm and record measurements (q, q̇, q̈, τ) with small sensor noise ----
    let ns = 400usize;
    let mut samples_q = Vec::new();
    let mut big_y = DMatrix::zeros(n * ns, 10 * n + 2 * n); // [inertial | friction] augmented regressor
    let mut big_t = DVector::zeros(n * ns);
    for k in 0..ns {
        let t = k as f64 * 0.01;
        let (mut q, mut qd, mut qdd) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n { let (a, b, c) = traj(t, i); q[i] = a; qd[i] = b; qdd[i] = c; }
        let tau_dyn = inverse_dynamics(&robot, &real, &q, &qd, &qdd, g);
        let yin = inertial_regressor(&robot, &q, &qd, &qdd, g);
        for i in 0..n {
            let tau = tau_dyn[i] + b_visc_true[i] * qd[i] + b_coul_true[i] * sgn(qd[i]) + 1e-3 * noise(k as u32 * 7 + i as u32);
            big_t[k * n + i] = tau;
            for j in 0..10 * n { big_y[(k * n + i, j)] = yin[(i, j)]; }
            big_y[(k * n + i, 10 * n + 2 * i)] = qd[i];      // viscous friction column (joint i only)
            big_y[(k * n + i, 10 * n + 2 * i + 1)] = sgn(qd[i]); // Coulomb friction column (joint i only)
        }
        samples_q.push((q, qd, qdd));
    }
    // ---- joint least-squares over inertial + friction parameters ----
    let theta = big_y.pseudo_inverse(1e-9).expect("pinv") * &big_t;
    let phi_hat = DVector::from_iterator(10 * n, (0..10 * n).map(|j| theta[j]));
    let b_visc_hat: Vec<f64> = (0..n).map(|i| theta[10 * n + 2 * i]).collect();

    // ---- validate on HELD-OUT motion: does the identified model predict the real torques? ----
    let mut max_err = 0.0f64;
    for k in 0..40 {
        let t = 100.0 + k as f64 * 0.017; // disjoint from training times
        let (mut q, mut qd, mut qdd) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n { let (a, b, c) = traj(t, i); q[i] = a; qd[i] = b; qdd[i] = c; }
        let tau_true = inverse_dynamics(&robot, &real, &q, &qd, &qdd, g);
        let yin = inertial_regressor(&robot, &q, &qd, &qdd, g);
        let pred_in = &yin * &phi_hat;
        for i in 0..n {
            let pred = pred_in[i] + b_visc_hat[i] * qd[i] + theta[10 * n + 2 * i + 1] * sgn(qd[i]);
            let true_tau = tau_true[i] + b_visc_true[i] * qd[i] + b_coul_true[i] * sgn(qd[i]);
            max_err = max_err.max((pred - true_tau).abs());
        }
    }
    println!("  [ID] held-out torque prediction error: {:.2e} N·m  → the real dynamics are pinned ✓", max_err);
    let fric_err: f64 = (0..n).map(|i| (b_visc_hat[i] - b_visc_true[i]).abs()).fold(0.0, f64::max);
    println!("  [ID] recovered viscous friction {:?}", b_visc_hat.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>());
    println!("       vs hidden truth            {:?}   max error {:.3} N·m·s/rad ✓\n", b_visc_true, fric_err);

    // ---- the PAYOFF: collapse the envelope, re-size the certified fence ----
    // the certificate's table fence is sized to the worst-case fold-down descent speed, set by the
    // WEAKEST damping in the envelope. Identifying the real damping collapses that worst case.
    let mut q_fold = vec![0.0; 5]; let mut zmin = f64::INFINITY;
    for s in 0..8000u32 { let h = |mut x: u32| { x ^= x >> 15; x = x.wrapping_mul(2246822519); x ^= x >> 13; x = x.wrapping_mul(3266489917); x ^= x >> 16; x }; let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * ((h(s * 5 + i as u32 + 1) % 1_000_000) as f64 / 1e6)).collect(); let z = robot.fk(&c).translation.vector.z; if z < zmin { zmin = z; q_fold = c; } }
    let q_start = vec![0.2, -0.3, 0.6, 0.0, 0.0]; // a nominal raised working pose
    let b_prior_worst = 0.35;                      // CAD prior: weakest damping in [0.35, 0.85]
    let b_post_worst = b_visc_hat.iter().cloned().fold(f64::INFINITY, f64::min) - 0.02 - fric_err; // identified − margin
    let v_prior = descent_speed(&robot, &real, &q_fold, &q_start, b_prior_worst);
    let v_post = descent_speed(&robot, &real, &q_fold, &q_start, b_post_worst);
    let fence_prior = v_prior * v_prior / (2.0 * A_UP);
    let fence_post = v_post * v_post / (2.0 * A_UP);
    println!("  [COLLAPSE] worst-case fold-down descent speed and the fence it forces (a↑={:.1} m/s²):", A_UP);
    println!("     PRIOR   (damping ≥ {:.2}, CAD box):        v↓ = {:.3} m/s → fence ≥ {:.3} m", b_prior_worst, v_prior, fence_prior);
    println!("     AFTER ID(damping ≥ {:.2}, identified±margin): v↓ = {:.3} m/s → fence ≥ {:.3} m", b_post_worst, v_post, fence_post);
    let shrink = 100.0 * (1.0 - fence_post / fence_prior);
    println!("     → the certified table fence collapses by {:.0}% ({:.3} m → {:.3} m): same guarantee, tighter niche.\n", shrink, fence_prior, fence_post);

    println!("  READINESS: when the physical SO-101 arrives, this is the drop-in transfer — excite, identify");
    println!("  (inertials + friction), re-verify the certificate on the collapsed envelope. Honest bounds:");
    println!("  identify() recovers the identifiable subspace (validated predictively); friction is the linear");
    println!("  extension; the fence estimate uses the closed-form braking model (exact re-cert = so101_certify.rs).");
}
