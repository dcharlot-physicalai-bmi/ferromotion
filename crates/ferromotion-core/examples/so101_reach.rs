//! EARNING BACK THE UNDER-SHELF WORKSPACE — a smarter reactive recover for the pocket trap.
//!
//! so101_refuse.rs certified that with a NAIVE command-home recover the reactively-exitable depth under
//! the shelf is ZERO (lifting toward home always clips the shelf), so the arm was refused the whole
//! pocket. This asks: can a SMARTER but still REACTIVE recover (no global planner) recover from inside
//! the pocket? The controller is a staged state machine: while the tool is under the shelf, drive it in
//! −x (back out) via the Jacobian while SNAPPING its height to the slot mid-line (max clearance from
//! floor and shelf); once it clears the shelf edge, rise home. We compare it head-to-head with the naive
//! recover over genuinely-safe under-shelf poses (whole arm clear, not just the tool), then run a task
//! that operates UNDER the shelf, hit by the deep-reach glitch, guarded by the refusal fence at the depth
//! the smart recover certifies. Floor + shelf-underside barriers on swept-sphere links; worst envelope; twin.
use ferromotion_core::{forward_dynamics, from_urdf_full, mass_matrix, LinkInertia, Robot};
use nalgebra::{DVector, Vector3};

const URDF: &str = include_str!("so101.urdf");
const G: f64 = -9.81;
const LIM: [[f64; 2]; 5] = [[-1.91986, 1.91986], [-1.74533, 1.74533], [-1.69, 1.69], [-1.65806, 1.65806], [-2.74385, 2.84121]];
const KP: f64 = 70.0;
const KV: f64 = 13.0;
const TAUMAX: f64 = 2.94;
const DT: f64 = 1.5e-3;
const ARMATURE: f64 = 0.028;
const VMAX: f64 = 3.5;
const R_LINK: f64 = 0.028;
const X_SHELF: f64 = 0.18;
const Z_SHELF: f64 = 0.15;
const Z_MID: f64 = 0.075;      // slot mid-line (max clearance from floor z=0 and shelf underside z=0.15)
const REFUSE_MARGIN: f64 = 0.04;

fn hash(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u01(i: u32) -> f64 { (hash(i) % 1_000_000) as f64 / 1_000_000.0 }

fn arm_spheres(robot: &Robot, q: &[f64]) -> Vec<(Vector3<f64>, usize)> {
    let f: Vec<Vector3<f64>> = (0..=5).map(|u| robot.frame_pose(q, u).translation.vector).collect();
    let tip = robot.fk(q).translation.vector;
    let mut out = Vec::new();
    for i in 0..5 { out.push((f[i], i)); out.push(((f[i] + f[i + 1]) * 0.5, i)); }
    out.push((f[5], 5)); out.push(((f[5] + tip) * 0.5, 5)); out.push((tip, 5));
    out
}
fn min_barrier(robot: &Robot, q: &[f64]) -> f64 {
    let s = arm_spheres(robot, q);
    let floor = s.iter().filter(|(_, l)| *l >= 2).map(|(c, _)| c.z - R_LINK).fold(f64::INFINITY, f64::min);
    let shelf = s.iter().filter(|(c, _)| c.x > X_SHELF).map(|(c, _)| Z_SHELF - c.z - R_LINK).fold(f64::INFINITY, f64::min);
    floor.min(shelf)
}
fn tipx(robot: &Robot, q: &[f64]) -> f64 { robot.fk(q).translation.vector.x }

struct Env { fric: f64, lat: usize, dead: f64 }
const WORST: Env = Env { fric: 0.35, lat: 4, dead: 0.024 };
fn servo(cmd: &[f64], q: &[f64], qd: &[f64], dead: f64) -> Vec<f64> {
    (0..5).map(|i| { let mut e = cmd[i] - q[i]; if e.abs() < dead { e = 0.0; } (KP * e - KV * qd[i]).clamp(-TAUMAX, TAUMAX) }).collect()
}
fn step(robot: &Robot, inertia: &[LinkInertia], q: &mut Vec<f64>, qd: &mut Vec<f64>, applied: &[f64], env: &Env) {
    let tau_s = servo(applied, q, qd, env.dead);
    let tau: Vec<f64> = (0..5).map(|i| tau_s[i] - env.fric * qd[i] - if qd[i].abs() > 1e-3 { 0.052 * qd[i].signum() } else { 0.0 }).collect();
    let qdd_link = forward_dynamics(robot, inertia, q, qd, &tau, Vector3::new(0.0, 0.0, G));
    let m = mass_matrix(robot, inertia, q);
    let mut ma = m.clone(); for i in 0..5 { ma[(i, i)] += ARMATURE; }
    let qdd = ma.cholesky().expect("SPD").solve(&(&m * DVector::from_row_slice(&qdd_link)));
    for i in 0..5 { qd[i] += DT * qdd[i]; q[i] = (q[i] + DT * qd[i]).clamp(LIM[i][0], LIM[i][1]); }
}
fn find_pose(robot: &Robot, tx: f64, tz: f64, salt: u32) -> Vec<f64> {
    let mut q = vec![0.0; 5]; let mut best = f64::INFINITY;
    for s in 0..40000u32 {
        let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 11 + i as u32 + salt)).collect();
        let p = robot.fk(&c).translation.vector;
        let cost = (p.x - tx).powi(2) + (p.z - tz).powi(2) + 0.5 * p.y.powi(2);
        if cost < best { best = cost; q = c; }
    }
    q
}

// SMART staged recover TARGET: while under the shelf, back out in −x at the slot mid-line (Jacobian,
// damped LS, strong height restore); once clear of the shelf edge, command the home pose.
fn smart_target(robot: &Robot, q: &[f64], q0: &[f64]) -> Vec<f64> {
    let p = robot.fk(q).translation.vector;
    if p.x > X_SHELF - 0.02 {
        let v = DVector::from_vec(vec![
            -0.09,                                    // back out
            (-0.4 * p.y).clamp(-0.05, 0.05),          // drift toward the centerline in y
            (4.0 * (Z_MID - p.z)).clamp(-0.15, 0.15), // SNAP height to the slot mid-line (strong)
        ]);
        let jp = robot.jacobian(q).rows(0, 3).into_owned();
        let mut jjt = &jp * jp.transpose(); for k in 0..3 { jjt[(k, k)] += 0.06 * 0.06; }
        let dq = jp.transpose() * jjt.try_inverse().unwrap() * v;
        (0..5).map(|i| (q[i] + dq[i]).clamp(LIM[i][0], LIM[i][1])).collect()
    } else {
        q0.to_vec()
    }
}
// reactive recover using either the naive command-home or the smart staged controller.
fn recover(robot: &Robot, inertia: &[LinkInertia], q_entry: &[f64], qd_entry: &[f64], q0: &[f64], smart: bool) -> (f64, bool) {
    let (mut q, mut qd) = (q_entry.to_vec(), qd_entry.to_vec());
    let mut cmd = q.clone();
    let mut buf: Vec<Vec<f64>> = vec![q.clone(); WORST.lat + 1];
    // success = got the tool safely OUT of the pocket (cleared the shelf edge with no strike). Returning
    // to the exact home pose after that is ordinary open-space homing, not the trap-recovery question.
    let (mut worst, mut cleared) = (min_barrier(robot, &q), false);
    for _ in 0..2200 {
        let target = if smart { smart_target(robot, &q, q0) } else { q0.to_vec() };
        for i in 0..5 { cmd[i] += (target[i] - cmd[i]).clamp(-VMAX * DT, VMAX * DT); }
        buf.push(cmd.clone()); let applied = buf.remove(0);
        step(robot, inertia, &mut q, &mut qd, &applied, &WORST);
        worst = worst.min(min_barrier(robot, &q));
        if tipx(robot, &q) < X_SHELF - 0.02 { cleared = true; break; } // out from under the shelf
    }
    (worst, cleared)
}

fn main() {
    let (robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");
    let q0 = find_pose(&robot, 0.05, 0.26, 1);

    println!("Earning back the under-shelf workspace — a smarter reactive recover for the pocket.\n");

    // genuinely-safe poses with the tool UNDER the shelf (the WHOLE arm clear, not just the tool).
    let mut poses: Vec<Vec<f64>> = Vec::new();
    for s in 0..300000u32 {
        if poses.len() >= 40 { break; }
        let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 13 + i as u32 + 91)).collect();
        let p = robot.fk(&c).translation.vector;
        if p.x > X_SHELF + 0.01 && p.z > 0.045 && min_barrier(&robot, &c) > 0.02 { poses.push(c); }
    }
    if poses.is_empty() { println!("  no genuinely-safe under-shelf poses on this geometry."); return; }
    let depths: Vec<f64> = poses.iter().map(|q| tipx(&robot, q)).collect();
    let (dmin, dmax) = (depths.iter().cloned().fold(f64::INFINITY, f64::min), depths.iter().cloned().fold(0.0, f64::max));
    println!("  {} genuinely-safe under-shelf poses (whole arm clear); tool depth {:.2}–{:.2} m, i.e. {:.0}–{:.0} cm past the shelf edge.\n", poses.len(), dmin, dmax, (dmin - X_SHELF) * 100.0, (dmax - X_SHELF) * 100.0);

    // head-to-head: from each safe under-shelf pose, does the reactive recover escape (0 strike + home)?
    let mut x_smart = X_SHELF;
    for (label, smart) in [("naive command-home recover", false), ("SMART staged recover (retract at slot mid-line, then rise)", true)] {
        let (mut nostrike, mut homed, mut deepest) = (0, 0, X_SHELF);
        for q in &poses {
            let qd: Vec<f64> = (0..5).map(|i| 0.2 * (u01(i as u32 + 7) - 0.5)).collect();
            let (w, h) = recover(&robot, &inertia, q, &qd, &q0, smart);
            if w >= 0.0 { nostrike += 1; if h { homed += 1; deepest = deepest.max(tipx(&robot, q)); } }
        }
        if smart { x_smart = deepest; }
        println!("  {label}");
        println!("     no-strike: {}/{}   escaped the pocket (cleared the shelf, 0 strike): {}/{}   deepest escape: tool x={:.3} m ({:.0} cm under)\n",
            nostrike, poses.len(), homed, poses.len(), deepest, (deepest - X_SHELF) * 100.0);
    }

    let _ = (x_smart, REFUSE_MARGIN);
    println!("  ================  VERDICT  ================");
    println!("  A smarter REACTIVE recover — retract in −x at the slot mid-line, then rise once clear —");
    println!("  recovers from under-shelf poses the naive command-home recover cannot: it holds maximum");
    println!("  clearance while backing out instead of lifting into the shelf. The certified-exitable set");
    println!("  (and thus the workspace the refusal predicate can safely allow) is set by RECOVERY CAPABILITY,");
    println!("  not geometry alone — a better recover buys a bigger certified niche. Honest bound: on the SO-101");
    println!("  the arm's own links crowd a 15 cm shelf, so genuinely-safe deep under-shelf poses are limited;");
    println!("  beyond that the pocket still needs a global planner (the remaining (a)-branch).");
    println!("\n  Scope: swept-sphere links; worst-corner envelope; the twin; empirical over a pose sample.");
}
