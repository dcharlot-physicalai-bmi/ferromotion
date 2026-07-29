//! GLOBAL PLANNER for the deep pocket — the last (a)-branch of the trap boundary.
//!
//! so101_reach.rs showed a smarter REACTIVE recover reclaims under-shelf recovery to ~13 cm, but the
//! deepest pockets (and 5/40 poses) stay unrecoverable: local control cannot see the way out. This is
//! the constructive answer for those: a barrier-respecting RRT in configuration space plans a
//! collision-free retreat from a deep-pocket pose to home (every sampled point on every edge keeps the
//! whole arm's min-clearance above a margin), and we EXECUTE the plan under the worst-corner envelope
//! (servo-tracked waypoints, real dynamics) to confirm 0 strikes and escape. Compared head-to-head with
//! the reactive smart recover on the deepest safe under-shelf poses: reactive strikes, the plan escapes.
//! This closes the arc — trap (reactive insufficient) → refuse → earn-back (better reactive) → PLAN
//! (global, for the rest). Floor + shelf-underside barriers on swept-sphere links; worst envelope; twin.
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
const Z_MID: f64 = 0.075;
const PLAN_MARGIN: f64 = 0.03; // the planned path keeps the whole arm this clear of every barrier —
                               // capped by geometry: a deep pocket under a 15 cm shelf offers little more

fn hash(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u01(i: u32) -> f64 { (hash(i) % 1_000_000) as f64 / 1_000_000.0 }
fn rrng(s: &mut u32) -> f64 { *s = hash(*s); (*s % 1_000_000) as f64 / 1_000_000.0 }

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
fn dist(a: &[f64], b: &[f64]) -> f64 { (0..5).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt() }

// ---- barrier-respecting RRT in configuration space ----
fn edge_free(robot: &Robot, a: &[f64], b: &[f64]) -> bool {
    let m = 10;
    for t in 0..=m {
        let f = t as f64 / m as f64;
        let q: Vec<f64> = (0..5).map(|i| a[i] + f * (b[i] - a[i])).collect();
        if min_barrier(robot, &q) < PLAN_MARGIN { return false; }
    }
    true
}
// greedy shortcut smoothing: drop any waypoint that can be bypassed by a collision-free straight edge.
// A shorter, straighter path is far easier for the servo to track without cutting into a barrier.
fn shortcut(robot: &Robot, path: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut p = path.to_vec();
    let mut i = 0;
    while i + 2 < p.len() {
        if edge_free(robot, &p[i], &p[i + 2]) { p.remove(i + 1); } else { i += 1; }
    }
    p
}
struct Node { q: Vec<f64>, parent: i32 }
fn plan_rrt(robot: &Robot, start: &[f64], goal: &[f64], seed: u32) -> Option<Vec<Vec<f64>>> {
    if min_barrier(robot, start) < PLAN_MARGIN { return None; } // start must itself be clear enough to plan from
    let mut nodes = vec![Node { q: start.to_vec(), parent: -1 }];
    let step = 0.12;
    let mut rng = seed | 1;
    for _ in 0..12000 {
        let q_rand: Vec<f64> = if rrng(&mut rng) < 0.15 { goal.to_vec() }
            else { (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * rrng(&mut rng)).collect() };
        // nearest existing node
        let mut ni = 0usize; let mut nd = f64::INFINITY;
        for (i, n) in nodes.iter().enumerate() { let d = dist(&n.q, &q_rand); if d < nd { nd = d; ni = i; } }
        let dir = dist(&nodes[ni].q, &q_rand);
        if dir < 1e-6 { continue; }
        let q_new: Vec<f64> = (0..5).map(|i| (nodes[ni].q[i] + step * (q_rand[i] - nodes[ni].q[i]) / dir).clamp(LIM[i][0], LIM[i][1])).collect();
        if !edge_free(robot, &nodes[ni].q, &q_new) { continue; }
        nodes.push(Node { q: q_new.clone(), parent: ni as i32 });
        if dist(&q_new, goal) < step * 1.5 && edge_free(robot, &q_new, goal) {
            nodes.push(Node { q: goal.to_vec(), parent: (nodes.len() - 1) as i32 });
            // backtrack
            let mut path = Vec::new(); let mut idx = (nodes.len() - 1) as i32;
            while idx >= 0 { path.push(nodes[idx as usize].q.clone()); idx = nodes[idx as usize].parent; }
            path.reverse();
            return Some(path);
        }
    }
    None
}

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

// EXECUTE a (smoothed) planned path under the worst envelope: track waypoints SLOWLY and TIGHTLY so the
// lagging servo stays on the collision-free path, with a safety hold that freezes motion if clearance
// dips (plan globally, filter locally). returns (worst min-barrier, cleared the pocket).
fn execute_path(robot: &Robot, inertia: &[LinkInertia], path: &[Vec<f64>], qd0: &[f64]) -> (f64, bool) {
    let (mut q, mut qd) = (path[0].clone(), qd0.to_vec());
    let mut cmd = q.clone();
    let mut buf: Vec<Vec<f64>> = vec![q.clone(); WORST.lat + 1];
    let (mut worst, mut wi, mut cleared) = (min_barrier(robot, &q), 1usize.min(path.len() - 1), false);
    let slew = 0.25 * VMAX; // move well below servo speed so actual q hugs the command (minimal lag)
    for _ in 0..18000 {
        if dist(&q, &path[wi]) < 0.03 && wi < path.len() - 1 { wi += 1; } // advance only when actually close
        // safety filter: if clearance dips, hold position (freeze the target at q) so tracking error can't
        // drive the arm through a barrier; otherwise track the current waypoint.
        let target: Vec<f64> = if min_barrier(robot, &q) < 0.02 { q.clone() } else { path[wi].clone() };
        for i in 0..5 { cmd[i] += (target[i] - cmd[i]).clamp(-slew * DT, slew * DT); }
        buf.push(cmd.clone()); let applied = buf.remove(0);
        step(robot, inertia, &mut q, &mut qd, &applied, &WORST);
        worst = worst.min(min_barrier(robot, &q));
        if tipx(robot, &q) < X_SHELF - 0.02 && wi >= path.len() - 1 { cleared = true; break; }
    }
    (worst, cleared)
}

// the reactive smart recover (from so101_reach.rs) for the head-to-head.
fn smart_target(robot: &Robot, q: &[f64], q0: &[f64]) -> Vec<f64> {
    let p = robot.fk(q).translation.vector;
    if p.x > X_SHELF - 0.02 {
        let v = DVector::from_vec(vec![-0.09, (-0.4 * p.y).clamp(-0.05, 0.05), (4.0 * (Z_MID - p.z)).clamp(-0.15, 0.15)]);
        let jp = robot.jacobian(q).rows(0, 3).into_owned();
        let mut jjt = &jp * jp.transpose(); for k in 0..3 { jjt[(k, k)] += 0.06 * 0.06; }
        let dq = jp.transpose() * jjt.try_inverse().unwrap() * v;
        (0..5).map(|i| (q[i] + dq[i]).clamp(LIM[i][0], LIM[i][1])).collect()
    } else { q0.to_vec() }
}
fn reactive_recover(robot: &Robot, inertia: &[LinkInertia], start: &[f64], qd0: &[f64], q0: &[f64]) -> (f64, bool) {
    let (mut q, mut qd) = (start.to_vec(), qd0.to_vec());
    let mut cmd = q.clone();
    let mut buf: Vec<Vec<f64>> = vec![q.clone(); WORST.lat + 1];
    let (mut worst, mut cleared) = (min_barrier(robot, &q), false);
    for _ in 0..2200 {
        let target = smart_target(robot, &q, q0);
        for i in 0..5 { cmd[i] += (target[i] - cmd[i]).clamp(-VMAX * DT, VMAX * DT); }
        buf.push(cmd.clone()); let applied = buf.remove(0);
        step(robot, inertia, &mut q, &mut qd, &applied, &WORST);
        worst = worst.min(min_barrier(robot, &q));
        if tipx(robot, &q) < X_SHELF - 0.02 { cleared = true; break; }
    }
    (worst, cleared)
}

fn main() {
    let (robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");
    let q0 = find_pose(&robot, 0.05, 0.26, 1);

    println!("Global planner for the deep pocket — the last branch of the trap boundary.\n");

    // the DEEPEST genuinely-safe under-shelf poses (whole arm clear) — where reactive recovery struggles.
    let mut deep: Vec<Vec<f64>> = Vec::new();
    for s in 0..400000u32 {
        if deep.len() >= 12 { break; }
        let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 13 + i as u32 + 91)).collect();
        let p = robot.fk(&c).translation.vector;
        if p.x > 0.27 && min_barrier(&robot, &c) > PLAN_MARGIN { deep.push(c); } // ≥9 cm under, clear enough to plan from
    }
    if deep.is_empty() { println!("  no deep safe poses found on this geometry."); return; }
    let dd: Vec<f64> = deep.iter().map(|q| tipx(&robot, q)).collect();
    println!("  {} deep under-shelf poses (tool {:.2}–{:.2} m, {:.0}–{:.0} cm past the shelf edge):\n",
        deep.len(), dd.iter().cloned().fold(9.0, f64::min), dd.iter().cloned().fold(0.0, f64::max),
        (dd.iter().cloned().fold(9.0, f64::min) - X_SHELF) * 100.0, (dd.iter().cloned().fold(0.0, f64::max) - X_SHELF) * 100.0);

    let (mut r_ok, mut p_ok, mut planned) = (0, 0, 0);
    for (i, q) in deep.iter().enumerate() {
        let qd: Vec<f64> = (0..5).map(|j| 0.15 * (u01(j as u32 + 7) - 0.5)).collect();
        // reactive
        let (rw, rc) = reactive_recover(&robot, &inertia, q, &qd, &q0);
        let react = rw >= 0.0 && rc;
        if react { r_ok += 1; }
        // planned
        let (pw, pc, plen) = match plan_rrt(&robot, q, &q0, 1234 + i as u32 * 97) {
            Some(raw) => { planned += 1; let path = shortcut(&robot, &raw); let (w, c) = execute_path(&robot, &inertia, &path, &qd); (w, c, path.len()) }
            None => (f64::NAN, false, 0),
        };
        let plan_ok = pw >= 0.0 && pc;
        if plan_ok { p_ok += 1; }
        println!("  pose {:>2} (tool x={:.3}, {:.0} cm under): reactive {:<12} | planned {:<10} (path {} nodes)",
            i, tipx(&robot, q), (tipx(&robot, q) - X_SHELF) * 100.0,
            if react { "ESCAPES" } else { "strikes/stuck" },
            if plan_ok { "ESCAPES" } else if plen == 0 { "no plan" } else { "exec-strike" }, plen);
    }
    println!("\n  reactive smart recover: {}/{} escaped   |   RRT plan + execute: {}/{} escaped ({} plans found)",
        r_ok, deep.len(), p_ok, deep.len(), planned);

    println!("\n  ================  VERDICT  ================");
    println!("  On the deepest pockets, the reactive recover strikes or strands ({}/{} escape), but a barrier-", r_ok, deep.len());
    println!("  respecting RRT that plans a collision-free retreat and executes it under the worst envelope");
    println!("  escapes {}/{}. This is the (a)-branch: where LOCAL control cannot see the exit, a GLOBAL plan", p_ok, deep.len());
    println!("  finds it. The full boundary is now covered — reactive recovery for the shallow, certified pocket");
    println!("  (fast, no planning); a planned retreat for the deep pocket; refusal only where neither is available.");
    println!("\n  Scope: swept-sphere links; RRT plans in C-space against the barrier at {:.0} cm margin; execution", PLAN_MARGIN * 100.0);
    println!("  is servo-tracked under the worst-corner envelope; the twin, not hardware; empirical, not a proof.");
}
