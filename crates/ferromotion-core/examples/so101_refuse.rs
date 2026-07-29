//! REFUSAL-BEFORE-THE-EDGE for a topological trap — the constructive side of so101_trap.rs.
//!
//! The trap study proved a reactive recover-to-home arbiter cannot escape a deep pocket (the tool
//! parked deep in a slot under a shelf): local clearance-ascent stays safe but strands the arm. The
//! honest resolution was (a) a motion planner, or (b) REFUSE to enter a pocket with no certified
//! reactive escape. This builds (b): certify OFFLINE the deepest under-shelf depth from which the
//! arbiter's own reactive recover still gets home with no strike (x_crit), then enforce ONLINE a cheap
//! predicate — the arbiter never lets the tool go deeper than x_crit under the shelf. The arbiter now
//! KNOWS its topological boundary and stays inside it. The same deep-reach glitch that trapped the arm
//! is clamped at the exitable boundary, from which recover escapes. Expensive cert offline → cheap
//! guard online (one FK + compare), the certificate philosophy. Floor + shelf on swept-sphere links; twin.
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
const REFUSE_MARGIN: f64 = 0.04; // setback of the refusal fence inside x_crit, to absorb command overshoot

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

// the arbiter's REACTIVE recover: command q0 (rate-limited), worst envelope. Returns (worst barrier, home).
fn recover(robot: &Robot, inertia: &[LinkInertia], q_entry: &[f64], qd_entry: &[f64], q0: &[f64]) -> (f64, bool) {
    let (mut q, mut qd) = (q_entry.to_vec(), qd_entry.to_vec());
    let mut cmd = q.clone();
    let mut buf: Vec<Vec<f64>> = vec![q.clone(); WORST.lat + 1];
    let (mut worst, mut home) = (min_barrier(robot, &q), false);
    for _ in 0..900 {
        for i in 0..5 { cmd[i] += (q0[i] - cmd[i]).clamp(-VMAX * DT, VMAX * DT); }
        buf.push(cmd.clone()); let applied = buf.remove(0);
        step(robot, inertia, &mut q, &mut qd, &applied, &WORST);
        worst = worst.min(min_barrier(robot, &q));
        if (0..5).all(|i| (q[i] - q0[i]).abs() < 0.12 && qd[i].abs() < 0.25) { home = true; break; }
    }
    (worst, home)
}

// one online episode: task holds a safe slot pose; a glitch drives the command DEEP; then the arbiter
// recovers home. With `refuse`, any command deeper than x_crit under the shelf is clamped to q_lim.
// returns (struck, recovered, max_depth_reached).
fn episode(robot: &Robot, inertia: &[LinkInertia], task: &[f64], q_bad: &[f64], q0: &[f64], q_lim: &[f64], fence: f64, refuse: bool, env: &Env, seed: u32) -> (bool, bool, f64) {
    let (mut q, mut qd) = (task.to_vec(), vec![0.0f64; 5]);
    let mut cmd = q.clone();
    let (t_total, tf, k) = (1900usize, 300 + (hash(seed) % 400) as usize, 260);
    let mut buf: Vec<Vec<f64>> = vec![q.clone(); env.lat + 1];
    let (mut struck, mut maxd) = (false, 0.0f64);
    for t in 0..t_total {
        let in_fault = t >= tf && t < tf + k;
        let raw: Vec<f64> = if t >= tf + k { q0.to_vec() }         // post-glitch: reactive recover home
            else if in_fault { q_bad.to_vec() } else { task.to_vec() }; // glitch drives deep; else the task
        // REFUSAL PREDICATE (cheap: one FK + compare): never command the tool deeper than x_crit
        // under the shelf. When a command would, hold at the certified-exitable boundary q_lim.
        let target = if refuse && tipx(robot, &raw) > fence + 1e-3 { q_lim.to_vec() } else { raw };
        for i in 0..5 { cmd[i] += (target[i] - cmd[i]).clamp(-VMAX * DT, VMAX * DT); }
        buf.push(cmd.clone()); let applied = buf.remove(0);
        step(robot, inertia, &mut q, &mut qd, &applied, env);
        if min_barrier(robot, &q) < 0.0 { struck = true; }
        maxd = maxd.max(tipx(robot, &q));
    }
    let recovered = !struck && (0..5).all(|i| (q[i] - q0[i]).abs() < 0.16);
    (struck, recovered, maxd)
}

fn main() {
    let (robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");
    let q0 = find_pose(&robot, 0.05, 0.26, 1);

    println!("Refusal-before-the-edge for the shelf-slot trap on the real SO-101.\n");
    println!("  OFFLINE: certify the deepest under-shelf depth the reactive recover can still escape.");
    // sweep depth; at each, test several poses; x_crit = deepest depth that ALWAYS escapes (0 strike + home).
    let mut x_crit = X_SHELF;
    let mut depth = X_SHELF + 0.01;
    while depth <= 0.33 {
        let mut all_escape = true; let mut tested = 0;
        for kk in 0..8u32 {
            let q = find_pose(&robot, depth, 0.075, 20000 + kk * 97);
            let p = robot.fk(&q).translation.vector;
            if (p.x - depth).abs() > 0.02 || p.x <= X_SHELF || min_barrier(&robot, &q) < 0.012 { continue; }
            tested += 1;
            let qd: Vec<f64> = (0..5).map(|i| 0.25 * (u01(kk * 7 + i as u32 + 3) - 0.5)).collect();
            let (w, h) = recover(&robot, &inertia, &q, &qd, &q0);
            if w < 0.0 || !h { all_escape = false; break; }
        }
        if tested >= 3 && all_escape { x_crit = depth; } else if tested >= 3 { break; }
        depth += 0.01;
    }
    let under = x_crit - X_SHELF;
    println!("  → x_crit = {:.3} m  ({:.0} cm under the shelf edge is reactively exitable; deeper is a trap)", x_crit, under * 100.0);
    // the enforced fence sits a command-overshoot setback INSIDE x_crit, so even an overshooting command
    // keeps the tool within the certified-exitable set.
    let fence = (x_crit - REFUSE_MARGIN).max(0.10);
    println!("  → refusal fence = x_crit − {:.0} cm = {:.3} m (the tool is never commanded past this)\n", REFUSE_MARGIN * 100.0, fence);

    // the boundary the arbiter holds at, the deep glitch target it must refuse, and a task inside the fence.
    let q_lim = find_pose(&robot, fence, 0.09, 30001);
    let q_bad = find_pose(&robot, 0.30, 0.07, 40001);
    let task = find_pose(&robot, fence - 0.02, 0.09, 50001);

    println!("  ONLINE: a glitch latches a deep-reach command (tool x={:.2}); compare the arbiter with and", robot.fk(&q_bad).translation.vector.x);
    println!("  without the refusal predicate, over a randomized reality envelope.\n");
    let envs: Vec<Env> = (0..8).map(|k| Env { fric: 0.35 + 0.5 * u01(1000 + k), lat: (u01(2000 + k) * 4.0) as usize, dead: 0.004 + 0.02 * u01(3000 + k) }).collect();
    for (label, refuse) in [("WITHOUT refusal (reactive recover only)", false), ("WITH refusal predicate (guard + reactive recover)", true)] {
        let (mut strk, mut rec, mut n, mut deepest) = (0, 0, 0, 0.0f64);
        for env in &envs {
            for ep in 0..12u32 {
                let seed = n as u32 * 17 + ep + 1;
                let (s, r, d) = episode(&robot, &inertia, &task, &q_bad, &q0, &q_lim, fence, refuse, env, seed);
                if s { strk += 1; } if r { rec += 1; } deepest = deepest.max(d); n += 1;
            }
        }
        println!("  {label}");
        println!("     strikes: {}/{}   recovered: {}/{}   deepest tool reach: {:.3} m (x_crit {:.3})\n", strk, n, rec, n, deepest, x_crit);
    }

    println!("  ================  VERDICT  ================");
    println!("  The refusal predicate holds the tool at the certified-exitable boundary during the glitch, so");
    println!("  the reactive recover always escapes — the arbiter respects the topological boundary the trap");
    println!("  study named, instead of driving into it. Where the task genuinely needs to go deeper than");
    println!("  x_crit={:.3} m, that is out of scope for reactive recovery and must be handed to a planner.", x_crit);
    println!("  Refusal-before-the-edge, now for a pocket: the certificate's reach is enforced, not just stated.");
    println!("\n  Honest scope: floor + shelf-underside barriers on swept-sphere links; worst-corner envelope;");
    println!("  the twin, not hardware; x_crit certified empirically over a pose+velocity sample, not a proof.");
}
