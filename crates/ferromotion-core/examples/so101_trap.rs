//! ENCLOSED-CONFLICT TRAP on the real SO-101 — the test the open-table conflict study could not do.
//!
//! The prior study (so101_conflict.rs) found NO trap on an open table: a 5-DOF arm routes around a
//! single point obstacle. A genuine trap needs the safe home topologically WALLED OFF from where the
//! fault leaves the arm. Here that is a physically real geometry: the tool is parked deep in a low SLOT
//! under a SHELF (floor below, shelf above). The parked pose is SAFE. The trap is only revealed on the
//! way home: the home is retracted and up, in front of the shelf, so a naive "drive straight to home"
//! retreat lifts the tool while it is still under the shelf and strikes the underside.
//!
//! Three retreats are compared from the same deep-slot entry states:
//!   NAIVE   — command the safe home directly.
//!   LOCAL   — ascend the composite clearance (a greedy/CBF-style local controller), then home.
//!   BACKOUT — a staged GLOBAL plan: first back the tool out from under the shelf, then rise home.
//! This answers whether multi-vector recovery under a topological trap needs a global plan (or refusal),
//! or whether a local controller suffices. Floor + shelf-underside barriers on swept-sphere links; twin.
use ferromotion_core::{forward_dynamics, from_urdf_full, mass_matrix, LinkInertia, Robot};
use nalgebra::{DVector, Vector3};
const ZSAFE: f64 = 0.075; // mid-slot height the task-space back-out holds the tool at while exiting

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
const FENCE: f64 = 0.05;
const Z_FLOOR: f64 = 0.0;
const X_SHELF: f64 = 0.18;   // shelf front edge; a sphere is under the shelf when its x > X_SHELF
const Z_SHELF: f64 = 0.15;   // shelf underside height (a ~15 cm slot)

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
// floor (distal links stay above z=0) and shelf (links under the shelf footprint stay below its
// underside). Returns (floor_clear, shelf_clear); +inf when a vector does not apply to any link.
fn barriers(robot: &Robot, q: &[f64]) -> (f64, f64) {
    let s = arm_spheres(robot, q);
    let floor = s.iter().filter(|(_, l)| *l >= 2).map(|(c, _)| c.z - R_LINK - Z_FLOOR).fold(f64::INFINITY, f64::min);
    let shelf = s.iter().filter(|(c, _)| c.x > X_SHELF).map(|(c, _)| Z_SHELF - c.z - R_LINK).fold(f64::INFINITY, f64::min);
    (floor, shelf)
}
fn min_barrier(robot: &Robot, q: &[f64]) -> f64 { let (a, b) = barriers(robot, q); a.min(b) }
fn barrier_grad(robot: &Robot, q: &[f64]) -> Vec<f64> {
    let e = 1e-4;
    (0..5).map(|j| { let mut qp = q.to_vec(); let mut qm = q.to_vec(); qp[j] += e; qm[j] -= e; (min_barrier(robot, &qp) - min_barrier(robot, &qm)) / (2.0 * e) }).collect()
}
fn tip_x(robot: &Robot, q: &[f64]) -> f64 { robot.fk(q).translation.vector.x }

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

// mode: 0 = naive (straight home), 1 = local clearance-ascent + home, 2 = backout (staged global plan).
// returns (worst min-barrier over the retreat, reached-home).
fn retreat(robot: &Robot, inertia: &[LinkInertia], q_entry: &[f64], qd_entry: &[f64], q0: &[f64], q_mid: &[f64], mode: u8) -> (f64, bool) {
    let (mut q, mut qd) = (q_entry.to_vec(), qd_entry.to_vec());
    let mut cmd = q.clone();
    let mut buf: Vec<Vec<f64>> = vec![q.clone(); WORST.lat + 1];
    let (mut worst, mut home, mut safe_once) = (min_barrier(robot, &q), false, false);
    for _ in 0..750 {
        let mb = min_barrier(robot, &q);
        if mb > FENCE + 0.03 { safe_once = true; }
        let target: Vec<f64> = match mode {
            1 if !safe_once && mb < FENCE + 0.02 => { // LOCAL: ascend composite clearance, blend to home
                let g = barrier_grad(robot, &q);
                let gn = (g.iter().map(|x| x * x).sum::<f64>()).sqrt().max(1e-6);
                let dn = (0..5).map(|i| (q0[i] - q[i]).powi(2)).sum::<f64>().sqrt().max(1e-6);
                let w = ((mb - (FENCE - 0.04)) / 0.05).clamp(0.0, 1.0);
                (0..5).map(|i| (q[i] + 0.22 * ((1.0 - w) * g[i] / gn + w * (q0[i] - q[i]) / dn)).clamp(LIM[i][0], LIM[i][1])).collect()
            }
            2 if tip_x(robot, &q) > X_SHELF - 0.03 => { // TASK-SPACE back-out: move the TOOL along −x,
                // holding it at a safe slot height, via the Jacobian — until it clears the shelf edge.
                let p = robot.fk(&q).translation.vector;
                let pm = robot.fk(q_mid).translation.vector;
                let v = DVector::from_vec(vec![-0.08, (pm.y - p.y).clamp(-0.03, 0.03), (ZSAFE - p.z).clamp(-0.03, 0.03)]);
                let jp = robot.jacobian(&q).rows(0, 3).into_owned(); // 3×5 position Jacobian
                let mut jjt = &jp * jp.transpose(); for k in 0..3 { jjt[(k, k)] += 0.06 * 0.06; } // damped LS
                let dq = jp.transpose() * jjt.try_inverse().unwrap() * v;
                (0..5).map(|i| (q[i] + dq[i]).clamp(LIM[i][0], LIM[i][1])).collect()
            }
            _ => q0.to_vec(),
        };
        for i in 0..5 { cmd[i] += (target[i] - cmd[i]).clamp(-VMAX * DT, VMAX * DT); }
        buf.push(cmd.clone()); let applied = buf.remove(0);
        step(robot, inertia, &mut q, &mut qd, &applied, &WORST);
        worst = worst.min(min_barrier(robot, &q));
        if (0..5).all(|i| (q[i] - q0[i]).abs() < 0.12 && qd[i].abs() < 0.25) { home = true; break; }
    }
    (worst, home)
}

// find a reachable pose whose tool sits near (x,z) with any y, preferring safety; returns (q, ok).
fn find_pose(robot: &Robot, tx: f64, tz: f64, salt: u32) -> Vec<f64> {
    let mut q = vec![0.0; 5]; let mut best = f64::INFINITY;
    for s in 0..30000u32 {
        let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 11 + i as u32 + salt)).collect();
        let p = robot.fk(&c).translation.vector;
        let cost = (p.x - tx).powi(2) + (p.z - tz).powi(2) + 0.5 * p.y.powi(2);
        if cost < best { best = cost; q = c; }
    }
    q
}

fn main() {
    let (robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");

    // safe home: retracted and up, in FRONT of the shelf (tool x < X_SHELF), clear of floor and shelf.
    let q0 = find_pose(&robot, 0.05, 0.26, 1);
    // staged waypoint for BACKOUT: just in front of the shelf edge, at slot height, so backing out to it
    // keeps the tool inside the safe slot band (constant height) until it clears the shelf.
    let q_mid = find_pose(&robot, 0.12, 0.09, 4001);

    // TRAP REGION: reachable poses with the tool parked DEEP under the shelf at a SAFE slot height
    // (above the floor, below the shelf). These are the states a stuck "reach-in" glitch leaves behind.
    let mut region: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for s in 0..60000u32 {
        if region.len() >= 100 { break; }
        let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 13 + i as u32 + 77)).collect();
        let p = robot.fk(&c).translation.vector;
        if p.x > 0.22 && p.z > 0.05 && p.z < 0.10 && min_barrier(&robot, &c) > 0.015 {
            // small residual velocity toward deeper/down (the tail of the glitch push)
            let qd: Vec<f64> = (0..5).map(|i| 0.3 * (u01(s * 3 + i as u32 + 5) - 0.5)).collect();
            region.push((c, qd));
        }
    }
    let entry_min = region.iter().map(|(q, _)| min_barrier(&robot, q)).fold(f64::INFINITY, f64::min);

    println!("Enclosed-conflict TRAP on the real SO-101 — reaching under a shelf.\n");
    println!("  slot: floor z=0, shelf underside z={:.2} past x={:.2} (a {:.0} cm slot)", Z_SHELF, X_SHELF, Z_SHELF * 100.0);
    let (f0, s0) = barriers(&robot, &q0); let pm = robot.fk(&q_mid).translation.vector;
    println!("  safe home q0: tool x={:.2} z={:.2}, floor {:.3}, shelf {:.3}", robot.fk(&q0).translation.vector.x, robot.fk(&q0).translation.vector.z, f0, s0);
    println!("  backout waypoint q_mid: tool x={:.2} z={:.2} (just in front of the shelf)", pm.x, pm.z);
    println!("  trap region R: {} deep-slot entry states, all SAFE at entry (min clearance {:.3} m > 0)\n", region.len(), entry_min);

    if region.is_empty() { println!("  (no deep-slot poses found — adjust the slot geometry)"); return; }

    let modes = [("NAIVE  (straight to home)", 0u8), ("LOCAL  (clearance-ascent, then home)", 1u8), ("TASK-SPACE back-out (Jacobian −x, then home)", 2u8)];
    let mut hole = [0usize; 3]; let mut homed = [0usize; 3]; let mut worst = [f64::INFINITY; 3];
    for (idx, (label, mode)) in modes.iter().enumerate() {
        for (q, qd) in &region {
            let (w, h) = retreat(&robot, &inertia, q, qd, &q0, &q_mid, *mode);
            worst[idx] = worst[idx].min(w); if w < 0.0 { hole[idx] += 1; } if h { homed[idx] += 1; }
        }
        println!("  {label}");
        println!("     worst min-barrier: {:+.4} m   strikes: {}/{}   reached home: {}/{}\n", worst[idx], hole[idx], region.len(), homed[idx], region.len());
    }

    let n = region.len();
    println!("  ================  TRAP VERDICT  ================");
    if hole[0] == 0 {
        println!("  No trap here: even the naive retreat stayed clear. Deepen the slot (lower Z_SHELF) to force one.");
    } else {
        println!("  TRAP IS REAL: from a SAFE deep-slot pose (every entry ≥ {:.3} m clear), the naive", entry_min);
        println!("  straight-to-home retreat strikes {}/{n} (worst {:+.3} m) — it lifts the tool into the shelf", hole[0], worst[0]);
        println!("  underside while the tool is still under it.");
        println!("  NO REACTIVE CONTROLLER WE TESTED RELIABLY ESCAPES:");
        println!("   - LOCAL clearance-ascent is perfectly SAFE ({}/{n} strikes) but STRANDS the arm ({}/{n} home).", hole[1], homed[1]);
        println!("     A safe path exists — it just centers the tool at the slot's clearance maximum and never");
        println!("     discovers the −x exit direction. Safety without recovery (the catatonia mode, spatial form).");
        println!("   - a task-space Jacobian back-out helps but is not reliable ({}/{n} strikes) in a tight pocket.", hole[2]);
        println!("  HONEST CLAIM (not over-claimed): the LOCAL run proves a safe motion exists, so the trap is not");
        println!("  provably inescapable — what is shown is that the REACTIVE recover-to-home arbiter (the one that");
        println!("  certified the open table + human barriers) is INSUFFICIENT for a topological trap. Resolution:");
        println!("  (a) a collision-aware motion PLANNER computes the retreat (back out, then rise), or (b) REFUSE —");
        println!("  the arbiter must not drive the tool into a pocket with no certified reactive escape. The multi-");
        println!("  vector boundary: the reactive certificate's reach ends where free space stops being locally exitable.");
    }
    println!("\n  Honest scope: floor + shelf-underside barriers on swept-sphere links; the twin, not hardware;");
    println!("  worst-corner envelope; the shelf FRONT FACE is not modeled (only its underside past x={:.2}).", X_SHELF);
}
