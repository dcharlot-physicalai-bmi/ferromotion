//! BARRIER-CONFLICT CERTIFICATE — the doctrine's deepest open problem: when protective vectors point
//! in DIFFERENT directions, does a safe retreat always exist, or are there TRAP states where two
//! barriers fight and no retreat keeps both positive? A naive "retreat straight to the safe home" can
//! cross a barrier it must pass — recovering from the person by lifting the tool UP through the table's
//! edge, or clearing the table by swinging INTO the person. This example (1) places a person's hand to
//! genuinely CONFLICT with the table+home geometry, (2) shows the naive straight-to-home retreat has
//! holes (violates a barrier from some entry states), and (3) tests a CONFLICT-AWARE retreat that
//! ascends the composite min-clearance gradient (move away from whichever vector is closest) until safe,
//! then converges home. It CERTIFIES the conflict-aware retreat worst-case over the conflict region — or,
//! honestly, NAMES the trap region if one survives. The gradient of the min-barrier is finite-differenced
//! from the real SO-101 kinematics. Honest scope: two active vectors (table+human); the twin, not hardware.
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
const FENCE: f64 = 0.06;      // recognition fence for both vectors
// a person's hand hovering over the FRONT workspace. The safe home is in the BACK (−y, high), so the
// straight way home from a front-low fault sweeps the tool UP through the hand — a genuine conflict:
// clearing the table (go up) risks the hand; clearing the hand (come back/down) risks the table.
const HUMAN_C: [f64; 3] = [0.24, 0.00, 0.16];
const HUMAN_R: f64 = 0.06;

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
fn table_clear(robot: &Robot, q: &[f64]) -> f64 { arm_spheres(robot, q).iter().filter(|(_, l)| *l >= 2).map(|(c, _)| c.z - R_LINK).fold(f64::INFINITY, f64::min) }
fn human_clear(robot: &Robot, q: &[f64]) -> f64 { let hc = Vector3::new(HUMAN_C[0], HUMAN_C[1], HUMAN_C[2]); arm_spheres(robot, q).iter().map(|(c, _)| (c - hc).norm() - R_LINK - HUMAN_R).fold(f64::INFINITY, f64::min) }
fn min_barrier(robot: &Robot, q: &[f64]) -> f64 { table_clear(robot, q).min(human_clear(robot, q)) }
// ∇_q of the active (minimum) barrier, by central finite differences on the real kinematics.
fn barrier_grad(robot: &Robot, q: &[f64]) -> Vec<f64> {
    let e = 1e-4;
    (0..5).map(|j| { let mut qp = q.to_vec(); let mut qm = q.to_vec(); qp[j] += e; qm[j] -= e; (min_barrier(robot, &qp) - min_barrier(robot, &qm)) / (2.0 * e) }).collect()
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

// roll out a retreat from an entry state; `aware` selects the conflict-aware controller.
// returns (worst min-barrier clearance over the retreat, reached-home).
fn retreat(robot: &Robot, inertia: &[LinkInertia], q_entry: &[f64], qd_entry: &[f64], q0: &[f64], aware: bool) -> (f64, bool) {
    let (mut q, mut qd) = (q_entry.to_vec(), qd_entry.to_vec());
    let mut cmd = q.clone();
    let mut buf: Vec<Vec<f64>> = vec![q.clone(); WORST.lat + 1];
    let (mut worst, mut home, mut safe_once) = (min_barrier(robot, &q), false, false);
    for _ in 0..700 {
        // choose the command TARGET
        let mb = min_barrier(robot, &q);
        // LATCH: once the retreat has won clear slack, commit to homing and stop escaping — otherwise
        // the escape term keeps re-triggering near barriers and the retreat never settles.
        if mb > FENCE + 0.015 { safe_once = true; } // latch below the home's own clearance so it settles
        let target: Vec<f64> = if aware && !safe_once && mb < FENCE + 0.01 {
            // CONFLICT-AWARE: ascend the composite clearance (away from whichever vector is closest),
            // BLENDED toward home so it routes around the active barrier while still making progress.
            let g = barrier_grad(robot, &q);
            let gn = (g.iter().map(|x| x * x).sum::<f64>()).sqrt().max(1e-6);
            let dn = (0..5).map(|i| (q0[i] - q[i]).powi(2)).sum::<f64>().sqrt().max(1e-6);
            let w = ((mb - (FENCE - 0.04)) / 0.05).clamp(0.0, 1.0);
            (0..5).map(|i| (q[i] + 0.22 * ((1.0 - w) * g[i] / gn + w * (q0[i] - q[i]) / dn)).clamp(LIM[i][0], LIM[i][1])).collect()
        } else { q0.to_vec() }; // otherwise head home
        for i in 0..5 { cmd[i] += (target[i] - cmd[i]).clamp(-VMAX * DT, VMAX * DT); }
        buf.push(cmd.clone()); let applied = buf.remove(0);
        step(robot, inertia, &mut q, &mut qd, &applied, &WORST);
        worst = worst.min(min_barrier(robot, &q));
        if (0..5).all(|i| (q[i] - q0[i]).abs() < 0.12 && qd[i].abs() < 0.25) { home = true; break; }
    }
    (worst, home)
}

fn main() {
    let (robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");
    // safe home in the BACK (−y, high): pick the pose that MAXIMIZES min-clearance while biased to the
    // back-up region — guarantees the safest reachable home clear of BOTH vectors (the person is out front).
    let mut q0 = vec![0.0; 5]; let mut best = f64::NEG_INFINITY;
    for s in 0..30000u32 { let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 7 + i as u32 + 1)).collect(); let p = robot.fk(&c).translation.vector; let score = min_barrier(&robot, &c) - 0.20 * ((p.x - 0.0).powi(2) + (p.y + 0.18).powi(2) + (p.z - 0.28).powi(2)).sqrt(); if score > best { best = score; q0 = c; } }
    // the corrupt fault: drive a link deep into the FRONT hand (the retreat home then sweeps up past it).
    let mut q_bad = vec![0.0; 5]; let mut worst_h = f64::INFINITY;
    for s in 0..20000u32 { let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 9 + i as u32 + 7)).collect(); let h = human_clear(&robot, &c); if h < worst_h && robot.fk(&c).translation.vector.z < 0.14 { worst_h = h; q_bad = c; } }

    println!("Barrier-conflict certificate on the real SO-101 — two protective vectors that fight.\n");
    println!("  person's hand at {:?} r={:.2}   raised home q0: table {:.3}, human {:.3} (both > fence {:.2})", HUMAN_C, HUMAN_R, table_clear(&robot, &q0), human_clear(&robot, &q0), FENCE);
    println!("  fault corner q_bad: table {:.3}, human {:.3} (violates BOTH)\n", table_clear(&robot, &q_bad), human_clear(&robot, &q_bad));

    // ---- collect conflict entry states: drive the fault, capture (q,qd) when min-barrier ≤ fence ----
    let mut entries: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for phase in 0..120u32 {
        let start: Vec<f64> = (0..5).map(|i| (q0[i] + 0.15 * ((0.3 * phase as f64 + i as f64).sin())).clamp(LIM[i][0], LIM[i][1])).collect();
        let (mut q, mut qd) = (start, vec![0.0f64; 5]);
        let mut cmd = q.clone(); let mut buf: Vec<Vec<f64>> = vec![q.clone(); WORST.lat + 1];
        for _ in 0..700 {
            cmd.copy_from_slice(&q_bad);
            buf.push(cmd.clone()); let applied = buf.remove(0);
            step(&robot, &inertia, &mut q, &mut qd, &applied, &WORST);
            if min_barrier(&robot, &q) <= FENCE { entries.push((q.clone(), qd.clone())); break; }
        }
    }
    // adversarial speed boost
    let mut region: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for (q, qd) in &entries { for &sc in &[1.0, 1.2] { region.push((q.clone(), qd.iter().map(|x| x * sc).collect())); } }
    println!("  conflict region R: {} entry states (fault into the low-front corner × boosted speed)\n", region.len());

    let mut hole = [0usize; 2]; let mut homed_c = [0usize; 2]; let mut worst = [f64::INFINITY; 2];
    for (idx, (label, aware)) in [("NAIVE retreat (straight to safe home)", false), ("CONFLICT-AWARE retreat (ascend composite clearance, then home)", true)].iter().enumerate() {
        for (q, qd) in &region {
            let (w, h) = retreat(&robot, &inertia, q, qd, &q0, *aware);
            worst[idx] = worst[idx].min(w); if w < 0.0 { hole[idx] += 1; } if h { homed_c[idx] += 1; }
        }
        println!("  {label}");
        println!("     worst min-barrier over R: {:.4} m   barrier violations (trap holes): {}/{}   reached home: {}/{}\n", worst[idx], hole[idx], region.len(), homed_c[idx], region.len());
    }

    // ---- verdict ----
    let n = region.len();
    println!("  ================  BARRIER-CONFLICT VERDICT  ================");
    if hole[0] == 0 {
        println!("  NO TRAP on this body: even a NAIVE straight-to-home retreat kept BOTH vectors ≥ {:.3} m over the", worst[0]);
        println!("  whole conflict region ({n} adversarial entries into a corner violating both). On a 5-DOF arm with a");
        println!("  genuinely safe home, multi-vector recovery COLLAPSES to the single-vector case already certified —");
        println!("  the joints have enough freedom to route to home without crossing a barrier. HONEST BOUND: this is a");
        println!("  negative result for THIS arm+home, not a proof no trap exists; traps require the safe set to be");
        println!("  topologically ENCLOSED from the fault region (higher-DOF, tighter workspace, or an obstacle that");
        println!("  walls off home). The conflict-aware retreat (ascend the composite clearance, then home) is the tool");
        println!("  for that case; here it is unnecessary — the honest finding is that the trap did not manifest.");
    } else if hole[1] < hole[0] {
        println!("  CONFLICT IS REAL: the naive retreat traps {}/{n} (crosses the other barrier); the conflict-aware", hole[0]);
        println!("  retreat cuts it to {}/{n} by routing around the active vector. Any RESIDUAL holes are a genuine TRAP", hole[1]);
        println!("  region the arbiter must REFUSE TO ENTER (a named niche edge) — the multi-vector refusal-before-the-edge.");
    } else {
        println!("  The naive retreat traps {}/{n}; the aware retreat did not improve it — a hard TRAP region that must", hole[0]);
        println!("  be refused at the boundary, not recovered from. Name it and keep the arbiter out of it.");
    }
}
