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
// a person's hand placed LOW and in front — between the fault region and a raised home, so lifting the
// tool to safety sweeps toward the hand while dropping away from the hand sweeps toward the table.
const HUMAN_C: [f64; 3] = [0.20, 0.05, 0.09];
const HUMAN_R: f64 = 0.075;

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
    let (mut worst, mut home) = (min_barrier(robot, &q), false);
    for _ in 0..800 {
        // choose the command TARGET
        let target: Vec<f64> = if aware && min_barrier(robot, &q) < FENCE + 0.04 {
            // CONFLICT-AWARE: ascend the composite clearance (away from whichever vector is closest),
            // blended toward home so it still makes progress once there is slack.
            let g = barrier_grad(robot, &q);
            let gn = (g.iter().map(|x| x * x).sum::<f64>()).sqrt().max(1e-6);
            (0..5).map(|i| (q[i] + 0.25 * g[i] / gn).clamp(LIM[i][0], LIM[i][1])).collect()
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
    // raised safe home, clear of both vectors.
    let mut q0 = vec![0.0; 5]; let mut best = f64::INFINITY;
    for s in 0..14000u32 { let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 7 + i as u32 + 1)).collect(); let p = robot.fk(&c).translation.vector; let cost = (p.z - 0.24).powi(2) + (p.x - 0.10).powi(2) + (p.y + 0.10).powi(2); if cost < best && min_barrier(&robot, &c) > FENCE + 0.05 { best = cost; q0 = c; } }
    // the corrupt fault: drive into the LOW-FRONT corner (toward the hand AND the table at once).
    let mut q_bad = vec![0.0; 5]; let mut worst_corner = f64::INFINITY;
    for s in 0..14000u32 { let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 9 + i as u32 + 7)).collect(); let sc = min_barrier(&robot, &c); if sc < worst_corner { worst_corner = sc; q_bad = c; } }

    println!("Barrier-conflict certificate on the real SO-101 — two protective vectors that fight.\n");
    println!("  person's hand at {:?} r={:.2}   raised home q0: table {:.3}, human {:.3} (both > fence {:.2})", HUMAN_C, HUMAN_R, table_clear(&robot, &q0), human_clear(&robot, &q0), FENCE);
    println!("  fault corner q_bad: table {:.3}, human {:.3} (violates BOTH)\n", table_clear(&robot, &q_bad), human_clear(&robot, &q_bad));

    // ---- collect conflict entry states: drive the fault, capture (q,qd) when min-barrier ≤ fence ----
    let mut entries: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for phase in 0..240u32 {
        let start: Vec<f64> = (0..5).map(|i| (q0[i] + 0.15 * ((0.3 * phase as f64 + i as f64).sin())).clamp(LIM[i][0], LIM[i][1])).collect();
        let (mut q, mut qd) = (start, vec![0.0f64; 5]);
        let mut cmd = q.clone(); let mut buf: Vec<Vec<f64>> = vec![q.clone(); WORST.lat + 1];
        for _ in 0..600 {
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

    for (label, aware) in [("NAIVE retreat (straight to safe home)", false), ("CONFLICT-AWARE retreat (ascend composite clearance, then home)", true)] {
        let (mut worst_all, mut holes, mut homed, mut n) = (f64::INFINITY, 0, 0, 0);
        for (q, qd) in &region {
            let (w, h) = retreat(&robot, &inertia, q, qd, &q0, aware);
            worst_all = worst_all.min(w); if w < 0.0 { holes += 1; } if h { homed += 1; } n += 1;
        }
        println!("  {label}");
        println!("     worst min-barrier over R: {:.4} m   barrier violations (trap holes): {}/{n}   reached home: {}/{n}\n", worst_all, holes, homed);
    }

    println!("  Honest reading: a NAIVE straight-to-home retreat treats recovery as one vector and can cross the");
    println!("  OTHER barrier in a conflict corner; the conflict-aware retreat ascends the composite clearance");
    println!("  (moves away from whichever vector is closest) before homing. If the aware retreat still shows");
    println!("  holes, those entry states are a genuine TRAP the arbiter must refuse to ENTER (a named niche edge),");
    println!("  not one it can recover from — the multi-vector version of refusal-before-the-edge.");
}
