//! MULTI-VECTOR arbiter on the real SO-101 — the doctrine's deeper crux: many protective vectors at
//! once, not one. Here THREE geometric protective vectors + a resource + the generative reach:
//!   V_table  — the tool/links stay above the work surface (z ≥ 0)               [self-not-harm]
//!   V_human  — a person's hand reaches into the workspace; NOTHING may strike it [NO-HARM-TO-OTHERS]
//!   V_self   — the arm's own links may not collide with each other              [self-not-harm]
//!   milk     — STS3215 thermal budget (self-caused untracked state)             [resource]
//!   reach    — a generative tabletop wander                                     [GENERATIVE flow]
//! The barrier is the swept-sphere clearance of the REAL links (placed by frame_pose) against an
//! SdfScene {table Plane, human Sphere} plus link-vs-link self clearance. The fault (Feetech comms
//! glitch) latches a corrupt command that swings the arm INTO the person or DOWN through the table —
//! the two protective vectors can point in DIFFERENT directions, so the arbiter must recognize which
//! is threatened and retreat to a pose safe under ALL of them. This is #2 (the full Cartesian barrier)
//! and #1 (arbitrating + recovering over several vectors). Honest scope: spheres approximate the links
//! (conservative if they enclose the geometry); the twin, not a physical unit.
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
const R_LINK: f64 = 0.028;   // swept-sphere radius approximating the SO-101 link thickness
const MARG: f64 = 0.05;      // safety clearance the barrier must keep (refusal-before-the-edge)
const HUMAN_C: [f64; 3] = [0.12, 0.26, 0.15]; // a person's hand reaching into the workspace (+y side)
const HUMAN_R: f64 = 0.09;
const HWARN: f64 = 0.7;
const HKILL: f64 = 1.6;

fn hash(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u01(i: u32) -> f64 { (hash(i) % 1_000_000) as f64 / 1_000_000.0 }

// swept-sphere model of the arm: each sphere tagged with its proximal link index (0=base..5=tool).
fn arm_spheres(robot: &Robot, q: &[f64]) -> Vec<(Vector3<f64>, usize)> {
    let f: Vec<Vector3<f64>> = (0..=5).map(|u| robot.frame_pose(q, u).translation.vector).collect();
    let tip = robot.fk(q).translation.vector;
    let mut out = Vec::new();
    for i in 0..5 { out.push((f[i], i)); out.push(((f[i] + f[i + 1]) * 0.5, i)); }  // segments base..wrist
    out.push((f[5], 5)); out.push(((f[5] + tip) * 0.5, 5)); out.push((tip, 5));     // wrist→tool
    out
}
// the three protective vectors as signed clearances (>0 = safe). Returns (table, human, self).
fn barriers(robot: &Robot, q: &[f64]) -> (f64, f64, f64) {
    let s = arm_spheres(robot, q);
    // table: only the MOVING distal links (forearm/wrist/tool, link ≥2) — the base+shoulder legitimately
    // sit at the mount plane; the barrier is the tool/forearm crashing onto the surface away from the base.
    let table = s.iter().filter(|(_, l)| *l >= 2).map(|(c, _)| c.z - R_LINK).fold(f64::INFINITY, f64::min);
    let hc = Vector3::new(HUMAN_C[0], HUMAN_C[1], HUMAN_C[2]);
    let human = s.iter().map(|(c, _)| (c - hc).norm() - R_LINK - HUMAN_R).fold(f64::INFINITY, f64::min);
    // self: the fold-back collision — tool/wrist (link ≥4) against base/shoulder (link ≤1).
    let mut selfc = f64::INFINITY;
    for (a, la) in &s { for (b, lb) in &s { if *la >= 4 && *lb <= 1 { selfc = selfc.min((a - b).norm() - 2.0 * R_LINK); } } }
    (table, human, selfc)
}
fn bmin(robot: &Robot, q: &[f64]) -> f64 { let (t, h, s) = barriers(robot, q); t.min(h).min(s) }

struct Env { fric: f64, lat: usize, dead: f64 }
fn servo(cmd: &[f64], q: &[f64], qd: &[f64], throttle: f64, dead: f64) -> Vec<f64> {
    (0..5).map(|i| { let mut e = cmd[i] - q[i]; if e.abs() < dead { e = 0.0; } (KP * e - KV * qd[i]).clamp(-TAUMAX * throttle, TAUMAX * throttle) }).collect()
}
fn step(robot: &Robot, inertia: &[LinkInertia], q: &mut [f64], qd: &mut [f64], applied: &[f64], env: &Env, throttle: f64) -> Vec<f64> {
    let tau_s = servo(applied, q, qd, throttle, env.dead);
    let tau: Vec<f64> = (0..5).map(|i| tau_s[i] - env.fric * qd[i] - if qd[i].abs() > 1e-3 { 0.052 * qd[i].signum() } else { 0.0 }).collect();
    let qdd_link = forward_dynamics(robot, inertia, q, qd, &tau, Vector3::new(0.0, 0.0, G));
    let m = mass_matrix(robot, inertia, q);
    let mut ma = m.clone(); for i in 0..5 { ma[(i, i)] += ARMATURE; }
    let qdd = ma.cholesky().expect("SPD").solve(&(&m * DVector::from_row_slice(&qdd_link)));
    for i in 0..5 { qd[i] += DT * qdd[i]; q[i] = (q[i] + DT * qd[i]).clamp(LIM[i][0], LIM[i][1]); }
    tau_s
}

struct Task { q0: Vec<f64>, bad_table: Vec<f64>, bad_human: Vec<f64> }

// returns (struck, which_barrier, recovered)  — struck = a protective vector went negative (contact).
fn episode(robot: &Robot, inertia: &[LinkInertia], task: &Task, arbiter: bool, env: &Env, seed: u32) -> (bool, u8, bool) {
    let q_play = |t: usize| -> Vec<f64> { (0..5).map(|i| (task.q0[i] + 0.06 * (0.02 * t as f64 + i as f64).sin()).clamp(LIM[i][0], LIM[i][1])).collect() };
    let (mut q, mut qd) = (q_play(0), vec![0.0f64; 5]);
    let mut cmd = q.clone();
    let mut heat = 0.0f64;
    let (t_total, tf, k) = (1600usize, 350 + (hash(seed) % 700) as usize, 220);
    let mut buf: Vec<Vec<f64>> = vec![q.clone(); env.lat + 1];
    let mut mode_recover = false;
    let (mut struck, mut which, mut resumed) = (false, 0u8, false);
    let toward_human = hash(seed) & 1 == 0; // the glitch swings into the person, or folds through the table
    for t in 0..t_total {
        let in_fault = t >= tf && t < tf + k;
        let raw: Vec<f64> = if in_fault { if toward_human { task.bad_human.clone() } else { task.bad_table.clone() } } else { q_play(t) };
        let target: Vec<f64> = if arbiter {
            let cmd_b = bmin(robot, &raw);   // where would this command put the arm? (feedforward)
            let cur_b = bmin(robot, &q);
            if in_fault || cmd_b < MARG || cur_b < MARG { mode_recover = true; } // RECOGNITION across ALL vectors
            if mode_recover {
                if (0..5).all(|i| (q[i] - task.q0[i]).abs() < 0.10 && qd[i].abs() < 0.2) { mode_recover = false; if t > tf { resumed = true; } }
                task.q0.clone() // retreat to the pose safe under EVERY protective vector
            } else if heat > HWARN { q.clone() } else { raw }
        } else { raw };
        if arbiter { for i in 0..5 { cmd[i] += (target[i] - cmd[i]).clamp(-VMAX * DT, VMAX * DT); } } else { cmd.copy_from_slice(&target); }
        buf.push(cmd.clone()); let applied = buf.remove(0);
        let throttle = if heat > 1.0 { 0.35 } else { 1.0 };
        let tau_s = step(robot, inertia, &mut q, &mut qd, &applied, env, throttle);
        heat += 2.2e-4 * tau_s.iter().map(|x| x * x).sum::<f64>(); heat *= 0.9985;
        let (bt, bh, bs) = barriers(robot, &q);
        if !struck { if bh < 0.0 { struck = true; which = 2; } else if bt < 0.0 { struck = true; which = 1; } else if bs < 0.0 { struck = true; which = 3; } }
        if heat > HKILL { struck = true; which = 4; break; }
    }
    let recovered = arbiter && !struck && resumed;
    (struck, which, recovered)
}

fn main() {
    let (robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");

    // safe home q0: reach forward, tool up, AWAY from the person (−y), clear of table & self.
    let mut q0 = vec![0.0; 5]; let mut best = f64::INFINITY;
    for s in 0..12000u32 {
        let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 7 + i as u32 + 1)).collect();
        let p = robot.fk(&c).translation.vector;
        let cost = (p.z - 0.20).powi(2) + (p.x - 0.18).powi(2) + (p.y + 0.06).powi(2);
        if cost < best && bmin(&robot, &c) > MARG + 0.03 { best = cost; q0 = c; }
    }
    // corrupt fault targets: fold through the table (min tip-z), and swing INTO the person (min human clearance).
    let mut bad_table = vec![0.0; 5]; let mut zmin = f64::INFINITY;
    for s in 0..8000u32 { let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 5 + i as u32 + 5)).collect(); let z = robot.fk(&c).translation.vector.z; if z < zmin { zmin = z; bad_table = c; } }
    let mut bad_human = vec![0.0; 5]; let mut hmin = f64::INFINITY;
    for s in 0..12000u32 { let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 9 + i as u32 + 7)).collect(); let (_, bh, _) = barriers(&robot, &c); if bh < hmin { hmin = bh; bad_human = c; } }
    let task = Task { q0: q0.clone(), bad_table, bad_human };

    println!("Multi-vector arbiter on the real SO-101 — three protective vectors at once.\n");
    let (t0, h0, s0) = barriers(&robot, &q0);
    println!("  human hand keep-out: center {:?} r={:.2} m", HUMAN_C, HUMAN_R);
    println!("  safe home q0 clearances: table {:.3}  human {:.3}  self {:.3}  (all > fence {:.2}) ✓", t0, h0, s0, MARG);
    let (btt, _, _) = barriers(&robot, &task.bad_table);
    let (_, bhh, _) = barriers(&robot, &task.bad_human);
    println!("  fault A (fold-down): table clearance {:.3} m → strikes the table", btt);
    println!("  fault B (into hand): human clearance {:.3} m → strikes the person\n", bhh);
    let _ = hmin;

    let envs: Vec<Env> = (0..8).map(|k| Env { fric: 0.35 + 0.5 * u01(1000 + k), lat: (u01(2000 + k) * 4.0) as usize, dead: 0.004 + 0.02 * u01(3000 + k) }).collect();
    for (label, arb) in [("SINGLE-VECTOR (execute task/fault)", false), ("MULTI-VECTOR ARBITER (3 barriers ⟂ reach + milk + recover)", true)] {
        let (mut strk, mut rec, mut n) = (0, 0, 0);
        let (mut ht, mut tb, mut sf, mut th) = (0, 0, 0, 0);
        for (ei, env) in envs.iter().enumerate() {
            for ep in 0..14 {
                let seed = ei as u32 * 131 + ep * 7 + 1;
                let (s, w, r) = episode(&robot, &inertia, &task, arb, env, seed);
                n += 1; if s { strk += 1; match w { 2 => ht += 1, 1 => tb += 1, 3 => sf += 1, 4 => th += 1, _ => {} } }
                if r { rec += 1; }
            }
        }
        println!("  {label}");
        println!("     barrier strikes: {:>3}/{n}   (human {ht}, table {tb}, self {sf}, thermal {th}){}",
            strk, if arb { format!("   recovered+resumed: {rec}/{n}") } else { String::new() });
    }
    println!("\n(8 randomized reality models × 14 episodes; each glitch swings into the person OR through the table)");
}
