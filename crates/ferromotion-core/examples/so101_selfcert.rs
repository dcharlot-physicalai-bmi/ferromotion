//! SELF-COLLISION on the SO-101 — the last uncertified protective vector, and first the honest question:
//! CAN this arm even fold onto itself within its joint limits? The table and human barriers were real
//! irreversibilities; self-collision "monitored but never bound" in the multi-vector demo. Rather than
//! engineer a fault to force it, we first PROBE the reachable set: the minimum clearance between distal
//! links (wrist/tool) and proximal links (base/shoulder) over the whole joint space. If that minimum
//! stays positive, the SO-101 physically cannot self-collide — the barrier is vacuous, hardware-protected
//! like the joint limits, and certifying it would be manufacturing a failure. If it goes negative, self-
//! collision is real and we certify the recover keeps the arm off itself worst-case. Swept-sphere links; twin.
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
const R_LINK: f64 = 0.028;   // swept-sphere radius (link half-thickness); self-collision when two spheres overlap

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
// self-clearance: min gap between non-adjacent links (chain distance ≥ 3) — the pairs that could actually
// collide (distal wrist/tool folding back onto base/shoulder), excluding structurally-adjacent segments.
fn self_clear(robot: &Robot, q: &[f64]) -> f64 {
    let s = arm_spheres(robot, q);
    let mut m = f64::INFINITY;
    for i in 0..s.len() { for j in 0..s.len() {
        if (s[i].1 as i32 - s[j].1 as i32) >= 3 { m = m.min((s[i].0 - s[j].0).norm() - 2.0 * R_LINK); }
    } }
    m
}

struct Env { fric: f64, lat: usize, dead: f64 }
const WORST: Env = Env { fric: 0.35, lat: 4, dead: 0.024 };
fn servo(cmd: &[f64], q: &[f64], qd: &[f64], dead: f64) -> Vec<f64> {
    (0..5).map(|i| { let mut e = cmd[i] - q[i]; if e.abs() < dead { e = 0.0; } (KP * e - KV * qd[i]).clamp(-TAUMAX, TAUMAX) }).collect()
}
fn step(robot: &Robot, inertia: &[LinkInertia], q: &mut [f64], qd: &mut [f64], applied: &[f64], env: &Env) {
    let tau_s = servo(applied, q, qd, env.dead);
    let tau: Vec<f64> = (0..5).map(|i| tau_s[i] - env.fric * qd[i] - if qd[i].abs() > 1e-3 { 0.052 * qd[i].signum() } else { 0.0 }).collect();
    let qdd_link = forward_dynamics(robot, inertia, q, qd, &tau, Vector3::new(0.0, 0.0, G));
    let m = mass_matrix(robot, inertia, q);
    let mut ma = m.clone(); for i in 0..5 { ma[(i, i)] += ARMATURE; }
    let qdd = ma.cholesky().expect("SPD").solve(&(&m * DVector::from_row_slice(&qdd_link)));
    for i in 0..5 { qd[i] += DT * qdd[i]; q[i] = (q[i] + DT * qd[i]).clamp(LIM[i][0], LIM[i][1]); }
}

fn main() {
    let (robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");

    println!("Self-collision on the SO-101 — is the barrier even reachable?\n");
    // PROBE 1: dense random scan for the minimum self-clearance over the joint box.
    let (mut worst_sc, mut q_worst) = (f64::INFINITY, vec![0.0; 5]);
    for s in 0..600000u32 {
        let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 7 + i as u32 + 1)).collect();
        let sc = self_clear(&robot, &c);
        if sc < worst_sc { worst_sc = sc; q_worst = c; }
    }
    println!("  random scan (600k poses): min self-clearance = {:+.3} m at q=[{}]",
        worst_sc, q_worst.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>().join(", "));

    // PROBE 2: local descent from the worst random pose (min self-clearance) to sharpen the estimate —
    // gradient-free coordinate search toward less clearance, to catch a collision the random scan missed.
    let mut q = q_worst.clone();
    let mut cur = worst_sc;
    for _ in 0..2000 {
        let mut improved = false;
        for i in 0..5 { for &d in &[-0.03f64, 0.03] {
            let mut c = q.clone(); c[i] = (c[i] + d).clamp(LIM[i][0], LIM[i][1]);
            let sc = self_clear(&robot, &c);
            if sc < cur { cur = sc; q = c; improved = true; }
        } }
        if !improved { break; }
    }
    println!("  local descent from there:              min self-clearance = {:+.3} m\n", cur);

    if cur > 0.0 {
        println!("  ================  VERDICT: BARRIER VACUOUS  ================");
        println!("  The SO-101 CANNOT fold onto itself within its joint limits — the closest any distal link");
        println!("  (wrist/tool) gets to a proximal link (base/shoulder) is {:.1} cm of clearance, everywhere in", cur * 100.0);
        println!("  the joint box. Self-collision is not a reachable failure mode for this arm; the joint limits");
        println!("  and link geometry already prevent it, exactly as the STS3215 torque ceiling prevents joint-");
        println!("  limit slams. Certifying a recover for it would be manufacturing a failure. The barrier stays");
        println!("  MONITORED (a cheap runtime check, {:.0} mm radius spheres), but it is hardware-protected, not", R_LINK * 1000.0);
        println!("  a vector the arbiter must actively recover from — the honest closure of the barrier trilogy:");
        println!("  table (certified) ⟂ human (certified) ⟂ self (VACUOUS on this geometry).");
        return;
    }

    // self-collision IS reachable → certify the recover keeps the arm off itself.
    println!("  self-collision IS reachable (min {:+.3} m < 0) — certifying the recover.\n", cur);
    let q_bad = q.clone(); // the deepest self-collision pose = the corrupt fold-onto-self target
    // safe home = the pose with the MOST self-clearance (fully extended, arm open).
    let (mut best_sc, mut q0) = (f64::NEG_INFINITY, vec![0.0; 5]);
    for s in 0..200000u32 {
        let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 9 + i as u32 + 5)).collect();
        let sc = self_clear(&robot, &c);
        if sc > best_sc { best_sc = sc; q0 = c; }
    }
    println!("  fold-onto-self fault target: self-clearance {:+.3} m ; safe home (extended): {:+.3} m", self_clear(&robot, &q_bad), best_sc);

    // certify: from entry states the fault delivers near self-contact (self_clear ≤ fence), does commanding
    // the extended home keep self_clear ≥ 0 worst-case, and re-open the arm?
    const FENCE: f64 = 0.04;
    let mut entries: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for phase in 0..160u32 {
        let start: Vec<f64> = (0..5).map(|i| (q0[i] + 0.2 * ((0.3 * phase as f64 + i as f64).sin())).clamp(LIM[i][0], LIM[i][1])).collect();
        let (mut qq, mut qd) = (start, vec![0.0f64; 5]);
        let mut cmd = qq.clone(); let mut buf: Vec<Vec<f64>> = vec![qq.clone(); WORST.lat + 1];
        for _ in 0..700 {
            cmd.copy_from_slice(&q_bad);
            buf.push(cmd.clone()); let applied = buf.remove(0);
            step(&robot, &inertia, &mut qq, &mut qd, &applied, &WORST);
            if self_clear(&robot, &qq) <= FENCE { entries.push((qq.clone(), qd.clone())); break; }
        }
    }
    let (mut strk, mut homed, mut worst, mut n) = (0, 0, f64::INFINITY, 0);
    for (q_e, qd_e) in &entries { for &sc in &[1.0, 1.2] {
        let (mut qq, mut qd) = (q_e.clone(), qd_e.iter().map(|x| x * sc).collect::<Vec<_>>());
        let mut cmd = qq.clone(); let mut buf: Vec<Vec<f64>> = vec![qq.clone(); WORST.lat + 1];
        let (mut w, mut home) = (self_clear(&robot, &qq), false);
        for _ in 0..900 {
            for i in 0..5 { cmd[i] += (q0[i] - cmd[i]).clamp(-VMAX * DT, VMAX * DT); }
            buf.push(cmd.clone()); let applied = buf.remove(0);
            step(&robot, &inertia, &mut qq, &mut qd, &applied, &WORST);
            w = w.min(self_clear(&robot, &qq));
            if (0..5).all(|i| (qq[i] - q0[i]).abs() < 0.14) { home = true; break; }
        }
        worst = worst.min(w); if w < 0.0 { strk += 1; }
        if home { homed += 1; } n += 1;
    } }
    println!("\n  ================  SELF-COLLISION RECOVER  ================");
    println!("  over {n} adversarial fold-onto-self entries, worst envelope: self-contact strikes {strk}/{n},");
    println!("  re-opened to the extended home {homed}/{n}, worst self-clearance {:+.3} m.", worst);
}
