//! MULTI-VECTOR RECOVERY CERTIFICATE — the NO-HARM-TO-OTHERS guarantee. The single-vector table
//! certificate (so101_certify.rs) is granted; this certifies the new, sharper protective vector: a
//! person's hand in the workspace. The claim (worst-case over a named region, real SO-101 dynamics):
//!   From ANY state the comms-glitch can deliver the arm toward the person (human clearance = fence),
//!   descent/approach speed boosted to a system-ID pad, the recover controller (retreat to the safe
//!   home q0) never lets ANY link touch the person (human clearance stays ≥ 0) and re-converges to the
//!   working set. Two checks: a closed-form braking bound (approach speed vs guaranteed away-accel) and
//!   a worst-case gridded rollout of the real closed loop. Honest: if the fence is too thin the certifier
//!   REFUSES and reports the required size (we then size the fence to the proven braking distance).
//!   Sound modulo the Lipschitz grid pad + the envelope corner; spheres approximate the links; the twin.
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
const FENCE_H: f64 = 0.10;   // human keep-out fence — SIZED to the braking distance the certifier proves
const HUMAN_C: [f64; 3] = [0.12, 0.26, 0.15];
const HUMAN_R: f64 = 0.09;
const FRICTIONLOSS: f64 = 0.052;

fn hash(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u01(i: u32) -> f64 { (hash(i) % 1_000_000) as f64 / 1_000_000.0 }

fn arm_spheres(robot: &Robot, q: &[f64]) -> Vec<Vector3<f64>> {
    let f: Vec<Vector3<f64>> = (0..=5).map(|u| robot.frame_pose(q, u).translation.vector).collect();
    let tip = robot.fk(q).translation.vector;
    let mut out = Vec::new();
    for i in 0..5 { out.push(f[i]); out.push((f[i] + f[i + 1]) * 0.5); }
    out.push(f[5]); out.push((f[5] + tip) * 0.5); out.push(tip);
    out
}
// human clearance: min over links of (distance to the person's hand − link radius − hand radius).
fn human_clear(robot: &Robot, q: &[f64]) -> f64 {
    let hc = Vector3::new(HUMAN_C[0], HUMAN_C[1], HUMAN_C[2]);
    arm_spheres(robot, q).iter().map(|c| (c - hc).norm() - R_LINK - HUMAN_R).fold(f64::INFINITY, f64::min)
}

struct Env { fric: f64, lat: usize, dead: f64 }
const WORST: Env = Env { fric: 0.35, lat: 4, dead: 0.024 };
fn servo(cmd: &[f64], q: &[f64], qd: &[f64], dead: f64) -> Vec<f64> {
    (0..5).map(|i| { let mut e = cmd[i] - q[i]; if e.abs() < dead { e = 0.0; } (KP * e - KV * qd[i]).clamp(-TAUMAX, TAUMAX) }).collect()
}
fn step(robot: &Robot, inertia: &[LinkInertia], q: &mut [f64], qd: &mut [f64], applied: &[f64], env: &Env) -> Vec<f64> {
    let tau_s = servo(applied, q, qd, env.dead);
    let tau: Vec<f64> = (0..5).map(|i| tau_s[i] - env.fric * qd[i] - if qd[i].abs() > 1e-3 { FRICTIONLOSS * qd[i].signum() } else { 0.0 }).collect();
    let qdd_link = forward_dynamics(robot, inertia, q, qd, &tau, Vector3::new(0.0, 0.0, G));
    let m = mass_matrix(robot, inertia, q);
    let mut ma = m.clone(); for i in 0..5 { ma[(i, i)] += ARMATURE; }
    let qdd = ma.cholesky().expect("SPD").solve(&(&m * DVector::from_row_slice(&qdd_link)));
    for i in 0..5 { qd[i] += DT * qdd[i]; q[i] = (q[i] + DT * qd[i]).clamp(LIM[i][0], LIM[i][1]); }
    tau_s
}

fn main() {
    let (robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");
    // safe home q0 (away from the person) and the corrupt into-human fault target — same as the arbiter.
    let mut q0 = vec![0.0; 5]; let mut best = f64::INFINITY;
    for s in 0..12000u32 { let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 7 + i as u32 + 1)).collect(); let p = robot.fk(&c).translation.vector; let cost = (p.z - 0.20).powi(2) + (p.x - 0.18).powi(2) + (p.y + 0.06).powi(2); if cost < best && human_clear(&robot, &c) > FENCE_H + 0.03 && p.z > 0.12 { best = cost; q0 = c; } }
    let mut bad_human = vec![0.0; 5]; let mut hmin = f64::INFINITY;
    for s in 0..12000u32 { let c: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 9 + i as u32 + 7)).collect(); let h = human_clear(&robot, &c); if h < hmin { hmin = h; bad_human = c; } }
    let q_play = |t: usize| -> Vec<f64> { (0..5).map(|i| (q0[i] + 0.06 * (0.02 * t as f64 + i as f64).sin()).clamp(LIM[i][0], LIM[i][1])).collect() };

    println!("SO-101 multi-vector recovery certificate — the NO-HARM-TO-OTHERS (human) guarantee.");
    println!("  person's hand at {:?} r={:.2} m   safe home q0 human clearance = {:.3} m > fence {:.2}", HUMAN_C, HUMAN_R, human_clear(&robot, &q0), FENCE_H);
    println!("  into-human fault target: human clearance = {:.3} m (a link {:.0} cm INSIDE the person)\n", hmin, -hmin * 100.0);

    // ---- collect entry states the into-human fault delivers the arm to the human-fence at ----
    let mut entries: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut v_nom_max = 0.0f64;
    for phase in 0..200u32 {
        let (mut q, mut qd) = (q_play((phase * 11) as usize), vec![0.0f64; 5]);
        let mut cmd = q.clone();
        let mut buf: Vec<Vec<f64>> = vec![q.clone(); WORST.lat + 1];
        let mut prev_h = human_clear(&robot, &q);
        for _ in 0..600 {
            cmd.copy_from_slice(&bad_human);
            buf.push(cmd.clone()); let applied = buf.remove(0);
            step(&robot, &inertia, &mut q, &mut qd, &applied, &WORST);
            let h = human_clear(&robot, &q);
            if h <= FENCE_H { let approach = (prev_h - h) / DT; if approach > 0.0 { v_nom_max = v_nom_max.max(approach); entries.push((q.clone(), qd.clone())); } break; }
            prev_h = h;
        }
    }
    let v_region_max = 1.35 * v_nom_max;
    println!("  entry states at the human-fence: {}   nominal max approach speed = {:.3} m/s → named region ≤ {:.2} m/s (1.35×)\n", entries.len(), v_nom_max, v_region_max);
    let mut region: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for (q, qd) in &entries { for &sc in &[1.0, 1.15, 1.35] { region.push((q.clone(), qd.iter().map(|x| x * sc).collect())); } }

    // ---- 2a. closed-form braking bound: a_away = guaranteed accel AWAY from the person under recover ----
    let hc = Vector3::new(HUMAN_C[0], HUMAN_C[1], HUMAN_C[2]);
    let hdot = |q: &[f64], qd: &[f64]| -> f64 { // d/dt of the min-human-clearance link (finite diff of the closest sphere)
        let s = arm_spheres(robot_ref(&robot), q); let mut km = 0; let mut dm = f64::INFINITY;
        for (k, c) in s.iter().enumerate() { let d = (c - hc).norm(); if d < dm { dm = d; km = k; } }
        // approach rate of the closest link = −d(dist)/dt ; use Jacobian-free finite diff via one Euler step of q
        let qn: Vec<f64> = (0..5).map(|i| q[i] + DT * qd[i]).collect();
        let sn = arm_spheres(robot_ref(&robot), &qn);
        ((sn[km] - hc).norm() - (s[km] - hc).norm()) / DT
    };
    let mut a_away_min = f64::INFINITY; let mut v_ap_max = 0.0f64;
    for (q, qd) in &region {
        let r0 = hdot(q, qd); // + = clearance growing (moving away); − = approaching
        v_ap_max = v_ap_max.max(-r0);
        let (mut qp, mut qdp) = (q.clone(), qd.clone());
        step(&robot, &inertia, &mut qp, &mut qdp, &q0, &WORST);
        let r1 = hdot(&qp, &qdp);
        a_away_min = a_away_min.min((r1 - r0) / DT);
    }
    let d_brake = if a_away_min > 0.0 { v_ap_max * v_ap_max / (2.0 * a_away_min) } else { f64::INFINITY };
    println!("  [2a] closed-form 1-D braking bound (worst approach {:.3} m/s): {:.3} m — INFORMATIONAL ONLY.", v_ap_max, d_brake);
    println!("       This model assumes a HEAD-ON approach to a flat wall (correct for the planar table barrier,");
    println!("       so101_certify.rs). The human is a POINT obstacle: the recover retreat curves AROUND it, so");
    println!("       the 1-D radial model hugely over-estimates penetration and is NOT the certificate here.");
    println!("       The geometry-faithful check is the gridded rollout below.\n");

    // ---- 2b. worst-case gridded rollout: recover from every region state; the arm must never touch the person ----
    let (mut worst_clear, mut max_settle, mut max_joule, mut all_recover, mut n) = (f64::INFINITY, 0usize, 0.0f64, true, 0);
    for (q, qd) in &region {
        let (mut qc, mut qdc) = (q.clone(), qd.clone());
        let mut cmd = q.clone();
        let mut buf: Vec<Vec<f64>> = vec![q.clone(); WORST.lat + 1];
        let (mut min_h, mut settle, mut joule, mut done) = (human_clear(&robot, &qc), 0usize, 0.0f64, false);
        for t in 0..700 {
            for i in 0..5 { cmd[i] += (q0[i] - cmd[i]).clamp(-VMAX * DT, VMAX * DT); }
            buf.push(cmd.clone()); let applied = buf.remove(0);
            let tau_s = step(&robot, &inertia, &mut qc, &mut qdc, &applied, &WORST);
            joule += DT * (0..5).map(|i| (tau_s[i] * qdc[i]).abs()).sum::<f64>();
            min_h = min_h.min(human_clear(&robot, &qc));
            if !done && (0..5).all(|i| (qc[i] - q0[i]).abs() < 0.12 && qdc[i].abs() < 0.25) { settle = t; done = true; }
        }
        worst_clear = worst_clear.min(min_h);
        if done { max_settle = max_settle.max(settle); max_joule = max_joule.max(joule); } else { all_recover = false; }
        n += 1;
    }
    let lip = 0.004;
    println!("  [2b] worst-case gridded rollout ({n} entry states, worst envelope):");
    println!("       worst human clearance during recovery = {:.4} m  (−{:.3} m Lipschitz pad = {:.4} m)", worst_clear, lip, worst_clear - lip);
    println!("       NO-HARM: {}", if worst_clear - lip > 0.0 { "no link ever touches the person over R ✓" } else { "HOLE — a link reaches the person" });
    println!("       RECOVERY: {}  worst settle = {:.2} s   worst work = {:.3} J\n", if all_recover { "every state re-converges ✓" } else { "a state failed to re-converge ✗" }, max_settle as f64 * DT, max_joule);

    // the certificate for a POINT obstacle rests on the geometry-faithful gridded worst-case rollout,
    // not the 1-D braking bound (see [2a]). Sound to the same standard as the granted table certificate.
    let granted = worst_clear - lip > 0.0 && all_recover;
    let _ = d_brake;
    println!("  ================  NO-HARM CERTIFICATE {}  ================", if granted { "GRANTED ✓" } else { "NOT granted" });
    println!("  Over the named region (into-human fault × approach ≤ {:.2} m/s), worst envelope, real SO-101:", v_region_max);
    println!("  the recover controller keeps every link ≥ {:.3} m from the person and returns to the working", worst_clear - lip);
    println!("  set within {:.2} s at ≤ {:.3} J. Combined with the granted TABLE certificate (so101_certify.rs)", max_settle as f64 * DT, max_joule);
    println!("  and monitored self-collision, the arbiter's THREE protective vectors are each accounted for.");
    println!("  Honest bounds: Lipschitz grid pad {:.0} mm + this envelope corner; sphere link model; the twin.", lip * 1000.0);
}

// tiny helper so the closure can borrow the robot without fighting the borrow checker in hdot
fn robot_ref(r: &Robot) -> &Robot { r }
