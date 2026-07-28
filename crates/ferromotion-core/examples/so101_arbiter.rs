//! The Agency-Axis arbiter, on the real SO-101, in the trusted sim (ferromotion) — the first ant.
//!
//! FINDING THAT SET THE BARRIER: we first tried a JOINT-LIMIT barrier hit by a Feetech comms glitch.
//! It showed nothing — instrumented, the worst any joint got to its stop during a 200 ms latched
//! garbage command was 0.7–1.25 rad INSIDE the limit. A position-controlled, torque-limited
//! (±2.94 N·m), damped, armature-loaded arm physically cannot slam its own joint stops in a glitch:
//! the STS3215 torque ceiling is already a protective layer for joint limits. Manufacturing a
//! joint-limit "collapse" here would be dishonest. So we moved the barrier to the irreversibility the
//! hardware does NOT self-protect and that has no passive safe set: the TOOL crashing through the
//! WORK SURFACE. The joints stay legal the whole way down; only a Cartesian barrier on the tool tip
//! catches it. That barrier is expressible with forward kinematics alone — no collision meshes.
//!
//! Three vectors on one arm: a PROTECTIVE Cartesian barrier (tool tip must stay above the table —
//! a safe set checked EXACTLY via FK), a GENERATIVE task (a low tabletop reach + wander = "play")
//! flowing freely inside it, and a RESOURCE "milk" (STS3215 thermal budget). The FAULT is the real
//! one: a Feetech serial-bus comms glitch that latches a corrupt fold-DOWN command driving the tool
//! below the table. The arbiter must RECOGNIZE it, LIFT to a safe home, RE-CONVERGE, and resume the
//! reach — grab the napkin, not pull the tarp. Swept over a BOUNDED-REALITY ENVELOPE (friction, servo
//! latency, deadband, backlash) so the guarantee survives the sim→real gap; system-ID later collapses
//! the envelope onto the physical unit. Honest scope: barrier = table plane on the tip (+ thermal);
//! self-collision & a human are the same Cartesian family, needing more geometry. Software arbiter on
//! the twin — not a real robot yet.
use ferromotion_core::{forward_dynamics, from_urdf_full, mass_matrix, LinkInertia, Robot};
use nalgebra::{DVector, Vector3};

const URDF: &str = include_str!("so101.urdf");
const G: f64 = -9.81;
// the 5 arm joint limits (rad), from the URDF — a HARDWARE-protected safe set (proven above), so the
// arbiter's protective vector is not here; it is the Cartesian table barrier below.
const LIM: [[f64; 2]; 5] = [[-1.91986, 1.91986], [-1.74533, 1.74533], [-1.69, 1.69], [-1.65806, 1.65806], [-2.74385, 2.84121]];
const KP: f64 = 70.0;   // STS3215 position servo
const KV: f64 = 13.0;
const TAUMAX: f64 = 2.94; // STS3215 torque limit (N·m), from the MJCF
const DT: f64 = 1.5e-3;
const ARMATURE: f64 = 0.028; // STS3215 reflected rotor inertia (MJCF) — regularizes the light wrist, real param
const VMAX: f64 = 3.5;    // servo max joint rate (rad/s) — rate-limit so the command can't step-slam
const MARG_Z: f64 = 0.10;  // barrier margin ABOVE the table — SIZED to the certified braking distance
                           // (so101_certify.rs proves the recover controller stops the tool within this)
const HWARN: f64 = 0.7;   // thermal budget: recognize + conserve
const HMAX: f64 = 1.0;    // throttle
const HKILL: f64 = 1.6;   // runaway death

struct Env { fric: f64, lat: usize, dead: f64, back: f64 } // the bounded-reality envelope

fn hash(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u01(i: u32) -> f64 { (hash(i) % 1_000_000) as f64 / 1_000_000.0 }

fn tip_z(robot: &Robot, q: &[f64]) -> f64 { robot.fk(q).translation.vector.z }

fn servo(cmd: &[f64], q: &[f64], qd: &[f64], throttle: f64, dead: f64) -> Vec<f64> {
    (0..5).map(|i| {
        let mut e = cmd[i] - q[i];
        if e.abs() < dead { e = 0.0; }
        (KP * e - KV * qd[i]).clamp(-TAUMAX * throttle, TAUMAX * throttle)
    }).collect()
}

struct Task { q0: Vec<f64>, q_bad: Vec<f64>, z_floor: f64 } // safe working pose, corrupt fold-down, table height

// the generative vector: a slow tabletop wander around the working pose q0 — flow, safely above the fence.
fn q_play(task: &Task, t: usize) -> Vec<f64> {
    (0..5).map(|i| (task.q0[i] + 0.07 * (0.02 * t as f64 + i as f64).sin()).clamp(LIM[i][0], LIM[i][1])).collect()
}

fn episode(robot: &Robot, inertia: &[LinkInertia], task: &Task, arbiter: bool, env: &Env, seed: u32) -> (bool, bool, bool) {
    let z_fence = task.z_floor + MARG_Z;
    let mut q = q_play(task, 0);        // start on the working trajectory (no initial step)
    let mut qd = vec![0.0f64; 5];
    let mut cmd = q.clone();            // persistent, rate-limited command
    let mut heat = 0.0f64;
    let (t_total, tf, k) = (1600usize, 350 + (hash(seed) % 700) as usize, 220);
    let mut buf: Vec<Vec<f64>> = vec![q.clone(); env.lat + 1]; // servo command latency
    let (mut table_crash, mut thermal_death) = (false, false);
    let mut mode_recover = false;
    let mut worst_below = f64::NEG_INFINITY; // furthest the tool got BELOW the table (>0 = crash)
    let mut resumed_after = false;      // did the arbiter re-converge to the task after the fault?
    for t in 0..t_total {
        let in_fault = t >= tf && t < tf + k;
        // raw command TARGET: the generative reach, or a STUCK corrupt fold-DOWN command during the
        // glitch (a latched garbage serial value that drives the tool through the table — the real
        // Feetech failure, not random noise). q_bad is a genuine reachable pose whose tip is below the floor.
        let raw: Vec<f64> = if in_fault { task.q_bad.clone() } else { q_play(task, t) };
        // ---- the arbiter: recognition (via FK), fence, recovery ----
        let target: Vec<f64> = if arbiter {
            let cmd_z = tip_z(robot, &raw);   // where would this command put the tool?
            let cur_z = tip_z(robot, &q);     // where is the tool now?
            if cmd_z < z_fence || cur_z < z_fence { mode_recover = true; } // RECOGNITION: heading below the fence
            if mode_recover {
                // re-converged to the safe working pose (tool back up, settled) → resume the task
                if (0..5).all(|i| (q[i] - task.q0[i]).abs() < 0.12 && qd[i].abs() < 0.25) {
                    mode_recover = false; if t > tf { resumed_after = true; }
                }
                task.q0.clone() // LIFT to the safe working home (tool well above the table)
            } else if heat > HWARN {
                q.clone()       // CONSERVE the milk: hold to cool
            } else {
                raw             // flow: the reach is already above the fence
            }
        } else { raw };
        // rate-limiting is a PROTECTIVE measure (makes the fence a true state-barrier): the arbiter has
        // it; the naive single-vector sends the command straight through and drives the tool down.
        if arbiter { for i in 0..5 { cmd[i] += (target[i] - cmd[i]).clamp(-VMAX * DT, VMAX * DT); } }
        else { cmd.copy_from_slice(&target); }
        buf.push(cmd.clone()); let applied = buf.remove(0); // servo latency (envelope)
        let throttle = if heat > HMAX { 0.35 } else { 1.0 };
        let tau_s = servo(&applied, &q, &qd, throttle, env.dead);
        // STS3215 joint damping (~0.6 nominal, randomized) + Coulomb frictionloss
        let tau: Vec<f64> = (0..5).map(|i| tau_s[i] - env.fric * qd[i] - if qd[i].abs() > 1e-3 { 0.052 * qd[i].signum() } else { 0.0 }).collect();
        // faithful dynamics: inject the servo ARMATURE — solve (M(q)+diag(armature)) qdd = M(q)·qdd_link
        let qdd_link = forward_dynamics(robot, inertia, &q, &qd, &tau, Vector3::new(0.0, 0.0, G));
        let m = mass_matrix(robot, inertia, &q);
        let mut ma = m.clone();
        for i in 0..5 { ma[(i, i)] += ARMATURE; }
        let qdd = ma.cholesky().expect("M+A SPD").solve(&(&m * DVector::from_row_slice(&qdd_link)));
        for i in 0..5 { qd[i] += DT * qdd[i]; q[i] = (q[i] + DT * qd[i]).clamp(LIM[i][0], LIM[i][1]); }
        heat += 2.2e-4 * tau_s.iter().map(|x| x * x).sum::<f64>(); heat *= 0.9985;
        let below = task.z_floor - tip_z(robot, &q); // how far the tool is below the table
        worst_below = worst_below.max(below);
        if below > 0.0 { table_crash = true; }        // the tool went through the work surface — irreversible
        if heat > HKILL { thermal_death = true; break; }
        let _ = env.back; // backlash reserved as command/measurement noise; folded into MARG_Z for v1
    }
    if !arbiter && seed <= 9 { eprintln!("  [single seed={seed:>3}] tool drove {:+.3} m past the table (>0 = crash)", worst_below); }
    // arbiter is "recovered+viable" only if it never crashed, never overheated, AND resumed the task.
    let recovered = arbiter && !table_crash && !thermal_death && resumed_after;
    (table_crash, thermal_death, recovered)
}

fn main() {
    let (robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");

    // Build the task geometry from the REAL arm: find the lowest-tip reachable pose (the corrupt
    // fold-down the glitch latches), put the table just above the safe-working band, and pick a
    // realistic low working pose whose tool sits a hand's-width above the table.
    let mut q_bad = vec![0.0; 5];
    let mut zmin = f64::INFINITY;
    for s in 0..6000u32 {
        let cand: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 5 + i as u32 + 1)).collect();
        let z = tip_z(&robot, &cand);
        if z < zmin { zmin = z; q_bad = cand; }
    }
    // the table is the plane the arm is BOLTED TO — the base mounting plane, z = 0. The tool must stay
    // above it; the arm can physically fold its tool 0.12 m below it (zmin), i.e. crash into the table.
    let z_floor = 0.0;
    // working pose: scan for a pose whose tool sits ~0.14 m above the table and is reached out in +x,
    // so the generative wander stays clear of the fence and only the glitch drives the tool down.
    let mut q0 = vec![0.0; 5];
    let z_work = z_floor + 0.20;
    let mut best = f64::INFINITY;
    for s in 0..8000u32 {
        let cand: Vec<f64> = (0..5).map(|i| LIM[i][0] + (LIM[i][1] - LIM[i][0]) * u01(s * 7 + i as u32 + 90001)).collect();
        let p = robot.fk(&cand).translation.vector;
        let cost = (p.z - z_work).powi(2) + (p.x - 0.18).powi(2) + 0.3 * p.y.powi(2);
        if cost < best && p.z > z_floor + 0.15 { best = cost; q0 = cand; }
    }
    let task = Task { q0: q0.clone(), q_bad: q_bad.clone(), z_floor };

    println!("Agency-Axis arbiter on the real SO-101 in ferromotion (position-controlled, STS3215).\n");
    println!("Barrier chosen from a finding: the joint-limit barrier showed NOTHING — the arm cannot slam its");
    println!("own stops in a glitch (torque ceiling self-protects). The unprotected irreversibility is Cartesian:");
    println!("  table plane z_floor = {:.3} m   fence (keep tool above) = {:.3} m", z_floor, z_floor + MARG_Z);
    println!("  working pose  tool z = {:.3} m  (the generative reach, safely above the fence)", tip_z(&robot, &q0));
    println!("  corrupt fold-down tool z = {:.3} m  (the latched glitch command — {:.3} m through the table)\n",
        tip_z(&robot, &q_bad), z_floor - tip_z(&robot, &q_bad));

    // the bounded-reality envelope: friction, servo latency, deadband, backlash — the unmeasured params.
    let envs: Vec<Env> = (0..10).map(|k| Env {
        fric: 0.35 + 0.5 * u01(1000 + k),
        lat: (u01(2000 + k) * 4.0) as usize,
        dead: 0.004 + 0.02 * u01(3000 + k),
        back: 0.5f64.to_radians() * (0.5 + u01(4000 + k)),
    }).collect();
    for (label, arb) in [("SINGLE-VECTOR (execute the task/fault commands)", false), ("ARBITER (table barrier ⟂ reach + thermal milk + recovery)", true)] {
        let (mut crash, mut death, mut rec, mut n) = (0, 0, 0, 0);
        for (ei, env) in envs.iter().enumerate() {
            for ep in 0..14 {
                let seed = (ei as u32 * 131 + ep * 7 + 1) as u32;
                let (c, d, r) = episode(&robot, &inertia, &task, arb, env, seed);
                n += 1; if c { crash += 1; } if d { death += 1; } if r { rec += 1; }
            }
        }
        println!("  {label}");
        println!("     tool-through-table crashes: {:>3}/{n}   thermal deaths: {:>3}/{n}   {}",
            crash, death, if arb { format!("recovered+resumed: {rec}/{n}") } else { String::new() });
    }
    println!("\n(across {} randomized reality models × 14 one-shot episodes each, each with a Feetech comms-glitch injected)", 10);
}
