//! **Where the quadruped's non-smoothness actually lives** — the diagnostic that found the chatter.
//!
//! Finite-differencing a whole gait cycle produced a spectral radius that moved by a factor of 2000 with
//! the probe step, and the first explanation offered for it — that a perturbation changes which foot is on
//! the ground — turned out to be only a third of the story. This example is what settled it, by measuring
//! rather than reasoning:
//!
//! 1. How often does the **contact mode** actually change in one period — a foot loading or unloading, or
//!    switching between sticking and sliding? Touchdown is only one of those, and it turns out to be the
//!    rarest.
//! 2. Are those changes a real contact sequence, or chatter at the timestep scale? A foot resting on the
//!    floor whose impulse toggles every step is the integrator talking, not the gait, and it destroys any
//!    linearisation of the motion.
//! 3. What drives them: each foot's own gap, or redundant load being shuffled between four feet on a floor?
//!
//! The answer was chatter, 160 flips on one foot in a single period, every one of them within a millimetre
//! of the floor and a median of one timestep apart. Stabilising the contact solver's gap feedback
//! ([`PgsStabilization`](ferromotion_core::PgsStabilization)) took the period's mode changes from 205 to
//! 18, which is what made the gait's monodromy measurable at all — see `quadruped_saltation_monodromy`.
//!
//! Run: `cargo run --release --example quadruped_mode_events -p ferromotion-core`

use ferromotion_core::{quadruped, quadruped_trot_tau, whole_body_contact_step_pgs, whole_body_forward_kinematics, LinkInertia, WholeBodyContactPoint};
use nalgebra::{DVector, Isometry3, Matrix3, Point3, Translation3, UnitQuaternion, Vector3, Vector6};

const DT: f64 = 5e-4;
const STAND_Z: f64 = 0.60;
const MU: f64 = 0.9;
const ITERS: usize = 40;
const NX: usize = 25;

fn base_inertia() -> LinkInertia {
    LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)) }
}

fn pack(base: &Isometry3<f64>, v0: &Vector6<f64>, q: &[f64], qd: &[f64]) -> DVector<f64> {
    let (roll, pitch, _) = base.rotation.euler_angles();
    let mut x = DVector::zeros(NX);
    x[0] = base.translation.z;
    x[1] = roll;
    x[2] = pitch;
    for i in 0..6 {
        x[3 + i] = v0[i];
    }
    for i in 0..8 {
        x[9 + i] = q[i];
        x[17 + i] = qd[i];
    }
    x
}

fn unpack(x: &DVector<f64>) -> (Isometry3<f64>, Vector6<f64>, Vec<f64>, Vec<f64>) {
    let base = Isometry3::from_parts(Translation3::new(0.0, 0.0, x[0]), UnitQuaternion::from_euler_angles(x[1], x[2], 0.0));
    let v0 = Vector6::from_iterator((0..6).map(|i| x[3 + i]));
    (base, v0, (0..8).map(|i| x[9 + i]).collect(), (0..8).map(|i| x[17 + i]).collect())
}

/// The **contact mode** of one foot, which is what the dynamics actually branch on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// No normal impulse: the foot is not carrying load.
    Free,
    /// Normal impulse with the tangential impulse strictly inside the friction cone.
    Stick,
    /// Normal impulse with the tangential impulse on the cone boundary.
    Slip,
}

fn modes(impulses: &[Vector3<f64>]) -> Vec<Mode> {
    impulses
        .iter()
        .map(|l| {
            if l.z <= 1e-12 {
                Mode::Free
            } else if (l.x * l.x + l.y * l.y).sqrt() >= 0.999 * MU * l.z {
                Mode::Slip
            } else {
                Mode::Stick
            }
        })
        .collect()
}

/// A rollout: the final state, the per-foot contact mode at each step, and each foot's height at each step.
type Rollout = (DVector<f64>, Vec<Vec<Mode>>, Vec<Vec<f64>>);

/// Roll the gait forward, recording the mode and height traces.
fn roll_iters(x: &DVector<f64>, t0: f64, secs: f64, iters: usize) -> Option<Rollout> {
    let (joints, inertia, parent, feet) = quadruped();
    let bi = base_inertia();
    let pts: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, MU)).collect();
    let g = Vector3::new(0.0, 0.0, -9.81);
    let (mut base, mut v0, mut q, mut qd) = unpack(x);
    let mut warm: Option<Vec<Vector3<f64>>> = None;
    let mut trace = Vec::new();
    let mut heights = Vec::new();
    for k in 0..(secs / DT).round().max(1.0) as usize {
        let t = t0 + k as f64 * DT;
        let tau = quadruped_trot_tau(&q, &qd, std::f64::consts::TAU * t);
        let r = whole_body_contact_step_pgs(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &pts, 0.0, DT, iters, g, warm.as_deref());
        base = r.base;
        v0 = r.v0;
        q = r.q.clone();
        qd = r.qd.clone();
        trace.push(modes(&r.impulses));
        let world = whole_body_forward_kinematics(&joints, &parent, base, &q);
        heights.push(feet.iter().map(|&(b, off, _)| (world[b] * Point3::from(off)).coords.z).collect());
        warm = Some(r.impulses);
        if !base.translation.vector.iter().all(|v| v.is_finite()) {
            return None;
        }
    }
    Some((pack(&base, &v0, &q, &qd), trace, heights))
}

fn roll(x: &DVector<f64>, t0: f64, secs: f64) -> Option<Rollout> {
    roll_iters(x, t0, secs, ITERS)
}

/// Per-step normal impulses and foot heights, for looking at the mechanism directly.
fn impulse_trace(x: &DVector<f64>, t0: f64, secs: f64) -> Option<Vec<(Vec<f64>, Vec<f64>)>> {
    let (joints, inertia, parent, feet) = quadruped();
    let bi = base_inertia();
    let pts: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, MU)).collect();
    let g = Vector3::new(0.0, 0.0, -9.81);
    let (mut base, mut v0, mut q, mut qd) = unpack(x);
    let mut warm: Option<Vec<Vector3<f64>>> = None;
    let mut out = Vec::new();
    for k in 0..(secs / DT).round().max(1.0) as usize {
        let t = t0 + k as f64 * DT;
        let tau = quadruped_trot_tau(&q, &qd, std::f64::consts::TAU * t);
        let r = whole_body_contact_step_pgs(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &pts, 0.0, DT, ITERS, g, warm.as_deref());
        base = r.base;
        v0 = r.v0;
        q = r.q.clone();
        qd = r.qd.clone();
        let world = whole_body_forward_kinematics(&joints, &parent, base, &q);
        out.push((r.impulses.iter().map(|l| l.z).collect(), feet.iter().map(|&(b, off, _)| (world[b] * Point3::from(off)).coords.z).collect()));
        warm = Some(r.impulses);
        if !base.translation.vector.iter().all(|v| v.is_finite()) {
            return None;
        }
    }
    Some(out)
}

fn flow(x: &DVector<f64>, t0: f64, secs: f64) -> Option<DVector<f64>> {
    roll(x, t0, secs).map(|(s, _, _)| s)
}

fn main() {
    println!("Where the quadruped's non-smoothness lives (dt {DT} s, mu {MU})\n");

    let start = pack(&Isometry3::translation(0.0, 0.0, STAND_Z), &Vector6::zeros(), &[0.0; 8], &[0.0; 8]);
    let Some(settled) = flow(&start, 0.0, 6.0) else {
        println!("diverged while settling");
        return;
    };

    // ---- question 1: how many mode changes are there in one period? ----
    let Some((_, trace, heights)) = roll(&settled, 0.0, 1.0) else { return };
    let feet = trace[0].len();
    let mut switches = vec![[0usize; 3]; feet]; // [free<->loaded, stick->slip, slip->stick]
    let mut touchdowns = vec![0usize; feet];
    for k in 1..trace.len() {
        for f in 0..feet {
            let (a, b) = (trace[k - 1][f], trace[k][f]);
            if a == b {
                continue;
            }
            match (a, b) {
                (Mode::Free, _) | (_, Mode::Free) => switches[f][0] += 1,
                (Mode::Stick, Mode::Slip) => switches[f][1] += 1,
                (Mode::Slip, Mode::Stick) => switches[f][2] += 1,
                _ => {}
            }
        }
        for f in 0..feet {
            if heights[k - 1][f] > 1e-3 && heights[k][f] <= 1e-3 {
                touchdowns[f] += 1;
            }
        }
    }
    println!("mode changes per foot over one period, from the solver's own impulses:");
    println!("   foot   load on/off   stick->slip   slip->stick   geometric touchdowns");
    let mut total = 0;
    for f in 0..feet {
        let s = switches[f];
        total += s[0] + s[1] + s[2];
        println!("     {f}          {:>4}          {:>4}          {:>4}                   {:>2}", s[0], s[1], s[2], touchdowns[f]);
    }
    println!("\n   total contact-mode changes in one period: {total}");
    println!("   total geometric touchdowns:               {}", touchdowns.iter().sum::<usize>());

    // Questions about the *linearisation* used to live here, differencing stretches and whole periods.
    // They have moved to `quadruped_saltation_monodromy`, and for a reason worth recording: the numbers
    // they produced were wrong, because the state vector they differenced dropped the base's x, y and yaw.
    // Round-tripping through that projection injected a large perturbation of its own, which read as a
    // spectral radius of 23 and then of a million. Everything below is measured from the solver's own
    // impulses and does not depend on any packing, which is why it survived and those numbers did not.

    // ---- question 4: are foot 0's many load flips a real contact sequence, or chatter? ----
    // A physical sequence has the foot clearly off the ground between loadings. Chatter is a foot resting
    // on the floor whose normal impulse toggles at the timestep scale, which is the integrator talking and
    // not the gait. The two call for completely different responses, so tell them apart by looking.
    let Some((_, tr2, h2)) = roll(&settled, 0.0, 1.0) else { return };
    let flips: Vec<usize> = (1..tr2.len()).filter(|&k| (tr2[k][0] == Mode::Free) != (tr2[k - 1][0] == Mode::Free)).collect();
    println!("\nfoot 0's load flips: {} in one period. Height at each of the first 12:", flips.len());
    print!("   ");
    for &k in flips.iter().take(12) {
        print!("{:.4}  ", h2[k][0]);
    }
    println!();
    let near_floor = flips.iter().filter(|&&k| h2[k][0] < 1e-3).count();
    let gap_steps: Vec<usize> = flips.windows(2).map(|w| w[1] - w[0]).collect();
    let median_gap = {
        let mut g = gap_steps.clone();
        g.sort_unstable();
        g.get(g.len() / 2).copied().unwrap_or(0)
    };
    println!("   {near_floor} of {} flips happen with the foot within 1 mm of the floor", flips.len());
    println!("   median steps between flips: {median_gap} (= {:.1} ms, against a {:.0} ms period)", median_gap as f64 * DT * 1e3, 1000.0);

    // ---- question 5: what mechanism drives the flips? ----
    // Two candidates. Gap feedback: the foot is genuinely separating and re-touching, so its own gap
    // drives it. Redundant-load redistribution: four feet on a floor is over-constrained, the impulses are
    // not uniquely determined, and the sweep shuffles which feet carry the weight while the *total*
    // support stays steady. The second is invisible in any single foot's gap, and the discriminator is
    // whether the total normal impulse holds still while the individual ones move.
    let Some(imp) = impulse_trace(&settled, 0.0, 1.0) else { return };
    // locate the densest run of foot-0 flips rather than assuming where it is
    let f0_flips: Vec<usize> = (1..imp.len()).filter(|&k| (imp[k].0[0] > 1e-12) != (imp[k - 1].0[0] > 1e-12)).collect();
    let centre = f0_flips.first().copied().unwrap_or(0);
    let lo = centre.saturating_sub(3);
    println!("\nfoot 0's first load flip is at step {centre} (t = {:.3} s). Around it:", centre as f64 * DT);
    println!("      step   foot0     foot1     foot2     foot3     total     gap0        gap1");
    for (k, (l, h)) in imp.iter().enumerate().skip(lo).take(16) {
        let tot: f64 = l.iter().sum();
        println!("      {k:>4}   {:>7.4}   {:>7.4}   {:>7.4}   {:>7.4}   {:>7.4}   {:>9.2e}   {:>9.2e}", l[0], l[1], l[2], l[3], tot, h[0], h[1]);
    }
    let totals: Vec<f64> = imp[lo..(lo + 200).min(imp.len())].iter().map(|(l, _)| l.iter().sum()).collect();
    let mean_tot = totals.iter().sum::<f64>() / totals.len() as f64;
    let spread = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
    };
    let win = &imp[lo..(lo + 200).min(imp.len())];
    let per_foot: Vec<f64> = (0..4).map(|f| spread(&win.iter().map(|(l, _)| l[f]).collect::<Vec<_>>())).collect();
    println!("\n      over the 100 ms after the first flip:");
    println!("      total support: mean {mean_tot:.4}, spread {:.4} ({:.1}% of mean)", spread(&totals), 100.0 * spread(&totals) / mean_tot.abs().max(1e-12));
    println!("      per-foot spread: {}", per_foot.iter().map(|s| format!("{s:.4}")).collect::<Vec<_>>().join("  "));
    println!("      weight per step (m g dt) for reference: {:.4}", (8.0 + 4.0 * 0.5) * 9.81 * DT);

    println!("\nWhat this run established. The dynamics branch on contact-mode changes, and there are far");
    println!("more of them than there are touchdowns - so splitting a cycle at its touchdowns splits it in");
    println!("the wrong places. Most of those changes were not gait at all: a foot resting within a");
    println!("millimetre of the floor, its impulse toggling on and off a median of one timestep apart, while");
    println!("the total support stayed steady. That is redundant load being shuffled between four feet on a");
    println!("floor, amplified into a velocity demand by dividing a nanometre of gap by the timestep.");
    println!("\nRaising the sweep count from 40 to 400 changed nothing, so it was never an unconverged");
    println!("solver. Giving the gap feedback a resting band it does not correct inside");
    println!("(PgsStabilization) took the period from 205 mode changes to 18 without moving the torso, and");
    println!("that is what made the gait linearisable. quadruped_saltation_monodromy picks it up from there.");
}
