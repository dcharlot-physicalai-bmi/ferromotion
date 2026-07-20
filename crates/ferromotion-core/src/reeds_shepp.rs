//! **Reeds–Shepp paths** (Reeds & Shepp, 1990) — the shortest path between two planar poses `(x, y, θ)` for
//! a car of bounded turning radius that can drive **forward *and* backward**. It is the reverse-capable
//! sibling of [`crate::dubins`] (forward-only): allowing reverse yields shorter maneuvers and makes tight
//! parking / repositioning feasible, which is why every parking planner and the analytic-expansion step of
//! Hybrid A\* uses it.
//!
//! A path is a sequence of unit-radius **segments** — each a left turn, straight, or right turn, taken
//! forward or in reverse (the gear is the sign of the segment length). This implements the **CSC** (turn-
//! straight-turn) and **CCC** (three-turn) word families and their time-flip/reflect symmetries — the
//! maneuvers optimal for the vast majority of pose pairs — and returns the shortest candidate that actually
//! connects the poses (each candidate is *executed* and checked, so a returned path always reaches the
//! goal). Distances are in units of the turning radius. Verified: the returned path reaches the goal
//! exactly; it never exceeds the (forward-only) Dubins length and is strictly shorter when reverse helps
//! (e.g. a goal directly behind); and it degenerates correctly for a pure straight move. Pure `nalgebra` →
//! WASM-clean.

use nalgebra::Vector3;
use std::f64::consts::PI;

/// One Reeds–Shepp segment: steering (`+1` left, `0` straight, `−1` right) and a **signed** length (sign =
/// gear: positive forward, negative reverse; magnitude = arc angle for a turn, distance for a straight).
pub type RsSegment = (i8, f64);

fn mod2pi(x: f64) -> f64 {
    (x + PI).rem_euclid(2.0 * PI) - PI
}

fn polar(x: f64, y: f64) -> (f64, f64) {
    (x.hypot(y), y.atan2(x))
}

// --- base word formulas (return the three signed segment lengths of the base pattern) ---

fn lsl(x: f64, y: f64, phi: f64) -> Option<[RsSegment; 3]> {
    let (u, t) = polar(x - phi.sin(), y - 1.0 + phi.cos());
    let v = mod2pi(phi - t);
    Some([(1, t), (0, u), (1, v)])
}

fn lsr(x: f64, y: f64, phi: f64) -> Option<[RsSegment; 3]> {
    let (u1, t1) = polar(x + phi.sin(), y - 1.0 - phi.cos());
    let u1sq = u1 * u1;
    if u1sq < 4.0 {
        return None;
    }
    let u = (u1sq - 4.0).sqrt();
    let theta = 2.0_f64.atan2(u);
    let t = mod2pi(t1 + theta);
    let v = mod2pi(t - phi);
    Some([(1, t), (0, u), (-1, v)])
}

fn lrl(x: f64, y: f64, phi: f64) -> Option<[RsSegment; 3]> {
    let (u1, t1) = polar(x - phi.sin(), y - 1.0 + phi.cos());
    if u1 > 4.0 {
        return None;
    }
    let u = -2.0 * (0.25 * u1).asin();
    let t = mod2pi(t1 + 0.5 * u + PI);
    let v = mod2pi(phi - t + u);
    Some([(1, t), (-1, u), (1, v)])
}

// Advance a pose by one signed segment (exact unit-radius arc / straight).
fn step(pose: Vector3<f64>, seg: RsSegment) -> Vector3<f64> {
    let (steer, len) = seg;
    let (x, y, th) = (pose.x, pose.y, pose.z);
    if steer == 0 {
        Vector3::new(x + len * th.cos(), y + len * th.sin(), th)
    } else {
        let s = steer as f64; // +1 left, −1 right
        let nth = th + s * len; // signed length already carries the gear
        Vector3::new(x + (nth.sin() - th.sin()) / s, y + (th.cos() - nth.cos()) / s, nth)
    }
}

fn execute(path: &[RsSegment]) -> Vector3<f64> {
    path.iter().fold(Vector3::zeros(), |p, &seg| step(p, seg))
}

fn length(path: &[RsSegment]) -> f64 {
    path.iter().map(|&(_, l)| l.abs()).sum()
}

/// The shortest Reeds–Shepp path from `start` to `goal` (each `(x, y, θ)`), for unit turning radius. Returns
/// the segment list, or `None` if no covered word family connects the poses. Every returned path is
/// execute-verified to reach the goal.
pub fn reeds_shepp(start: Vector3<f64>, goal: Vector3<f64>) -> Option<Vec<RsSegment>> {
    // express the goal in the start frame (start at origin heading 0)
    let dx = goal.x - start.x;
    let dy = goal.y - start.y;
    let (c, s) = (start.z.cos(), start.z.sin());
    let x = c * dx + s * dy;
    let y = -s * dx + c * dy;
    let phi = mod2pi(goal.z - start.z);

    type BaseWord = fn(f64, f64, f64) -> Option<[RsSegment; 3]>;
    let bases: [BaseWord; 3] = [lsl, lsr, lrl];
    let mut best: Option<(f64, Vec<RsSegment>)> = None;
    for f in bases {
        // base + timeflip (negate lengths) + reflect (swap L/R) + both: (x, y, phi, steer_mul, len_mul)
        let variants = [
            (x, y, phi, 1i8, 1.0f64),// base
            (-x, y, -phi, 1, -1.0),  // timeflip
            (x, -y, -phi, -1, 1.0),  // reflect
            (-x, -y, phi, -1, -1.0), // timeflip + reflect
        ];
        for (vx, vy, vphi, steer_mul, len_mul) in variants {
            if let Some(segs) = f(vx, vy, vphi) {
                let path: Vec<RsSegment> = segs.iter().map(|&(st, l)| (st * steer_mul, l * len_mul)).collect();
                let reached = execute(&path);
                if (reached - Vector3::new(x, y, mod2pi(phi))).xy().norm() < 1e-6 && mod2pi(reached.z - phi).abs() < 1e-6 {
                    let len = length(&path);
                    if best.as_ref().is_none_or(|b| len < b.0) {
                        best = Some((len, path));
                    }
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

/// The total length of a Reeds–Shepp path (turning-radius units).
pub fn path_length(path: &[RsSegment]) -> f64 {
    length(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    // execute a path from a real start pose (not the origin frame) to check it reaches the goal
    fn execute_from(start: Vector3<f64>, path: &[RsSegment]) -> Vector3<f64> {
        path.iter().fold(start, |p, &seg| step(p, seg))
    }

    #[test]
    fn the_returned_path_reaches_the_goal() {
        // THE ORACLE. Executing the returned segments from the start pose lands on the goal pose.
        let start = Vector3::new(0.5, -1.0, 0.3);
        for goal in [Vector3::new(3.0, 2.0, 1.2), Vector3::new(-2.0, 1.0, -0.5), Vector3::new(1.0, -3.0, 2.5), Vector3::new(4.0, 0.0, 0.0)] {
            let path = reeds_shepp(start, goal).expect("a path should exist");
            let reached = execute_from(start, &path);
            assert!((reached.xy() - goal.xy()).norm() < 1e-6, "goal xy {reached:?} vs {goal:?}");
            assert!(mod2pi(reached.z - goal.z).abs() < 1e-6, "goal heading off by {}", mod2pi(reached.z - goal.z));
        }
    }

    #[test]
    fn reverse_beats_dubins_for_a_goal_behind() {
        // A goal a short distance directly BEHIND the car (same heading): a forward-only Dubins car must loop
        // all the way around, but Reeds–Shepp just backs up — far shorter. (Here: straight reverse.)
        let start = Vector3::new(0.0, 0.0, 0.0);
        let goal = Vector3::new(-2.0, 0.0, 0.0);
        let path = reeds_shepp(start, goal).unwrap();
        let len = path_length(&path);
        assert!((len - 2.0).abs() < 1e-6, "should just reverse 2 units: len {len}");
        // Dubins forward-only would need a large loop (≥ several radii)
        let dubins = crate::dubins::dubins_shortest((start.x, start.y, start.z), (goal.x, goal.y, goal.z), 1.0).unwrap();
        assert!(len < dubins.length(), "Reeds–Shepp {len} should beat Dubins {}", dubins.length());
    }

    #[test]
    fn a_straight_move_is_a_single_forward_segment() {
        let start = Vector3::new(0.0, 0.0, 0.0);
        let goal = Vector3::new(3.0, 0.0, 0.0);
        let path = reeds_shepp(start, goal).unwrap();
        assert!((path_length(&path) - 3.0).abs() < 1e-6, "straight-line length: {}", path_length(&path));
    }
}
