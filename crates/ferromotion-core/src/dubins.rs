//! **Dubins paths** (Dubins, 1957) — the shortest path between two planar poses `(x, y, θ)` for a
//! *forward-only* car with a minimum turning radius. Dubins proved the optimum is always one of six words
//! built from left turns `L`, right turns `R`, and straight segments `S` — `{LSL, RSR, LSR, RSL, RLR,
//! LRL}` — each at maximum curvature or zero. This is the **nonholonomic steering function** a car-like
//! planner needs: RRT/PRM connect states with it, and it gives the exact cost-to-go for kinodynamic
//! search. (Reeds–Shepp adds reverse gears — 48 words — as the natural extension.)
//!
//! Each candidate word has a closed form (Shkel & Lumelsky 2001 / Walker's `dubins`); the shortest feasible
//! one wins. Verified end-to-end by *integrating* the chosen path and checking it lands on the goal pose,
//! plus the length lower bound and the collinear straight-line limit. Pure Rust → WASM-clean.

use std::f64::consts::TAU;

/// Closed form of one Dubins word: `(α, β, d) → (t, p, q)` normalized segment lengths, or `None`.
type WordFn = fn(f64, f64, f64) -> Option<(f64, f64, f64)>;

/// A path segment: left turn, right turn, or straight — at the car's fixed curvature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seg {
    L,
    R,
    S,
}

/// A pose `(x, y, θ)`.
pub type Pose = (f64, f64, f64);

/// A Dubins path: the three segment types, their *normalized* lengths (turn angle in radians / straight
/// distance in radii), the turning radius, and the start pose.
#[derive(Clone, Copy, Debug)]
pub struct DubinsPath {
    pub word: [Seg; 3],
    pub lengths: [f64; 3],
    pub radius: f64,
    pub start: Pose,
}

fn mod2pi(x: f64) -> f64 {
    let r = x.rem_euclid(TAU);
    if r < 0.0 {
        r + TAU
    } else {
        r
    }
}

// Each word: closed form for (t, p, q) in normalized units, or None if infeasible.
fn lsl(a: f64, b: f64, d: f64) -> Option<(f64, f64, f64)> {
    let tmp0 = d + a.sin() - b.sin();
    let p_sq = 2.0 + d * d - 2.0 * (a - b).cos() + 2.0 * d * (a.sin() - b.sin());
    if p_sq < 0.0 {
        return None;
    }
    let tmp1 = (b.cos() - a.cos()).atan2(tmp0);
    Some((mod2pi(-a + tmp1), p_sq.sqrt(), mod2pi(b - tmp1)))
}
fn rsr(a: f64, b: f64, d: f64) -> Option<(f64, f64, f64)> {
    let tmp0 = d - a.sin() + b.sin();
    let p_sq = 2.0 + d * d - 2.0 * (a - b).cos() + 2.0 * d * (b.sin() - a.sin());
    if p_sq < 0.0 {
        return None;
    }
    let tmp1 = (a.cos() - b.cos()).atan2(tmp0);
    Some((mod2pi(a - tmp1), p_sq.sqrt(), mod2pi(-b + tmp1)))
}
fn lsr(a: f64, b: f64, d: f64) -> Option<(f64, f64, f64)> {
    let p_sq = -2.0 + d * d + 2.0 * (a - b).cos() + 2.0 * d * (a.sin() + b.sin());
    if p_sq < 0.0 {
        return None;
    }
    let p = p_sq.sqrt();
    let tmp = (-a.cos() - b.cos()).atan2(d + a.sin() + b.sin()) - (-2.0_f64).atan2(p);
    Some((mod2pi(-a + tmp), p, mod2pi(-mod2pi(b) + tmp)))
}
fn rsl(a: f64, b: f64, d: f64) -> Option<(f64, f64, f64)> {
    let p_sq = -2.0 + d * d + 2.0 * (a - b).cos() - 2.0 * d * (a.sin() + b.sin());
    if p_sq < 0.0 {
        return None;
    }
    let p = p_sq.sqrt();
    let tmp = (a.cos() + b.cos()).atan2(d - a.sin() - b.sin()) - 2.0_f64.atan2(p);
    Some((mod2pi(a - tmp), p, mod2pi(b - tmp)))
}
fn rlr(a: f64, b: f64, d: f64) -> Option<(f64, f64, f64)> {
    let tmp = (6.0 - d * d + 2.0 * (a - b).cos() + 2.0 * d * (a.sin() - b.sin())) / 8.0;
    if tmp.abs() > 1.0 {
        return None;
    }
    let p = mod2pi(TAU - tmp.acos());
    let t = mod2pi(a - (a.cos() - b.cos()).atan2(d - a.sin() + b.sin()) + p / 2.0);
    Some((t, p, mod2pi(a - b - t + p)))
}
fn lrl(a: f64, b: f64, d: f64) -> Option<(f64, f64, f64)> {
    let tmp = (6.0 - d * d + 2.0 * (a - b).cos() + 2.0 * d * (-a.sin() + b.sin())) / 8.0;
    if tmp.abs() > 1.0 {
        return None;
    }
    let p = mod2pi(TAU - tmp.acos());
    let t = mod2pi(-a - (a.cos() - b.cos()).atan2(d + a.sin() - b.sin()) + p / 2.0);
    Some((t, p, mod2pi(mod2pi(b) - a - t + p)))
}

const WORDS: [([Seg; 3], WordFn); 6] = [
    ([Seg::L, Seg::S, Seg::L], lsl),
    ([Seg::R, Seg::S, Seg::R], rsr),
    ([Seg::L, Seg::S, Seg::R], lsr),
    ([Seg::R, Seg::S, Seg::L], rsl),
    ([Seg::R, Seg::L, Seg::R], rlr),
    ([Seg::L, Seg::R, Seg::L], lrl),
];

/// The shortest Dubins path from `start` to `goal` with turning radius `radius`.
pub fn dubins_shortest(start: Pose, goal: Pose, radius: f64) -> Option<DubinsPath> {
    let (dx, dy) = (goal.0 - start.0, goal.1 - start.1);
    let dist = dx.hypot(dy);
    let d = dist / radius;
    let theta = mod2pi(dy.atan2(dx));
    let a = mod2pi(start.2 - theta);
    let b = mod2pi(goal.2 - theta);

    let mut best: Option<DubinsPath> = None;
    for (word, f) in WORDS {
        if let Some((t, p, q)) = f(a, b, d) {
            let total = t + p + q;
            if best.is_none_or(|bp: DubinsPath| total < bp.lengths.iter().sum::<f64>()) {
                best = Some(DubinsPath { word, lengths: [t, p, q], radius, start });
            }
        }
    }
    best
}

impl DubinsPath {
    /// Total path length (real units).
    pub fn length(&self) -> f64 {
        self.radius * self.lengths.iter().sum::<f64>()
    }

    /// The pose at real arc-length `s` along the path (integrating the fixed-curvature segments).
    pub fn sample(&self, s: f64) -> Pose {
        let mut pose = self.start;
        let mut remaining = (s / self.radius).max(0.0); // in normalized units
        for (seg, &len) in self.word.iter().zip(&self.lengths) {
            let step = remaining.min(len);
            pose = advance(pose, *seg, step, self.radius);
            remaining -= step;
            if remaining <= 0.0 {
                break;
            }
        }
        pose
    }

    /// The pose the path ends at (should equal the goal).
    pub fn endpoint(&self) -> Pose {
        self.sample(self.length())
    }
}

/// Advance a pose by a segment of normalized length `len` (turn angle / straight distance in radii).
fn advance((x, y, th): Pose, seg: Seg, len: f64, radius: f64) -> Pose {
    match seg {
        Seg::S => (x + len * radius * th.cos(), y + len * radius * th.sin(), th),
        Seg::L => {
            let nth = th + len;
            (x + radius * (nth.sin() - th.sin()), y + radius * (th.cos() - nth.cos()), nth)
        }
        Seg::R => {
            let nth = th - len;
            (x + radius * (th.sin() - nth.sin()), y + radius * (nth.cos() - th.cos()), nth)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn ang_diff(a: f64, b: f64) -> f64 {
        let d = (a - b).rem_euclid(TAU);
        (d).min(TAU - d)
    }

    #[test]
    fn the_shortest_path_actually_reaches_the_goal() {
        // THE ORACLE. Integrate the chosen path and confirm it lands on the goal pose — this validates
        // every closed-form word against the actual car kinematics.
        let radius = 1.5;
        let cases: [(Pose, Pose); 5] = [
            ((0.0, 0.0, 0.0), (4.0, 2.0, PI / 2.0)),
            ((1.0, -1.0, 0.3), (-2.0, 3.0, -1.0)),
            ((0.0, 0.0, 0.0), (0.5, 0.0, PI)), // tight U-turn-ish (needs RLR/LRL)
            ((2.0, 2.0, 1.5), (2.0, -2.0, 0.0)),
            ((0.0, 0.0, PI), (-5.0, 0.5, PI + 0.2)),
        ];
        for (start, goal) in cases {
            let path = dubins_shortest(start, goal, radius).expect("a Dubins path always exists");
            let end = path.endpoint();
            assert!((end.0 - goal.0).hypot(end.1 - goal.1) < 1e-6, "position missed: {:?} vs {:?}", (end.0, end.1), (goal.0, goal.1));
            assert!(ang_diff(end.2, goal.2) < 1e-6, "heading missed: {} vs {}", end.2, goal.2);
        }
    }

    #[test]
    fn the_length_is_at_least_the_straight_line_distance() {
        let radius = 2.0;
        let start = (0.0, 0.0, 0.4);
        let goal = (6.0, -3.0, 1.2);
        let path = dubins_shortest(start, goal, radius).unwrap();
        let euclid = (goal.0 - start.0).hypot(goal.1 - start.1);
        assert!(path.length() >= euclid - 1e-9, "path {} shorter than the straight line {euclid}", path.length());
    }

    #[test]
    fn aligned_far_poses_give_essentially_a_straight_line() {
        // Same heading, goal far ahead along it ⇒ the optimum is ~a straight segment (length ≈ distance).
        let radius = 1.0;
        let start = (0.0, 0.0, 0.0);
        let goal = (20.0, 0.0, 0.0);
        let path = dubins_shortest(start, goal, radius).unwrap();
        assert!((path.length() - 20.0).abs() < 1e-6, "aligned path should be ~straight: {}", path.length());
        assert_eq!(path.word[1], Seg::S, "the middle segment should be straight");
    }

    #[test]
    fn it_returns_the_minimum_over_all_feasible_words() {
        let (radius, start, goal) = (1.2, (0.0, 0.0, 0.5), (3.0, 1.0, -0.5));
        let best = dubins_shortest(start, goal, radius).unwrap();
        let (dx, dy) = (goal.0 - start.0, goal.1 - start.1);
        let d = dx.hypot(dy) / radius;
        let theta = mod2pi(dy.atan2(dx));
        let (a, b) = (mod2pi(start.2 - theta), mod2pi(goal.2 - theta));
        for (_, f) in WORDS {
            if let Some((t, p, q)) = f(a, b, d) {
                assert!(best.lengths.iter().sum::<f64>() <= t + p + q + 1e-9, "a word was shorter than the reported best");
            }
        }
    }
}
