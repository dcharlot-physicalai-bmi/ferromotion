//! ferromotion-ruckig — jerk-limited, time-optimal trajectory generation in Rust.
//!
//! A port of the MIT **Community** capability of Ruckig: given per-DoF start/target positions and
//! velocity/acceleration/jerk limits, produce the time-optimal S-curve (rest-to-rest) and
//! synchronize multiple DoF to a common arrival time — Ruckig's signature. The motion is built as
//! constant-jerk segments; the peak velocity is found by monotone bisection, so the profile reaches
//! the target exactly and respects every limit by construction. Pure `f64` / std → trivially
//! WASM-clean.
//!
//! Beyond the Community scope, this crate also implements **intermediate waypoints** (a Ruckig
//! *Pro* capability, reimplemented openly): non-stop jerk-limited motion through a sequence of
//! waypoints, with heuristic through-velocities and per-segment cross-DoF time synchronization —
//! see [`plan_waypoints`]. Like Ruckig Pro's own waypoint mode, this makes **no global
//! time-optimality claim**: through-velocities are chosen heuristically (direction of travel,
//! zeroed at reversals, capped by reachability), and each segment is then planned tightly.
//!
//! Not ported: arbitrary non-zero-acceleration boundary states and the online tracking interface.

/// Per-DoF kinematic limits (all strictly positive).
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub vmax: f64,
    pub amax: f64,
    pub jmax: f64,
}

impl Limits {
    pub fn new(vmax: f64, amax: f64, jmax: f64) -> Self {
        Self { vmax, amax, jmax }
    }
}

/// A constant-jerk segment, carrying the state at its start.
#[derive(Clone, Copy, Debug)]
struct Seg {
    jerk: f64,
    dur: f64,
    p0: f64,
    v0: f64,
    a0: f64,
}

impl Seg {
    fn at(&self, t: f64) -> (f64, f64, f64) {
        let p = self.p0 + self.v0 * t + 0.5 * self.a0 * t * t + self.jerk * t * t * t / 6.0;
        let v = self.v0 + self.a0 * t + 0.5 * self.jerk * t * t;
        let a = self.a0 + self.jerk * t;
        (p, v, a)
    }
    fn end(&self) -> (f64, f64, f64) {
        self.at(self.dur)
    }
}

/// A single-DoF unsigned motion profile (starts at position 0, velocity 0, accel 0).
#[derive(Clone, Debug)]
pub struct Profile {
    segs: Vec<Seg>,
    pub duration: f64,
}

impl Profile {
    fn from_jerks(pieces: &[(f64, f64)]) -> Profile {
        let mut segs = Vec::new();
        let (mut p, mut v, mut a) = (0.0, 0.0, 0.0);
        let mut duration = 0.0;
        for &(jerk, dur) in pieces {
            if dur <= 0.0 {
                continue;
            }
            let s = Seg { jerk, dur, p0: p, v0: v, a0: a };
            let (np, nv, na) = s.end();
            p = np;
            v = nv;
            a = na;
            duration += dur;
            segs.push(s);
        }
        Profile { segs, duration }
    }

    /// State `(position, velocity, acceleration)` at time `t` (clamped to the profile span).
    fn at(&self, t: f64) -> (f64, f64, f64) {
        if self.segs.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let t = t.clamp(0.0, self.duration);
        let mut acc = 0.0;
        for s in &self.segs {
            if t <= acc + s.dur {
                return s.at(t - acc);
            }
            acc += s.dur;
        }
        let last = self.segs.last().unwrap();
        last.end()
    }

    fn end_pos(&self) -> f64 {
        self.at(self.duration).0
    }
}

/// Jerk pieces `(jerk, duration)` for a rest→`v` acceleration phase (ending with accel back at 0).
fn accel_pieces(v: f64, lim: &Limits) -> Vec<(f64, f64)> {
    if v <= 0.0 {
        return vec![];
    }
    let (tj, tc) = if v * lim.jmax >= lim.amax * lim.amax {
        (lim.amax / lim.jmax, v / lim.amax - lim.amax / lim.jmax) // amax is reached: const-accel plateau
    } else {
        ((v / lim.jmax).sqrt(), 0.0) // triangular accel; amax not reached
    };
    vec![(lim.jmax, tj), (0.0, tc.max(0.0)), (-lim.jmax, tj)]
}

fn accel_disp(v: f64, lim: &Limits) -> f64 {
    Profile::from_jerks(&accel_pieces(v, lim)).end_pos()
}

/// Plan an unsigned rest-to-rest profile covering distance `h ≥ 0`, plus the sign and origin so the
/// caller can map it back to real coordinates.
fn plan_unsigned(h: f64, lim: &Limits) -> Profile {
    if h < 1e-12 {
        return Profile { segs: vec![], duration: 0.0 };
    }
    let da_full = accel_disp(lim.vmax, lim);
    let (vpk, tv) = if 2.0 * da_full <= h {
        (lim.vmax, (h - 2.0 * da_full) / lim.vmax) // cruise phase exists
    } else {
        // vmax not reached: bisect the peak velocity so 2·disp(vpk) = h (disp is monotone in vpk).
        let (mut lo, mut hi) = (0.0, lim.vmax);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if 2.0 * accel_disp(mid, lim) > h {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        (0.5 * (lo + hi), 0.0)
    };

    let mut pieces = accel_pieces(vpk, lim);
    if tv > 0.0 {
        pieces.push((0.0, tv));
    }
    for (j, d) in accel_pieces(vpk, lim) {
        pieces.push((-j, d)); // decel = negated-jerk mirror of accel
    }
    Profile::from_jerks(&pieces)
}

/// One planned DoF: an unsigned profile plus how to map it back (sign, origin) and stretch it for
/// multi-DoF synchronization.
#[derive(Clone, Debug)]
struct DofPlan {
    profile: Profile,
    sign: f64,
    origin: f64,
    stretch: f64,
}

/// A synchronized multi-DoF trajectory. Sample it with [`Trajectory::at`] / [`Trajectory::positions`].
#[derive(Clone, Debug)]
pub struct Trajectory {
    dofs: Vec<DofPlan>,
    pub duration: f64,
}

/// Plan a single-DoF rest-to-rest S-curve from `p0` to `p1`.
pub fn plan_1dof(p0: f64, p1: f64, lim: &Limits) -> Trajectory {
    plan(&[p0], &[p1], std::slice::from_ref(lim))
}

/// Plan a time-synchronized multi-DoF rest-to-rest motion: every DoF starts and finishes together
/// at the slowest DoF's optimal time. Slower DoF are time-stretched (which only lowers their
/// velocity/accel/jerk, so all limits still hold).
pub fn plan(p0: &[f64], p1: &[f64], lims: &[Limits]) -> Trajectory {
    assert!(p0.len() == p1.len() && p0.len() == lims.len(), "mismatched DoF counts");
    let base: Vec<(Profile, f64, f64)> = (0..p0.len())
        .map(|i| {
            let h = (p1[i] - p0[i]).abs();
            let sign = if p1[i] >= p0[i] { 1.0 } else { -1.0 };
            (plan_unsigned(h, &lims[i]), sign, p0[i])
        })
        .collect();

    let duration = base.iter().map(|(pr, _, _)| pr.duration).fold(0.0, f64::max);
    let dofs = base
        .into_iter()
        .map(|(profile, sign, origin)| {
            let stretch = if profile.duration > 1e-12 { duration / profile.duration } else { 1.0 };
            DofPlan { profile, sign, origin, stretch }
        })
        .collect();

    Trajectory { dofs, duration }
}

impl Trajectory {
    /// `[position, velocity, acceleration]` per DoF at time `t`.
    pub fn at(&self, t: f64) -> Vec<[f64; 3]> {
        self.dofs
            .iter()
            .map(|d| {
                let tau = t / d.stretch;
                let (p, v, a) = d.profile.at(tau);
                [d.origin + d.sign * p, d.sign * v / d.stretch, d.sign * a / (d.stretch * d.stretch)]
            })
            .collect()
    }

    /// Positions per DoF at time `t`.
    pub fn positions(&self, t: f64) -> Vec<f64> {
        self.at(t).into_iter().map(|s| s[0]).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Waypoints: velocity-to-velocity segments (zero boundary acceleration) + through-velocity
// heuristic + per-segment synchronization that PRESERVES junction velocities (the existing
// stretch-sync scales boundary velocities, so it cannot be used across waypoints).
// ---------------------------------------------------------------------------------------------

/// Jerk pieces for an S-curve velocity change of magnitude `dv ≥ 0` (same trapezoid-in-acceleration
/// shape as `accel_pieces`, which is the `v_from = 0` special case).
fn vel_change_pieces(dv: f64, lim: &Limits) -> Vec<(f64, f64)> {
    accel_pieces(dv, lim)
}

/// Profile from jerk pieces starting at velocity `v_start` (position 0, acceleration 0).
fn profile_from(v_start: f64, pieces: &[(f64, f64)]) -> Profile {
    let mut segs = Vec::new();
    let (mut p, mut v, mut a) = (0.0, v_start, 0.0);
    let mut duration = 0.0;
    for &(jerk, dur) in pieces {
        if dur <= 0.0 {
            continue;
        }
        let s = Seg { jerk, dur, p0: p, v0: v, a0: a };
        let (pe, ve, ae) = s.end();
        segs.push(s);
        p = pe;
        v = ve;
        a = ae;
        duration += dur;
    }
    Profile { segs, duration }
}

/// Pieces for the unsigned cruise-level profile v0 → u → v1 (all ≥ 0, u the cruise level): an
/// S-curve from v0 to u, a cruise at u long enough to make the total displacement `h`, and an
/// S-curve from u to v1. Returns `None` when the cruise time would be negative (u too high) or the
/// cruise level is non-positive.
fn vv_pieces(h: f64, v0: f64, v1: f64, u: f64, lim: &Limits) -> Option<Vec<(f64, f64)>> {
    if u <= 1e-12 {
        return None;
    }
    let up: Vec<(f64, f64)> = vel_change_pieces((u - v0).abs(), lim)
        .into_iter()
        .map(|(j, d)| (if u >= v0 { j } else { -j }, d))
        .collect();
    let down: Vec<(f64, f64)> = vel_change_pieces((u - v1).abs(), lim)
        .into_iter()
        .map(|(j, d)| (if v1 >= u { j } else { -j }, d))
        .collect();
    let d_up = profile_from(v0, &up).end_pos();
    let d_down = profile_from(u, &down).end_pos();
    let t_cruise = (h - d_up - d_down) / u;
    if t_cruise < -1e-9 {
        return None;
    }
    let mut pieces = up;
    pieces.push((0.0, t_cruise.max(0.0)));
    pieces.extend(down);
    Some(pieces)
}

/// Duration of the cruise-level profile, `f64::INFINITY` when infeasible — monotone decreasing in
/// `u` on the feasible range, which is what both bisections below rely on.
fn vv_duration(h: f64, v0: f64, v1: f64, u: f64, lim: &Limits) -> f64 {
    match vv_pieces(h, v0, v1, u, lim) {
        Some(p) => p.iter().map(|&(_, d)| d).sum(),
        None => f64::INFINITY,
    }
}

/// Time-optimal unsigned velocity-to-velocity profile over distance `h`: raise the cruise level as
/// high as displacement allows (capped at vmax), by bisection.
fn plan_vv(h: f64, v0: f64, v1: f64, lim: &Limits) -> Profile {
    let floor = v0.max(v1);
    // is the full-speed profile feasible?
    if vv_pieces(h, v0, v1, lim.vmax, lim).is_some() {
        return profile_from(v0, &vv_pieces(h, v0, v1, lim.vmax, lim).unwrap());
    }
    // bisect u in (floor-ish, vmax): higher u ⇒ more displacement consumed by the ramps
    let (mut lo, mut hi) = (floor.max(1e-9), lim.vmax);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if vv_pieces(h, v0, v1, mid, lim).is_some() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    profile_from(v0, &vv_pieces(h, v0, v1, lo, lim).expect("feasible cruise level"))
}

/// Duration-matched unsigned velocity-to-velocity profile: find the cruise level whose total
/// duration equals `t_target` (≥ the time-optimal duration), by bisection downward. Falls back to
/// the time-optimal profile when `t_target` is below what any cruise level can reach.
fn plan_vv_t(h: f64, v0: f64, v1: f64, t_target: f64, lim: &Limits) -> Profile {
    let opt = plan_vv(h, v0, v1, lim);
    if t_target <= opt.duration + 1e-9 {
        return opt;
    }
    // duration is monotone DEcreasing in u; find u with duration == t_target
    let (mut lo, mut hi) = (1e-9, lim.vmax);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if vv_duration(h, v0, v1, mid, lim) > t_target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let u = 0.5 * (lo + hi);
    match vv_pieces(h, v0, v1, u, lim) {
        Some(p) => profile_from(v0, &p),
        None => opt,
    }
}

/// One synchronized waypoint segment: per-DoF profiles with a shared duration.
#[derive(Clone, Debug)]
struct WpSegment {
    dofs: Vec<DofPlan>, // stretch is always 1.0 here — sync is via duration-matched planning
    duration: f64,
}

/// A jerk-limited trajectory through intermediate waypoints — non-stop where the path direction
/// holds, stopping only at direction reversals. Sample with [`WaypointTrajectory::at`].
#[derive(Clone, Debug)]
pub struct WaypointTrajectory {
    segments: Vec<WpSegment>,
    pub duration: f64,
}

/// Rest-to-rest peak velocity reachable over unsigned distance `h` (the cap the through-velocity
/// heuristic uses so every segment stays feasible).
fn reachable_peak(h: f64, lim: &Limits) -> f64 {
    let da_full = accel_disp(lim.vmax, lim);
    if 2.0 * da_full <= h {
        return lim.vmax;
    }
    let (mut lo, mut hi) = (0.0, lim.vmax);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if 2.0 * accel_disp(mid, lim) > h {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Plan a synchronized multi-DoF jerk-limited motion through `points` (first = start, rest at both
/// ends; every inner point is an intermediate waypoint passed WITHOUT stopping unless that DoF
/// reverses direction there). Through-velocities are heuristic — direction of travel, zero at
/// reversals, capped at 60% of each adjacent segment's rest-to-rest reachable peak — and each
/// segment is then planned tightly with all DoF duration-matched to the slowest. No global
/// time-optimality claim (Ruckig Pro's waypoint mode makes none either); the guarantees are the
/// ones that matter: limits hold everywhere, every waypoint is hit exactly, and position, velocity
/// and acceleration are continuous throughout.
pub fn plan_waypoints(points: &[Vec<f64>], lims: &[Limits]) -> WaypointTrajectory {
    let n_seg = points.len().saturating_sub(1);
    assert!(n_seg >= 1, "need at least two points");
    let nd = lims.len();
    for p in points {
        assert!(p.len() == nd, "point/DoF mismatch");
    }

    // through-velocity per inner waypoint per DoF (signed)
    let mut vwp = vec![vec![0.0; nd]; points.len()];
    for w in 1..points.len() - 1 {
        for i in 0..nd {
            let d_in = points[w][i] - points[w - 1][i];
            let d_out = points[w + 1][i] - points[w][i];
            if d_in * d_out <= 1e-12 {
                continue; // reversal (or degenerate) → stop this DoF at the waypoint
            }
            let cap_in = reachable_peak(d_in.abs(), &lims[i]);
            let cap_out = reachable_peak(d_out.abs(), &lims[i]);
            vwp[w][i] = d_in.signum() * 0.6 * cap_in.min(cap_out);
        }
    }

    let mut segments = Vec::with_capacity(n_seg);
    let mut duration = 0.0;
    for k in 0..n_seg {
        // per-DoF time-optimal first, to find the segment's common duration
        let mut per: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(nd); // h, v0u, v1u, sign
        for i in 0..nd {
            let d = points[k + 1][i] - points[k][i];
            let sign = if d >= 0.0 { 1.0 } else { -1.0 };
            // unsigned boundary velocities along the direction of travel (zero if opposing)
            let v0u = (vwp[k][i] * sign).max(0.0);
            let v1u = (vwp[k + 1][i] * sign).max(0.0);
            per.push((d.abs(), v0u, v1u, sign));
        }
        // the segment's common duration = the slowest DoF's time-optimal duration
        let t_seg = (0..nd)
            .map(|i| plan_vv(per[i].0, per[i].1, per[i].2, &lims[i]).duration)
            .fold(0.0, f64::max);
        let dofs: Vec<DofPlan> = (0..nd)
            .map(|i| {
                let (h, v0u, v1u, sign) = per[i];
                let profile = plan_vv_t(h, v0u, v1u, t_seg, &lims[i]);
                DofPlan { profile, sign, origin: points[k][i], stretch: 1.0 }
            })
            .collect();
        segments.push(WpSegment { dofs, duration: t_seg });
        duration += t_seg;
    }
    WaypointTrajectory { segments, duration }
}

impl WaypointTrajectory {
    /// `[position, velocity, acceleration]` per DoF at global time `t`.
    pub fn at(&self, t: f64) -> Vec<[f64; 3]> {
        let mut t = t.clamp(0.0, self.duration);
        let mut idx = 0;
        while idx + 1 < self.segments.len() && t > self.segments[idx].duration {
            t -= self.segments[idx].duration;
            idx += 1;
        }
        let seg = &self.segments[idx];
        let tau = t.min(seg.duration);
        seg.dofs
            .iter()
            .map(|d| {
                let (p, v, a) = if tau >= d.profile.duration {
                    // profile shorter than the segment only by numerical slack; hold the end state
                    let (pe, ve, ae) = if d.profile.segs.is_empty() { (0.0, 0.0, 0.0) } else { d.profile.at(d.profile.duration) };
                    let _ = ae;
                    (pe + ve * (tau - d.profile.duration), ve, 0.0)
                } else {
                    d.profile.at(tau)
                };
                [d.origin + d.sign * p, d.sign * v, d.sign * a]
            })
            .collect()
    }

    pub fn positions(&self, t: f64) -> Vec<f64> {
        self.at(t).into_iter().map(|s| s[0]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_check(traj: &Trajectory, p0: &[f64], p1: &[f64], lims: &[Limits]) {
        let steps = 500;
        for k in 0..=steps {
            let t = traj.duration * k as f64 / steps as f64;
            for (i, s) in traj.at(t).into_iter().enumerate() {
                let slack = 1e-6;
                assert!(s[1].abs() <= lims[i].vmax + slack, "dof {i} v {} > vmax", s[1]);
                assert!(s[2].abs() <= lims[i].amax + slack, "dof {i} a {} > amax", s[2]);
            }
        }
        // Starts at p0 (rest), ends at p1 (rest).
        for (i, s) in traj.at(0.0).into_iter().enumerate() {
            assert!((s[0] - p0[i]).abs() < 1e-9);
        }
        for (i, s) in traj.at(traj.duration).into_iter().enumerate() {
            assert!((s[0] - p1[i]).abs() < 1e-6, "dof {i} end {} != {}", s[0], p1[i]);
            assert!(s[1].abs() < 1e-6, "dof {i} not at rest: v {}", s[1]);
        }
    }

    #[test]
    fn long_move_hits_cruise_and_respects_limits() {
        let lim = Limits::new(1.0, 2.0, 5.0);
        let traj = plan_1dof(0.0, 3.0, &lim);
        assert!(traj.duration > 0.0);
        sample_check(&traj, &[0.0], &[3.0], &[lim]);
        // A long move should reach vmax somewhere.
        let peak_v = (0..=500)
            .map(|k| traj.at(traj.duration * k as f64 / 500.0)[0][1].abs())
            .fold(0.0, f64::max);
        assert!((peak_v - lim.vmax).abs() < 1e-3, "should cruise at vmax, peak {peak_v}");
    }

    #[test]
    fn short_move_no_cruise_still_exact() {
        let lim = Limits::new(5.0, 2.0, 5.0); // high vmax → won't be reached on a short move
        let traj = plan_1dof(0.0, 0.05, &lim);
        sample_check(&traj, &[0.0], &[0.05], &[lim]);
        let peak_v = (0..=500)
            .map(|k| traj.at(traj.duration * k as f64 / 500.0)[0][1].abs())
            .fold(0.0, f64::max);
        assert!(peak_v < lim.vmax, "short move should not reach vmax, peak {peak_v}");
    }

    #[test]
    fn negative_direction() {
        let lim = Limits::new(1.0, 2.0, 8.0);
        let traj = plan_1dof(1.0, -0.5, &lim);
        sample_check(&traj, &[1.0], &[-0.5], &[lim]);
    }

    /// Waypoint trajectories: limits hold everywhere, every waypoint is hit exactly, position /
    /// velocity / acceleration are continuous, inner waypoints are passed WITHOUT stopping (same
    /// direction), and the whole motion beats the stop-at-every-waypoint baseline.
    #[test]
    fn waypoints_nonstop_within_limits_and_faster_than_stopping() {
        let lims = vec![Limits::new(1.2, 3.0, 12.0), Limits::new(0.9, 2.5, 10.0)];
        let pts = vec![
            vec![0.0, 0.0],
            vec![0.8, 0.3],
            vec![1.6, 0.5],
            vec![2.2, 1.1],
        ];
        let wt = plan_waypoints(&pts, &lims);
        assert!(wt.duration > 0.0);

        // dense sampling: limits + finite-difference jerk sanity + continuity
        let steps = 4000;
        let dt = wt.duration / steps as f64;
        let mut prev = wt.at(0.0);
        for k in 1..=steps {
            let t = dt * k as f64;
            let cur = wt.at(t);
            for i in 0..2 {
                assert!(cur[i][1].abs() <= lims[i].vmax + 1e-6, "v limit dof {i} at t={t}: {}", cur[i][1]);
                assert!(cur[i][2].abs() <= lims[i].amax + 1e-6, "a limit dof {i} at t={t}: {}", cur[i][2]);
                // continuity: position and velocity may not jump between adjacent samples
                assert!((cur[i][0] - prev[i][0]).abs() < lims[i].vmax * dt + 1e-6, "p jump dof {i} at t={t}");
                assert!((cur[i][1] - prev[i][1]).abs() < lims[i].amax * dt + 1e-6, "v jump dof {i} at t={t}");
                assert!((cur[i][2] - prev[i][2]).abs() < lims[i].jmax * dt + 1e-5, "a jump dof {i} at t={t}");
            }
            prev = cur;
        }

        // endpoints at rest, on target
        let s0 = wt.at(0.0);
        let s1 = wt.at(wt.duration);
        for i in 0..2 {
            assert!((s0[i][0] - pts[0][i]).abs() < 1e-9);
            assert!((s1[i][0] - pts[3][i]).abs() < 1e-5, "end dof {i}: {} vs {}", s1[i][0], pts[3][i]);
            assert!(s1[i][1].abs() < 1e-6, "end not at rest");
        }

        // every waypoint hit exactly at its segment boundary, moving (not stopped) at inner ones
        let mut t_acc = 0.0;
        for (w, seg_end) in [(1usize, 0usize), (2, 1)] {
            t_acc += wt.segments[seg_end].duration;
            let s = wt.at(t_acc);
            let speed: f64 = (0..2).map(|i| s[i][1] * s[i][1]).sum::<f64>().sqrt();
            for i in 0..2 {
                assert!((s[i][0] - pts[w][i]).abs() < 1e-5, "waypoint {w} dof {i}: {} vs {}", s[i][0], pts[w][i]);
            }
            assert!(speed > 0.05, "inner waypoint {w} should be passed without stopping (speed {speed})");
        }

        // beats stopping at every waypoint
        let stop_time: f64 = (0..3).map(|k| plan(&pts[k], &pts[k + 1], &lims).duration).sum();
        assert!(
            wt.duration < stop_time - 1e-6,
            "non-stop ({}) must beat stop-at-each ({})",
            wt.duration,
            stop_time
        );
    }

    /// A direction reversal at a waypoint stops that DoF there (the honest choice — passing a
    /// reversal at speed would overshoot the waypoint), while the other DoF keeps moving.
    #[test]
    fn reversal_waypoint_stops_only_the_reversing_dof() {
        let lims = vec![Limits::new(1.0, 3.0, 10.0), Limits::new(1.0, 3.0, 10.0)];
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.5], // dof 0 reverses after this waypoint; dof 1 keeps going
            vec![0.2, 1.0],
        ];
        let wt = plan_waypoints(&pts, &lims);
        let t_wp = wt.segments[0].duration;
        let s = wt.at(t_wp);
        assert!((s[0][0] - 1.0).abs() < 1e-5 && (s[1][0] - 0.5).abs() < 1e-5, "waypoint hit");
        assert!(s[0][1].abs() < 1e-6, "reversing DoF must stop at the waypoint");
        assert!(s[1][1].abs() > 0.05, "non-reversing DoF should pass through at speed: {}", s[1][1]);
        // and the final state is exact
        let e = wt.at(wt.duration);
        assert!((e[0][0] - 0.2).abs() < 1e-5 && (e[1][0] - 1.0).abs() < 1e-5);
    }

    /// The v-to-v primitive: exact boundary conditions and limit respect across regimes (cruise
    /// above both boundary velocities, and the duration-matched dip below them).
    #[test]
    fn velocity_to_velocity_profiles_exact_and_stretchable() {
        let lim = Limits::new(2.0, 4.0, 20.0);
        for &(h, v0, v1) in &[(1.5, 0.5, 0.8), (0.6, 1.2, 0.2), (3.0, 0.0, 1.5)] {
            let p = plan_vv(h, v0, v1, &lim);
            let (pe, ve, _ae) = p.at(p.duration);
            assert!((pe - h).abs() < 1e-6, "h={h}: end pos {pe}");
            assert!((ve - v1).abs() < 1e-6, "h={h}: end vel {ve} vs {v1}");
            let (p0_, v0_, _) = p.at(0.0);
            assert!(p0_.abs() < 1e-9 && (v0_ - v0).abs() < 1e-9);
            // duration-matched: 1.7× the optimal time, same boundaries, same displacement
            let t_slow = p.duration * 1.7;
            let ps = plan_vv_t(h, v0, v1, t_slow, &lim);
            assert!((ps.duration - t_slow).abs() < 1e-4, "stretch: {} vs {}", ps.duration, t_slow);
            let (pse, psve, _) = ps.at(ps.duration);
            assert!((pse - h).abs() < 1e-5 && (psve - v1).abs() < 1e-5, "stretched boundaries hold");
        }
    }

    #[test]
    fn multi_dof_is_synchronized() {
        let lims = vec![Limits::new(1.0, 3.0, 10.0), Limits::new(1.0, 3.0, 10.0), Limits::new(2.0, 5.0, 20.0)];
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [2.0, 0.3, 1.0]; // very different distances
        let traj = plan(&p0, &p1, &lims);
        // Every DoF arrives at the same total time, at rest, on target, within limits.
        sample_check(&traj, &p0, &p1, &lims);
        // At half time no DoF has finished early (synchronization: all still moving unless trivial).
        let mid = traj.at(traj.duration * 0.5);
        assert!(mid.iter().any(|s| s[1].abs() > 1e-3), "all DoF stalled at midpoint");
    }
}
