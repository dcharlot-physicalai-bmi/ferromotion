//! ferromotion-ruckig — jerk-limited, time-optimal trajectory generation in Rust.
//!
//! A port of the MIT **Community** capability of Ruckig: given per-DoF start/target positions and
//! velocity/acceleration/jerk limits, produce the time-optimal S-curve (rest-to-rest) and
//! synchronize multiple DoF to a common arrival time — Ruckig's signature. The motion is built as
//! constant-jerk segments; the peak velocity is found by monotone bisection, so the profile reaches
//! the target exactly and respects every limit by construction. Pure `f64` / std → trivially
//! WASM-clean.
//!
//! Not ported (Ruckig **Pro**, closed-source): arbitrary non-rest boundary states, intermediate
//! waypoints, and the tracking interface. Arbitrary-state online retargeting is the next extension.

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
