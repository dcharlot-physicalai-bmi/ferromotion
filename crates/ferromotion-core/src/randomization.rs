//! **Domain randomisation with ranges the data chose** — and a correction about which ranges those are.
//!
//! Nothing in this workspace randomised model parameters: no distribution, no fitted ranges, no schedule. The usual fix
//! is a hand-picked box, plus or minus twenty percent on every mass, and the intended improvement here was to take the
//! ranges from [`identify_with_covariance`](crate::identify_with_covariance) instead — narrow where the data resolved a
//! direction, wide where it did not.
//!
//! **Half of that was wrong, and the measurement says so plainly.** The premise was that the regressor's null space is
//! "model error the data could not rule out", and therefore the prize to randomise over. It is not. For inertial
//! identification the null space is **structural**, not a consequence of which trajectories were sampled. Perturbing the
//! estimate by a norm of `38.665` along it changes joint torques by **nothing measurable**: the test asserts the worst
//! change over every trajectory is `< 1e-6 N m`, and it lands ten orders of magnitude under that. One run's transcript,
//! for scale:
//!
//! ```text
//!   training samples          ~1e-13 N m       (2.903e-13 on this machine, 1.976e-13 on another)
//!   held out, same speeds     ~1e-13           (2.816e-13 / 2.078e-13)
//!   held out, 10x faster      ~1e-12           (6.228e-12 / 3.361e-12)
//!   held out, 100x faster     ~1e-10           (4.945e-10 / 5.045e-10)
//! ```
//!
//! **Do not quote those mantissas as measurements.** They are numerically zero, and the digits move run to run because
//! the 15-dimensional null space is degenerate: which basis SVD returns inside it is not unique, so the perturbation
//! direction — and the rounding noise it drags along — differs between builds. The reproducible facts are the
//! perturbation norm `38.665`, the enforced bound `< 1e-6 N m`, and the conclusion. Those 15 of 30 parameter directions
//! have no observable effect on the joint torques of *any* trajectory, so randomising them buys no robustness — it
//! perturbs coordinates the dynamics cannot see. What it would buy is wasted compute and a misleading sense of coverage.
//!
//! The same measurement explains something about [`identification`](crate::identify_consistent) worth stating directly:
//! `|truth - estimate| = 1.8889` (reproducible) while the largest component along any *identifiable* direction is at the
//! `1e-16` level — again noise, not a figure to quote, and the test enforces `< 1e-9`. Link 0's mass is identified as
//! `0.0000` against a true `1.5000` and the estimate is **dynamically equivalent** for joint torques. Recovering the
//! true inertial parameters is therefore not the goal of identification, and the reason
//! [`identify_consistent`](crate::identify_consistent) matters is not that the fit was wrong — it is that a simulator
//! needing mass for contact, collision or energy gets nonsense from the unconstrained estimate.
//!
//! So the honest support to randomise over has two parts, and neither is the null space:
//!
//! 1. **Estimation uncertainty on the identifiable directions** — [`ParameterDistribution::from_identification`]. This is
//!    the part the data quantifies, and the estimate is unbiased there, so a `k`-sigma support covers the truth at the
//!    expected rate.
//! 2. **Explicitly declared unmodelled effects** — [`ParameterDistribution::with_extra_axes`]. Joint friction, backlash,
//!    an unknown payload. These are where real sim-to-real error lives, and no amount of staring at the regressor
//!    reveals them, because the model does not contain them.

use crate::IdentifiedParams;
use nalgebra::DVector;

/// A distribution over model parameters, shaped by whatever produced it.
#[derive(Clone, Debug)]
pub struct ParameterDistribution {
    /// The nominal parameters.
    pub centre: DVector<f64>,
    /// `(direction, half-width)` pairs. Sampling perturbs the centre along each direction by a uniform draw scaled by
    /// its half-width, so the support is a parallelepiped rather than a box in parameter coordinates.
    pub axes: Vec<(DVector<f64>, f64)>,
    /// Multiplier applied to every half-width — the single knob an [`AdrSchedule`] turns.
    pub scale: f64,
}

impl ParameterDistribution {
    /// **A hand-picked box**: one axis per parameter, each with its own half-width. The baseline to measure against.
    pub fn from_box(centre: DVector<f64>, half_widths: &[f64]) -> Option<ParameterDistribution> {
        if half_widths.len() != centre.len() {
            return None;
        }
        let n = centre.len();
        let axes = (0..n)
            .map(|i| {
                let mut e = DVector::zeros(n);
                e[i] = 1.0;
                (e, half_widths[i].abs())
            })
            .collect();
        Some(ParameterDistribution { centre, axes, scale: 1.0 })
    }

    /// **A distribution the identification chose**, over the directions it actually resolved.
    ///
    /// Each identifiable direction gets a half-width of `sigmas` standard errors. The unidentifiable directions are
    /// deliberately **excluded**: they are structurally unobservable (see the module documentation), so widening them
    /// changes no torque on any trajectory and would only inflate the apparent coverage.
    pub fn from_identification(id: &IdentifiedParams, sigmas: f64) -> Option<ParameterDistribution> {
        if id.phi.len() != id.parameters || sigmas < 0.0 {
            return None;
        }
        Some(ParameterDistribution {
            centre: id.phi.clone(),
            axes: id.identifiable.iter().map(|(v, se)| (v.clone(), sigmas * se)).collect(),
            scale: 1.0,
        })
    }

    /// **Add axes for effects the model does not contain** — joint friction, backlash, an unknown payload mass.
    ///
    /// These are the directions real sim-to-real error lives along, and they cannot be read off a regressor whose model
    /// omits them. Each axis is a parameter-space direction with a half-width; they are appended, so estimation
    /// uncertainty and declared model error compose.
    pub fn with_extra_axes(mut self, axes: &[(DVector<f64>, f64)]) -> Option<ParameterDistribution> {
        for (dir, half) in axes {
            if dir.len() != self.centre.len() || *half < 0.0 {
                return None;
            }
            let norm = dir.norm();
            if norm < 1e-12 {
                return None;
            }
            self.axes.push((dir / norm, *half));
        }
        Some(self)
    }

    /// One sample: the centre perturbed along every axis by an independent uniform draw.
    pub fn sample(&self, rng: &mut Lcg) -> DVector<f64> {
        let mut out = self.centre.clone();
        for (dir, half) in &self.axes {
            out += (self.scale * half * rng.uniform_signed()) * dir;
        }
        out
    }

    /// `sum log(2 * scale * half_width)` — the log-volume of the support, and the quantity a schedule grows. Axes of
    /// zero width are skipped rather than sending it to negative infinity.
    ///
    /// **Comparable only at a fixed axis count.** Volume in `R^15` and volume in `R^16` are not the same kind of
    /// quantity, so adding an axis can lower this number while strictly enlarging the support — an axis of extent below
    /// `1` contributes a negative log. Use it to compare scales of one distribution, not distributions of different
    /// dimension.
    pub fn log_volume(&self) -> f64 {
        self.axes
            .iter()
            .filter(|(_, h)| *h > 0.0)
            .map(|(_, h)| (2.0 * self.scale * h).ln())
            .sum()
    }

    /// Whether `phi` lies inside the support, to `tol` relative on each axis.
    pub fn contains(&self, phi: &DVector<f64>, tol: f64) -> bool {
        if phi.len() != self.centre.len() {
            return false;
        }
        let d = phi - &self.centre;
        // the axes are orthonormal by construction in both constructors, so the coordinate is the projection
        self.axes.iter().all(|(dir, half)| {
            let w = self.scale * half;
            w <= 0.0 || d.dot(dir).abs() <= w * (1.0 + tol)
        })
    }

    pub fn with_scale(&self, scale: f64) -> ParameterDistribution {
        ParameterDistribution { scale: scale.max(0.0), ..self.clone() }
    }
}

/// A small deterministic generator, so a randomisation run is reproducible and a test is not at the mercy of a global
/// seed.
#[derive(Clone, Debug)]
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Lcg {
        Lcg(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform on `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform on `[-1, 1)` — genuinely centred. The pre-existing `lcg` helper in `sysid`'s tests returns `[-1, 0)`,
    /// which as measurement noise gave a DC bias that drove interval coverage to 21% instead of 95%.
    pub fn uniform_signed(&mut self) -> f64 {
        2.0 * self.uniform() - 1.0
    }
}

/// **Automatic domain randomisation**: widen the distribution while the policy is succeeding, narrow it when it is not.
#[derive(Clone, Copy, Debug)]
pub struct AdrSchedule {
    /// Success rate to hold. Above it the support widens, below it the support narrows.
    pub target_success: f64,
    /// Multiplicative step per update.
    pub step: f64,
    pub min_scale: f64,
    pub max_scale: f64,
}

impl Default for AdrSchedule {
    fn default() -> Self {
        AdrSchedule { target_success: 0.7, step: 1.15, min_scale: 0.05, max_scale: 20.0 }
    }
}

impl AdrSchedule {
    /// The next scale given the success rate measured at the current one.
    ///
    /// This is a proportional controller on a monotone-decreasing plant (wider support means lower success), so it has
    /// a fixed point at the target and reaches it from either side. It is **not** guaranteed to be monotone on the way,
    /// and the test below reports the settled value rather than asserting the path.
    pub fn update(&self, scale: f64, measured_success: f64) -> f64 {
        let next = if measured_success > self.target_success { scale * self.step } else { scale / self.step };
        next.clamp(self.min_scale, self.max_scale)
    }

    /// Run the schedule to a settled scale, given something that measures success at a scale.
    pub fn settle(&self, mut scale: f64, rounds: usize, mut success_at: impl FnMut(f64) -> f64) -> (f64, f64) {
        let mut last = 0.0;
        for _ in 0..rounds {
            last = success_at(scale);
            scale = self.update(scale, last);
        }
        (scale, last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{identify_with_covariance, inertial_regressor, inverse_dynamics, params_from_inertia, IdSample, PARAMS_PER_LINK};
    use crate::{from_urdf_full, Robot};
    use nalgebra::Vector3;

    const ARM3: &str = r#"<robot name="a3">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0.1 0.05"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0.001" ixz="0.002" iyy="0.03" iyz="0.0015" izz="0.025"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.25 0 0.05"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0.001" iyy="0.012" iyz="0" izz="0.011"/></inertial></link>
      <link name="l3"><inertial><origin xyz="0.15 0.02 0"/><mass value="0.6"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.006" iyz="0.0005" izz="0.005"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.5 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0.4 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="0.3 0 0"/></joint>
    </robot>"#;

    fn fixture(count: usize, seed: u64) -> (Robot, Vec<IdSample>, DVector<f64>, Vector3<f64>) {
        fixture_full(count, seed, 1.0, 0.0)
    }

    fn fixture_scaled(count: usize, seed: u64, scale: f64) -> (Robot, Vec<IdSample>, DVector<f64>, Vector3<f64>) {
        fixture_full(count, seed, scale, 0.0)
    }

    fn fixture_noisy(count: usize, seed: u64, noise: f64) -> (Robot, Vec<IdSample>, DVector<f64>, Vector3<f64>) {
        fixture_full(count, seed, 1.0, noise)
    }

    fn fixture_full(count: usize, seed: u64, scale: f64, noise: f64) -> (Robot, Vec<IdSample>, DVector<f64>, Vector3<f64>) {
        let (robot, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let truth = params_from_inertia(&inertia);
        let mut rng = Lcg::new(seed);
        let samples = (0..count)
            .map(|_| {
                let q: Vec<f64> = (0..3).map(|_| scale * rng.uniform_signed()).collect();
                let qd: Vec<f64> = (0..3).map(|_| scale * rng.uniform_signed()).collect();
                let qdd: Vec<f64> = (0..3).map(|_| scale * rng.uniform_signed()).collect();
                let mut tau = inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
                for v in &mut tau {
                    *v += noise * (rng.uniform_signed() + rng.uniform_signed() + rng.uniform_signed());
                }
                IdSample { q, qd, qdd, tau }
            })
            .collect();
        (robot, samples, truth, g)
    }

    /// Gram-Schmidt a null-space basis out of the identifiable span.
    fn null_basis(id: &IdentifiedParams) -> Vec<DVector<f64>> {
        let n = id.parameters;
        let mut out: Vec<DVector<f64>> = Vec::new();
        for i in 0..n {
            let mut e = DVector::zeros(n);
            e[i] = 1.0;
            for (v, _) in &id.identifiable {
                let p = e.dot(v);
                e -= p * v;
            }
            for v in &out {
                let p = e.dot(v);
                e -= p * v;
            }
            if e.norm() > 1e-6 {
                out.push(e.normalize());
            }
        }
        out
    }

    /// **The correction, as a test.** The regressor's null space is structurally unobservable: a large perturbation
    /// along it changes joint torques on no trajectory, including ones far outside the training distribution. So it is
    /// not "uncertainty the data left open" and randomising it buys nothing.
    #[test]
    fn the_null_space_is_structurally_unobservable_not_merely_unfitted() {
        let (robot, train, truth, g) = fixture(60, 7);
        let id = identify_with_covariance(&robot, &train, g, 1e-8).expect("identifies");
        let null = null_basis(&id);
        eprintln!("rank {} of {} parameters, null-space dimension {}", id.rank, id.parameters, null.len());
        assert_eq!(null.len(), id.parameters - id.rank);

        // perturb hard along every null direction
        let mut phi = id.phi.clone();
        for (k, v) in null.iter().enumerate() {
            phi += (2.0 + k as f64) * v;
        }
        let pert = (&phi - &id.phi).norm();
        eprintln!("   perturbation norm {pert:.3}");
        assert!(pert > 30.0, "the perturbation is large, not marginal");

        eprintln!("   {:>26}  {:>14}", "trajectory distribution", "worst |dtau|");
        let mut worst_overall = 0.0f64;
        for (name, scale, seed) in [("training samples", 1.0, 0u64), ("held out, same speeds", 1.0, 4242), ("held out, 10x faster", 10.0, 999), ("held out, 100x faster", 100.0, 31337)] {
            let (_, set, _, _) = if seed == 0 { fixture(60, 7) } else { fixture_scaled(40, seed, scale) };
            let mut worst = 0.0f64;
            for s in &set {
                let y = inertial_regressor(&robot, &s.q, &s.qd, &s.qdd, g);
                let pred = &y * &phi;
                for i in 0..3 {
                    worst = worst.max((pred[i] - s.tau[i]).abs());
                }
            }
            eprintln!("   {name:>26}  {worst:>14.3e}");
            worst_overall = worst_overall.max(worst);
        }
        assert!(worst_overall < 1e-6, "a 38-unit perturbation is invisible everywhere: {worst_overall:.3e}");
        eprintln!("   Invisible everywhere. Those {} directions are not uncertainty, they are coordinates the", null.len());
        eprintln!("   dynamics cannot see - so randomising them is wasted compute and false coverage.");

        // and the consequence for identification: the whole discrepancy lives in that invisible subspace
        let d = &truth - &id.phi;
        let along_identifiable = id.identifiable.iter().map(|(v, _)| d.dot(v).abs()).fold(0.0, f64::max);
        eprintln!("\n   |truth - estimate| = {:.4}, largest identifiable component {along_identifiable:.3e}", d.norm());
        eprintln!("   link 0 mass: true {:.4}, identified {:.4} - dynamically equivalent for joint torques.", truth[0], id.phi[0]);
        assert!(along_identifiable < 1e-9, "the estimate is right about everything the data could see");
        assert!(d.norm() > 1.0, "and differs by more than a kilogram in a direction that does not matter here");
    }

    /// **The part the data does quantify.** On the identifiable directions the estimate is unbiased, so a `k`-sigma
    /// support covers the truth's component at about the expected rate. That is the statistically justified support.
    #[test]
    fn a_k_sigma_support_covers_the_identifiable_directions() {
        let (mut hits, mut total) = (0usize, 0usize);
        for trial in 0..80u64 {
            let (robot, train, truth, g) = fixture_noisy(60, 500 + trial * 13, 0.02);
            let Some(id) = identify_with_covariance(&robot, &train, g, 1e-8) else { continue };
            let dist = ParameterDistribution::from_identification(&id, 2.0).expect("builds");
            for (v, half) in dist.axes.iter().take(5) {
                let coord = (&truth - &id.phi).dot(v);
                total += 1;
                hits += usize::from(coord.abs() <= *half);
            }
        }
        let rate = hits as f64 / total as f64;
        eprintln!("2-sigma support over identifiable directions: {hits}/{total} covered = {:.1}%", 100.0 * rate);
        assert!(total > 300, "enough samples to judge: {total}");
        assert!(rate > 0.90, "a 2-sigma support should cover most of the time, got {:.3}", rate);
    }

    /// Declared unmodelled effects are the other half of the support, and they compose with the estimation uncertainty.
    #[test]
    fn declared_model_error_composes_with_estimation_uncertainty() {
        let (robot, train, _t, g) = fixture(60, 7);
        let id = identify_with_covariance(&robot, &train, g, 1e-8).expect("identifies");
        let base = ParameterDistribution::from_identification(&id, 2.0).expect("builds");
        let n = id.parameters;
        // an unknown payload: extra mass on the last link
        let mut payload = DVector::zeros(n);
        payload[2 * PARAMS_PER_LINK] = 1.0;
        let combined = base.clone().with_extra_axes(&[(payload, 0.25)]).expect("appends");
        eprintln!("estimation only: {} axes, log-volume {:.3}", base.axes.len(), base.log_volume());
        eprintln!("plus a declared payload axis: {} axes, log-volume {:.3}", combined.axes.len(), combined.log_volume());
        assert_eq!(combined.axes.len(), base.axes.len() + 1, "the support gains a dimension");
        // NOT a volume comparison: 15-dimensional and 16-dimensional volumes are not the same kind of quantity, and an
        // axis of extent below 1 lowers a sum of logs while strictly enlarging the support. My first version of this
        // test asserted the log-volume grew, which is simply the wrong statement.
        let (a, b) = (base.with_scale(2.0).log_volume(), base.log_volume());
        assert!(a > b, "within one distribution, scaling up does raise the log-volume: {a:.3} vs {b:.3}");

        // and the payload axis actually changes the dynamics, unlike a null direction
        let mut rng = Lcg::new(5);
        let mut worst = 0.0f64;
        for _ in 0..20 {
            let phi = combined.sample(&mut rng);
            for s in train.iter().take(5) {
                let y = inertial_regressor(&robot, &s.q, &s.qd, &s.qdd, g);
                let pred = &y * &phi;
                for i in 0..3 {
                    worst = worst.max((pred[i] - s.tau[i]).abs());
                }
            }
        }
        eprintln!("   worst torque change from sampling the declared axis: {worst:.3e} N m");
        assert!(worst > 1e-3, "a declared payload is observable, which is why it is worth randomising: {worst:.3e}");
    }

    /// The schedule settles at its target success rate from either side.
    #[test]
    fn the_schedule_settles_at_its_target() {
        let sched = AdrSchedule { target_success: 0.7, step: 1.10, min_scale: 1e-3, max_scale: 1e3 };
        let plant = |scale: f64| (1.0 / (1.0 + (scale / 2.0).powi(2))).clamp(0.0, 1.0);
        eprintln!("{:>12}  {:>10}  {:>10}", "start scale", "settled", "success");
        for start in [0.05, 0.5, 2.0, 50.0] {
            let (settled, success) = sched.settle(start, 400, plant);
            eprintln!("{start:>12.2}  {settled:>10.4}  {success:>10.4}");
            assert!((success - sched.target_success).abs() < 0.06, "settles near the target from {start}: {success:.4}");
        }
        eprintln!("   the exact fixed point is {:.4}", 2.0 * (3.0f64 / 7.0).sqrt());
    }

    /// Log-volume grows with scale as `scale^n`, which is what makes it usable as the schedule's objective.
    #[test]
    fn log_volume_is_monotone_in_scale() {
        let dist = ParameterDistribution::from_box(DVector::from_element(6, 1.0), &[0.1; 6]).expect("builds");
        let mut prev = f64::NEG_INFINITY;
        for scale in [0.25, 0.5, 1.0, 2.0, 4.0] {
            let v = dist.with_scale(scale).log_volume();
            eprintln!("scale {scale:>5.2}: log-volume {v:>9.4}");
            assert!(v > prev);
            prev = v;
        }
        let (a, b) = (dist.with_scale(1.0).log_volume(), dist.with_scale(2.0).log_volume());
        assert!((b - a - 6.0 * 2.0f64.ln()).abs() < 1e-12, "the volume scales as scale^n");
    }

    /// Degenerate input is refused.
    #[test]
    fn degenerate_input_is_refused() {
        assert!(ParameterDistribution::from_box(DVector::from_element(4, 0.0), &[0.1; 3]).is_none(), "width count must match");
        let (robot, train, _t, g) = fixture(20, 3);
        let id = identify_with_covariance(&robot, &train, g, 1e-8).unwrap();
        assert!(ParameterDistribution::from_identification(&id, -1.0).is_none(), "a negative sigma count");
        let d = ParameterDistribution::from_identification(&id, 1.0).unwrap();
        let n = id.parameters;
        assert!(d.clone().with_extra_axes(&[(DVector::zeros(n), 0.1)]).is_none(), "a zero-length direction");
        assert!(d.clone().with_extra_axes(&[(DVector::zeros(n + 1), 0.1)]).is_none(), "a direction of the wrong width");
        let point = ParameterDistribution::from_box(DVector::from_element(3, 1.0), &[0.0; 3]).unwrap();
        assert!(point.contains(&DVector::from_element(3, 1.0), 1e-12));
        assert_eq!(point.log_volume(), 0.0, "no positive-width axes, so the log-volume is an empty sum");
    }
}
