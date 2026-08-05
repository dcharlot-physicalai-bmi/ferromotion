//! **What a grasp analysis leaves out** — the on-device lab behind the grasp-and-contact lesson.
//!
//! A grasp gets judged by a number: force closure, or the radius of the largest wrench ball it resists. This lab
//! exists because two things that number hides are measurable, and both of them were surprises when measured.
//!
//! **One: a planar analysis of a spatial grasp is wrong, but not in the way people expect.** The intuition is that
//! coplanar contacts cannot resist out-of-plane wrenches, so their spatial quality should collapse to zero. It does
//! not. A friction cone at a contact whose normal lies in the plane still contains out-of-plane directions, because
//! the tangent `n x t1` leaves the plane. So a coplanar grasp is full rank 6 with a positive wrench ball.
//!
//! Which way the planar number errs turns out to depend on the arrangement rather than being a law: measured here it
//! reads `0.92x` the spatial value at three contacts and `1.10x` at four. The lesson is that the two metrics disagree
//! and neither is a safe proxy for the other, not that one of them is reliably optimistic. What *is* consistent is
//! more useful: the spatial wrench ball does not improve at all as coplanar contacts are added, while the planar
//! number wanders. Adding fingers in the same plane buys nothing in six dimensions.
//!
//! **Two: a grasp that is holding can already be slipping.** Coulomb friction says stuck or sliding, and a real
//! elastic contact says neither: a stick region shrinks from the edge inward as tangential load rises
//! ([Cattaneo-Mindlin](crate::grasp_reality_lab)), so partial slip begins long before the grasp lets go. The stick
//! radius goes as `(1 - Q/muP)^(1/3)`, which means it has already lost a third of its area at about a third of
//! capacity. A controller watching only for gross slip gets no warning; one watching the stick fraction gets a lot.
//!
//! Everything is computed live from `ferromotion-core`'s spatial grasp module and `ferromotion-tactile`'s shear
//! model, against the same contact set the readout describes.

use ferromotion_core::{
    force_closure_q1_planar_subspace, force_closure_q1_spatial, force_closure_soft_spatial, grasp_split, wrench_rank,
    GraspContact3,
};
use ferromotion_tactile::shear::{slip_signal, slip_state, ShearLoad, SlipState};
use nalgebra::{Vector3, Vector6};
use wasm_bindgen::prelude::*;

/// Facets per friction cone. Enough that the polyhedral cone is not the dominant approximation, few enough that the
/// whole sweep stays interactive.
const FACETS: usize = 8;
/// Probe directions for the wrench-ball radius.
const DIRS: usize = 64;
/// Object radius, metres. A graspable sphere.
const RADIUS: f64 = 0.05;

#[wasm_bindgen]
pub struct GraspRealityLab {
    contacts: usize,
    mu: f64,
    /// 0 = all contacts on one great circle (coplanar), 1 = spread over the sphere.
    spread: f64,
    /// Normal load per contact, newtons.
    normal_load: f64,
    /// Tangential load as a fraction of the Coulomb limit `mu * N`.
    load_fraction: f64,
}

#[wasm_bindgen]
impl GraspRealityLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GraspRealityLab {
        GraspRealityLab { contacts: 3, mu: 0.5, spread: 0.0, normal_load: 5.0, load_fraction: 0.35 }
    }

    pub fn set_contacts(&mut self, n: f64) {
        self.contacts = (n.max(1.0) as usize).clamp(1, 8);
    }

    pub fn set_mu(&mut self, mu: f64) {
        self.mu = mu.clamp(0.05, 1.5);
    }

    /// Move the contacts from a single great circle out over the sphere. At `0` they are coplanar, which is the
    /// arrangement the planar-versus-spatial comparison is about.
    pub fn set_spread(&mut self, s: f64) {
        self.spread = s.clamp(0.0, 1.0);
    }

    pub fn set_load_fraction(&mut self, f: f64) {
        self.load_fraction = f.clamp(0.0, 1.2);
    }

    pub fn contacts(&self) -> f64 {
        self.contacts as f64
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn spread(&self) -> f64 {
        self.spread
    }

    pub fn load_fraction(&self) -> f64 {
        self.load_fraction
    }

    /// Contacts on the sphere: evenly spaced in azimuth, tilted out of the equatorial plane by `spread`. Normals point
    /// inward, which is what a fingertip pressing on a sphere does.
    fn contact_set(&self) -> Vec<GraspContact3> {
        (0..self.contacts)
            .map(|i| {
                let az = 2.0 * core::f64::consts::PI * (i as f64) / (self.contacts as f64);
                // Alternate above and below the equator as spread rises, so the set genuinely leaves the plane
                // instead of rotating within it.
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                let el = sign * self.spread * 0.9;
                let (ce, se) = (el.cos(), el.sin());
                let pos = Vector3::new(RADIUS * ce * az.cos(), RADIUS * ce * az.sin(), RADIUS * se);
                let normal = -pos.normalize();
                GraspContact3::hard(pos, normal, self.mu)
            })
            .collect()
    }

    /// Position of contact `i`, for drawing.
    pub fn contact_x(&self, i: usize) -> f64 {
        self.contact_set().get(i).map_or(f64::NAN, |c| c.pos.x)
    }

    pub fn contact_y(&self, i: usize) -> f64 {
        self.contact_set().get(i).map_or(f64::NAN, |c| c.pos.y)
    }

    pub fn contact_z(&self, i: usize) -> f64 {
        self.contact_set().get(i).map_or(f64::NAN, |c| c.pos.z)
    }

    /// **The rank of the wrench space.** The claim to check: coplanar contacts should be rank-deficient. They are not.
    pub fn wrench_rank(&self) -> f64 {
        wrench_rank(&self.contact_set(), FACETS) as f64
    }

    /// The spatial wrench-ball radius: the honest 6-D quality of the grasp.
    pub fn q1_spatial(&self) -> f64 {
        force_closure_q1_spatial(&self.contact_set(), FACETS, DIRS)
    }

    /// The same metric restricted to the planar subspace: what a 2-D analysis would report.
    pub fn q1_planar(&self) -> f64 {
        force_closure_q1_planar_subspace(&self.contact_set(), FACETS, DIRS)
    }

    /// **Planar quality divided by spatial quality.** Deliberately not called an overstatement: which way a planar
    /// analysis errs depends on the arrangement, and on a coplanar sphere grasp it reads *low* (about 0.92), not high.
    /// The finding is that the two metrics disagree and that neither is a safe proxy, not that one is optimistic.
    pub fn planar_ratio(&self) -> f64 {
        let (p, s) = (self.q1_planar(), self.q1_spatial());
        if s > 0.0 { p / s } else { f64::INFINITY }
    }

    /// A smoothed quality, for when the hard metric's zero is uninformative about which way to move.
    pub fn q1_soft(&self) -> f64 {
        force_closure_soft_spatial(&self.contact_set(), FACETS, DIRS, 40.0)
    }

    /// Dimension of the internal-force space: squeezes that produce no net wrench on the object. A grasp with zero
    /// internal dimension cannot preload, so it cannot resist anything by squeezing harder.
    pub fn internal_dim(&self) -> f64 {
        grasp_split(&self.contact_set(), FACETS).internal_dim as f64
    }

    pub fn grasp_rank(&self) -> f64 {
        grasp_split(&self.contact_set(), FACETS).rank as f64
    }

    fn shear_load(&self) -> ShearLoad {
        // Hertzian contact radius for a fingertip-scale patch. Fixed, so the slider isolates the load ratio.
        ShearLoad {
            normal: self.normal_load,
            tangential: self.load_fraction * self.mu * self.normal_load,
            mu: self.mu,
            contact_radius: 3e-3,
        }
    }

    /// **The slip state**: `0 = stuck, 1 = partial slip, 2 = gross slip`. Coulomb friction only has states 0 and 2.
    pub fn slip_code(&self) -> i32 {
        match slip_state(&self.shear_load()) {
            Some(SlipState::Stuck) => 0,
            Some(SlipState::PartialSlip { .. }) => 1,
            Some(SlipState::GrossSlip) => 2,
            None => -1,
        }
    }

    /// Fraction of the contact patch still stuck. Falls as `(1 - Q/muP)^(1/3)`, so it is already well below one when a
    /// Coulomb model still says "stuck".
    pub fn stick_fraction(&self) -> f64 {
        slip_state(&self.shear_load()).map_or(f64::NAN, |s| s.stick_fraction())
    }

    /// The stick *radius* ratio `c/a = (1 - Q/muP)^(1/3)`. The area fraction reported by [`Self::stick_fraction`] is
    /// this squared, and confusing the two is easy: at a tenth of capacity the radius has fallen to 0.966 and the area
    /// to 0.932.
    pub fn stick_radius_ratio(&self) -> f64 {
        match slip_state(&self.shear_load()) {
            Some(SlipState::Stuck) => 1.0,
            Some(SlipState::PartialSlip { stick_radius_ratio, .. }) => stick_radius_ratio,
            Some(SlipState::GrossSlip) => 0.0,
            None => f64::NAN,
        }
    }

    /// Whether the slip signal would warn now, at the given stick-fraction threshold.
    pub fn warns_at(&self, threshold: f64) -> bool {
        let t = self.shear_load().tangential;
        slip_signal(self.normal_load, Vector3::new(t, 0.0, 0.0), self.mu, 3e-3)
            .is_some_and(|s| s.incipient(threshold))
    }

    /// **The load fraction at which a stick-fraction monitor first warns.** Compare against `1.0`, which is where a
    /// Coulomb monitor first warns: the difference is the warning you get for free.
    pub fn warning_load_fraction(&self, threshold: f64) -> f64 {
        let mut probe = GraspRealityLab { ..*self };
        let mut lo = 0.0;
        let mut hi = 1.0;
        // The stick fraction is monotone in the load, so bisection is exact to the returned precision.
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            probe.load_fraction = mid;
            if probe.warns_at(threshold) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Shear traction at radius `r` within the patch, for drawing the pressure profile.
    pub fn traction_at(&self, r: f64) -> f64 {
        ferromotion_tactile::shear::shear_traction(&self.shear_load(), r).unwrap_or(f64::NAN)
    }

    pub fn contact_radius(&self) -> f64 {
        3e-3
    }

    /// The net wrench a unit squeeze produces, so a learner can see a balanced grasp produce nothing.
    pub fn net_wrench_norm(&self) -> f64 {
        let contacts = self.contact_set();
        let n = contacts.len() * FACETS;
        let mags = vec![1.0 / (n as f64); n];
        ferromotion_core::net_wrench(&contacts, FACETS, &mags)
            .map_or(f64::NAN, |w: Vector6<f64>| w.norm())
    }
}

impl Default for GraspRealityLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The refuted claim, pinned.** The intuition is that coplanar contacts cannot resist out-of-plane wrenches, so
    /// their spatial quality collapses to zero. Measured, they are full rank 6 with a positive wrench ball, because the
    /// tangent `n x t1` leaves the plane.
    ///
    /// This test asserts the structural facts and *reports* the planar-to-spatial ratio rather than bounding it. An
    /// earlier version asserted the ratio exceeded one, importing a `1.5x` figure measured on a different contact set;
    /// on this geometry it is `0.92`. The direction is a property of the arrangement, not a law.
    #[test]
    fn coplanar_contacts_are_full_rank_with_a_positive_wrench_ball() {
        let mut lab = GraspRealityLab::new();
        lab.set_spread(0.0);
        for n in [3.0, 4.0, 5.0] {
            lab.set_contacts(n);
            let (rank, s, p) = (lab.wrench_rank(), lab.q1_spatial(), lab.q1_planar());
            eprintln!("{n} coplanar contacts: rank {rank}, Q1 spatial {s:.5}, planar {p:.5}, ratio {:.3}x", lab.planar_ratio());
            assert_eq!(rank, 6.0, "coplanar contacts should still span all six wrench directions");
            assert!(s > 0.0, "spatial Q1 should be positive, not zero: {s}");
            assert!(p > 0.0, "planar Q1 should also be positive: {p}");
            // The two metrics must genuinely disagree, which is the reason not to use one as a proxy for the other.
            assert!((lab.planar_ratio() - 1.0).abs() > 0.02, "the metrics agree too closely to make the point");
        }
    }

    /// Spreading the contacts off a single circle must improve the spatial grasp. This is the direction a designer
    /// actually cares about, and it should be visible in the honest metric.
    #[test]
    fn spreading_the_contacts_improves_the_spatial_grasp() {
        let mut lab = GraspRealityLab::new();
        lab.set_contacts(4.0);
        lab.set_spread(0.0);
        let flat = lab.q1_spatial();
        lab.set_spread(1.0);
        let spread = lab.q1_spatial();
        eprintln!("Q1 spatial: coplanar {flat:.5} -> spread {spread:.5}");
        assert!(spread > flat, "spreading should help: {flat:.5} -> {spread:.5}");
    }

    /// **Adding coplanar contacts buys nothing in 6-D.** The binding direction for a coplanar set is out of the plane,
    /// and it is set by the friction-cone half-angle rather than by how many fingers are on the circle. A designer
    /// reading only the planar metric would see the number move and conclude otherwise.
    #[test]
    fn more_coplanar_contacts_do_not_improve_the_spatial_grasp() {
        let mut lab = GraspRealityLab::new();
        lab.set_spread(0.0);
        let mut spatial = Vec::new();
        let mut planar = Vec::new();
        for n in [3.0, 4.0, 5.0, 6.0, 7.0] {
            lab.set_contacts(n);
            spatial.push(lab.q1_spatial());
            planar.push(lab.q1_planar());
        }
        eprintln!("coplanar 3..7 contacts, Q1 spatial: {spatial:.5?}");
        eprintln!("coplanar 3..7 contacts, Q1 planar:  {planar:.5?}");
        let (lo, hi) = spatial.iter().fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(hi - lo < 1e-9, "the spatial ball should not move with contact count: {lo:.6} to {hi:.6}");
        // ... while the planar number does move, which is exactly the trap.
        let (plo, phi) = planar.iter().fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(phi - plo > 1e-3, "the planar number should visibly move: {plo:.6} to {phi:.6}");
    }

    /// **The early-warning result.** A stick-fraction monitor warns at a fraction of capacity where a Coulomb monitor
    /// is still reporting "stuck".
    #[test]
    fn partial_slip_warns_well_before_gross_slip() {
        let mut lab = GraspRealityLab::new();
        let at = lab.warning_load_fraction(0.65);
        eprintln!("a 65% stick-fraction monitor first warns at {:.3} of Coulomb capacity", at);
        assert!(at < 0.5, "the warning should arrive well before capacity, got {at}");

        // And a Coulomb model reports nothing at that load: it is not gross slip.
        lab.set_load_fraction(at);
        assert_eq!(lab.slip_code(), 1, "the state should be partial slip, not stuck or gross");
        assert!(lab.stick_fraction() < 1.0 && lab.stick_fraction() > 0.0);

        // Past capacity it is gross slip and the stick region is gone.
        lab.set_load_fraction(1.05);
        assert_eq!(lab.slip_code(), 2);
        assert_eq!(lab.stick_fraction(), 0.0);
    }

    /// The stick region has to follow the Cattaneo-Mindlin law, because the whole reason the warning is early is the
    /// shape of that curve. Two quantities, easy to confuse and checked separately: the stick *radius* goes as the cube
    /// root, and the stick *area* as its square.
    #[test]
    fn the_stick_region_follows_the_cattaneo_mindlin_law() {
        let mut lab = GraspRealityLab::new();
        for f in [0.1, 0.3, 0.5, 0.7, 0.9] {
            lab.set_load_fraction(f);
            let want_r = (1.0f64 - f).cbrt();
            let want_a = want_r * want_r;
            let (got_r, got_a) = (lab.stick_radius_ratio(), lab.stick_fraction());
            eprintln!("load {f:.1} of capacity: stick radius {got_r:.4} (law {want_r:.4}), area {got_a:.4} (law {want_a:.4})");
            assert!((got_r - want_r).abs() < 1e-12, "radius at {f}: {got_r} vs {want_r}");
            assert!((got_a - want_a).abs() < 1e-12, "area at {f}: {got_a} vs {want_a}");
        }
        // The early warning comes from the curve's shape: a tenth of the load costs nearly a tenth of the area.
        lab.set_load_fraction(0.1);
        assert!(lab.stick_fraction() < 0.94, "a tenth of capacity already costs 6.8% of the patch");
    }
}
