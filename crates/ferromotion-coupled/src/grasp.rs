//! **Rigid↔FEM grasp** — a rigid parallel-jaw gripper closing on a soft (FEM) object. The two jaws
//! are half-planes at `x = center.x ± half_width` with inward normals `±x`, driven kinematically.
//! Every FEM vertex that penetrates a jaw feels a penalty **normal** force (pushing it out of the jaw)
//! plus regularized **Coulomb friction** opposing its slip relative to the jaw. Friction is what makes
//! it a *grasp*: to lift the object against gravity the available friction `2·μ·N` must exceed its
//! weight `m·g` — squeeze hard enough and the object is retained; squeeze too little and it slips.
//! This is the deformable-object counterpart to the rigid grasp metrics, and closes the last leg of
//! the perceive → reach → grasp loop. Pure `nalgebra` → WASM-clean.

use ferromotion_fem::FemSim;
use nalgebra::Vector3;

/// A parallel-jaw gripper grasping a soft body.
pub struct GraspFemSim {
    pub fem: FemSim,
    /// Gripper centre (the midpoint between the jaws); moved by `jaw_vel` each step.
    pub center: Vector3<f64>,
    /// Half the jaw opening — each jaw sits at `center.x ± half_width`.
    pub half_width: f64,
    /// Prescribed gripper velocity (drives the lift; also sets the friction slip reference).
    pub jaw_vel: Vector3<f64>,
    pub k_contact: f64,
    /// Coulomb friction coefficient at the jaw ↔ object contact.
    pub mu: f64,
    /// Unit closing axis of the jaws in the world: the jaws sit at `center ± half_width·grip_axis`
    /// with inward normals `∓grip_axis`. Defaults to `+x`; set it from a wrist pose for a mounted gripper.
    pub grip_axis: Vector3<f64>,
    /// Finite jaw radius: a vertex is gripped only if its offset *perpendicular* to `grip_axis` is
    /// within `pad` of the centre. Defaults to `∞` (full half-planes — the standalone gripper always
    /// straddles its object). Set it to the object size for a *mounted* gripper that approaches from
    /// afar, so its jaw planes don't clip the object before it arrives.
    pub pad: f64,
    pub gravity: Vector3<f64>,
    pub floor: Option<f64>,
    /// Friction regularization velocity: below this slip speed friction ramps linearly (a stiff,
    /// static-like hold) instead of chattering.
    pub eps_v: f64,
}

impl GraspFemSim {
    pub fn new(mut fem: FemSim, center: Vector3<f64>, half_width: f64, k_contact: f64, mu: f64) -> Self {
        // this integrator owns gravity + floor; silence the sub-sim's own copies
        fem.gravity = Vector3::zeros();
        fem.floor = None;
        GraspFemSim { fem, center, half_width, jaw_vel: Vector3::zeros(), k_contact, mu, grip_axis: Vector3::new(1.0, 0.0, 0.0), pad: f64::INFINITY, gravity: Vector3::new(0.0, 0.0, -9.81), floor: None, eps_v: 1e-3 }
    }

    /// Total mass of the soft body (all vertices share `fem.mass`).
    pub fn object_mass(&self) -> f64 {
        self.fem.mass * self.fem.n_verts() as f64
    }

    /// Centre of mass of the soft body.
    pub fn object_centroid(&self) -> Vector3<f64> {
        let n = self.fem.n_verts().max(1) as f64;
        self.fem.x.iter().sum::<Vector3<f64>>() / n
    }

    /// Mean velocity of the soft body.
    pub fn object_velocity(&self) -> Vector3<f64> {
        let n = self.fem.n_verts().max(1) as f64;
        self.fem.v.iter().sum::<Vector3<f64>>() / n
    }

    /// One semi-implicit step of the grasp.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) {
        let dt = self.fem.dt;
        // move the gripper (kinematic)
        self.center += self.jaw_vel * dt;

        let nv = self.fem.n_verts();
        let mut f = self.fem.forces(); // elastic

        // gravity
        for fi in f.iter_mut() {
            *fi += self.fem.mass * self.gravity;
        }

        // jaw contact: normal penalty + Coulomb friction against the jaw's motion. The jaws are the
        // two planes at `center ± half_width` along `grip_axis`; a vertex's position along that axis
        // (relative to the centre) decides which jaw, if any, it has entered.
        let g = self.grip_axis;
        let gamma = 0.5 * (self.k_contact * self.fem.mass).sqrt();
        for i in 0..nv {
            let p = self.fem.x[i];
            let vrel = self.fem.v[i] - self.jaw_vel;
            let s = (p - self.center).dot(&g); // signed distance along the closing axis
            // which jaw (if any) is this vertex inside?
            let contact = if s > self.half_width {
                Some((-g, s - self.half_width)) // past the right jaw: inward normal −g, penetration
            } else if s < -self.half_width {
                Some((g, -self.half_width - s))
            } else {
                None
            };
            // finite jaw: skip vertices beyond the pad radius (perpendicular to the closing axis)
            let contact = contact.filter(|_| {
                if self.pad.is_infinite() {
                    return true;
                }
                let perp = (p - self.center) - s * g;
                perp.norm() <= self.pad
            });
            if let Some((n, pen)) = contact {
                let vn = vrel.dot(&n); // normal component of slip (negative = compressing)
                let n_mag = (self.k_contact * pen - gamma * vn.min(0.0)).max(0.0); // normal force magnitude
                let f_n = n_mag * n;
                // tangential slip and regularized Coulomb friction opposing it
                let v_t = vrel - vn * n;
                let s = v_t.norm();
                let f_t = if s > 1e-12 { -(self.mu * n_mag) * (v_t / (s + self.eps_v)) } else { Vector3::zeros() };
                f[i] += f_n + f_t;
            }
        }

        // optional floor
        if let Some(fz) = self.floor {
            for i in 0..nv {
                let pen = fz - self.fem.x[i].z;
                if pen > 0.0 {
                    let vnf = self.fem.v[i].z.min(0.0);
                    f[i].z += self.k_contact * pen - gamma * vnf;
                }
            }
        }

        // integrate (semi-implicit Euler), pins held
        let inv_m = 1.0 / self.fem.mass;
        let fdamp = self.fem.damping;
        for i in 0..nv {
            if self.fem.pinned[i] {
                self.fem.v[i] = Vector3::zeros();
                continue;
            }
            self.fem.v[i] = (self.fem.v[i] + dt * f[i] * inv_m) * (1.0 - fdamp);
            self.fem.x[i] += dt * self.fem.v[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A soft cube and a gripper centred on it, closing along `grip`. Returns (sim, object
    /// half-extent along that axis).
    fn cube_and_gripper(squeeze: f64, mu: f64, grip: Vector3<f64>) -> (GraspFemSim, f64) {
        // A firm, light little block: stiff enough to stay compact under its own weight while the
        // jaws squeeze it (so the centroid tracks the grip), still visibly deformable.
        let mut fem = FemSim::box_grid(3, 3, 3, 0.07, 0.02, 1.0e4, 6.0e3, 2.0e-4);
        fem.damping = 0.02; // the soft body dissipates its own vibration
        let g = grip.normalize();
        let n = fem.n_verts() as f64;
        let centroid: Vector3<f64> = fem.x.iter().sum::<Vector3<f64>>() / n;
        let smin = fem.x.iter().map(|p| (p - centroid).dot(&g)).fold(f64::INFINITY, f64::min);
        let smax = fem.x.iter().map(|p| (p - centroid).dot(&g)).fold(f64::NEG_INFINITY, f64::max);
        let half = 0.5 * (smax - smin);
        // jaws squeeze `squeeze` metres past each face along the grip axis
        let mut sim = GraspFemSim::new(fem, centroid, half - squeeze, 5.0e4, mu);
        sim.grip_axis = g;
        (sim, half)
    }

    /// **Grasp-retention oracle.** Squeeze the soft cube, then lift the gripper: with enough normal
    /// force, friction carries the object up with the jaws — its centroid rises with the gripper and
    /// it never slips free. The physics that must hold: `2·μ·N ≳ m·g`.
    #[test]
    fn firm_grasp_lifts_the_soft_object() {
        let (mut sim, _half_x) = cube_and_gripper(0.018, 0.9, Vector3::new(1.0, 0.0, 0.0));
        let z0 = sim.object_centroid().z;

        // Phase 1 — settle the squeeze while holding position against gravity.
        for _ in 0..6000 {
            sim.step();
        }
        let z_held = sim.object_centroid().z;
        let held_drop = z0 - z_held;
        eprintln!("grasp: settle drop {held_drop:.4} m (held in place by friction)");
        assert!(held_drop < 0.03, "object slipped while merely held: dropped {held_drop:.3} m");

        // Phase 2 — lift the gripper; the object should ride up with it.
        let lift = 0.12;
        sim.jaw_vel = Vector3::new(0.0, 0.0, 0.4);
        let steps = (lift / 0.4 / sim.fem.dt) as usize;
        let cx0 = sim.center.x;
        for _ in 0..steps {
            sim.step();
        }
        // settle a moment at the top
        sim.jaw_vel = Vector3::zeros();
        for _ in 0..2000 {
            sim.step();
        }
        let z_top = sim.object_centroid().z;
        let rose = z_top - z_held;
        let x_off = (sim.object_centroid().x - sim.center.x).abs();
        eprintln!("grasp: gripper lifted {lift:.3} m, object rose {rose:.3} m, x-offset {x_off:.3}, jaw moved {:.3}", sim.center.x - cx0);
        assert!(rose > 0.7 * lift, "object did not ride up with the gripper: rose {rose:.3} of {lift:.3}");
        assert!(x_off < 0.05, "object slipped out of the jaws laterally: {x_off:.3}");
    }

    /// **Negative control — friction is real.** A barely-touching grip (tiny normal force, so
    /// `2·μ·N < m·g`) cannot hold the object: it slips down under gravity. This is what makes the
    /// positive result meaningful rather than the jaws merely pinning the object geometrically.
    #[test]
    fn weak_grasp_slips() {
        // jaws open slightly WIDER than the object (only the deformed corners graze them)
        let (mut sim, _half_x) = cube_and_gripper(-0.004, 0.5, Vector3::new(1.0, 0.0, 0.0));
        let z0 = sim.object_centroid().z;
        for _ in 0..6000 {
            sim.step();
        }
        let drop = z0 - sim.object_centroid().z;
        eprintln!("weak grasp: object dropped {drop:.3} m under gravity (as it must)");
        assert!(drop > 0.05, "a barely-touching grip should not hold the object, but drop was only {drop:.3}");
    }

    /// **Oriented-jaw oracle.** The same grasp closing along a *different* axis than the default
    /// world-x — here world-y, a face the arm might present after reaching. A mounted gripper arrives
    /// at whatever orientation the arm gives it, so the jaws must hold and lift for a non-default
    /// `grip_axis`. Verifies the general projection path end to end. (A cube only offers full faces
    /// along ±x/±y/±z; an off-face diagonal contacts only corners and — correctly — grips weakly.)
    #[test]
    fn oriented_grasp_lifts_along_a_nondefault_axis() {
        let g = Vector3::new(0.0, 1.0, 0.0); // close along world-y, not the default x
        let (mut sim, _half) = cube_and_gripper(0.018, 0.9, g);
        let z0 = sim.object_centroid().z;
        for _ in 0..6000 {
            sim.step();
        }
        let z_held = sim.object_centroid().z;
        assert!(z0 - z_held < 0.03, "oriented grip did not hold: dropped {:.3}", z0 - z_held);

        sim.jaw_vel = Vector3::new(0.0, 0.0, 0.4);
        for _ in 0..(0.12 / 0.4 / sim.fem.dt) as usize {
            sim.step();
        }
        sim.jaw_vel = Vector3::zeros();
        for _ in 0..2000 {
            sim.step();
        }
        let rose = sim.object_centroid().z - z_held;
        // slip along the (horizontal) grip axis stays bounded — the object doesn't squirt out sideways
        let along = (sim.object_centroid() - sim.center).dot(&g).abs();
        eprintln!("oriented grasp: rose {rose:.3} m along a world-y grip, axis-offset {along:.3}");
        assert!(rose > 0.7 * 0.12, "oriented grip did not lift the object: rose {rose:.3}");
        assert!(along < 0.03, "object slipped along the grip axis: {along:.3}");
    }
}
