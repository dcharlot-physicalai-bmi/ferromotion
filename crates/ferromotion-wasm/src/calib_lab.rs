//! **Calibration lab** — the rig behind the "the model meets reality" lesson. It drives the real
//! [`ferromotion_learn::calib`] pipeline: a 2-DOF arm whose TRUE masses and joint friction are
//! hidden, a WRONG starting model (±35% off), and recorded excitation torques. The reader presses
//! *Calibrate* and watches exact dual-number gradients through RNEA pull every parameter onto its
//! true value — the torque-prediction error collapsing by orders of magnitude as it happens. This
//! is gradient-based real-to-sim: the single most validated use of differentiable dynamics.

use ferromotion_core::{Iso, Joint, LinkInertia, Robot};
use ferromotion_learn::calib::{calibrate, excite, CalibSample, CalibSpec, JointFriction};
use nalgebra::{Matrix3, Translation3, UnitQuaternion, Vector3};
use wasm_bindgen::prelude::*;

const G: [f64; 3] = [0.0, 0.0, -9.81];

fn arm() -> Robot {
    // planar 2-link arm: both joints about y, links extending in +z
    let mk = |z: f64| Iso::from_parts(Translation3::new(0.0, 0.0, z), UnitQuaternion::identity());
    Robot {
        joints: vec![Joint::revolute(mk(0.05), Vector3::y()), Joint::revolute(mk(0.30), Vector3::y())],
        ee_offset: Iso::identity(),
    }
}

fn truth() -> (Vec<LinkInertia>, Vec<JointFriction>) {
    let inertia = vec![
        LinkInertia {
            mass: 2.4,
            com: Vector3::new(0.0, 0.0, 0.15),
            inertia: Matrix3::from_diagonal(&Vector3::new(0.045, 0.045, 0.008)),
        },
        LinkInertia {
            mass: 1.3,
            com: Vector3::new(0.0, 0.0, 0.12),
            inertia: Matrix3::from_diagonal(&Vector3::new(0.020, 0.020, 0.004)),
        },
    ];
    let friction = vec![
        JointFriction { coulomb: 0.45, viscous: 0.20 },
        JointFriction { coulomb: 0.30, viscous: 0.12 },
    ];
    (inertia, friction)
}

fn wrong_start() -> (Vec<LinkInertia>, Vec<JointFriction>) {
    let (mut inertia, mut friction) = truth();
    inertia[0].mass *= 1.35;
    inertia[1].mass *= 0.65;
    friction[0].coulomb *= 0.5;
    friction[0].viscous *= 1.8;
    friction[1].coulomb *= 1.7;
    friction[1].viscous *= 0.45;
    (inertia, friction)
}

#[wasm_bindgen]
pub struct CalibLab {
    robot: Robot,
    samples: Vec<CalibSample>,
    inertia: Vec<LinkInertia>,
    friction: Vec<JointFriction>,
    rms0: f64,
    rms: f64,
    iters: u32,
}

#[wasm_bindgen]
impl CalibLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CalibLab {
        let robot = arm();
        let (true_inertia, true_friction) = truth();
        let samples = excite(&robot, &true_inertia, &true_friction, G, 160, 0.0);
        let (inertia, friction) = wrong_start();
        let mut lab = CalibLab { robot, samples, inertia, friction, rms0: 0.0, rms: 0.0, iters: 0 };
        lab.rms = lab.measure_rms();
        lab.rms0 = lab.rms;
        lab
    }

    fn measure_rms(&self) -> f64 {
        // rms via a zero-iteration calibrate call would be wasteful; compute directly
        use ferromotion_core::gendyn::GenModel;
        let model = GenModel::<f64>::from_robot(&self.robot, &self.inertia, G);
        let (mut se, mut cnt) = (0.0, 0.0);
        for s in &self.samples {
            let mut tau = model.rnea(&s.q, &s.qd, &s.qdd);
            for j in 0..2 {
                tau[j] += self.friction[j].coulomb
                    * (s.qd[j] / ferromotion_learn::calib::COULOMB_SMOOTH).tanh()
                    + self.friction[j].viscous * s.qd[j];
                se += (tau[j] - s.tau[j]).powi(2);
                cnt += 1.0;
            }
        }
        (se / cnt).sqrt()
    }

    /// Run `n` gradient iterations of the real calibration (masses + friction), continuing from
    /// the current estimate. Returns the RMS torque error after.
    pub fn calibrate(&mut self, n: u32) -> f64 {
        let spec = CalibSpec {
            fit_mass: true,
            fit_com: false,
            fit_inertia_diag: false,
            fit_friction: true,
            iters: n as usize,
            lr: 0.05,
        };
        let rep = calibrate(&self.robot, &self.inertia, &self.friction, &self.samples, G, spec);
        self.inertia = rep.inertia;
        self.friction = rep.friction;
        self.rms = rep.rms_after;
        self.iters += n;
        self.rms
    }

    pub fn reset(&mut self) {
        let (inertia, friction) = wrong_start();
        self.inertia = inertia;
        self.friction = friction;
        self.iters = 0;
        self.rms = self.measure_rms();
        self.rms0 = self.rms;
    }

    pub fn iters(&self) -> u32 {
        self.iters
    }
    pub fn rms(&self) -> f64 {
        self.rms
    }
    pub fn rms_initial(&self) -> f64 {
        self.rms0
    }

    /// Current estimates and truths, packed `[est, true]` per parameter:
    /// m1, m2, coulomb1, viscous1, coulomb2, viscous2.
    pub fn params(&self) -> Vec<f64> {
        let (ti, tf) = truth();
        vec![
            self.inertia[0].mass, ti[0].mass,
            self.inertia[1].mass, ti[1].mass,
            self.friction[0].coulomb, tf[0].coulomb,
            self.friction[0].viscous, tf[0].viscous,
            self.friction[1].coulomb, tf[1].coulomb,
            self.friction[1].viscous, tf[1].viscous,
        ]
    }

    /// Measured vs predicted torque trace for joint `j` over the first `n` samples (pairs
    /// `[measured, predicted]`) — the "does the model explain the data" view.
    pub fn torque_trace(&self, j: usize, n: usize) -> Vec<f64> {
        use ferromotion_core::gendyn::GenModel;
        let model = GenModel::<f64>::from_robot(&self.robot, &self.inertia, G);
        self.samples
            .iter()
            .take(n)
            .flat_map(|s| {
                let mut tau = model.rnea(&s.q, &s.qd, &s.qdd);
                tau[j] += self.friction[j].coulomb
                    * (s.qd[j] / ferromotion_learn::calib::COULOMB_SMOOTH).tanh()
                    + self.friction[j].viscous * s.qd[j];
                [s.tau[j], tau[j]]
            })
            .collect()
    }
}

impl Default for CalibLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lab converges the way the lesson promises: RMS collapses and masses land on truth.
    #[test]
    fn lab_calibration_converges() {
        let mut lab = CalibLab::new();
        assert!(lab.rms_initial() > 0.3, "wrong start must visibly mispredict: {}", lab.rms_initial());
        lab.calibrate(500);
        assert!(lab.rms() < lab.rms_initial() / 100.0, "rms must collapse: {} → {}", lab.rms_initial(), lab.rms());
        let p = lab.params();
        for k in [0, 2] {
            let (est, tru) = (p[k * 2], p[k * 2 + 1]);
            assert!(((est - tru) / tru).abs() < 0.03, "param {k}: {est} vs {tru}");
        }
    }
}
