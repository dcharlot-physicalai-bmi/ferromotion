// Reference submission: velocity Verlet — a second-order symplectic integrator. Best energy behavior; PASSES.
use crate::bench::{accel, Meta, Model};

pub const META: Meta = Meta { name: "velocity-verlet", author: "reference", kind: "structured" };

pub struct M;
impl Model for M {
    fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64) {
        let a = accel(th);
        let tn = th + dt * w + 0.5 * dt * dt * a;
        let wn = w + 0.5 * dt * (a + accel(tn));
        (tn, wn)
    }
}
