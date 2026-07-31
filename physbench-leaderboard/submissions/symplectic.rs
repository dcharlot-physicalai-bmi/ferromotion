// Reference submission: semi-implicit (symplectic) Euler — updates velocity first, then position.
// Symplectic integrators conserve energy over long rollouts, so this PASSES.
use crate::bench::{accel, Meta, Model};

pub const META: Meta = Meta { name: "symplectic", author: "reference", kind: "structured" };

pub struct M;
impl Model for M {
    fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64) {
        let w2 = w + dt * accel(th);
        (th + dt * w2, w2)
    }
}
