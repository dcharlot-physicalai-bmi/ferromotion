// Reference submission: explicit (forward) Euler. Accurate per step, but not symplectic — it PUMPS energy
// over a rollout and FAILS. Same one-step accuracy as symplectic, opposite verdict: that is the whole point.
use crate::bench::{accel, Meta, Model};

pub const META: Meta = Meta { name: "explicit-euler", author: "reference", kind: "no structure" };

pub struct M;
impl Model for M {
    fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64) {
        (th + dt * w, w + dt * accel(th))
    }
}
