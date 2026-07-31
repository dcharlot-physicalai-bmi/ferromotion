// Reference submission: a plausible-looking model that silently bleeds energy every step (a stand-in for a
// learned model that "looks smooth" but modeled the wrong system). It LEAKS energy and FAILS the invariant.
use crate::bench::{accel, Meta, Model};

pub const META: Meta = Meta { name: "lossy", author: "reference", kind: "wrong invariant" };

pub struct M;
impl Model for M {
    fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64) {
        let w2 = (w + dt * accel(th)) * 0.999;
        (th + dt * w2, w2)
    }
}
