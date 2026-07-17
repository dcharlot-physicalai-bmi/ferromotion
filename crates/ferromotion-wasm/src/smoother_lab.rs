//! **Smoother lab** — the rig behind the textbook chapter on revising the past. Drives the real
//! [`ferromotion_core::LegSmoother`] (SE(2) pose-graph fixed-lag smoother). A robot drives a loop; its
//! odometry drifts so the dead-reckoned path never closes; a single **loop-closure** factor — "I am back
//! where I started" — then re-optimizes the whole window and snaps every past pose into alignment. A
//! filter cannot do that; a smoother does.

use ferromotion_core::LegSmoother;
use wasm_bindgen::prelude::*;

fn wrap(a: f64) -> f64 {
    a.sin().atan2(a.cos())
}
fn rel(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let (c, s) = (a[2].cos(), a[2].sin());
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    [c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])]
}

#[wasm_bindgen]
pub struct SmootherLab {
    n: usize,
    gt: Vec<[f64; 3]>,
    drift: f64,
}

#[wasm_bindgen]
impl SmootherLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SmootherLab {
        // Ground truth: a closed loop (circle) the robot drives and returns to the start.
        let n = 28;
        let r = 1.0;
        let gt: Vec<[f64; 3]> = (0..n)
            .map(|k| {
                let ph = TAU * k as f64 / (n - 1) as f64;
                [r * ph.sin(), r * (1.0 - ph.cos()), ph] // starts at origin heading +x, loops back
            })
            .collect();
        SmootherLab { n, gt, drift: 0.03 }
    }

    pub fn set_drift(&mut self, d: f64) {
        self.drift = d;
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn gt_xy(&self) -> Vec<f64> {
        self.gt.iter().flat_map(|p| [p[0], p[1]]).collect()
    }

    /// Biased odometry between consecutive ground-truth keyframes (a constant heading bias that
    /// accumulates into drift, scaled by `drift`).
    fn odom(&self, k: usize) -> [f64; 3] {
        let mut r = rel(self.gt[k], self.gt[k + 1]);
        r[2] += self.drift; // systematic under/over-rotation each step → the loop spirals open
        r[0] += self.drift * 0.3;
        r
    }

    /// The dead-reckoned trajectory (pure odometry, no smoothing) — what a filter's mean would trace.
    fn dead_reckon(&self) -> Vec<[f64; 3]> {
        let mut poses = vec![self.gt[0]];
        let mut cur = self.gt[0];
        for k in 0..self.n - 1 {
            let o = self.odom(k);
            let (c, s) = (cur[2].cos(), cur[2].sin());
            cur = [cur[0] + c * o[0] - s * o[1], cur[1] + s * o[0] + c * o[1], wrap(cur[2] + o[2])];
            poses.push(cur);
        }
        poses
    }

    fn build(&self, with_closure: bool) -> Vec<[f64; 3]> {
        let mut sm = LegSmoother::new(self.n);
        sm.add_prior(0, self.gt[0], 100.0);
        for k in 0..self.n - 1 {
            sm.add_odometry(k, k + 1, self.odom(k), 20.0);
        }
        if with_closure {
            // the robot recognizes it is back at the start: an accurate closure between last & first
            sm.add_contact(self.n - 1, 0, rel(self.gt[self.n - 1], self.gt[0]), 60.0);
        }
        let dr = self.dead_reckon();
        let x0: Vec<f64> = dr.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        sm.solve(&x0)
    }

    /// The estimate as interleaved `[x,y,…]`. Without closure this is the drifted dead-reckoning; with
    /// closure the smoother snaps the whole loop shut.
    pub fn estimate_xy(&self, with_closure: bool) -> Vec<f64> {
        self.build(with_closure).iter().flat_map(|p| [p[0], p[1]]).collect()
    }

    /// RMS position error of an estimate vs ground truth.
    pub fn rms_error(&self, with_closure: bool) -> f64 {
        let est = self.build(with_closure);
        let s: f64 = (0..self.n).map(|k| (est[k][0] - self.gt[k][0]).powi(2) + (est[k][1] - self.gt[k][1]).powi(2)).sum();
        (s / self.n as f64).sqrt()
    }

    /// How far the loop fails to close (last keyframe vs first) — the visible gap a closure removes.
    pub fn loop_gap(&self, with_closure: bool) -> f64 {
        let est = self.build(with_closure);
        ((est[self.n - 1][0] - est[0][0]).powi(2) + (est[self.n - 1][1] - est[0][1]).powi(2)).sqrt()
    }
}

const TAU: f64 = std::f64::consts::TAU;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_closure_cuts_the_drift() {
        // THE CHAPTER. A drifting odometry loop does not close; the loop-closure factor snaps it shut
        // and slashes the whole-trajectory error — the smoother revises the past.
        let mut lab = SmootherLab::new();
        lab.set_drift(0.04);
        let open = lab.rms_error(false);
        let closed = lab.rms_error(true);
        assert!(closed < 0.4 * open, "loop closure should slash trajectory error: {closed} vs {open}");
        let gap_open = lab.loop_gap(false);
        let gap_closed = lab.loop_gap(true);
        assert!(gap_closed < 0.25 * gap_open, "closure should shut the loop: gap {gap_closed} vs {gap_open}");
    }

    #[test]
    fn no_drift_recovers_the_truth() {
        let mut lab = SmootherLab::new();
        lab.set_drift(0.0);
        assert!(lab.rms_error(true) < 1e-4, "with no drift the estimate should match ground truth");
    }

    #[test]
    fn more_drift_opens_a_wider_loop() {
        let mut lab = SmootherLab::new();
        lab.set_drift(0.02);
        let small = lab.loop_gap(false);
        lab.set_drift(0.06);
        let big = lab.loop_gap(false);
        assert!(big > small, "more odometry drift should open a wider loop gap: {big} vs {small}");
    }
}
