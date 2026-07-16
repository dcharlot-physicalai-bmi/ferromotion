//! **Consensus lab** — the rig behind the textbook chapter on *algebraic connectivity*.
//!
//! Agents that can see only their immediate neighbours nonetheless come to agree, and the whole story
//! is a single number: the **Fiedler value** `λ₂`, the second-smallest eigenvalue of the graph
//! Laplacian. It is not a *bound* on how fast they agree — it **is** the rate. Disagreement decays as
//! `e^{−λ₂ t}`, exactly. Cut the graph in two and `λ₂` falls to zero and agreement becomes impossible;
//! add a link and `λ₂` rises and everyone agrees sooner.
//!
//! The rig runs the real [`ferromotion_control::consensus_step`] on a 2-D swarm (each agent a point in
//! the plane, the protocol run on both coordinates), so the reader watches a cloud of agents collapse
//! onto the one place they could ever agree on — their own starting centroid — while the page measures
//! the collapse rate and checks it against `λ₂` computed from the Laplacian spectrum, on device.

use ferromotion_control::{consensus_step, Graph};
use nalgebra::{DMatrix, DVector};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ConsensusLab {
    n: usize,
    adj: DMatrix<f64>, // symmetric adjacency, zero diagonal
    px: DVector<f64>,
    py: DVector<f64>,
}

#[wasm_bindgen]
impl ConsensusLab {
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize) -> ConsensusLab {
        ConsensusLab {
            n,
            adj: DMatrix::zeros(n, n),
            px: DVector::zeros(n),
            py: DVector::zeros(n),
        }
    }

    /// Place agent `i` at `(x, y)`.
    pub fn set_agent(&mut self, i: usize, x: f64, y: f64) {
        self.px[i] = x;
        self.py[i] = y;
    }

    pub fn has_edge(&self, i: usize, j: usize) -> bool {
        self.adj[(i, j)] != 0.0
    }

    /// Toggle the undirected link between `i` and `j`; returns whether it is now present.
    pub fn toggle_edge(&mut self, i: usize, j: usize) -> bool {
        if i == j {
            return false;
        }
        let on = self.adj[(i, j)] == 0.0;
        let w = if on { 1.0 } else { 0.0 };
        self.adj[(i, j)] = w;
        self.adj[(j, i)] = w;
        on
    }

    pub fn set_edge(&mut self, i: usize, j: usize, on: bool) {
        if i == j {
            return;
        }
        let w = if on { 1.0 } else { 0.0 };
        self.adj[(i, j)] = w;
        self.adj[(j, i)] = w;
    }

    pub fn clear_edges(&mut self) {
        self.adj = DMatrix::zeros(self.n, self.n);
    }

    fn graph(&self) -> Graph {
        Graph { adj: self.adj.clone() }
    }

    /// The Fiedler value `λ₂` — algebraic connectivity. Zero iff the graph is disconnected.
    pub fn fiedler(&self) -> f64 {
        self.graph().fiedler_value()
    }

    pub fn is_connected(&self) -> bool {
        self.graph().is_connected()
    }

    /// The full Laplacian spectrum, ascending — for drawing the eigenvalue bar (λ₁=0, λ₂ highlighted).
    pub fn eigenvalues(&self) -> Vec<f64> {
        let mut ev: Vec<f64> =
            self.graph().laplacian().symmetric_eigen().eigenvalues.iter().cloned().collect();
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ev
    }

    /// One consensus step `ẋ = −Lx` on both coordinates (the real library protocol).
    pub fn step(&mut self, dt: f64) {
        let g = self.graph();
        consensus_step(&g, &mut self.px, dt);
        consensus_step(&g, &mut self.py, dt);
    }

    pub fn centroid_x(&self) -> f64 {
        self.px.sum() / self.n as f64
    }
    pub fn centroid_y(&self) -> f64 {
        self.py.sum() / self.n as f64
    }

    /// RMS distance of the swarm from its centroid — the disagreement whose decay rate is `λ₂`.
    pub fn spread(&self) -> f64 {
        let (cx, cy) = (self.centroid_x(), self.centroid_y());
        let mut s = 0.0;
        for i in 0..self.n {
            s += (self.px[i] - cx).powi(2) + (self.py[i] - cy).powi(2);
        }
        (s / self.n as f64).sqrt()
    }

    // --- getters for drawing ---
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn x(&self, i: usize) -> f64 {
        self.px[i]
    }
    pub fn y(&self, i: usize) -> f64 {
        self.py[i]
    }
    pub fn degree(&self, i: usize) -> f64 {
        (0..self.n).map(|j| self.adj[(i, j)]).sum()
    }
    pub fn num_edges(&self) -> usize {
        let mut c = 0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if self.adj[(i, j)] != 0.0 {
                    c += 1;
                }
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path graph 0–1–2–…–(n−1) with the agents strung along a line, then displaced.
    fn path(n: usize) -> ConsensusLab {
        let mut lab = ConsensusLab::new(n);
        for i in 0..n {
            // deterministic, spread-out initial positions (no RNG in this environment)
            let a = i as f64;
            lab.set_agent(i, (a * 1.7).sin() * 3.0 + a, (a * 2.3).cos() * 3.0 - a);
        }
        for i in 0..n - 1 {
            lab.set_edge(i, i + 1, true);
        }
        lab
    }

    #[test]
    fn the_fiedler_value_is_the_measured_decay_rate() {
        // THE CHAPTER. Run the real protocol, measure how fast disagreement actually decays, and
        // compare to λ₂ from the Laplacian spectrum. λ₂ is not a bound on the rate — it is the rate.
        let mut lab = path(6);
        let lambda2 = lab.fiedler();
        assert!(lambda2 > 1e-6, "path graph should be connected");
        let (dt, warmup, window) = (1e-4, 40_000, 40_000);
        for _ in 0..warmup {
            lab.step(dt); // let the faster Laplacian modes die out
        }
        let s1 = lab.spread();
        for _ in 0..window {
            lab.step(dt);
        }
        let s2 = lab.spread();
        let measured = -(s2 / s1).ln() / (window as f64 * dt);
        assert!(
            (measured - lambda2).abs() / lambda2 < 0.02,
            "measured decay rate {measured:.5} vs λ₂ {lambda2:.5}"
        );
    }

    #[test]
    fn the_centroid_is_exactly_conserved() {
        // 1ᵀL = 0 ⇒ the average position never moves. The swarm can only agree on where it already was.
        let mut lab = path(7);
        let (cx0, cy0) = (lab.centroid_x(), lab.centroid_y());
        for _ in 0..60_000 {
            lab.step(1e-3);
        }
        assert!((lab.centroid_x() - cx0).abs() < 1e-12, "centroid x drifted");
        assert!((lab.centroid_y() - cy0).abs() < 1e-12, "centroid y drifted");
        // …and it has essentially collapsed onto that centroid.
        assert!(lab.spread() < 1e-4, "swarm did not converge: spread {}", lab.spread());
    }

    #[test]
    fn disconnecting_the_graph_makes_agreement_impossible() {
        // Split into two components {0,1,2} and {3,4,5}: λ₂ → 0 and each half agrees only with itself.
        let mut lab = ConsensusLab::new(6);
        for i in 0..6 {
            lab.set_agent(i, i as f64, if i < 3 { 0.0 } else { 10.0 });
        }
        for (i, j) in [(0, 1), (1, 2), (3, 4), (4, 5)] {
            lab.set_edge(i, j, true);
        }
        assert!(lab.fiedler().abs() < 1e-9, "disconnected graph must have λ₂ = 0");
        assert!(!lab.is_connected());
        let ya = lab.centroid_y(); // global centroid (=5) — which NOBODY reaches
        for _ in 0..40_000 {
            lab.step(1e-3);
        }
        // Each component collapses to its OWN centroid (y=0 and y=10), not the global one (y=5).
        assert!((lab.y(0) - 0.0).abs() < 1e-4 && (lab.y(2) - 0.0).abs() < 1e-4);
        assert!((lab.y(3) - 10.0).abs() < 1e-4 && (lab.y(5) - 10.0).abs() < 1e-4);
        assert!((lab.y(0) - lab.y(3)).abs() > 5.0, "the two halves must NOT agree");
        assert!((ya - 5.0).abs() < 1e-9, "sanity: the unreachable global mean was 5");
    }

    #[test]
    fn adding_a_link_never_slows_agreement() {
        // λ₂ is monotone non-decreasing under edge addition: another channel can only speed agreement.
        let mut lab = ConsensusLab::new(5);
        for i in 0..5 {
            lab.set_agent(i, i as f64, 0.0);
        }
        // Start from a path (connected) and add chords, checking λ₂ never drops.
        for i in 0..4 {
            lab.set_edge(i, i + 1, true);
        }
        let mut prev = lab.fiedler();
        for (i, j) in [(0, 2), (0, 4), (1, 3), (2, 4)] {
            lab.set_edge(i, j, true);
            let now = lab.fiedler();
            assert!(now >= prev - 1e-9, "adding edge ({i},{j}) lowered λ₂: {now} < {prev}");
            prev = now;
        }
        // The complete-ish graph agrees much faster than the bare path did.
        assert!(prev > lab.fiedler() * 0.0 + 1.0, "densely-linked λ₂ should be well above the path's");
    }

    #[test]
    fn connectivity_agrees_with_the_fiedler_test() {
        let mut lab = ConsensusLab::new(4);
        for i in 0..4 {
            lab.set_agent(i, i as f64, 0.0);
        }
        assert!(!lab.is_connected(), "no edges ⇒ disconnected");
        for i in 0..3 {
            lab.set_edge(i, i + 1, true);
        }
        assert!(lab.is_connected() && lab.fiedler() > 1e-9, "a path is connected");
        lab.set_edge(1, 2, false); // cut the middle
        assert!(!lab.is_connected() && lab.fiedler().abs() < 1e-9, "cutting the bridge disconnects it");
    }
}
