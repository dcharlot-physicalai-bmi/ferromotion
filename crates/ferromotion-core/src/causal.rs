//! **Causal sufficiency** — why a world model that predicts perfectly can still be wrong about what
//! happens when you act.
//!
//! A learned dynamics model is fitted to observed data and scored on predicting more observed data.
//! Control does something the data never did: it *intervenes*. Those are different queries, and a model
//! can be exactly right about the first and wrong about the second, because observational data leaves
//! the direction of an association undetermined and says nothing about what breaks when you seize a
//! variable and set it yourself.
//!
//! The concrete failure has a name. If a confounder drives both an action and an outcome, the
//! observed correlation between them mixes the effect of acting with the effect of the confounder. A
//! model trained to reproduce that correlation will predict the wrong response to a commanded action —
//! and it will keep passing its validation set, because its validation set is observational too.
//!
//! This module works with linear-Gaussian structural causal models, where every quantity of interest is
//! available in closed form and so can be checked rather than sampled:
//!
//! * [`Scm::observational_cov`] — what a passive observer sees.
//! * [`Scm::intervene`] — Pearl's `do` operator, which *cuts* the incoming edges of the seized
//!   variable rather than conditioning on it.
//! * [`Scm::backdoor_adjustment`] — the effect recovered from observational data when a sufficient
//!   adjustment set is available, and [`Scm::satisfies_backdoor`] to check that it is.

use nalgebra::{DMatrix, DVector};

/// A linear-Gaussian structural causal model: `x = B x + e`, with `B[i][j]` the direct effect of `j` on
/// `i` (so `B` is lower-triangular in a topological order) and independent noise of variance
/// `noise[i]`.
#[derive(Clone, Debug)]
pub struct Scm {
    pub b: DMatrix<f64>,
    pub noise: DVector<f64>,
}

impl Scm {
    /// A model from a direct-effect matrix and noise variances. `None` if the shapes disagree or the
    /// system is not solvable (a cyclic model with unit gain).
    pub fn new(b: DMatrix<f64>, noise: DVector<f64>) -> Option<Scm> {
        if b.nrows() != b.ncols() || b.nrows() != noise.len() {
            return None;
        }
        let n = b.nrows();
        (DMatrix::identity(n, n) - &b).try_inverse()?;
        Some(Scm { b, noise })
    }

    pub fn dim(&self) -> usize {
        self.b.nrows()
    }

    /// `(I − B)⁻¹`, which maps noise to observed values and whose entries are the *total* effects.
    fn reduced_form(&self) -> DMatrix<f64> {
        let n = self.dim();
        (DMatrix::identity(n, n) - &self.b).try_inverse().expect("checked at construction")
    }

    /// The covariance a passive observer measures: `(I−B)⁻¹ diag(noise) (I−B)⁻ᵀ`.
    pub fn observational_cov(&self) -> DMatrix<f64> {
        let a = self.reduced_form();
        &a * DMatrix::from_diagonal(&self.noise) * a.transpose()
    }

    /// The **total causal effect** of `cause` on `effect`: the change in `effect` per unit of `cause`
    /// when `cause` is set by intervention. This is the ground truth an interventional experiment would
    /// measure, and the number a controller needs.
    pub fn total_effect(&self, cause: usize, effect: usize) -> f64 {
        self.reduced_form()[(effect, cause)]
    }

    /// **Pearl's `do` operator**: seize `var` and set it, which *deletes* its incoming edges rather than
    /// conditioning on its observed value. The returned model is the post-intervention world; its noise
    /// for `var` is zero because the variable is no longer free to vary.
    pub fn intervene(&self, var: usize) -> Scm {
        let mut b = self.b.clone();
        for j in 0..self.dim() {
            b[(var, j)] = 0.0; // nothing upstream influences a variable you are holding
        }
        let mut noise = self.noise.clone();
        noise[var] = 0.0;
        Scm { b, noise }
    }

    /// The **naive regression coefficient** of `effect` on `cause` from observational data alone:
    /// `Cov(cause, effect) / Var(cause)`. This is what fitting a model to logged data recovers, and it
    /// equals the causal effect only when nothing confounds the pair.
    pub fn observational_regression(&self, cause: usize, effect: usize) -> f64 {
        let s = self.observational_cov();
        s[(effect, cause)] / s[(cause, cause)]
    }

    /// Whether `adjust` is a valid **backdoor adjustment set** for `cause → effect`: no member is a
    /// descendant of `cause`, and the set blocks every path into `cause` that could confound the pair.
    /// Checked structurally on the graph, which is the only place the answer lives — no amount of data
    /// settles it.
    pub fn satisfies_backdoor(&self, cause: usize, effect: usize, adjust: &[usize]) -> bool {
        let desc = self.descendants(cause);
        if adjust.iter().any(|a| desc.contains(a) || *a == cause || *a == effect) {
            return false; // adjusting on a descendant of the cause opens new bias
        }
        // every parent of `cause` that also reaches `effect` other than through `cause` must be covered
        for p in self.parents(cause) {
            if adjust.contains(&p) {
                continue;
            }
            if self.reaches_avoiding(p, effect, cause) {
                return false; // an open backdoor remains
            }
        }
        true
    }

    /// Effect of `cause` on `effect` recovered from observational data by **backdoor adjustment**: the
    /// partial regression coefficient of `effect` on `cause` holding `adjust` fixed. When the set is
    /// sufficient this equals the interventional effect, which is the whole point — it is how a causal
    /// quantity is obtained without running the experiment. `None` if the linear algebra is singular.
    pub fn backdoor_adjustment(&self, cause: usize, effect: usize, adjust: &[usize]) -> Option<f64> {
        let s = self.observational_cov();
        // regress `effect` on [cause, adjust...] and read the coefficient on `cause`
        let mut idx = vec![cause];
        idx.extend_from_slice(adjust);
        let k = idx.len();
        let mut xx = DMatrix::zeros(k, k);
        let mut xy = DVector::zeros(k);
        for a in 0..k {
            xy[a] = s[(idx[a], effect)];
            for bb in 0..k {
                xx[(a, bb)] = s[(idx[a], idx[bb])];
            }
        }
        let sol = xx.lu().solve(&xy)?;
        Some(sol[0])
    }

    fn parents(&self, v: usize) -> Vec<usize> {
        (0..self.dim()).filter(|&j| j != v && self.b[(v, j)].abs() > 0.0).collect()
    }

    /// Every variable reachable from `v` by following edges forward, `v` excluded.
    fn descendants(&self, v: usize) -> Vec<usize> {
        let n = self.dim();
        let mut seen = vec![false; n];
        let mut stack = vec![v];
        while let Some(x) = stack.pop() {
            for (c, flag) in seen.iter_mut().enumerate() {
                if self.b[(c, x)].abs() > 0.0 && !*flag {
                    *flag = true;
                    stack.push(c);
                }
            }
        }
        (0..n).filter(|&i| seen[i] && i != v).collect()
    }

    /// Whether `from` reaches `to` forward without passing through `avoid`.
    fn reaches_avoiding(&self, from: usize, to: usize, avoid: usize) -> bool {
        let n = self.dim();
        let mut seen = vec![false; n];
        let mut stack = vec![from];
        seen[from] = true;
        while let Some(x) = stack.pop() {
            if x == to {
                return true;
            }
            for (c, flag) in seen.iter_mut().enumerate() {
                if self.b[(c, x)].abs() > 0.0 && !*flag && c != avoid {
                    *flag = true;
                    stack.push(c);
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The confounded triangle: `Z → A`, `Z → Y`, `A → Y`. This is the shape of every logged-teleop
    /// dataset where an operator's unrecorded intent drives both what they did and how it turned out.
    fn confounded() -> Scm {
        // order: 0 = Z (confounder), 1 = A (action), 2 = Y (outcome)
        let mut b = DMatrix::zeros(3, 3);
        b[(1, 0)] = 2.0; // Z -> A
        b[(2, 0)] = 3.0; // Z -> Y
        b[(2, 1)] = 0.5; // A -> Y, the true causal effect
        Scm::new(b, DVector::from_row_slice(&[1.0, 1.0, 1.0])).unwrap()
    }

    /// The central failure, in numbers: fitting the observed data recovers a coefficient that is not the
    /// causal effect, so a model validated on logged data mispredicts what a commanded action does.
    #[test]
    fn observational_fit_recovers_the_wrong_effect_under_confounding() {
        let m = confounded();
        let truth = m.total_effect(1, 2); // A -> Y by intervention
        let naive = m.observational_regression(1, 2);
        eprintln!("confounded action: true causal effect {truth:.4}, observational fit {naive:.4}  (off by {:.1}x)", naive / truth);
        assert!((truth - 0.5).abs() < 1e-12, "the model's direct A->Y effect is 0.5, got {truth}");
        assert!((naive - truth).abs() > 0.5, "the naive fit must be badly wrong here, got {naive}");
    }

    /// Adjusting on the confounder recovers the causal effect from the very same observational data.
    /// Nothing new was measured; the graph supplied what the data alone could not.
    #[test]
    fn backdoor_adjustment_recovers_the_causal_effect() {
        let m = confounded();
        assert!(m.satisfies_backdoor(1, 2, &[0]), "Z is a valid adjustment set for A -> Y");
        let adjusted = m.backdoor_adjustment(1, 2, &[0]).unwrap();
        let truth = m.total_effect(1, 2);
        eprintln!("after adjusting for the confounder: {adjusted:.6} vs truth {truth:.6}");
        assert!((adjusted - truth).abs() < 1e-9, "adjustment should recover {truth}, got {adjusted}");
        // and the empty set is not sufficient, which is why the naive fit failed
        assert!(!m.satisfies_backdoor(1, 2, &[]), "with the backdoor open, no adjustment set is empty-valid");
    }

    /// `do` is not conditioning. Intervening on the action cuts the confounder's influence on it, so the
    /// post-intervention model has a different covariance than the observational one — and in it, the
    /// regression finally reads the causal effect.
    #[test]
    fn intervention_cuts_incoming_edges_rather_than_conditioning() {
        let m = confounded();
        let done = m.intervene(1);
        assert!(done.b.row(1).iter().all(|v| *v == 0.0), "A must have no parents after do(A)");
        // in the intervened world the confounder no longer reaches A, so Cov(A,Y)/Var(A) is the effect
        // (A now has zero variance, so drive it explicitly instead: total effect is unchanged)
        assert!((done.total_effect(1, 2) - 0.5).abs() < 1e-12, "the causal path A->Y survives intervention");
        // Z -> Y is untouched by do(A)
        assert!((done.total_effect(0, 2) - 3.0).abs() < 1e-12, "do(A) must not alter Z -> Y");
        // and the observational covariance genuinely differs from the interventional one
        let diff = (m.observational_cov() - done.observational_cov()).amax();
        eprintln!("observational and interventional covariances differ by {diff:.4} — the two queries are not the same");
        assert!(diff > 1e-6, "if these agreed there would be no confounding problem");
    }

    /// Adjusting on a descendant of the action is rejected: it is the classic mistake that introduces
    /// bias rather than removing it, and it cannot be detected from data.
    #[test]
    fn adjusting_on_a_descendant_is_rejected() {
        let m = confounded();
        assert!(!m.satisfies_backdoor(1, 2, &[2]), "Y is the outcome, never an adjustment set");
        // a mediator: A -> M -> Y
        let mut b = DMatrix::zeros(4, 4);
        b[(1, 0)] = 1.0; // Z -> A
        b[(2, 1)] = 1.0; // A -> M
        b[(3, 2)] = 1.0; // M -> Y
        b[(3, 0)] = 1.0; // Z -> Y
        let med = Scm::new(b, DVector::from_row_slice(&[1.0; 4])).unwrap();
        assert!(!med.satisfies_backdoor(1, 3, &[2]), "M is a descendant of A, so adjusting on it is invalid");
        assert!(med.satisfies_backdoor(1, 3, &[0]), "Z is the confounder and is valid");
    }

    /// With no confounding, the observational fit is already the causal effect — the case where a
    /// learned model happens to be safe to act on, and the reason the failure is easy to miss.
    #[test]
    fn without_confounding_the_observational_fit_is_correct() {
        let mut b = DMatrix::zeros(3, 3);
        b[(2, 1)] = 0.75; // A -> Y only; nothing drives A
        let m = Scm::new(b, DVector::from_row_slice(&[1.0, 1.0, 1.0])).unwrap();
        let (truth, naive) = (m.total_effect(1, 2), m.observational_regression(1, 2));
        assert!(m.satisfies_backdoor(1, 2, &[]), "no backdoor path, so no adjustment needed");
        assert!((truth - naive).abs() < 1e-9, "unconfounded: {naive} should equal {truth}");
    }
}
