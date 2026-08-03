//! **Partial observability and active perception** — deciding what to *look at*, not only what to do.
//!
//! State estimation and partial observability are different problems, and having the first does not
//! give the second. A filter answers "given these measurements, where am I"; partial observability asks
//! "which measurement should I take next, and is this state distinguishable at all". A robot that
//! cannot answer the second will confidently average over two hypotheses it should have separated by
//! moving its head.
//!
//! The pieces here work on a discrete belief, which is the honest setting for the ambiguity that
//! actually breaks embodied systems: a symmetric corridor, a part that is either present or absent, a
//! door that is either latched or not. Those are multi-modal beliefs, and a Gaussian filter cannot
//! represent them at all.
//!
//! * [`Belief`] — a distribution over discrete states, with Bayesian [`update`](Belief::update) and
//!   [`predict`](Belief::predict).
//! * [`expected_information_gain`] — how much an action is expected to reduce uncertainty, *before*
//!   taking it, by averaging over the observations it might return.
//! * [`best_sensing_action`] — greedy active perception: take the measurement that tells you the most.
//!
//! Information gain is mutual information between the state and the observation the action would
//! produce, which is non-negative for every action: looking never hurts in expectation, though a
//! particular look may disappoint. That non-negativity is a property worth testing, because a sign
//! error in this kind of code produces a robot that avoids information.

/// A belief over a finite state space, held as normalised probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct Belief {
    pub p: Vec<f64>,
}

impl Belief {
    /// A uniform belief over `n` states: maximum entropy, nothing known.
    pub fn uniform(n: usize) -> Belief {
        Belief { p: vec![1.0 / n as f64; n] }
    }

    /// From unnormalised weights. `None` if they do not sum to something positive.
    pub fn from_weights(w: &[f64]) -> Option<Belief> {
        let s: f64 = w.iter().filter(|x| **x > 0.0).sum();
        if s <= 0.0 {
            return None;
        }
        Some(Belief { p: w.iter().map(|x| x.max(0.0) / s).collect() })
    }

    pub fn len(&self) -> usize {
        self.p.len()
    }
    pub fn is_empty(&self) -> bool {
        self.p.is_empty()
    }

    /// Shannon entropy in bits — the amount still unknown.
    pub fn entropy(&self) -> f64 {
        -self.p.iter().filter(|x| **x > 0.0).map(|x| x * x.log2()).sum::<f64>()
    }

    /// The most likely state and its probability.
    pub fn mode(&self) -> (usize, f64) {
        let mut best = (0usize, f64::NEG_INFINITY);
        for (i, &v) in self.p.iter().enumerate() {
            if v > best.1 {
                best = (i, v);
            }
        }
        best
    }

    /// **Bayesian update** on observing `obs`, given `likelihood[state][obs] = P(obs | state)`.
    /// `None` when the observation is impossible under this belief, which is itself useful information:
    /// it means the model is wrong, not that the update is hard.
    pub fn update(&self, likelihood: &[Vec<f64>], obs: usize) -> Option<Belief> {
        let w: Vec<f64> = self.p.iter().enumerate().map(|(s, &ps)| ps * likelihood.get(s).and_then(|r| r.get(obs)).copied().unwrap_or(0.0)).collect();
        Belief::from_weights(&w)
    }

    /// **Prediction** through a transition model `t[from][to] = P(to | from, action)`.
    pub fn predict(&self, t: &[Vec<f64>]) -> Belief {
        let n = self.p.len();
        let mut q = vec![0.0; n];
        for (from, &pf) in self.p.iter().enumerate() {
            if pf <= 0.0 {
                continue;
            }
            if let Some(row) = t.get(from) {
                for (to, &pt) in row.iter().enumerate() {
                    if to < n {
                        q[to] += pf * pt;
                    }
                }
            }
        }
        Belief::from_weights(&q).unwrap_or_else(|| self.clone())
    }
}

/// **Expected information gain** of a sensing action, in bits: the mutual information between the state
/// and the observation the action would return,
///
/// `I = H(belief) − Σ_o P(o) · H(belief | o)`.
///
/// Computed before acting, by averaging the posterior entropy over every observation the action might
/// produce. Non-negative for every action and every model — an expectation over observations cannot
/// increase uncertainty, even though one particular observation can.
pub fn expected_information_gain(b: &Belief, likelihood: &[Vec<f64>]) -> f64 {
    let n_obs = likelihood.first().map(|r| r.len()).unwrap_or(0);
    let prior_h = b.entropy();
    let mut expected_h = 0.0;
    for o in 0..n_obs {
        // P(o) = Σ_s P(s) P(o|s)
        let p_o: f64 = b.p.iter().enumerate().map(|(s, &ps)| ps * likelihood[s].get(o).copied().unwrap_or(0.0)).sum();
        if p_o <= 1e-15 {
            continue;
        }
        if let Some(post) = b.update(likelihood, o) {
            expected_h += p_o * post.entropy();
        }
    }
    (prior_h - expected_h).max(0.0)
}

/// **Greedy active perception**: of the available sensing actions, the one expected to reduce
/// uncertainty most, with its gain in bits. `None` when there are no actions.
pub fn best_sensing_action(b: &Belief, actions: &[Vec<Vec<f64>>]) -> Option<(usize, f64)> {
    actions
        .iter()
        .enumerate()
        .map(|(i, l)| (i, expected_information_gain(b, l)))
        .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sensor that reports the state perfectly collapses the belief; the update is exact Bayes.
    #[test]
    fn a_perfect_sensor_collapses_the_belief() {
        let b = Belief::uniform(3);
        assert!((b.entropy() - 3.0f64.log2()).abs() < 1e-12, "uniform over 3 is log2(3) bits");
        let perfect = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let after = b.update(&perfect, 1).unwrap();
        assert!((after.p[1] - 1.0).abs() < 1e-12 && after.entropy() < 1e-12, "belief should collapse");
        assert_eq!(after.mode().0, 1);
    }

    /// An observation with zero probability under the belief is refused rather than silently producing
    /// a renormalised fiction. That is a model error the caller needs to see.
    #[test]
    fn an_impossible_observation_is_refused() {
        let b = Belief::from_weights(&[1.0, 0.0]).unwrap();
        let l = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(b.update(&l, 1).is_none(), "state 1 has no mass, so obs 1 is impossible");
    }

    /// Information gain is never negative, whatever the sensor and whatever the belief. A sign error
    /// here would produce a robot that actively avoids looking.
    #[test]
    fn information_gain_is_never_negative() {
        let beliefs = [Belief::uniform(4), Belief::from_weights(&[0.7, 0.2, 0.05, 0.05]).unwrap(), Belief::from_weights(&[1.0, 0.0, 0.0, 0.0]).unwrap()];
        // a spread of sensors: perfect, useless, noisy, and one that only separates the first pair
        let sensors: Vec<Vec<Vec<f64>>> = vec![
            (0..4).map(|s| (0..4).map(|o| if s == o { 1.0 } else { 0.0 }).collect()).collect(),
            (0..4).map(|_| vec![0.25, 0.25, 0.25, 0.25]).collect(),
            (0..4).map(|s| (0..4).map(|o| if s == o { 0.7 } else { 0.1 }).collect()).collect(),
            vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5], vec![0.5, 0.5]],
        ];
        for b in &beliefs {
            for l in &sensors {
                let g = expected_information_gain(b, l);
                assert!(g >= -1e-12, "negative information gain {g}");
                assert!(g <= b.entropy() + 1e-9, "gain {g} cannot exceed the {} bits available", b.entropy());
            }
        }
    }

    /// A useless sensor gains nothing and a perfect one gains everything — the two ends of the scale.
    #[test]
    fn gain_spans_from_a_useless_to_a_perfect_sensor() {
        let b = Belief::uniform(4);
        let useless: Vec<Vec<f64>> = (0..4).map(|_| vec![0.25; 4]).collect();
        let perfect: Vec<Vec<f64>> = (0..4).map(|s| (0..4).map(|o| if s == o { 1.0 } else { 0.0 }).collect()).collect();
        let (gu, gp) = (expected_information_gain(&b, &useless), expected_information_gain(&b, &perfect));
        eprintln!("information gain: useless sensor {gu:.4} bits, perfect sensor {gp:.4} bits (2 available)");
        assert!(gu < 1e-9, "a sensor independent of the state tells you nothing, got {gu}");
        assert!((gp - 2.0).abs() < 1e-9, "a perfect sensor resolves all 2 bits, got {gp}");
    }

    /// Active perception, the point of the module: given a belief that is ambiguous in one particular
    /// way, the chosen action is the one that disambiguates *that* — not the one that is best on
    /// average over all beliefs. This is what a fixed sensing schedule cannot do.
    #[test]
    fn active_perception_picks_the_action_that_resolves_this_ambiguity() {
        // four states; the belief is torn between 0 and 1, and certain it is not 2 or 3
        let b = Belief::from_weights(&[0.5, 0.5, 0.0, 0.0]).unwrap();
        // action A separates {0} from {1}; action B separates {0,1} from {2,3} and so says nothing here
        let a_sep01 = vec![vec![0.95, 0.05], vec![0.05, 0.95], vec![0.5, 0.5], vec![0.5, 0.5]];
        let b_sep23 = vec![vec![0.95, 0.05], vec![0.95, 0.05], vec![0.05, 0.95], vec![0.05, 0.95]];
        let actions = vec![a_sep01.clone(), b_sep23.clone()];
        let (pick, gain) = best_sensing_action(&b, &actions).unwrap();
        let other = expected_information_gain(&b, &b_sep23);
        eprintln!("torn between states 0 and 1: chose action {pick} for {gain:.4} bits (the other offers {other:.4})");
        assert_eq!(pick, 0, "must choose the action that separates the two live hypotheses");
        assert!(gain > 0.5 && other < 1e-9, "gain {gain} vs {other}");

        // and if the ambiguity moves, so does the choice — the same fixed schedule would now be wrong
        let b2 = Belief::from_weights(&[0.5, 0.0, 0.5, 0.0]).unwrap();
        let (pick2, gain2) = best_sensing_action(&b2, &actions).unwrap();
        eprintln!("torn between states 0 and 2: chose action {pick2} for {gain2:.4} bits");
        assert_eq!(pick2, 1, "the other action is now the informative one");
    }

    /// Prediction through a transition model spreads a belief, raising entropy, which is the cost of
    /// acting without looking. A filter that did not do this would be overconfident by construction.
    #[test]
    fn prediction_without_observation_loses_information() {
        let b = Belief::from_weights(&[1.0, 0.0, 0.0]).unwrap();
        // a noisy move: mostly forward, sometimes stays
        let t = vec![vec![0.2, 0.8, 0.0], vec![0.0, 0.2, 0.8], vec![0.0, 0.0, 1.0]];
        let after = b.predict(&t);
        eprintln!("entropy after one blind move: {:.4} bits (was {:.4})", after.entropy(), b.entropy());
        assert!(after.entropy() > b.entropy(), "moving blind must not reduce uncertainty");
        // the mass went where the model says
        assert!((after.p[1] - 0.8).abs() < 1e-12 && (after.p[0] - 0.2).abs() < 1e-12);
    }
}
