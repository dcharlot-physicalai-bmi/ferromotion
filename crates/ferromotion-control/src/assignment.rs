//! **Linear assignment — the Hungarian algorithm** (Kuhn 1955; the O(n³) Jonker–Volgenant
//! shortest-augmenting-path form). Given an `n×n` cost matrix, find the one-to-one matching of rows to
//! columns that **minimizes total cost**. It is the workhorse of multi-robot **task allocation** (assign
//! robots to goals to minimize total travel), **data association** in multi-target tracking (match
//! detections to tracks), and matching problems throughout robotics and vision.
//!
//! This is the exact, polynomial-time optimum — not a heuristic. It complements the crate's *sampling* and
//! *game-theoretic* multi-agent methods (`swarm`, `orca`, `algames`) with the combinatorial-optimal
//! allocation layer. Verified against brute-force minimization over all permutations, checked for pairwise
//! swap-optimality, and shown to minimize total travel on a robots-to-goals instance. Pure `nalgebra`,
//! integer-free control flow → WASM-clean.

use nalgebra::DMatrix;

/// The optimal assignment: `assign[i]` is the column matched to row `i`, and `cost` is the total.
#[derive(Clone, Debug, PartialEq)]
pub struct Assignment {
    pub assign: Vec<usize>,
    pub cost: f64,
}

/// Minimize total cost of a one-to-one row→column matching of a square cost matrix, via the O(n³)
/// Hungarian / Jonker–Volgenant shortest-augmenting-path method with dual potentials. Costs may be any
/// finite reals (negative allowed). `cost` must be square.
pub fn hungarian(cost: &DMatrix<f64>) -> Assignment {
    let n = cost.nrows();
    assert_eq!(n, cost.ncols(), "cost matrix must be square");
    if n == 0 {
        return Assignment { assign: Vec::new(), cost: 0.0 };
    }
    let inf = f64::INFINITY;
    // 1-indexed working arrays; p[j] = row currently assigned to column j (0 = unassigned sentinel).
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        // Dijkstra-like augmenting search minimizing reduced cost.
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[(i0 - 1, j - 1)] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            // update the potentials by the slack delta
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // walk the augmenting path back, flipping matches
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assign = vec![0usize; n];
    for j in 1..=n {
        assign[p[j] - 1] = j - 1;
    }
    let total = (0..n).map(|i| cost[(i, assign[i])]).sum();
    Assignment { assign, cost: total }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::icem::Rng;

    // Brute-force minimum over all n! permutations — the exact oracle.
    fn brute(cost: &DMatrix<f64>) -> f64 {
        let n = cost.nrows();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut best = f64::INFINITY;
        // Heap's algorithm over permutations
        fn go(k: usize, perm: &mut Vec<usize>, cost: &DMatrix<f64>, best: &mut f64) {
            if k == 1 {
                let c: f64 = (0..perm.len()).map(|i| cost[(i, perm[i])]).sum();
                *best = best.min(c);
                return;
            }
            for i in 0..k {
                go(k - 1, perm, cost, best);
                if k.is_multiple_of(2) {
                    perm.swap(i, k - 1);
                } else {
                    perm.swap(0, k - 1);
                }
            }
        }
        go(n, &mut perm, cost, &mut best);
        best
    }

    #[test]
    fn it_matches_brute_force_on_random_matrices() {
        // THE ORACLE. Over many deterministic random cost matrices, the Hungarian optimum equals the
        // brute-force minimum over all permutations.
        let mut rng = Rng::new(12345);
        for n in 1..=6 {
            for _ in 0..20 {
                let m = DMatrix::from_fn(n, n, |_, _| (rng.gaussian() * 5.0).round());
                let got = hungarian(&m);
                let want = brute(&m);
                assert!((got.cost - want).abs() < 1e-9, "n={n}: hungarian {} vs brute {want}", got.cost);
                // the returned assignment is a valid permutation
                let mut seen = vec![false; n];
                for &c in &got.assign {
                    assert!(!seen[c], "assignment must be a permutation");
                    seen[c] = true;
                }
            }
        }
    }

    #[test]
    fn it_picks_the_obvious_diagonal() {
        // A matrix cheap on the diagonal, expensive off it ⇒ the identity assignment.
        let m = DMatrix::from_fn(4, 4, |i, j| if i == j { 0.0 } else { 10.0 });
        let a = hungarian(&m);
        assert_eq!(a.assign, vec![0, 1, 2, 3]);
        assert!(a.cost.abs() < 1e-12);
    }

    #[test]
    fn no_pairwise_swap_improves_the_optimum() {
        // Optimality certificate: swapping any two rows' columns cannot lower the total.
        let mut rng = Rng::new(999);
        let n = 6;
        let m = DMatrix::from_fn(n, n, |_, _| rng.gaussian().abs() * 3.0);
        let a = hungarian(&m);
        for i in 0..n {
            for j in (i + 1)..n {
                let swapped = a.cost - m[(i, a.assign[i])] - m[(j, a.assign[j])] + m[(i, a.assign[j])] + m[(j, a.assign[i])];
                assert!(swapped >= a.cost - 1e-9, "swap ({i},{j}) improved cost: {swapped} < {}", a.cost);
            }
        }
    }

    #[test]
    fn it_minimizes_total_travel_for_robots_to_goals() {
        // THE APPLICATION. Four robots, four goals; cost = squared distance. The optimal allocation
        // minimizes total travel — and here it's the non-trivial matching, not the naive index order.
        let robots: [(f64, f64); 4] = [(0.0, 0.0), (0.0, 3.0), (3.0, 0.0), (3.0, 3.0)];
        let goals: [(f64, f64); 4] = [(3.0, 3.0), (3.0, 0.0), (0.0, 3.0), (0.0, 0.0)];
        let m = DMatrix::from_fn(4, 4, |i, j| {
            let (rx, ry) = robots[i];
            let (gx, gy) = goals[j];
            (rx - gx).powi(2) + (ry - gy).powi(2)
        });
        let a = hungarian(&m);
        // each robot goes to its nearest (identical) goal: R0→G3, R1→G2, R2→G1, R3→G0 (all at distance 0)
        assert_eq!(a.assign, vec![3, 2, 1, 0]);
        assert!(a.cost.abs() < 1e-9, "each robot reaches a coincident goal: {}", a.cost);
    }
}
