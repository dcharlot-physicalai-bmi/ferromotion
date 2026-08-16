//! **Philip Hall basis** — a basis for the free nilpotent Lie algebra on `m` generators.
//!
//! Murray, Li & Sastry (1994), *A Mathematical Introduction to Robotic Manipulation*, §7.4.3. Listing every
//! Lie product of a set of vector fields over-counts badly: skew-symmetry makes `[f, g] = −[g, f]`, and the
//! Jacobi identity makes one of any three cyclic brackets redundant. The Hall basis is the standard way to
//! pick a genuine basis that accounts for both.
//!
//! This is what nonholonomic controllability analysis is written in. The iterated brackets
//! `{g₁, g₂, ad^i_{g₁} g₂}` that make [`crate::ChainedForm`] controllable are Hall elements, and the
//! Lie-algebraic rank condition asks how many independent ones there are at a point.
//!
//! # The definition (MLS §7.4.3)
//!
//! An ordered set of Lie products `H = {Bᵢ}` with:
//!
//! 1. every generator `gᵢ ∈ H`;
//! 2. if `l(Bᵢ) < l(Bⱼ)` then `Bᵢ < Bⱼ` — the order refines length;
//! 3. `[Bᵢ, Bⱼ] ∈ H` **iff** `Bᵢ, Bⱼ ∈ H` and `Bᵢ < Bⱼ`, **and** either `Bⱼ` is a generator, or
//!    `Bⱼ = [B_l, B_r]` with `B_l ≤ Bᵢ`.
//!
//! Condition 3 is the whole content: it is what removes the Jacobi-redundant element. MLS's worked
//! Example 7.12 notes that `[g₁, [g₂, g₃]]` is absent from the order-3 basis on three generators precisely
//! because `[g₁,[g₂,g₃]] + [g₂,[g₃,g₁]] + [g₃,[g₁,g₂]] = 0` and the other two are already present.
//!
//! # Verified against Witt's formula, not just against the book's table
//!
//! The number of basis elements of length `n` on `m` generators is
//! `(1/n)·Σ_{d|n} μ(d)·m^{n/d}` — Witt's formula, from the theory rather than from this construction. For
//! `m = 3` it gives 3, 3, 8 at lengths 1, 2, 3, totalling the 14 elements MLS lists. Checking the built basis
//! against Witt at every length and several `m` tests the construction against mathematics; checking it
//! against the printed table alone would only test transcription.
//!
//! **One presentational wrinkle in the book.** Example 7.12's length-2 row reads `[g₁,g₂] [g₂,g₃] [g₃,g₁]`,
//! but `[g₃,g₁]` violates condition 3(a) — it needs `Bᵢ < Bⱼ`. The length-3 row settles it: it contains
//! `[g₁,[g₁,g₃]]`, which requires `[g₁,g₃] ∈ H`. So the printed `[g₃,g₁]` is a cyclic display choice for the
//! same element up to sign, and the basis element proper is `[g₁,g₃]`. This construction emits the latter.

use std::cmp::Ordering;

/// A formal Lie product over generators indexed `0..m`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LieProduct {
    /// A generator `gᵢ`.
    Generator(usize),
    /// `[left, right]`.
    Bracket(Box<LieProduct>, Box<LieProduct>),
}

impl LieProduct {
    /// `l(gᵢ) = 1`, `l([A,B]) = l(A) + l(B)` — the number of generators in the expansion.
    pub fn length(&self) -> usize {
        match self {
            LieProduct::Generator(_) => 1,
            LieProduct::Bracket(a, b) => a.length() + b.length(),
        }
    }

    /// Bracket halves, if this is a bracket.
    pub fn parts(&self) -> Option<(&LieProduct, &LieProduct)> {
        match self {
            LieProduct::Bracket(a, b) => Some((a, b)),
            LieProduct::Generator(_) => None,
        }
    }

    /// Bracket notation, e.g. `[g1,[g1,g2]]`. Generators are 1-indexed to match the book.
    pub fn to_notation(&self) -> String {
        match self {
            LieProduct::Generator(i) => format!("g{}", i + 1),
            LieProduct::Bracket(a, b) => format!("[{},{}]", a.to_notation(), b.to_notation()),
        }
    }
}

/// The Hall order: **length first** (MLS condition 2), then a fixed structural tiebreak.
///
/// Condition 2 pins only the across-length part; within a length any consistent total order gives a valid
/// basis, so the tiebreak is a choice rather than a theorem. Generators compare by index, generators precede
/// brackets, and brackets compare left half then right half.
impl Ord for LieProduct {
    fn cmp(&self, other: &Self) -> Ordering {
        self.length().cmp(&other.length()).then_with(|| match (self, other) {
            (LieProduct::Generator(a), LieProduct::Generator(b)) => a.cmp(b),
            (LieProduct::Generator(_), LieProduct::Bracket(..)) => Ordering::Less,
            (LieProduct::Bracket(..), LieProduct::Generator(_)) => Ordering::Greater,
            (LieProduct::Bracket(a1, b1), LieProduct::Bracket(a2, b2)) => {
                a1.cmp(a2).then_with(|| b1.cmp(b2))
            }
        })
    }
}
impl PartialOrd for LieProduct {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Whether `[bi, bj]` belongs in the basis, given both are already in it — **MLS condition 3**.
///
/// `bi < bj`, and either `bj` is a generator, or `bj = [b_l, b_r]` with `b_l ≤ bi`. That second clause is what
/// eliminates the Jacobi-redundant element: it is the reason `[g₁,[g₂,g₃]]` is excluded while `[g₂,[g₁,g₃]]`
/// is kept.
pub fn admissible(bi: &LieProduct, bj: &LieProduct) -> bool {
    if bi >= bj {
        return false;
    }
    match bj {
        LieProduct::Generator(_) => true,
        LieProduct::Bracket(bl, _) => bl.as_ref() <= bi,
    }
}

/// Build the Philip Hall basis on `m` generators, nilpotent of order `order` (all products of length `> order`
/// treated as zero).
///
/// Elements come out sorted by the Hall order, so they are grouped by length. Returns an empty vector for
/// `m == 0` or `order == 0`.
pub fn hall_basis(m: usize, order: usize) -> Vec<LieProduct> {
    if m == 0 || order == 0 {
        return Vec::new();
    }
    let mut basis: Vec<LieProduct> = (0..m).map(LieProduct::Generator).collect();
    // Build by increasing length: a length-n element brackets a length-i with a length-(n−i), both already
    // present, so one pass per length suffices.
    for n in 2..=order {
        let mut fresh = Vec::new();
        for bi in &basis {
            for bj in &basis {
                if bi.length() + bj.length() != n {
                    continue;
                }
                if admissible(bi, bj) {
                    fresh.push(LieProduct::Bracket(Box::new(bi.clone()), Box::new(bj.clone())));
                }
            }
        }
        basis.extend(fresh);
    }
    basis.sort();
    basis
}

/// **Witt's formula**: the number of Hall basis elements of length `n` on `m` generators,
/// `(1/n)·Σ_{d|n} μ(d)·m^{n/d}`.
///
/// An independent count from the theory of free Lie algebras, used to check [`hall_basis`] against
/// mathematics rather than against its own output.
pub fn witt_dimension(m: usize, n: usize) -> usize {
    if n == 0 || m == 0 {
        return 0;
    }
    // Möbius μ(d)
    fn mobius(mut d: usize) -> i64 {
        let mut primes = 0;
        let mut p = 2;
        while p * p <= d {
            if d.is_multiple_of(p) {
                d /= p;
                if d.is_multiple_of(p) {
                    return 0; // squared prime factor
                }
                primes += 1;
            }
            p += 1;
        }
        if d > 1 {
            primes += 1;
        }
        if primes % 2 == 0 { 1 } else { -1 }
    }
    let mut total: i64 = 0;
    for d in 1..=n {
        if n.is_multiple_of(d) {
            total += mobius(d) * (m as i64).pow((n / d) as u32);
        }
    }
    (total / n as i64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_basis_matches_witts_formula_at_every_length() {
        // THE test: an independent count from the theory of free Lie algebras. Checking against the book's
        // printed table alone would only verify transcription.
        for m in 2..=4 {
            for order in 1..=5 {
                let basis = hall_basis(m, order);
                for n in 1..=order {
                    let built = basis.iter().filter(|b| b.length() == n).count();
                    let witt = witt_dimension(m, n);
                    assert_eq!(
                        built, witt,
                        "m={m} order={order} length={n}: built {built} elements, Witt's formula gives {witt}"
                    );
                }
                let total: usize = (1..=order).map(|n| witt_dimension(m, n)).sum();
                assert_eq!(basis.len(), total, "m={m} order={order}: total size");
            }
        }
    }

    #[test]
    fn witt_reproduces_the_values_the_theory_fixes() {
        // Sanity-check the oracle itself before trusting it. m generators: n=1 gives m; n=2 gives
        // m(m-1)/2; n=3 gives (m^3 - m)/3.
        for m in 1..=5 {
            assert_eq!(witt_dimension(m, 1), m);
            assert_eq!(witt_dimension(m, 2), m * (m - 1) / 2);
            assert_eq!(witt_dimension(m, 3), (m * m * m - m) / 3);
        }
        // MLS Example 7.12: three generators, order 3 => 3 + 3 + 8 = 14.
        assert_eq!(witt_dimension(3, 1), 3);
        assert_eq!(witt_dimension(3, 2), 3);
        assert_eq!(witt_dimension(3, 3), 8);
    }

    #[test]
    fn it_reproduces_mls_example_7_12() {
        // Three generators, nilpotent of order 3. MLS lists 14 elements.
        let basis = hall_basis(3, 3);
        assert_eq!(basis.len(), 14, "MLS Example 7.12 lists 14 elements, got {}", basis.len());
        let names: Vec<String> = basis.iter().map(|b| b.to_notation()).collect();

        // Every element the book prints, in the canonical Bi < Bj orientation. Note [g3,g1] in the book's
        // length-2 row is a cyclic display choice: condition 3(a) requires Bi < Bj, and the length-3 row's
        // [g1,[g1,g3]] confirms [g1,g3] is the basis element.
        for want in [
            "g1", "g2", "g3",
            "[g1,g2]", "[g1,g3]", "[g2,g3]",
            "[g1,[g1,g2]]", "[g1,[g1,g3]]", "[g2,[g1,g2]]", "[g2,[g1,g3]]",
            "[g2,[g2,g3]]", "[g3,[g1,g2]]", "[g3,[g1,g3]]", "[g3,[g2,g3]]",
        ] {
            assert!(names.contains(&want.to_string()), "missing {want} from {names:?}");
        }

        // **The Jacobi exclusion, which is the point of condition 3.** [g1,[g2,g3]] must be ABSENT, because
        // [g1,[g2,g3]] + [g2,[g3,g1]] + [g3,[g1,g2]] = 0 and the other two are present.
        assert!(!names.contains(&"[g1,[g2,g3]]".to_string()), "Jacobi-redundant element should be excluded");
        // and the mechanism is condition 3(b): g1 is not >= g2, the left half of [g2,g3]
        let g1 = LieProduct::Generator(0);
        let g2g3 = LieProduct::Bracket(Box::new(LieProduct::Generator(1)), Box::new(LieProduct::Generator(2)));
        assert!(!admissible(&g1, &g2g3), "condition 3(b) is what excludes it");
    }

    #[test]
    fn skew_symmetry_is_never_double_counted() {
        // Condition 3(a) requires Bi < Bj, so a bracket and its negation can never both appear.
        let basis = hall_basis(4, 4);
        for b in &basis {
            if let Some((l, r)) = b.parts() {
                assert!(l < r, "{} violates Bi < Bj", b.to_notation());
                let flipped = LieProduct::Bracket(Box::new(r.clone()), Box::new(l.clone()));
                assert!(!basis.contains(&flipped), "both {} and its flip are present", b.to_notation());
            }
        }
        // and no duplicates at all
        let mut sorted = basis.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), basis.len(), "the basis contains duplicates");
    }

    #[test]
    fn the_order_refines_length_and_degenerate_input_is_empty() {
        // MLS condition 2: l(Bi) < l(Bj) implies Bi < Bj.
        let basis = hall_basis(3, 4);
        for w in basis.windows(2) {
            assert!(w[0] <= w[1], "not sorted");
            assert!(w[0].length() <= w[1].length(), "order must refine length");
        }
        assert!(hall_basis(0, 3).is_empty(), "no generators, no basis");
        assert!(hall_basis(3, 0).is_empty(), "order 0 admits nothing");
        assert_eq!(hall_basis(1, 5).len(), 1, "one generator brackets to zero, so only g1 survives");
    }
}
