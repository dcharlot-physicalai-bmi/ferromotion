//! **Hybrid zero dynamics** — the certificate reduction that makes legged verification tractable, and
//! the condition it depends on.
//!
//! A full-order legged robot has tens of states, and certifying a periodic gait in all of them is what
//! `quadruped_saltation_monodromy` had to do the hard way. The classical alternative is a reduction.
//! Impose relative-degree-two virtual constraints `y = h(q)` and drive them to zero; the closed loop then
//! lives on the **zero-dynamics manifold** `Z = {h = 0, L_f h = 0}`, and if the impact map also carries
//! `Z` into itself — *hybrid* invariance, `Δ(S ∩ Z) ⊆ Z` — then something strong follows.
//!
//! **Morris and Grizzle (2009):** under hybrid invariance the linearised full Poincaré map is
//! **block-triangular**, so a periodic orbit inside `Z` is exponentially stable *for the full hybrid model*
//! if and only if the restricted return map is. Full-order orbital stability collapses to a
//! one-dimensional question. That is the reduction the certified-control stack leans on.
//!
//! The load-bearing word is *if*. This module supplies the reduction and, just as importantly,
//! [`HzdReduction`] measures whether its hypothesis actually holds, because a controller that renders no
//! `Z` — a clocked gait, or a learned policy trained on output error — destroys block-triangularity, and
//! then reading stability off the restricted map gives the wrong answer. The test in this module
//! demonstrates exactly that failure: a reduction whose restricted block says "stable" while the full map
//! is unstable through the coupling the reduction assumed away.
//!
//! Also here, from the same family: the **minimum-phase obstruction**. Output tracking accuracy is not
//! stability. [`zero_dynamics`] extracts the internal dynamics left uncontrolled when the output is held
//! at zero, and [`is_minimum_phase`] decides whether stabilising the output stabilises the robot at all.

use nalgebra::DMatrix;

/// The reduction of a full monodromy to its zero-dynamics block, together with the evidence for whether
/// the reduction is valid. Produced by [`hzd_reduction`].
#[derive(Clone, Debug)]
pub struct HzdReduction {
    /// Spectral radius of the restricted return map — the reduced, low-dimensional certificate.
    pub restricted_rho: f64,
    /// Spectral radius of the transverse block: how the directions off `Z` behave.
    pub transverse_rho: f64,
    /// Spectral radius of the full monodromy, computed without any reduction.
    pub full_rho: f64,
    /// **How far the monodromy is from block-triangular**, relative to its own size: the norm of the block
    /// mapping `Z` directions into the *transverse* space, over the norm of the whole matrix.
    ///
    /// This is the `Z → W` block and not the `W → Z` one, which is the direction the theorem is about:
    /// hybrid invariance says `Π` carries `Z` into `Z`, so a vector starting in `Z` must have no transverse
    /// component afterwards. (Both blocks vanishing would split the eigenvalues, so the eigenvalue test alone
    /// cannot tell the two apart — only the invariance reading can, and it picks this one.)
    pub coupling: f64,
    /// Whether the reduction may be trusted, i.e. [`coupling`](Self::coupling) is within the tolerance
    /// supplied. When this is false, [`restricted_rho`](Self::restricted_rho) is **not** a certificate for
    /// the full system, and the difference can be the difference between stable and unstable.
    pub valid: bool,
}

impl HzdReduction {
    /// Whether the full orbit is certified exponentially stable *by the reduction*.
    ///
    /// Three conditions, and all three are load-bearing. The reduction must be valid (the monodromy really is
    /// block-triangular, so the split means something), the restricted map must contract, **and the transverse
    /// block must contract too**.
    ///
    /// That third one is easy to omit and unsound to omit. Block-triangularity makes the full spectral radius
    /// the *larger* of the two blocks, so a contracting restricted map alongside an expanding transverse block
    /// is an unstable orbit. A compass-gait run with a trained constraint hit exactly that: restricted 0.8248,
    /// transverse 25.81, and an earlier version of this predicate called it certified. Whether the transverse
    /// block contracts is what a RES-CLF's `ε` buys, and a bad `ε` is precisely how it fails.
    pub fn certified(&self) -> bool {
        self.valid && self.restricted_rho < 1.0 && self.transverse_rho < 1.0
    }
}

/// **Morris-Grizzle reduction of a monodromy.** `monodromy` is the linearised return map, `z_basis` an
/// orthonormal basis (columns) of the zero-dynamics manifold's tangent space at the fixed point.
///
/// Changes coordinates to `[Z W]` with `W` an orthonormal complement, reads off the two diagonal blocks,
/// and measures the off-diagonal block that block-triangularity requires to vanish. `tol` is the relative
/// coupling below which the reduction is accepted.
///
/// Returns `None` if the shapes disagree or `z_basis` is not of full column rank.
pub fn hzd_reduction(monodromy: &DMatrix<f64>, z_basis: &DMatrix<f64>, tol: f64) -> Option<HzdReduction> {
    let n = monodromy.nrows();
    if monodromy.ncols() != n || z_basis.nrows() != n || z_basis.ncols() == 0 || z_basis.ncols() >= n {
        return None;
    }
    let z = orthonormalize(z_basis)?;
    let w = orthonormal_complement(&z)?;
    let k = z.ncols();

    // Full change of basis. Orthonormal, so the inverse is the transpose and no conditioning is lost.
    let mut t = DMatrix::zeros(n, n);
    t.view_mut((0, 0), (n, k)).copy_from(&z);
    t.view_mut((0, k), (n, n - k)).copy_from(&w);
    let m = t.transpose() * monodromy * &t;

    // In these coordinates, with a vector written as (z, w):
    //   [ Z->Z    W->Z ]
    //   [ Z->W    W->W ]
    // Invariance of Z says that starting from (z, 0) the result has no w component, so the block that must
    // vanish is the lower-left one, Z->W. Reading the upper-right block instead is self-consistent and wrong:
    // it also splits the eigenvalues, so the spectral test cannot catch the mistake, and a genuinely reduced
    // gait then reports itself uncertifiable. That is exactly what the compass-gait run found.
    let zz = m.view((0, 0), (k, k)).into_owned();
    let zw = m.view((k, 0), (n - k, k)).into_owned();
    let ww = m.view((k, k), (n - k, n - k)).into_owned();

    let scale = m.norm().max(1e-30);
    let coupling = zw.norm() / scale;
    Some(HzdReduction { restricted_rho: spectral_radius(&zz), transverse_rho: spectral_radius(&ww), full_rho: spectral_radius(monodromy), coupling, valid: coupling <= tol })
}

/// **Hybrid invariance residual**: how far the reset map takes points of `S ∩ Z` off `Z`.
///
/// `Δ(S ∩ Z) ⊆ Z` is the condition that makes the zero-dynamics manifold survive the impact, and without
/// it the reduction above has no basis. Given the reset map's linearisation `reset` and an orthonormal
/// basis `z_basis` of `Z`, this returns the norm of the component of `Δ Z` lying outside `Z`, relative to
/// the size of `Δ Z`. Zero means the impact preserves `Z` exactly.
pub fn hybrid_invariance_residual(reset: &DMatrix<f64>, z_basis: &DMatrix<f64>) -> Option<f64> {
    let z = orthonormalize(z_basis)?;
    if reset.nrows() != z.nrows() || reset.ncols() != z.nrows() {
        return None;
    }
    let image = reset * &z;
    // component of the image orthogonal to Z
    let outside = &image - &z * (z.transpose() * &image);
    Some(outside.norm() / image.norm().max(1e-30))
}

/// The **restricted return map in momentum coordinates**, which the hybrid-zero-dynamics construction makes
/// affine: `ρ(ζ) = δ² ζ − V`.
///
/// The remarkable part is where the contraction comes from. `δ²` is set by the *impact*, not by continuous
/// feedback — the swing leg's collision with the ground is what shrinks the orbit's deviation. Continuous
/// feedback only holds the robot on `Z`.
#[derive(Clone, Copy, Debug)]
pub struct ZeroDynamicsReturnMap {
    /// The impact's contraction factor on squared momentum, `δ²_zero`.
    pub delta_sq: f64,
    /// The potential term, `V_zero`: the energy the gait must supply per step.
    pub v_zero: f64,
}

impl ZeroDynamicsReturnMap {
    pub fn apply(&self, zeta: f64) -> f64 {
        self.delta_sq * zeta - self.v_zero
    }

    /// The fixed point `ζ* = −V/(1 − δ²)`, i.e. the periodic gait, when one exists. `None` at `δ² = 1`,
    /// where the map is a pure translation and no fixed point exists.
    pub fn fixed_point(&self) -> Option<f64> {
        let d = 1.0 - self.delta_sq;
        if d.abs() < 1e-12 {
            return None;
        }
        Some(-self.v_zero / d)
    }

    /// Whether the orbit is exponentially stable **and physically realisable**: `0 < δ² < 1` for
    /// contraction, plus the energy condition that the fixed point is a positive squared momentum. A
    /// `δ² < 1` with a non-positive fixed point is a contracting map with no gait on it.
    pub fn exponentially_stable(&self) -> bool {
        self.delta_sq > 0.0 && self.delta_sq < 1.0 && self.fixed_point().is_some_and(|z| z > 0.0)
    }
}

/// **The zero dynamics** of a square LTI system `ẋ = Ax + Bu`, `y = Cx`: the internal dynamics that remain
/// when the output is held identically at zero.
///
/// Holding `y ≡ 0` forces `x ∈ ker C` and pins the input to `u = −(CB)⁻¹CAx`, leaving `ẋ = (I −
/// B(CB)⁻¹C)Ax` restricted to `ker C`. Returns that restricted matrix, whose eigenvalues are the system's
/// transmission zeros. `None` unless `CB` is invertible (relative degree one) and `C` has full row rank.
///
/// This is the object that decides whether output tracking is safe. A policy trained to match reference
/// *outputs* says nothing about these eigenvalues, and if any lies in the right half plane then driving the
/// output to zero drives the internal state to infinity.
pub fn zero_dynamics(a: &DMatrix<f64>, b: &DMatrix<f64>, c: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let m = b.ncols();
    if a.ncols() != n || b.nrows() != n || c.ncols() != n || c.nrows() != m || m >= n {
        return None;
    }
    let cb = c * b;
    let cb_inv = cb.clone().try_inverse()?; // relative degree one
    let p = DMatrix::identity(n, n) - b * cb_inv * c;
    let z = kernel_basis(c)?;
    // P A maps ker C into ker C (because C P = 0), so this restriction is exact rather than a projection.
    Some(z.transpose() * p * a * &z)
}

/// **The zero dynamics of a relative-degree-two output** — the case every mechanical system is in.
///
/// [`zero_dynamics`] requires `CB` invertible, which means the input reaches the output in one derivative. A
/// position output on a mechanical system does not: force reaches position through two integrations, so `CB = 0`
/// and that function returns `None`. Using a *velocity* output instead to dodge the requirement is a trap worth
/// naming — it constrains one velocity combination and leaves both positions free, so an unactuated tipping mode
/// survives at every choice of output weights and the sweep finds no safe one. A velocity constraint cannot pin a
/// position.
///
/// Here the output is held at zero along with its first derivative, so `y = Cx` and `ẏ = CAx` both vanish: two
/// constraints, leaving `n − 2` internal states. The input follows from `ÿ = 0`, giving
/// `u = −(CAB)⁻¹CA²x`, and the internal dynamics is the restriction of `A − B(CAB)⁻¹CA²` to `ker[C; CA]`.
///
/// `None` unless `CAB` is invertible — relative degree exactly two.
pub fn zero_dynamics_order2(a: &DMatrix<f64>, b: &DMatrix<f64>, c: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let m = b.ncols();
    if a.ncols() != n || b.nrows() != n || c.ncols() != n || c.nrows() != m || 2 * m >= n {
        return None;
    }
    let ca = c * a;
    let cab = &ca * b;
    let cab_inv = cab.clone().try_inverse()?; // relative degree exactly two
    let ca2 = &ca * a;
    let closed = a - b * cab_inv * &ca2;

    // the internal space is the joint kernel of C and CA
    let mut both = DMatrix::zeros(2 * m, n);
    both.view_mut((0, 0), (m, n)).copy_from(c);
    both.view_mut((m, 0), (m, n)).copy_from(&ca);
    let z = kernel_basis(&both)?;
    // The closed-loop map preserves that kernel by construction, so the restriction is exact.
    Some(z.transpose() * closed * &z)
}

/// Whether a relative-degree-two output is **minimum phase**: every zero-dynamics eigenvalue strictly in the left
/// half plane. The design question for a learned output-tracker on an underactuated robot — and answerable from
/// the plant, before any policy exists.
pub fn is_minimum_phase_order2(a: &DMatrix<f64>, b: &DMatrix<f64>, c: &DMatrix<f64>) -> Option<bool> {
    let zd = zero_dynamics_order2(a, b, c)?;
    Some(zd.complex_eigenvalues().iter().all(|l| l.re < 0.0))
}

/// Whether the system is **minimum phase**: every zero-dynamics eigenvalue strictly in the left half plane.
///
/// Input-output-linearising feedback stabilises the state if and only if this holds. When it fails, exact
/// causal tracking is internally unstable however small the output error becomes — and the minimum
/// achievable cost is bounded below by the energy needed to stabilise the unstable zero dynamics, a floor
/// no amount of controller tuning or policy training moves.
pub fn is_minimum_phase(a: &DMatrix<f64>, b: &DMatrix<f64>, c: &DMatrix<f64>) -> Option<bool> {
    let zd = zero_dynamics(a, b, c)?;
    Some(zd.complex_eigenvalues().iter().all(|l| l.re < 0.0))
}

/// Spectral radius: the largest eigenvalue magnitude.
fn spectral_radius(m: &DMatrix<f64>) -> f64 {
    m.complex_eigenvalues().iter().fold(0.0f64, |acc, l| acc.max(l.norm()))
}

/// Orthonormalise the columns, dropping none. `None` if the columns are rank-deficient.
fn orthonormalize(v: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let qr = v.clone().qr();
    let q = qr.q();
    let r = qr.r();
    let k = v.ncols();
    if (0..k).any(|i| r[(i, i)].abs() < 1e-12) {
        return None; // rank-deficient: not a basis
    }
    Some(q.columns(0, k).into_owned())
}

/// An orthonormal basis of the orthogonal complement of `z`'s column space.
fn orthonormal_complement(z: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = z.nrows();
    let k = z.ncols();
    if k >= n {
        return None;
    }
    // Project the identity off Z and pull an orthonormal basis out of the result by SVD, which handles the
    // rank drop cleanly rather than depending on which columns happen to be independent.
    let p = DMatrix::identity(n, n) - z * z.transpose();
    let svd = ferromotion_core::finite_svd(&p, true, false)?;
    let u = svd.u?;
    Some(u.columns(0, n - k).into_owned())
}

/// An orthonormal basis of `ker C`, as the orthogonal complement of `C`'s row space. Taken this way rather
/// than from the null rows of `V^T`, because nalgebra's SVD is thin: for a wide `C` the returned `V^T` has
/// only as many rows as there are singular values, so the null space is simply not in it.
fn kernel_basis(c: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = c.ncols();
    if c.nrows() >= n {
        return None; // no room for a kernel of a full-row-rank C
    }
    let row_space = orthonormalize(&c.transpose())?;
    orthonormal_complement(&row_space)
}

#[cfg(test)]
mod rd2_tests {
    use super::*;
    use nalgebra::DVector;

    /// **Relative degree two, against a hand-computable case.** A double integrator pair with the output taking a
    /// weighted difference: `ẍ₁ = u`, `ẍ₂ = k·x₂`, `y = x₁ + c·x₂`. Holding `y` and `ẏ` at zero forces
    /// `x₁ = −c·x₂`, leaving `x₂` with `ẍ₂ = k·x₂` — so the zero dynamics is `±√k` regardless of `c`, and the
    /// output choice cannot rescue an unactuated unstable mode it does not couple to.
    #[test]
    fn relative_degree_two_zero_dynamics_matches_a_hand_computable_case() {
        let k = 4.0f64;
        // state [x1, x2, v1, v2]
        let mut a = DMatrix::zeros(4, 4);
        a[(0, 2)] = 1.0;
        a[(1, 3)] = 1.0;
        a[(3, 1)] = k; // the unactuated unstable mode
        let mut b = DMatrix::zeros(4, 1);
        b[(2, 0)] = 1.0;

        for c in [0.0f64, 1.0, -2.5] {
            let cm = DMatrix::from_row_slice(1, 4, &[1.0, c, 0.0, 0.0]);
            let zd = zero_dynamics_order2(&a, &b, &cm).expect("relative degree two");
            assert_eq!(zd.shape(), (2, 2), "n - 2m = 2 internal states");
            let worst = zd.complex_eigenvalues().iter().fold(f64::NEG_INFINITY, |m, l| m.max(l.re));
            eprintln!("c = {c:>5}: zero-dynamics worst real part {worst:+.6} (analytic sqrt(k) = {:+.6})", k.sqrt());
            assert!((worst - k.sqrt()).abs() < 1e-9, "the zero dynamics must be +/-sqrt(k): got {worst}");
            assert!(!is_minimum_phase_order2(&a, &b, &cm).unwrap());
        }

        // A velocity output on the same plant has relative degree ONE and leaves three internal states - which is
        // the trap the doc comment warns about: it constrains a velocity and both positions stay free.
        let vel = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 1.0, 0.0]);
        let zd1 = zero_dynamics(&a, &b, &vel).expect("relative degree one");
        assert_eq!(zd1.shape(), (3, 3), "one constraint leaves three internal states");
        assert!(!is_minimum_phase(&a, &b, &vel).unwrap(), "and the unstable mode is still in there");
    }

    /// Holding the output at zero really does leave the internal dynamics the function predicts — checked by
    /// simulating the relative-degree-two feedback and comparing the growth rate to the eigenvalue.
    #[test]
    fn the_order_two_prediction_matches_a_simulation() {
        let k = 4.0f64;
        let mut a = DMatrix::zeros(4, 4);
        a[(0, 2)] = 1.0;
        a[(1, 3)] = 1.0;
        a[(3, 1)] = k;
        let mut b = DMatrix::zeros(4, 1);
        b[(2, 0)] = 1.0;
        let cm = DMatrix::from_row_slice(1, 4, &[1.0, 1.0, 0.0, 0.0]);
        let ca = &cm * &a;
        let cab = (&ca * &b)[0];
        let ca2 = &ca * &a;

        // start on the zero-dynamics manifold: y = 0 and ydot = 0
        let mut x = DVector::from_row_slice(&[-0.01, 0.01, -0.02, 0.02]);
        assert!((&cm * &x)[0].abs() < 1e-15 && (&ca * &x)[0].abs() < 1e-15);
        let dt = 1e-5;
        let mut worst_y = 0.0f64;
        for _ in 0..100_000 {
            let u = -(&ca2 * &x)[0] / cab;
            worst_y = worst_y.max((&cm * &x)[0].abs());
            x += (&a * &x + &b * DVector::from_row_slice(&[u])) * dt;
        }
        let growth = x.norm() / 0.0316_f64;
        eprintln!("held y to {worst_y:.2e} over 1 s; internal state grew by {growth:.3}x (analytic e^sqrt(k) = {:.3})", k.sqrt().exp());
        assert!(worst_y < 1e-9, "the output must stay at zero, drifted to {worst_y:.2e}");
        assert!((growth / k.sqrt().exp() - 1.0).abs() < 0.1, "the growth must match the zero-dynamics eigenvalue: {growth} vs {}", k.sqrt().exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn dm(r: usize, c: usize, v: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(r, c, v)
    }

    /// The affine restricted map, against its closed form. `δ² < 1` contracts and `δ² > 1` does not, and
    /// the fixed point is where the gait lives.
    #[test]
    fn the_restricted_return_map_matches_its_closed_form() {
        let m = ZeroDynamicsReturnMap { delta_sq: 0.5, v_zero: -1.0 };
        let zs = m.fixed_point().unwrap();
        assert!((zs - 2.0).abs() < 1e-12, "fixed point should be -V/(1-d2) = 2, got {zs}");
        assert!((m.apply(zs) - zs).abs() < 1e-12, "the fixed point must be fixed");
        assert!(m.exponentially_stable(), "0 < d2 < 1 with a positive fixed point is a stable gait");

        // iterating really does converge to it, at the rate d2
        let mut z = 5.0;
        for _ in 0..60 {
            z = m.apply(z);
        }
        assert!((z - 2.0).abs() < 1e-9, "iteration should converge to the fixed point, got {z}");

        // expansion at the impact means no stable gait, however the continuous feedback is tuned
        let unstable = ZeroDynamicsReturnMap { delta_sq: 1.5, v_zero: 1.0 };
        assert!(!unstable.exponentially_stable());
        // and a contracting map whose fixed point is a negative squared momentum carries no gait at all
        let empty = ZeroDynamicsReturnMap { delta_sq: 0.5, v_zero: 1.0 };
        assert_eq!(empty.fixed_point(), Some(-2.0));
        assert!(!empty.exponentially_stable(), "a negative squared momentum is not a gait");
        // delta_sq = 1 is a pure translation: no fixed point to be stable about
        assert!(ZeroDynamicsReturnMap { delta_sq: 1.0, v_zero: 0.3 }.fixed_point().is_none());
    }

    /// **The reduction, when its hypothesis holds.** Build a block-triangular map in `Z` coordinates,
    /// rotate it into general position, and check the reduction recovers both blocks and reports no
    /// coupling — and that the full spectral radius really is the larger of the two blocks'.
    #[test]
    fn the_reduction_is_exact_under_hybrid_invariance() {
        // In Z-coordinates: restricted block 0.4, transverse block [[0.7, 0.2], [0.1, 0.6]], and a W->Z
        // coupling that invariance permits. The Z->W block (the first column below the diagonal) is zero,
        // which is what invariance requires.
        let core = dm(3, 3, &[0.4, 1.3, -0.5, 0.0, 0.7, 0.2, 0.0, 0.1, 0.6]);
        // an orthogonal change of basis hides the structure without changing any eigenvalue
        let q = orthonormalize(&dm(3, 3, &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])).unwrap();
        let mono = &q * &core * q.transpose();
        let z_basis = q.columns(0, 1).into_owned(); // Z is the image of the first coordinate

        let r = hzd_reduction(&mono, &z_basis, 1e-9).unwrap();
        eprintln!("hybrid-invariant case: restricted {:.4}, transverse {:.4}, full {:.4}, coupling {:.2e}", r.restricted_rho, r.transverse_rho, r.full_rho, r.coupling);
        assert!(r.coupling < 1e-9, "the map is block-triangular, coupling should vanish: {:.2e}", r.coupling);
        assert!(r.valid && r.certified(), "a contracting restricted block with a contracting transverse block under invariance is a certificate");
        assert!((r.restricted_rho - 0.4).abs() < 1e-9, "restricted block is 0.4, got {}", r.restricted_rho);
        // the point of the reduction: the full answer is the max of the blocks, so the 1-D map suffices
        assert!((r.full_rho - r.restricted_rho.max(r.transverse_rho)).abs() < 1e-9, "full rho must be the larger block under block-triangularity");
    }

    /// **The reduction when its hypothesis fails, which is the whole reason to measure it.**
    ///
    /// Here the restricted block reads 0.4 — comfortably contracting — while the full map is unstable
    /// through the transverse-to-`Z` coupling that block-triangularity assumes away. Anyone reading
    /// stability off the reduced map without checking would certify an unstable gait.
    #[test]
    fn a_stable_restricted_block_does_not_imply_a_stable_full_map() {
        // Z->Z is 0.4, W->W is 0.5, but the two off-diagonal couplings multiply to something large. The Z->W
        // entry (lower-left, 3.0) is the one invariance forbids.
        let mono = dm(2, 2, &[0.4, 6.0, 3.0, 0.5]);
        let z_basis = dm(2, 1, &[1.0, 0.0]);
        let r = hzd_reduction(&mono, &z_basis, 1e-6).unwrap();
        eprintln!("coupled case: restricted {:.4}, transverse {:.4}, FULL {:.4}, coupling {:.3}", r.restricted_rho, r.transverse_rho, r.full_rho, r.coupling);

        assert!((r.restricted_rho - 0.4).abs() < 1e-12, "the restricted block is contracting");
        assert!(r.coupling > 0.1, "the coupling is large and must be reported: {}", r.coupling);
        assert!(!r.valid, "the reduction must refuse itself when the map is not block-triangular");
        assert!(!r.certified(), "and refuse to certify, even though the restricted block contracts");
        assert!(r.full_rho > 1.0, "the full map really is unstable: rho = {}", r.full_rho);
        // the sharp statement: the reduced answer and the true answer disagree about stability itself
        assert!(r.restricted_rho < 1.0 && r.full_rho > 1.0, "reduced says stable, full says unstable - which is why the hypothesis is load-bearing");
    }

    /// **A contracting restricted block is not enough: the transverse block has to contract too.**
    ///
    /// Under block-triangularity the full spectral radius is the larger of the two blocks, so a reduction that
    /// reports only the restricted one can certify an unstable orbit. This is the case that caught it — and it
    /// is not hypothetical: a trained compass-gait constraint produced restricted 0.8248 against transverse
    /// 25.81 at a RES-CLF `ε` that was too large.
    #[test]
    fn a_contracting_restricted_block_with_an_expanding_transverse_one_is_not_certified() {
        // exactly block-triangular in the sense invariance requires (Z->W is zero), restricted 0.6,
        // transverse 4.0 - so the reduction is *valid* and the orbit is still unstable
        let mono = dm(2, 2, &[0.6, 1.1, 0.0, 4.0]);
        let z_basis = dm(2, 1, &[1.0, 0.0]);
        let r = hzd_reduction(&mono, &z_basis, 1e-9).unwrap();
        eprintln!("valid reduction, expanding transverse: restricted {:.4}, transverse {:.4}, full {:.4}, coupling {:.2e}", r.restricted_rho, r.transverse_rho, r.full_rho, r.coupling);
        assert!(r.valid, "the reduction itself is legitimate here: the coupling really is zero");
        assert!(r.restricted_rho < 1.0, "and the restricted block contracts");
        assert!(r.transverse_rho > 1.0 && r.full_rho > 1.0, "but the orbit is unstable through the transverse block");
        assert!(!r.certified(), "so it must NOT be certified - this is the unsoundness the predicate has to avoid");
    }

    /// Hybrid invariance is a property of the reset map, and the residual detects its failure. A reset that
    /// keeps `Z` invariant scores zero; one that tilts `Z` out of itself does not.
    #[test]
    fn the_invariance_residual_detects_a_reset_that_leaves_the_manifold() {
        let z = dm(3, 1, &[1.0, 0.0, 0.0]);
        // a reset that maps the Z direction back into Z (scaling along it, plus anything on the complement)
        let good = dm(3, 3, &[0.8, 0.3, -0.1, 0.0, 0.5, 0.2, 0.0, 0.1, 0.7]);
        let r_good = hybrid_invariance_residual(&good, &z).unwrap();
        assert!(r_good < 1e-12, "this reset preserves Z, residual should vanish: {r_good:.2e}");

        // tilt it: now the image of Z has a component off Z
        let bad = dm(3, 3, &[0.8, 0.3, -0.1, 0.6, 0.5, 0.2, 0.0, 0.1, 0.7]);
        let r_bad = hybrid_invariance_residual(&bad, &z).unwrap();
        eprintln!("invariance residual: {r_good:.2e} for a Z-preserving reset, {r_bad:.4} for a tilted one");
        assert!(r_bad > 0.5, "a reset that takes Z off Z must be flagged: {r_bad}");
    }

    /// **Zero dynamics against an analytic transfer function.** The realisation below is
    /// `(s − 1)/(s² + 3s + 2)`, whose single zero sits at `s = +1`: non-minimum phase. The zero-dynamics
    /// eigenvalue must be exactly that zero.
    #[test]
    fn zero_dynamics_eigenvalues_are_the_transmission_zeros() {
        let a = dm(2, 2, &[0.0, 1.0, -2.0, -3.0]);
        let b = dm(2, 1, &[0.0, 1.0]);
        let c = dm(1, 2, &[-1.0, 1.0]); // numerator s - 1

        let zd = zero_dynamics(&a, &b, &c).unwrap();
        assert_eq!(zd.shape(), (1, 1), "n - m = 1 internal state");
        eprintln!("zero dynamics of (s-1)/(s^2+3s+2): eigenvalue {:.10} (analytic zero is +1)", zd[(0, 0)]);
        assert!((zd[(0, 0)] - 1.0).abs() < 1e-9, "the transmission zero is +1, got {}", zd[(0, 0)]);
        assert!(!is_minimum_phase(&a, &b, &c).unwrap(), "a right-half-plane zero is not minimum phase");

        // Flip the numerator to (s + 1): same A and B, same output-tracking problem, now minimum phase.
        // Nothing about the output error distinguishes these two systems, and only one is safe to invert.
        let c_min = dm(1, 2, &[1.0, 1.0]);
        let zd_min = zero_dynamics(&a, &b, &c_min).unwrap();
        assert!((zd_min[(0, 0)] + 1.0).abs() < 1e-9, "zero at -1, got {}", zd_min[(0, 0)]);
        assert!(is_minimum_phase(&a, &b, &c_min).unwrap());
    }

    /// Holding the output at zero really does drive the internal state the way [`zero_dynamics`] predicts.
    /// This simulates the closed loop rather than trusting the algebra: an unstable zero dynamics diverges
    /// while the tracked output stays pinned at zero, which is the failure that output error cannot see.
    #[test]
    fn output_held_at_zero_diverges_when_the_zero_dynamics_is_unstable() {
        let a = dm(2, 2, &[0.0, 1.0, -2.0, -3.0]);
        let b = dm(2, 1, &[0.0, 1.0]);
        let c = dm(1, 2, &[-1.0, 1.0]);
        let cb_inv = (&c * &b).try_inverse().unwrap();

        // start on ker C, apply the output-zeroing input exactly, integrate
        let mut x = DVector::from_row_slice(&[1.0, 1.0]); // C x = 0
        let dt = 1e-4;
        let mut worst_output = 0.0f64;
        for _ in 0..40_000 {
            let u = -&cb_inv * &c * &a * &x; // the input that holds y at zero
            let xd = &a * &x + &b * &u;
            x += xd * dt;
            worst_output = worst_output.max((&c * &x)[0].abs());
        }
        eprintln!("output held to within {worst_output:.2e} while the internal state reached {:.3e}", x.norm());
        assert!(worst_output < 1e-6, "the output was supposed to stay at zero, drifted to {worst_output:.2e}");
        // 4 s at rate +1 is a factor of about e^4 = 55
        assert!(x.norm() > 30.0, "an unstable zero dynamics must diverge: reached {:.3e}", x.norm());
    }
}
