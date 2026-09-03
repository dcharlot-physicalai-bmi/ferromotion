//! **Elementary transform sequences (ETS)**: a robot written as a string of pure translations and
//! rotations about the local `x`, `y`, `z` axes, each by a constant or by one joint variable.
//!
//! The object is the one Haviland & Corke define in *Manipulator Differential Kinematics Part 1*
//! (arXiv 2207.01796v2, eqs (1)–(5)), attributed there to Corke's "walk-through" (IEEE T-RO 23(3),
//! 2007): `⁰T_e = E_1(η_1) E_2(η_2) … E_M(η_M)` with each `E_i` one of `T_tx, T_ty, T_tz, T_Rx, T_Ry,
//! T_Rz` and each `η_i` either a constant `c_i` or a joint variable `q_j`, every `q_j` in exactly one
//! factor. A factor's joint may run in the negative sense of its axis (Part 1, remark after eq (22):
//! the Panda's `E_7` and `E_11`), written here as `Ry(-q4)`.
//!
//! # What this module is, and is not
//!
//! This crate's [`Joint`] `{ origin, axis, kind }` with [`Robot`] `{ joints, ee_offset }` is already
//! the collapsed form of an ETS: each joint is "constant factors, then one variable factor", and
//! `ee_offset` is the trailing tool constant. So ETS is provided as a **notation layer** — parse,
//! print, collapse ([`Ets::to_robot`]), expand ([`Ets::from_robot`]), and the DH-row expansion
//! ([`Ets::from_dh`]) — and as an **independent oracle** for the differential kinematics:
//! [`Ets::jacobian`] and [`Ets::hessian`] evaluate the papers' matrix-product equations as written
//! (Part 1 eqs (26)–(30); Part 2 eqs (6), (17), (19)), with 4×4 generator matrices and no cross
//! products, so that they and the vector-form [`Robot::jacobian`] / [`Robot::kinematic_hessian`]
//! are two differently structured derivations of the same quantities. Building a robot from a DH
//! table stays on the direct path in [`crate::dh`]; nothing here is on any hot path.
//!
//! # Notation accepted by [`Ets::parse`]
//!
//! Whitespace-separated factors `Tx(η) Ty(η) Tz(η) Rx(η) Ry(η) Rz(η)` (Corke 2007's `T_k`, `R_k`;
//! Part 1's `T_tk`, `T_Rk` spellings `tx` and `T_tx`, `T_Rx` are accepted as well). `η` is a decimal
//! constant (metres or radians), one of `pi`, `-pi`, `pi/2`, `-pi/2`, or a joint variable `q1 … qn`
//! (1-based, as the papers number them; stored 0-based as the index into `q`), optionally negated
//! for a negative-sense joint: `Ry(-q4)`. [`Ets`]'s `Display` prints exactly this form, so a string
//! written that way parses and prints back identically (verified on the Panda string below).
//!
//! # The Panda of Part 1, Figure 1
//!
//! ```text
//! Tz(0.333) Rz(q1) Ry(q2) Tz(0.316) Rz(q3) Tx(0.0825) Ry(-q4) Tx(-0.0825) Tz(0.384) Rz(q5) Ry(-q6) Tx(0.088) Rx(pi) Tz(0.107) Rz(q7)
//! ```
//!
//! The lengths are the ones Part 1 names for the figure (0.333, 0.316, 0.0825, 0.384, 0.088, 0.107 m);
//! `E_10 = Rz(q5)` gives the paper's `μ(5) = 10`, and `E_7`, `E_11` are its two negative-sense joints.
//! The string is *derived by hand* in the tests from Franka's published modified-DH table and
//! verified against it: the forward kinematics of the collapsed ETS and of the DH-built arm agree to
//! `5.1e-16` (largest homogeneous-matrix entry difference) at three configurations, including the
//! table's hand-computed `q = 0` answer `(0.088, 0, 0.926)` and its in-limits check pose
//! `(0.306891, 0, 0.590282)`.
//!
//! # What is verified, against what
//!
//! - [`Ets::jacobian`] and [`Ets::hessian`] against [`Robot::jacobian`] and
//!   [`Robot::kinematic_hessian`] on the Panda ETS at three configurations: largest entry-wise
//!   difference `5.6e-16` (Jacobian) and `1.0e-15` (Hessian) on entries of size up to `1.0`; and on a
//!   six-joint arm with prismatic joints along `x`, `y`, `z` (one negative-sense) and revolute joints
//!   about `x`, `y`: `2.2e-16` for both (Hessian entries up to `0.92`), with the oracle Jacobian also
//!   within `1.7e-10` of central differences of the ETS product at step `1e-6`. Tolerance in the
//!   tests is `1e-9`.
//! - [`Ets::from_dh`] against [`Robot::from_dh`] joint-for-joint on the arms `dh.rs` tests, both
//!   conventions: origins, axes, kinds and tool bit-identical (gap `0`), forward kinematics within
//!   `3.3e-16` over 30 poses.
//! - [`Ets::from_robot`] ∘ [`Ets::to_robot`] preserves forward kinematics: bit-identical on the Panda,
//!   within `8.9e-16` on an arm with a non-axis-aligned joint and a gimbal-locked origin. The
//!   constant decomposition itself recomposes within `6.7e-16` over 400 rotations (69 of them in the
//!   `Ry Rz Rx` branch).
//! - Negative-sense factors: `Ry(-q)` equals a joint with axis `−y` and equals `Ry(q)` at `−q`.
//!
//! Mutation-checked (each deliberately broken, tests observed failing, restored): ignoring the
//! negative-sense flag in the collapse (Panda cross-check fails at every non-zero `q`); dropping the
//! second-order term `H_R·ρ(T)ᵀ` of Part 2 eq (17), or the second-derivative factor `R̂²T_R` of
//! eq (6) (oracle Hessian test fails).

use crate::dh::{DhConvention, DhRow};
use crate::{Iso, Joint, JointKind, Robot};
use core::fmt;
use nalgebra::{DMatrix, Matrix3, Matrix4, Translation3, UnitQuaternion, Vector3};
use std::f64::consts::{FRAC_PI_2, PI};

/// The parameter `η_i` of one elementary transform: a constant, or one joint variable (Part 1 eq (3)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Eta {
    /// A fixed offset (metres) or rotation (radians).
    Constant(f64),
    /// The joint variable `q[index]` (printed 1-based as `q{index+1}`). `negative` marks a joint whose
    /// positive motion is a negative rotation about, or translation along, the axis (Part 1's `R̂ᵀ`
    /// rule after eq (22)); the factor is then `T(−q)`.
    Joint { index: usize, negative: bool },
}

/// One elementary transform `E_i` (Part 1 eq (2)): a pure translation along, or rotation about, the
/// local `x`, `y` or `z` axis by [`Eta`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Et {
    Tx(Eta),
    Ty(Eta),
    Tz(Eta),
    Rx(Eta),
    Ry(Eta),
    Rz(Eta),
}

/// An elementary transform sequence: the ordered factors of Part 1 eq (1).
#[derive(Clone, Debug, PartialEq)]
pub struct Ets(pub Vec<Et>);

// ---- the six elementary transforms (Part 1 Fig. 2; the standard right-handed active forms whose
// derivatives are the generators of eqs (20)–(25)) ----

fn tx(v: f64) -> Iso {
    Iso::from_parts(Translation3::new(v, 0.0, 0.0), UnitQuaternion::identity())
}
fn ty(v: f64) -> Iso {
    Iso::from_parts(Translation3::new(0.0, v, 0.0), UnitQuaternion::identity())
}
fn tz(v: f64) -> Iso {
    Iso::from_parts(Translation3::new(0.0, 0.0, v), UnitQuaternion::identity())
}
fn rx(v: f64) -> Iso {
    Iso::from_parts(Translation3::identity(), UnitQuaternion::from_axis_angle(&Vector3::x_axis(), v))
}
fn ry(v: f64) -> Iso {
    Iso::from_parts(Translation3::identity(), UnitQuaternion::from_axis_angle(&Vector3::y_axis(), v))
}
fn rz(v: f64) -> Iso {
    Iso::from_parts(Translation3::identity(), UnitQuaternion::from_axis_angle(&Vector3::z_axis(), v))
}

/// The rotation generators `R̂_x, R̂_y, R̂_z` of Part 1 eqs (20)–(22), transcribed from the spec.
fn r_hat(axis: usize) -> Matrix4<f64> {
    #[rustfmt::skip]
    let m = match axis {
        0 => Matrix4::new(0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, -1.0, 0.0,
                          0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0),
        1 => Matrix4::new(0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 0.0,
                          -1.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0),
        _ => Matrix4::new(0.0, -1.0, 0.0, 0.0,
                          1.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0),
    };
    m
}

/// The translation generators `t̂_x, t̂_y, t̂_z` of Part 1 eqs (23)–(25): a single 1 at `(axis, 4)`.
fn t_hat(axis: usize) -> Matrix4<f64> {
    let mut m = Matrix4::zeros();
    m[(axis, 3)] = 1.0;
    m
}

/// `ρ(T)`: the rotation block (Part 1 Fig. 4).
fn rho(t: &Matrix4<f64>) -> Matrix3<f64> {
    t.fixed_view::<3, 3>(0, 0).into_owned()
}

/// `τ(T)`: the translation column (Part 1 Fig. 4).
fn tau(t: &Matrix4<f64>) -> Vector3<f64> {
    t.fixed_view::<3, 1>(0, 3).into_owned()
}

/// `∨_×(S)`: the vector of a skew matrix (Part 1 Fig. 3), read off the entries the papers name.
fn vee(s: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(s[(2, 1)], s[(0, 2)], s[(1, 0)])
}

impl Et {
    /// `(axis 0|1|2, is_rotation, η)`.
    fn parts(&self) -> (usize, bool, Eta) {
        match *self {
            Et::Tx(e) => (0, false, e),
            Et::Ty(e) => (1, false, e),
            Et::Tz(e) => (2, false, e),
            Et::Rx(e) => (0, true, e),
            Et::Ry(e) => (1, true, e),
            Et::Rz(e) => (2, true, e),
        }
    }

    fn from_parts(axis: usize, rotation: bool, eta: Eta) -> Et {
        match (axis, rotation) {
            (0, false) => Et::Tx(eta),
            (1, false) => Et::Ty(eta),
            (2, false) => Et::Tz(eta),
            (0, true) => Et::Rx(eta),
            (1, true) => Et::Ry(eta),
            _ => Et::Rz(eta),
        }
    }

    /// The transform `E_i(η_i)` for a constant, or `E_i(±q_j)` for a joint factor.
    fn iso(&self, q: &[f64]) -> Iso {
        let (axis, rotation, eta) = self.parts();
        let v = match eta {
            Eta::Constant(c) => c,
            Eta::Joint { index, negative } => {
                if negative {
                    -q[index]
                } else {
                    q[index]
                }
            }
        };
        match (axis, rotation) {
            (0, false) => tx(v),
            (1, false) => ty(v),
            (2, false) => tz(v),
            (0, true) => rx(v),
            (1, true) => ry(v),
            _ => rz(v),
        }
    }

    /// `E`, `dE/dq_j` or `d²E/dq_j²` as a homogeneous matrix — Part 1 eqs (20)–(25) and Part 2
    /// eqs (10)–(15). A negative-sense rotation uses `R̂ᵀ` (Part 1, after eq (22)); a negative-sense
    /// translation's derivative carries the sign of its axis, `−t̂`, which is what makes the factor
    /// `T_t(−q)` differentiate correctly (verified against central differences in the tests).
    fn matrix(&self, q: &[f64], order: u8) -> Matrix4<f64> {
        let e = self.iso(q).to_homogeneous();
        let (axis, rotation, eta) = self.parts();
        let negative = matches!(eta, Eta::Joint { negative: true, .. });
        match (order, rotation) {
            (0, _) => e,
            (1, true) => {
                let g = if negative { r_hat(axis).transpose() } else { r_hat(axis) };
                g * e
            }
            (1, false) => {
                if negative {
                    -t_hat(axis)
                } else {
                    t_hat(axis)
                }
            }
            (_, true) => {
                let g = if negative { r_hat(axis).transpose() } else { r_hat(axis) };
                g * g * e
            }
            (_, false) => Matrix4::zeros(),
        }
    }
}

impl fmt::Display for Eta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Eta::Constant(c) if c == PI => write!(f, "pi"),
            Eta::Constant(c) if c == -PI => write!(f, "-pi"),
            Eta::Constant(c) if c == FRAC_PI_2 => write!(f, "pi/2"),
            Eta::Constant(c) if c == -FRAC_PI_2 => write!(f, "-pi/2"),
            Eta::Constant(c) => write!(f, "{c}"),
            Eta::Joint { index, negative: false } => write!(f, "q{}", index + 1),
            Eta::Joint { index, negative: true } => write!(f, "-q{}", index + 1),
        }
    }
}

impl fmt::Display for Et {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (axis, rotation, eta) = self.parts();
        let name = ["x", "y", "z"][axis];
        write!(f, "{}{name}({eta})", if rotation { "R" } else { "T" })
    }
}

impl fmt::Display for Ets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, e) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{e}")?;
        }
        Ok(())
    }
}

fn parse_eta(s: &str) -> Option<Eta> {
    let s = s.trim();
    let (negative, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest.trim()),
        None => (false, s),
    };
    if let Some(n) = body.strip_prefix('q') {
        let n: usize = n.parse().ok()?;
        return (n >= 1).then(|| Eta::Joint { index: n - 1, negative });
    }
    let magnitude = if let Some(rest) = body.strip_prefix("pi") {
        match rest.strip_prefix('/') {
            None if rest.is_empty() => PI,
            Some(d) => PI / d.trim().parse::<f64>().ok()?,
            _ => return None,
        }
    } else {
        body.parse::<f64>().ok()?
    };
    let c = if negative { -magnitude } else { magnitude };
    c.is_finite().then_some(Eta::Constant(c))
}

fn parse_et(token: &str) -> Option<Et> {
    let open = token.find('(')?;
    let close = token.strip_suffix(')')?;
    let (head, arg) = (&token[..open], &close[open + 1..]);
    let (rotation, axis) = match head {
        "Tx" | "tx" | "T_tx" => (false, 0),
        "Ty" | "ty" | "T_ty" => (false, 1),
        "Tz" | "tz" | "T_tz" => (false, 2),
        "Rx" | "rx" | "T_Rx" => (true, 0),
        "Ry" | "ry" | "T_Ry" => (true, 1),
        "Rz" | "rz" | "T_Rz" => (true, 2),
        _ => return None,
    };
    Some(Et::from_parts(axis, rotation, parse_eta(arg)?))
}

/// Decompose a constant transform into translation factors `Tx Ty Tz` followed by three rotation
/// factors, dropping exact zeros.
///
/// The rotation is written `Rz(a)·Ry(b)·Rx(c)` (Z-Y-X Euler angles, from `r31 = −sin b`,
/// `r32 = cos b sin c`, `r33 = cos b cos c`, `r11 = cos b cos a`, `r21 = cos b sin a`) unless
/// `|r31| > 0.9`, near that order's singularity at `b = ±π/2`, where it is written `Ry(a)·Rz(b)·Rx(c)`
/// instead (from `r21 = sin b`, `r22 = cos b cos c`, `r23 = −cos b sin c`, `r11 = cos b cos a`,
/// `r31 = −cos b sin a`), whose own singularity `|r21| = 1` cannot occur there since
/// `r21² + r31² ≤ 1`. Both branches therefore see `cos b ≥ 0.43` and the recomposition holds to
/// rounding; the tests measure `6.7e-16` over 400 rotations including exact `Ry(±π/2)`. One of many
/// valid decompositions (see [`Ets::from_robot`]).
fn constant_factors(t: &Iso, out: &mut Vec<Et>) {
    let p = t.translation.vector;
    let r = t.rotation.to_rotation_matrix().into_inner();
    let factors: [(usize, bool, f64); 3] = if r[(2, 0)].abs() > 0.9 {
        let b = r[(1, 0)].atan2((r[(0, 0)].powi(2) + r[(2, 0)].powi(2)).sqrt());
        let a = (-r[(2, 0)]).atan2(r[(0, 0)]);
        let c = (-r[(1, 2)]).atan2(r[(1, 1)]);
        [(1, true, a), (2, true, b), (0, true, c)]
    } else {
        let b = (-r[(2, 0)]).atan2((r[(0, 0)].powi(2) + r[(1, 0)].powi(2)).sqrt());
        let a = r[(1, 0)].atan2(r[(0, 0)]);
        let c = r[(2, 1)].atan2(r[(2, 2)]);
        [(2, true, a), (1, true, b), (0, true, c)]
    };
    for (axis, rotation, v) in [(0, false, p.x), (1, false, p.y), (2, false, p.z)].into_iter().chain(factors) {
        if v != 0.0 {
            out.push(Et::from_parts(axis, rotation, Eta::Constant(v)));
        }
    }
}

impl Ets {
    /// Parse the notation described in the module documentation. Returns `None` for an unknown
    /// factor name, an unbalanced parenthesis, a non-finite constant, or a joint number below 1.
    /// Structural validity (each joint once, in order) is checked by [`Ets::to_robot`] and the
    /// oracle, not here, so that a string can still be printed and inspected.
    pub fn parse(s: &str) -> Option<Ets> {
        let factors = s.split_whitespace().map(parse_et).collect::<Option<Vec<_>>>()?;
        (!factors.is_empty()).then_some(Ets(factors))
    }

    /// Joint-variable indices in sequence order.
    fn joint_indices(&self) -> Vec<usize> {
        self.0
            .iter()
            .filter_map(|e| match e.parts().2 {
                Eta::Joint { index, .. } => Some(index),
                Eta::Constant(_) => None,
            })
            .collect()
    }

    /// Number of joints if the sequence is valid: every constant finite, at least one joint, and the
    /// joint variables `q1, q2, …` appearing exactly once each in increasing order (Part 1's `μ(j)` is
    /// a function; the collapse needs joint index = ETS order).
    fn dof(&self) -> Option<usize> {
        if self.0.iter().any(|e| matches!(e.parts().2, Eta::Constant(c) if !c.is_finite())) {
            return None;
        }
        let idx = self.joint_indices();
        let ordered = idx.iter().enumerate().all(|(k, &j)| j == k);
        (!idx.is_empty() && ordered).then_some(idx.len())
    }

    /// `μ(j)`: the position in the sequence of the factor that depends on `q_j` (Part 1, after eq (5)).
    fn mu(&self, j: usize) -> usize {
        self.0.iter().position(|e| matches!(e.parts().2, Eta::Joint { index, .. } if index == j)).expect("validated by dof()")
    }

    /// Part 1 eq (1): the plain left-to-right product of the factors at `q`.
    pub fn fk(&self, q: &[f64]) -> Iso {
        self.0.iter().fold(Iso::identity(), |t, e| t * e.iso(q))
    }

    /// Collapse into a [`Robot`] by associativity of the product (the mapping the spec states):
    /// constant factors accumulate into the next joint's `origin`, each joint factor becomes a
    /// [`Joint`] about `±x`, `±y` or `±z` (the sign realising a negative-sense factor), and trailing
    /// constants become `ee_offset`. `Robot::fk` then equals [`Ets::fk`] exactly.
    ///
    /// Returns `None` for an empty sequence, a non-finite constant, or joint variables that are
    /// repeated or out of order. Joint limits are not part of the notation and are left `None`.
    pub fn to_robot(&self) -> Option<Robot> {
        let n = self.dof()?;
        let mut joints = Vec::with_capacity(n);
        let mut carry = Iso::identity();
        for e in &self.0 {
            let (axis, rotation, eta) = e.parts();
            match eta {
                Eta::Constant(_) => carry *= e.iso(&[]),
                Eta::Joint { negative, .. } => {
                    let mut a = Vector3::zeros();
                    a[axis] = if negative { -1.0 } else { 1.0 };
                    joints.push(if rotation { Joint::revolute(carry, a) } else { Joint::prismatic(carry, a) });
                    carry = Iso::identity();
                }
            }
        }
        Some(Robot { joints, ee_offset: carry })
    }

    /// Write a [`Robot`] as an ETS. **The result is one of many**: each `origin` is decomposed as
    /// `Tx Ty Tz Rz Ry Rx` (translation, then Z-Y-X Euler angles; `Ry Rz Rx` near that order's
    /// singularity — see `constant_factors` — with exact zeros dropped), which
    /// round-trips through [`Ets::to_robot`] to the same forward kinematics but will not in general
    /// reproduce a hand-written string. The Panda of Part 1 prints back with its tool as
    /// `Tx(0.088) Ty(-0.0000000000000000131…) Tz(-0.107) Rx(pi) Rz(q7)`: the same map
    /// (`Rx(π)·Tz(0.107) = Tz(−0.107)·Rx(π)`), and a `1.3e-17` rounding residue of the origin product
    /// printed as-is (in `f64`'s positional `Display` form), because only exact zeros are dropped and
    /// the notation makes no tolerance decision on the caller's behalf. A joint whose axis is exactly `±x`, `±y` or `±z` becomes one variable
    /// factor (negative-sense when the sign is negative); any other axis `a` is expressed by the
    /// constant conjugation `R · Rz(q) · Rᵀ` with `R` a rotation taking `z` to `a`, which adds two
    /// constant runs around the variable. Limits, efforts and dynamics fields are not carried.
    pub fn from_robot(robot: &Robot) -> Ets {
        let mut out = Vec::new();
        for j in &robot.joints {
            constant_factors(&j.origin, &mut out);
            let a = j.axis.into_inner();
            let aligned = (0..3).find(|&k| a[k].abs() == 1.0 && (0..3).all(|m| m == k || a[m] == 0.0));
            let rotation = matches!(j.kind, JointKind::Revolute);
            match aligned {
                Some(axis) => out.push(Et::from_parts(axis, rotation, Eta::Joint { index: out_joint_count(&out), negative: a[axis] < 0.0 })),
                None => {
                    let r = UnitQuaternion::rotation_between(&Vector3::z(), &a).unwrap_or_else(UnitQuaternion::identity);
                    let conj = Iso::from_parts(Translation3::identity(), r);
                    constant_factors(&conj, &mut out);
                    out.push(Et::from_parts(2, rotation, Eta::Joint { index: out_joint_count(&out), negative: false }));
                    constant_factors(&conj.inverse(), &mut out);
                }
            }
        }
        constant_factors(&robot.ee_offset, &mut out);
        Ets(out)
    }

    /// Expand a Denavit–Hartenberg table into its factors, following Corke 2007 eq (1) for the
    /// standard convention (`Rz(θ)·Tz(d)·Tx(a)·Rx(α)`) and eq (4) for the modified one
    /// (`Rx(α)·Tx(a)·Rz(θ)·Tz(d)`), with the revolute variable `θ = θ₀ + q` split as
    /// `Rz(θ₀)·Rz(q)` (Corke 2007 eq (16)) and the prismatic `d = d₀ + q` as `Tz(d₀)·Tz(q)`. Constants
    /// that are exactly zero are omitted (rule (9)). A tool transform is appended by pushing constant
    /// factors onto the result. Limits are not carried; [`Robot::from_dh`] does carry them.
    ///
    /// Verified joint-for-joint against [`Robot::from_dh`] on the arms `dh.rs` tests, both conventions.
    pub fn from_dh(rows: &[DhRow], convention: DhConvention) -> Ets {
        let mut out = Vec::new();
        let push = |axis: usize, rotation: bool, v: f64, out: &mut Vec<Et>| {
            if v != 0.0 {
                out.push(Et::from_parts(axis, rotation, Eta::Constant(v)));
            }
        };
        for (j, row) in rows.iter().enumerate() {
            let q = Eta::Joint { index: j, negative: false };
            match (convention, row.kind) {
                (DhConvention::Standard, JointKind::Revolute) => {
                    push(2, true, row.theta, &mut out);
                    out.push(Et::Rz(q));
                    push(2, false, row.d, &mut out);
                    push(0, false, row.a, &mut out);
                    push(0, true, row.alpha, &mut out);
                }
                (DhConvention::Standard, JointKind::Prismatic) => {
                    push(2, true, row.theta, &mut out);
                    push(2, false, row.d, &mut out);
                    out.push(Et::Tz(q));
                    push(0, false, row.a, &mut out);
                    push(0, true, row.alpha, &mut out);
                }
                (DhConvention::Modified, JointKind::Revolute) => {
                    push(0, true, row.alpha, &mut out);
                    push(0, false, row.a, &mut out);
                    push(2, true, row.theta, &mut out);
                    out.push(Et::Rz(q));
                    push(2, false, row.d, &mut out);
                }
                (DhConvention::Modified, JointKind::Prismatic) => {
                    push(0, true, row.alpha, &mut out);
                    push(0, false, row.a, &mut out);
                    push(2, true, row.theta, &mut out);
                    push(2, false, row.d, &mut out);
                    out.push(Et::Tz(q));
                }
            }
        }
        Ets(out)
    }

    /// The product of the factors with the derivative of the given order substituted at the
    /// listed `(factor index, order)` slots — the bracketed products of Part 1 eq (26) and Part 2
    /// eq (6).
    fn product(&self, q: &[f64], slots: &[(usize, u8)]) -> Matrix4<f64> {
        let mut t = Matrix4::identity();
        for (i, e) in self.0.iter().enumerate() {
            let order = slots.iter().filter(|(k, _)| *k == i).map(|(_, o)| *o).sum();
            t *= e.matrix(q, order);
        }
        t
    }

    /// The manipulator Jacobian by Part 1 eqs (26)–(30), as written: column `j` is
    /// `∂T/∂q_j = [∏_{i<μ(j)} E_i]·[dE_μ(j)/dq_j]·[∏_{i>μ(j)} E_i]` (26), `J_ω,j = ∨_×(ρ(∂T/∂q_j)·ρ(T)ᵀ)`
    /// (27), `J_ν,j = τ(∂T/∂q_j)` (28), stacked `[J_ν; J_ω]` (29). World frame, `6 × n`, the same
    /// layout as [`Robot::jacobian`]. `O(n²)` matrix products; this is the oracle, not the fast path.
    ///
    /// `None` if the sequence is invalid (see [`Ets::to_robot`]) or `q.len()` is not the joint count.
    pub fn jacobian(&self, q: &[f64]) -> Option<DMatrix<f64>> {
        let n = self.dof()?;
        if q.len() != n {
            return None;
        }
        let t = self.product(q, &[]);
        let rt = rho(&t).transpose();
        let mut jac = DMatrix::zeros(6, n);
        for j in 0..n {
            let dt = self.product(q, &[(self.mu(j), 1)]);
            jac.fixed_view_mut::<3, 1>(0, j).copy_from(&tau(&dt));
            jac.fixed_view_mut::<3, 1>(3, j).copy_from(&vee(&(rho(&dt) * rt)));
        }
        Some(jac)
    }

    /// The manipulator Hessian by Part 2 eqs (6), (17), (19), as written: `∂²T/(∂q_j∂q_k)` is the
    /// product (6) with `dE/dq` in the slots `μ(j)` and `μ(k)` (or `d²E/dq²` in the one slot when
    /// `j = k`); `H_α,jk = ∨_×(H_R,jk·ρ(T)ᵀ + J_R,j·J_R,kᵀ)` (17) with `H_R = ρ(∂²T)`, `J_R = ρ(∂T)`;
    /// `H_a,jk = τ(∂²T/(∂q_j∂q_k))` (19). Returned in the layout of [`Robot::kinematic_hessian`]:
    /// element `k` is `∂J/∂q_k`, whose column `j` is `[H_a,jk; H_α,jk]`. `O(n³)` products.
    pub fn hessian(&self, q: &[f64]) -> Option<Vec<DMatrix<f64>>> {
        let n = self.dof()?;
        if q.len() != n {
            return None;
        }
        let t = self.product(q, &[]);
        let rt = rho(&t).transpose();
        let jr: Vec<Matrix3<f64>> = (0..n).map(|j| rho(&self.product(q, &[(self.mu(j), 1)]))).collect();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let mut h = DMatrix::zeros(6, n);
            for j in 0..n {
                let d2 = self.product(q, &[(self.mu(j), 1), (self.mu(k), 1)]);
                h.fixed_view_mut::<3, 1>(0, j).copy_from(&tau(&d2));
                h.fixed_view_mut::<3, 1>(3, j).copy_from(&vee(&(rho(&d2) * rt + jr[j] * jr[k].transpose())));
            }
            out.push(h);
        }
        Some(out)
    }
}

/// Joint factors emitted so far (the next joint's index).
fn out_joint_count(out: &[Et]) -> usize {
    out.iter().filter(|e| matches!(e.parts().2, Eta::Joint { .. })).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Unit;
    use std::f64::consts::FRAC_PI_4;

    /// Part 1, Figure 1 — the string in the module documentation.
    const PANDA: &str = "Tz(0.333) Rz(q1) Ry(q2) Tz(0.316) Rz(q3) Tx(0.0825) Ry(-q4) Tx(-0.0825) Tz(0.384) Rz(q5) Ry(-q6) Tx(0.088) Rx(pi) Tz(0.107) Rz(q7)";

    /// Franka's published modified-DH table for the Panda (FCI 'Robot and interface specifications',
    /// rows `(a_{i-1}, α_{i-1}, d_i, θ_i)`), inlined; flange `Tz(0.107)`.
    fn panda_dh() -> (Vec<DhRow>, Iso) {
        let rows = vec![
            DhRow::revolute(0.0, 0.333, 0.0, 0.0),
            DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2),
            DhRow::revolute(0.0, 0.316, 0.0, FRAC_PI_2),
            DhRow::revolute(0.0, 0.0, 0.0825, FRAC_PI_2),
            DhRow::revolute(0.0, 0.384, -0.0825, -FRAC_PI_2),
            DhRow::revolute(0.0, 0.0, 0.0, FRAC_PI_2),
            DhRow::revolute(0.0, 0.0, 0.088, FRAC_PI_2),
        ];
        (rows, tz(0.107))
    }

    /// Three Panda configurations: the table's `q = 0`, its in-limits check pose, and a generic one.
    const PANDA_Q: [[f64; 7]; 3] = [
        [0.0; 7],
        [0.0, -FRAC_PI_4, 0.0, -3.0 * FRAC_PI_4, 0.0, FRAC_PI_2, FRAC_PI_4],
        [0.3, -0.5, 0.7, -1.9, 0.4, 2.1, -0.8],
    ];

    fn iso_gap(a: &Iso, b: &Iso) -> f64 {
        (a.to_homogeneous() - b.to_homogeneous()).abs().max()
    }

    fn mat_gap(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        (a - b).abs().max()
    }

    /// **The Panda string of Part 1 Figure 1 parses, prints back identically, and is the same arm as
    /// Franka's modified-DH table.** The string is derived by hand from the table: row by row the DH
    /// product is `Rz(q1)Tz(.333) · Rx(-π/2)Rz(q2) · Rx(π/2)Rz(q3)Tz(.316) · Rx(π/2)Tx(.0825)Rz(q4) ·
    /// Rx(-π/2)Tx(-.0825)Rz(q5)Tz(.384) · Rx(π/2)Rz(q6) · Rx(π/2)Tx(.088)Rz(q7) · Tz(.107)`; conjugating
    /// `Rz(q)` by `Rx(∓π/2)` gives `Ry(±q)` (`Rx(-π/2)·ẑ = +ŷ`, `Rx(π/2)·ẑ = −ŷ`, hence the negative
    /// sense of joints 4 and 6), `Tz`/`Rz` commute, `Rx(π)` carries `Tx(0.088)` unchanged and sends
    /// the final `Rz(q7)Tz(.107)` to `Tz(.107)Rz(q7)` after itself. Known answer at `q = 0`, computed by
    /// hand from the table: flange at `(0.088, 0, 0.926)`, `R = diag(1,−1,−1)`.
    #[test]
    fn the_panda_ets_parses_prints_back_and_matches_the_modified_dh_table() {
        let ets = Ets::parse(PANDA).expect("the Panda string is valid notation");
        assert_eq!(ets.to_string(), PANDA, "print-back must be identical");
        assert_eq!(ets.0.len(), 15, "Part 1 numbers the Panda's factors E_1..E_15");
        assert!(matches!(ets.0[9], Et::Rz(Eta::Joint { index: 4, negative: false })), "mu(5) = 10");
        assert!(matches!(ets.0[6], Et::Ry(Eta::Joint { index: 3, negative: true })), "E_7 is negative-sense");
        assert!(matches!(ets.0[10], Et::Ry(Eta::Joint { index: 5, negative: true })), "E_11 is negative-sense");

        let via_ets = ets.to_robot().expect("collapses");
        let (rows, tool) = panda_dh();
        let via_dh = Robot::from_dh(&rows, DhConvention::Modified, tool).unwrap();
        assert_eq!(via_ets.dof(), 7);

        let home = via_ets.fk(&PANDA_Q[0]);
        let want = Vector3::new(0.088, 0.0, 0.926);
        assert!((home.translation.vector - want).norm() < 1e-12, "hand answer at q=0: {:?}", home.translation.vector);
        let r = home.rotation.to_rotation_matrix();
        assert!((r.matrix() - Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0))).abs().max() < 1e-12);

        let poses: Vec<Iso> = PANDA_Q.iter().map(|q| via_ets.fk(q)).collect();
        assert!(iso_gap(&poses[0], &poses[1]) > 0.1 && iso_gap(&poses[1], &poses[2]) > 0.1, "the three configurations must be distinct poses");
        let p1 = poses[1].translation.vector;
        assert!((p1 - Vector3::new(0.306891, 0.0, 0.590282)).norm() < 1e-6, "in-limits check pose from the table: {p1:?}");
        let mut worst = 0.0f64;
        for (q, pose) in PANDA_Q.iter().zip(&poses) {
            worst = worst.max(iso_gap(pose, &via_dh.fk(q)));
            assert!(iso_gap(pose, &ets.fk(q)) < 1e-14, "collapse must equal the literal product, eq (1)");
        }
        assert!(worst < 1e-12, "ETS vs DH Panda: worst {worst:e}");
    }

    /// **The DH-to-ETS expansion of the same table is a different string for the same map**: the
    /// paper's hand-written form and [`Ets::from_dh`]'s row expansion collapse to equal forward
    /// kinematics.
    #[test]
    fn the_panda_from_dh_expansion_agrees_with_the_hand_written_string() {
        let (rows, _) = panda_dh();
        let mut expanded = Ets::from_dh(&rows, DhConvention::Modified);
        expanded.0.push(Et::Tz(Eta::Constant(0.107)));
        let hand = Ets::parse(PANDA).unwrap();
        assert_ne!(expanded.to_string(), PANDA, "two different strings ...");
        for q in &PANDA_Q {
            assert!(iso_gap(&expanded.fk(q), &hand.fk(q)) < 1e-12, "... one map");
        }
    }

    /// **The oracle.** Part 1 eqs (26)–(28) and Part 2 eqs (6), (17), (19), evaluated as matrix
    /// products of generators, against the vector-form `z × (p_e − p)` construction in `lib.rs`, on
    /// the Panda at three configurations, to 1e-9 — two differently structured derivations of `J`
    /// and `∂J/∂q`. Non-vacuity: `J` has entries of order 1 and changes between the configurations.
    #[test]
    fn ets_jacobian_and_hessian_equal_the_robots_on_the_panda() {
        let ets = Ets::parse(PANDA).unwrap();
        let robot = ets.to_robot().unwrap();
        let (mut worst_j, mut worst_h, mut scale_j, mut scale_h) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut jacs = Vec::new();
        for q in &PANDA_Q {
            let (j_ets, j_rob) = (ets.jacobian(q).unwrap(), robot.jacobian(q));
            let (h_ets, h_rob) = (ets.hessian(q).unwrap(), robot.kinematic_hessian(q));
            worst_j = worst_j.max(mat_gap(&j_ets, &j_rob));
            scale_j = scale_j.max(j_rob.abs().max());
            assert_eq!(h_ets.len(), 7);
            for k in 0..7 {
                worst_h = worst_h.max(mat_gap(&h_ets[k], &h_rob[k]));
                scale_h = scale_h.max(h_rob[k].abs().max());
            }
            jacs.push(j_rob);
        }
        assert!(scale_j > 0.5 && scale_h > 0.5, "fixture: J scale {scale_j}, H scale {scale_h}");
        assert!(mat_gap(&jacs[0], &jacs[2]) > 0.1, "the Jacobian must vary across the configurations");
        assert!(worst_j < 1e-9, "Jacobian oracle vs Robot::jacobian: worst {worst_j:e} (scale {scale_j})");
        assert!(worst_h < 1e-9, "Hessian oracle vs Robot::kinematic_hessian: worst {worst_h:e} (scale {scale_h})");
    }

    /// **The oracle on prismatic joints and every axis.** The Panda has no prismatic joint and no
    /// joint about `x`, so this six-joint sequence has prismatic joints along `x`, `y`, `z` (one
    /// negative-sense) and revolute joints about `x` and `y` with a rotated tool. Compared against
    /// the `Robot` and, as a third reading, against central differences of the literal product (1).
    #[test]
    fn ets_oracle_covers_prismatic_and_negative_sense_translation_factors() {
        let ets = Ets::parse("Tz(0.2) Rx(q1) Tx(q2) Ry(pi/2) Ty(-q3) Rz(0.4) Ry(q4) Tz(q5) Rx(-pi/2) Tx(0.1) Rz(-q6) Ty(0.05) Rz(pi/2)").unwrap();
        let robot = ets.to_robot().unwrap();
        assert!(matches!(robot.joints[2].kind, JointKind::Prismatic) && robot.joints[2].axis.y == -1.0);
        let q = [0.4, 0.15, -0.2, 1.1, 0.3, -0.7];
        let (j_ets, j_rob) = (ets.jacobian(&q).unwrap(), robot.jacobian(&q));
        let (h_ets, h_rob) = (ets.hessian(&q).unwrap(), robot.kinematic_hessian(&q));
        assert!(j_rob.abs().max() > 0.5, "fixture: J scale {}", j_rob.abs().max());
        assert!(mat_gap(&j_ets, &j_rob) < 1e-9, "J: {:e}", mat_gap(&j_ets, &j_rob));
        let mut worst_h = 0.0f64;
        let mut scale_h = 0.0f64;
        for k in 0..6 {
            worst_h = worst_h.max(mat_gap(&h_ets[k], &h_rob[k]));
            scale_h = scale_h.max(h_rob[k].abs().max());
        }
        assert!(scale_h > 0.5 && worst_h < 1e-9, "H: worst {worst_h:e} on scale {scale_h}");
        // central differences of the literal product, eq (1): dT/dq_j -> J_nu = tau, J_omega = vee(dR R^T)
        let eps = 1e-6;
        let t = ets.fk(&q).to_homogeneous();
        let mut worst_fd = 0.0f64;
        for j in 0..6 {
            let (mut qp, mut qm) = (q, q);
            qp[j] += eps;
            qm[j] -= eps;
            let dt = (ets.fk(&qp).to_homogeneous() - ets.fk(&qm).to_homogeneous()) / (2.0 * eps);
            let col = j_ets.column(j);
            worst_fd = worst_fd.max((tau(&dt) - col.fixed_rows::<3>(0)).abs().max());
            worst_fd = worst_fd.max((vee(&(rho(&dt) * rho(&t).transpose())) - col.fixed_rows::<3>(3)).abs().max());
        }
        assert!(worst_fd < 1e-8, "oracle vs central differences: {worst_fd:e}");
    }

    /// **`from_dh` reproduces `Robot::from_dh` joint-for-joint and in forward kinematics**, both
    /// conventions, on the arms `dh.rs` tests (2R, spatial RRR, RP, θ/d offsets, the 5-joint arm
    /// with a prismatic row and a tool). Origins, axes, kinds and tool to 1e-14.
    #[test]
    fn from_dh_round_trips_the_dh_module_arms_in_both_conventions() {
        let (a1, a2) = (0.7, 0.4);
        let cases: Vec<(Vec<DhRow>, DhConvention, Iso)> = vec![
            (vec![DhRow::revolute(0.0, 0.0, a1, 0.0), DhRow::revolute(0.0, 0.0, a2, 0.0)], DhConvention::Standard, Iso::identity()),
            (vec![DhRow::revolute(0.0, 0.0, 0.0, 0.0), DhRow::revolute(0.0, 0.0, a1, 0.0)], DhConvention::Modified, tx(a2)),
            (vec![DhRow::revolute(0.0, 0.5, 0.0, FRAC_PI_2), DhRow::revolute(0.0, 0.0, 0.6, 0.0), DhRow::revolute(0.0, 0.0, 0.4, 0.0)], DhConvention::Standard, Iso::identity()),
            (vec![DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2), DhRow::prismatic(0.0, 0.2, 0.0, 0.0)], DhConvention::Standard, Iso::identity()),
            (vec![DhRow::revolute(0.0, 0.0, 0.0, 0.0), DhRow::prismatic(0.0, 0.2, 0.0, -FRAC_PI_2)], DhConvention::Modified, Iso::identity()),
            (vec![DhRow::revolute(FRAC_PI_2, 0.0, 0.5, 0.0)], DhConvention::Standard, Iso::identity()),
            (vec![DhRow::prismatic(0.0, 0.25, 0.0, 0.0)], DhConvention::Modified, Iso::identity()),
            (vec![DhRow::revolute(0.0, 0.3, 0.0, 0.0), DhRow::revolute(FRAC_PI_2, 0.0, 0.5, FRAC_PI_2)], DhConvention::Modified, tx(0.2)),
            (
                vec![
                    DhRow::revolute(0.0, 0.4, 0.0, FRAC_PI_2),
                    DhRow::revolute(0.0, 0.0, 0.45, 0.0),
                    DhRow::prismatic(FRAC_PI_2, 0.1, 0.0, FRAC_PI_2),
                    DhRow::revolute(0.0, 0.3, 0.0, -FRAC_PI_2),
                    DhRow::revolute(0.0, 0.0, 0.0, FRAC_PI_2),
                ],
                DhConvention::Standard,
                tx(0.08),
            ),
            (vec![DhRow::prismatic(0.3, 0.1, 0.2, FRAC_PI_2), DhRow::revolute(-0.4, 0.15, 0.1, -FRAC_PI_2)], DhConvention::Modified, ty(0.05)),
        ];
        let qs = [[0.0; 5], [0.3, -0.5, 0.12, 0.7, -0.4], [1.2, 0.8, -0.3, -2.0, 2.5]];
        let mut poses_seen = 0;
        for (rows, convention, tool) in &cases {
            let direct = Robot::from_dh(rows, *convention, *tool).unwrap();
            let mut ets = Ets::from_dh(rows, *convention);
            constant_factors(tool, &mut ets.0);
            let collapsed = ets.to_robot().unwrap();
            assert_eq!(collapsed.dof(), direct.dof());
            for (a, b) in collapsed.joints.iter().zip(&direct.joints) {
                assert!(iso_gap(&a.origin, &b.origin) < 1e-14, "origin: {:?} vs {:?}", a.origin, b.origin);
                assert!((a.axis.into_inner() - b.axis.into_inner()).norm() == 0.0 && a.kind == b.kind);
            }
            assert!(iso_gap(&collapsed.ee_offset, &direct.ee_offset) < 1e-14, "tool: {:?} vs {:?}", collapsed.ee_offset, direct.ee_offset);
            let n = rows.len();
            let mut prev: Option<Iso> = None;
            for q in &qs {
                let pose = direct.fk(&q[..n]);
                assert!(iso_gap(&collapsed.fk(&q[..n]), &pose) < 1e-13, "fk {convention:?}: {rows:?}");
                assert!(iso_gap(&ets.fk(&q[..n]), &pose) < 1e-13);
                if let Some(p) = prev {
                    assert!(iso_gap(&p, &pose) > 1e-3, "the configurations must move the arm");
                }
                prev = Some(pose);
                poses_seen += 1;
            }
        }
        assert_eq!(poses_seen, 30);
    }

    /// **`from_robot` ∘ `to_robot` preserves forward kinematics** — on the Panda, whose origins
    /// include `±π/2` twists that put the Euler decomposition at gimbal lock, and on an arm with a
    /// joint about the non-axis-aligned direction `(1,1,1)/√3` (expanded by conjugation). The
    /// expansion is documented as non-unique: the Panda's own string is not reproduced.
    #[test]
    fn from_robot_then_to_robot_preserves_forward_kinematics() {
        let hand = Ets::parse(PANDA).unwrap();
        let panda = hand.to_robot().unwrap();
        let printed = Ets::from_robot(&panda);
        assert_ne!(printed.to_string(), PANDA, "non-unique: the canonical decomposition is a different string");
        let back = printed.to_robot().expect("the printed string is a valid sequence");
        assert_eq!(back.dof(), 7);
        for q in &PANDA_Q {
            assert!(iso_gap(&back.fk(q), &panda.fk(q)) < 1e-14, "Panda round trip at {q:?}: {:e}", iso_gap(&back.fk(q), &panda.fk(q)));
        }
        // every joint factor is preceded only by constants since the previous joint factor
        let mut run = 0;
        for e in &printed.0 {
            match e.parts().2 {
                Eta::Constant(_) => run += 1,
                Eta::Joint { .. } => run = 0,
            }
        }
        assert!(run <= 6, "at most one Tx Ty Tz Rz Ry Rx run after the last joint");

        let skew = Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0));
        let g = Iso::from_parts(Translation3::new(0.1, -0.2, 0.3), UnitQuaternion::from_euler_angles(0.3, FRAC_PI_2, -0.4));
        let arm = Robot {
            joints: vec![Joint::revolute(tz(0.3), Vector3::z()), Joint::revolute(g, skew.into_inner()), Joint::prismatic(rx(FRAC_PI_2), -Vector3::y()), Joint::prismatic(tx(0.1), skew.into_inner())],
            ee_offset: ry(-FRAC_PI_2) * tz(0.2),
        };
        let ets = Ets::from_robot(&arm);
        let back = ets.to_robot().unwrap();
        assert_eq!(back.dof(), 4);
        assert!(back.joints.iter().all(|j| j.axis.into_inner().abs().max() == 1.0), "the collapse only ever emits axis-aligned joints");
        let mut prev = None;
        for q in [[0.0; 4], [0.4, -1.1, 0.2, 0.3], [2.0, 0.7, -0.4, -0.1]] {
            let (a, b) = (arm.fk(&q), back.fk(&q));
            assert!(iso_gap(&a, &b) < 1e-14, "skew-axis round trip at {q:?}: {:e}", iso_gap(&a, &b));
            if let Some(p) = prev {
                assert!(iso_gap(&p, &a) > 0.1);
            }
            prev = Some(a);
        }
        let reparsed = Ets::parse(&ets.to_string()).unwrap();
        assert_eq!(reparsed, ets, "Display and parse are inverse on a machine-generated string");
    }

    /// **The constant decomposition recomposes to rounding everywhere**, including the Z-Y-X
    /// singularity. 400 rotations: 396 from a deterministic LCG over the unit quaternions, plus exact
    /// `Ry(±π/2)` and `Ry(±π/2)` preceded by a yaw, which any single Euler order fails on.
    #[test]
    fn constant_decomposition_recomposes_to_rounding_including_gimbal_lock() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut rotations: Vec<UnitQuaternion<f64>> = (0..396).map(|_| UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(next(), next(), next(), next()))).collect();
        rotations.push(ry(FRAC_PI_2).rotation);
        rotations.push(ry(-FRAC_PI_2).rotation);
        rotations.push((rz(0.7) * ry(FRAC_PI_2)).rotation);
        rotations.push((rz(-1.9) * ry(-FRAC_PI_2) * rx(0.3)).rotation);
        let (mut worst, mut yzx_branch) = (0.0f64, 0);
        for rot in &rotations {
            let t = Iso::from_parts(Translation3::new(0.3, -0.1, 0.2), *rot);
            let mut factors = Vec::new();
            constant_factors(&t, &mut factors);
            if matches!(factors.iter().find(|e| e.parts().1), Some(Et::Ry(_))) {
                yzx_branch += 1;
            }
            let back = Ets(factors).fk(&[]);
            worst = worst.max(iso_gap(&back, &t));
        }
        assert!(yzx_branch >= 4 && yzx_branch < rotations.len() / 2, "both branches must be exercised: {yzx_branch} of {}", rotations.len());
        assert!(worst < 1e-14, "recomposition: worst {worst:e}");
    }

    /// **A negative-sense factor** is a joint with axis `−e` (the collapse), and equals the
    /// un-flipped factor at `−q` — for rotation and for translation (Part 1, after eq (22)).
    #[test]
    fn negative_sense_factors_are_the_negated_axis_and_the_negated_variable() {
        let flipped = Ets::parse("Tz(0.1) Ry(-q1) Tx(0.3) Tz(-q2) Rx(pi/2)").unwrap();
        let plain = Ets::parse("Tz(0.1) Ry(q1) Tx(0.3) Tz(q2) Rx(pi/2)").unwrap();
        let r = flipped.to_robot().unwrap();
        assert_eq!(r.joints[0].axis.into_inner(), -Vector3::y());
        assert_eq!(r.joints[1].axis.into_inner(), -Vector3::z());
        for (q1, q2) in [(0.7, 0.2), (-1.3, -0.4)] {
            let a = flipped.fk(&[q1, q2]);
            assert!(iso_gap(&a, &plain.fk(&[-q1, -q2])) < 1e-15);
            assert!(iso_gap(&a, &plain.fk(&[q1, q2])) > 0.1, "the sign must matter");
            let (jf, jp) = (flipped.jacobian(&[q1, q2]).unwrap(), plain.jacobian(&[-q1, -q2]).unwrap());
            assert!(mat_gap(&jf, &(-jp)) < 1e-15, "dT/dq of T(-q) is minus that of T(q) at -q");
        }
    }

    /// The parser refuses what the notation does not say, and the collapse refuses what `μ(j)`
    /// cannot be a function of.
    #[test]
    fn malformed_and_structurally_invalid_sequences_are_refused() {
        assert!(Ets::parse("").is_none());
        assert!(Ets::parse("Tw(0.1)").is_none(), "no w axis");
        assert!(Ets::parse("Tx(0.1").is_none(), "unbalanced");
        assert!(Ets::parse("Rz(q0)").is_none(), "joints are numbered from 1");
        assert!(Ets::parse("Rz(nan)").is_none() && Ets::parse("Tx(inf)").is_none(), "non-finite constants");
        assert!(Ets::parse("Rz(pie)").is_none());
        let pi_forms = Ets::parse("Rx(pi) Ry(-pi) Rz(pi/2) Rx(-pi/2) Ry(pi/4)").unwrap();
        assert_eq!(pi_forms.0[4], Et::Ry(Eta::Constant(FRAC_PI_4)));
        assert_eq!(pi_forms.to_string(), "Rx(pi) Ry(-pi) Rz(pi/2) Rx(-pi/2) Ry(0.7853981633974483)");
        assert!(Ets::parse("T_tz(0.5) T_Rz(q1) tx(0.2) rz(-q2)").is_some(), "the papers' spellings are accepted");

        assert!(Ets::parse("Tz(0.1) Rz(q1) Rz(q1)").unwrap().to_robot().is_none(), "a joint variable in two factors");
        assert!(Ets::parse("Rz(q2) Rz(q1)").unwrap().to_robot().is_none(), "out of order");
        assert!(Ets::parse("Rz(q1) Rz(q3)").unwrap().to_robot().is_none(), "a gap");
        assert!(Ets::parse("Tz(0.1) Tx(0.2)").unwrap().to_robot().is_none(), "no joint at all");
        let ok = Ets::parse("Rz(q1) Rz(q2)").unwrap();
        assert!(ok.jacobian(&[0.1]).is_none() && ok.hessian(&[0.1, 0.2, 0.3]).is_none(), "q must match the joint count");
        assert!(ok.to_robot().unwrap().ee_offset == Iso::identity(), "ending on a variable gives an identity tool");
    }
}
