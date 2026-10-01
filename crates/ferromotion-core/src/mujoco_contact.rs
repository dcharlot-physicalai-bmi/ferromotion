//! **MuJoCo's soft-constraint contact, as MuJoCo computes it** — the compliant rung of the contact-fidelity
//! ladder in the form 3.6 million installs a month actually run, measured against `mj_forward`.
//!
//! MuJoCo does not apply a spring-damper force. It solves, each step, for the constraint force `f` that
//! minimises `½ fᵀ(A + R)f + fᵀ(J·a₀ − a_ref)` over the friction cone, where `A = J M⁻¹ Jᵀ` is the inverse
//! inertia seen by the constraints, `R` a diagonal **regulariser** that makes the constraint soft, `a₀` the
//! unconstrained acceleration and `a_ref` a **reference acceleration** the constraint would like to have
//! (Todorov, *Convex and analytically-invertible dynamics with contacts and constraints*, ICRA 2014). The
//! acceleration is then `a = a₀ + M⁻¹Jᵀf`. Everything soft about it lives in `R` and `a_ref`, and both are
//! set by the two five- and two-number vectors every MJCF file carries, `solimp` and `solref`:
//!
//! * **impedance** `d(r) ∈ (0, 1)` from `solimp = (d₀, d_width, width, midpoint, power)`: a sigmoid of
//!   `x = |pos − margin| / width` — `y = a·xᵖ` below the midpoint, `1 − b·(1−x)ᵖ` above it — scaled to
//!   `d₀ + y·(d_width − d₀)`, saturating at `d_width` past `width` ([`mujoco_impedance`]);
//! * **stiffness and damping** from `solref = (timeconst, dampratio)`: `B = 2/(d_width·timeconst)`,
//!   `K = 1/(d_width²·timeconst²·dampratio²)`; or, when both are negative, directly `K = −solref₀/d_width²`,
//!   `B = −solref₁/d_width` ([`mujoco_kbip`]);
//! * **reference** `a_ref = −B·v − K·d·(pos − margin)` per row;
//! * **regulariser** `R = (1 − d)/d · Ā` where `Ā` is not the true diagonal of `A` but MuJoCo's
//!   *approximation* of it from per-body inverse weights computed once at `qpos0`
//!   ([`mujoco_diag_approx`]); for a pyramidal cone every facet row then takes the common
//!   `R = 2·μ²·(1 − d)/d · (Ā_n + μ²·Ā_n)` so that its friction impedance matches the elliptic model.
//!
//! All of that was read from `engine_core_constraint.c` (`getimpedance`, `mj_makeImpedance`,
//! `mj_diagApprox`, `mj_referenceConstraint`) and `engine_setconst.c` (`body_invweight0`), and every number
//! in the tests below is MuJoCo 3.13.0's own `efc_R`, `efc_aref`, `efc_force` and `qacc` on a sphere
//! pressed into a plane — at rest, closing, opening, with custom and with negative `solref`, frictionless and
//! with the default pyramidal cone at rest and sliding. The Euler integrator's semantics are what is
//! reproduced; MuJoCo's implicit integrators additionally divide `R` and `a_ref` by an implicit row factor
//! (`mj_isMetric`), which is not modelled here.
//!
//! What this is for: [`crate::contact_law_residuals`] can now be asked of MuJoCo's own model — a solver
//! residual is not a law residual, and this is the solver the field's results are computed with.

use nalgebra::{DMatrix, DVector};

/// MuJoCo's `solimp` — `(d₀, d_width, width, midpoint, power)`. MJCF default `0.9 0.95 0.001 0.5 2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolImp {
    pub d0: f64,
    pub d_width: f64,
    pub width: f64,
    pub midpoint: f64,
    pub power: f64,
}

impl SolImp {
    /// ⛔ MuJoCo CLIPS `solimp` where it USES it, not where it stores it: `getsolparam` forces `d0`,
    /// `d_width` and `midpoint` into `[mjMINIMP, mjMAXIMP] = [1e-4, 0.9999]`, `width` to ≥ 0 and `power` to
    /// ≥ 1. The model keeps what the file said — `seeed_rebot_devarm` writes `solimp="0.9999 0.99999 …"` and
    /// `eq_solimp` reports it unchanged — so a port that reads the stored value gets `1 − d = 1e-5` where
    /// MuJoCo has `1e-4`, and every regularisation on that row is out by exactly TEN.
    ///
    /// ⚠ It applies to the STIFFNESS as well as the impedance: `K` and `B` are built from `d_width`, so
    /// clipping only inside the sigmoid leaves `aref` out by 1.8e-4 with `efc_D` exact.
    pub fn clamped(&self) -> SolImp {
        SolImp {
            d0: self.d0.clamp(1e-4, 0.9999),
            d_width: self.d_width.clamp(1e-4, 0.9999),
            width: self.width.max(0.0),
            midpoint: self.midpoint.clamp(1e-4, 0.9999),
            power: self.power.max(1.0),
        }
    }
}

impl Default for SolImp {
    fn default() -> Self {
        Self { d0: 0.9, d_width: 0.95, width: 0.001, midpoint: 0.5, power: 2.0 }
    }
}

/// MuJoCo's `solref` — `(timeconst, dampratio)`, or `(−stiffness, −damping)` when both are negative. MJCF
/// default `0.02 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolRef(pub f64, pub f64);

impl Default for SolRef {
    fn default() -> Self {
        Self(0.02, 1.0)
    }
}

const MINVAL: f64 = 1e-15;

/// `getimpedance`: the impedance `d` and its derivative `d'` at constraint position `pos` (negative =
/// penetrating for a contact) and `margin`.
pub fn mujoco_impedance(s: &SolImp, pos: f64, margin: f64) -> (f64, f64) {
    // ⛔ MuJoCo CLIPS the impedance where it uses it, not where it stores it: `getsolparam` forces `d0`,
    // `d_width` and `midpoint` into [mjMINIMP, mjMAXIMP] = [1e-4, 0.9999], `width` to ≥ 0 and `power` to ≥ 1.
    // The model keeps what the file said — `seeed_rebot_devarm` writes `solimp="0.9999 0.99999 ..."` and
    // `eq_solimp` reports it unchanged — so a port that reads the stored value gets `1 − d = 1e-5` where
    // MuJoCo has `1e-4`, and every regularisation on that row is out by exactly TEN.
    let s = &s.clamped();
    if s.d0 == s.d_width || s.width <= MINVAL {
        return (0.5 * (s.d0 + s.d_width), 0.0);
    }
    let mut x = (pos - margin) / s.width;
    let mut sgn = 1.0;
    if x < 0.0 {
        x = -x;
        sgn = -1.0;
    }
    if x >= 1.0 || x <= 0.0 {
        return (if x >= 1.0 { s.d_width } else { s.d0 }, 0.0);
    }
    let (y, yp) = if s.power == 1.0 {
        (x, 1.0)
    } else if x <= s.midpoint {
        let a = 1.0 / s.midpoint.powf(s.power - 1.0);
        (a * x.powf(s.power), s.power * a * x.powf(s.power - 1.0))
    } else {
        let b = 1.0 / (1.0 - s.midpoint).powf(s.power - 1.0);
        (1.0 - b * (1.0 - x).powf(s.power), s.power * b * (1.0 - x).powf(s.power - 1.0))
    };
    (s.d0 + y * (s.d_width - s.d0), yp * sgn * (s.d_width - s.d0) / s.width)
}

/// MuJoCo's `efc_KBIP` for one row: `[K, B, impedance, impedance']`.
pub fn mujoco_kbip(r: &SolRef, s: &SolImp, pos: f64, margin: f64) -> [f64; 4] {
    let s = &s.clamped();
    let (imp, imp_p) = mujoco_impedance(s, pos, margin);
    let k = if r.0 > 0.0 { 1.0 / (s.d_width * s.d_width * r.0 * r.0 * r.1 * r.1).max(MINVAL) } else { -r.0 / (s.d_width * s.d_width).max(MINVAL) };
    let b = if r.1 > 0.0 { 2.0 / (s.d_width * r.0).max(MINVAL) } else { -r.1 / s.d_width.max(MINVAL) };
    [k, b, imp, imp_p]
}

/// A body's `body_invweight0` — the mean diagonal of its inverse spatial inertia `J M⁻¹ Jᵀ` at the body's
/// centre of mass, translation and rotation, evaluated at `qpos0`. For a free rigid body that is simply
/// `(1/m, mean(1/I_principal))`; for a link in a tree, compute `J_com M⁻¹ J_comᵀ` at `qpos0` and average the
/// two 3×3 diagonals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvWeight {
    pub tran: f64,
    pub rot: f64,
}

impl InvWeight {
    /// A free rigid body: `1/m` and the mean of the three principal inverse inertias.
    pub fn free_body(mass: f64, principal_inertia: [f64; 3]) -> Self {
        let rot = (1.0 / principal_inertia[0] + 1.0 / principal_inertia[1] + 1.0 / principal_inertia[2]) / 3.0;
        Self { tran: 1.0 / mass.max(MINVAL), rot }
    }
    /// A body welded to the world (static or mocap): MuJoCo gives it zero inverse weight.
    pub const STATIC: Self = Self { tran: 0.0, rot: 0.0 };
}

/// `mj_diagApprox` for one contact: the approximate inverse inertia of each of its rows, from the two
/// bodies' inverse weights. Frictionless: one value; pyramidal: `2·(condim−1)` values, one pair per friction
/// direction `k`, each `tran + μ_k²·(tran | rot)`.
pub fn mujoco_diag_approx(a: InvWeight, b: InvWeight, condim: usize, friction: &[f64], pyramidal: bool) -> Vec<f64> {
    let (tran, rot) = (a.tran + b.tran, a.rot + b.rot);
    if condim == 1 {
        return vec![tran];
    }
    if pyramidal {
        let mut out = Vec::with_capacity(2 * (condim - 1));
        for j in 0..condim - 1 {
            let fri = friction[j];
            let v = tran + fri * fri * if j < 2 { tran } else { rot };
            out.push(v);
            out.push(v);
        }
        out
    } else {
        (0..condim).map(|j| if j < 3 { tran } else { rot }).collect()
    }
}

/// `mjtCone`: which friction cone the constraint rows model. MJCF `<option cone>`, default pyramidal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cone {
    /// `2·(condim−1)` non-negative rows on opposing pyramid edges — a linearisation, and a box-constrained
    /// problem any projected solver can take.
    Pyramidal,
    /// `condim` rows, the contact frame itself, constrained to a second-order cone. Exact, and not a box.
    Elliptic,
}

/// How many constraint rows a contact of this condim occupies under this cone.
pub fn mujoco_cone_rows(cone: Cone, condim: usize) -> usize {
    match (cone, condim) {
        (_, 1) => 1,
        (Cone::Pyramidal, d) => 2 * (d - 1),
        (Cone::Elliptic, d) => d,
    }
}

/// **Constraint rows as `mj_makeConstraint` builds them**: the blocks that say what law each row obeys, the
/// Jacobian, `efc_aref` and `efc_D`, ready for [`solve_constraints_newton_blocks`].
#[derive(Clone, Debug, Default)]
pub struct AssembledRows {
    pub blocks: Vec<EfcBlock>,
    pub jac: DMatrix<f64>,
    pub aref: Vec<f64>,
    pub d: Vec<f64>,
}

impl AssembledRows {
    /// Put `other`'s rows after these. ⛔ MuJoCo's order is equality, friction, limits, contacts, and a
    /// row's LAW is read from its position — so this is not a set union, it is a concatenation with a
    /// meaning, and the caller is responsible for appending in that order.
    pub fn append(&mut self, other: AssembledRows) {
        let (n, m, nv) = (self.jac.nrows(), other.jac.nrows(), self.jac.ncols().max(other.jac.ncols()));
        let mut jac = DMatrix::zeros(n + m, nv);
        jac.view_mut((0, 0), (n, self.jac.ncols())).copy_from(&self.jac);
        jac.view_mut((n, 0), (m, other.jac.ncols())).copy_from(&other.jac);
        self.jac = jac;
        self.blocks.extend(other.blocks);
        self.aref.extend(other.aref);
        self.d.extend(other.d);
    }
}

/// **One contact, as `mj_instantiateContact` needs it.**
#[derive(Clone, Debug)]
pub struct ContactSpec {
    /// `condim × nv`: normal, the two tangents, then the torsional and rolling rows at condim 4 and 6.
    pub jac: DMatrix<f64>,
    pub dist: f64,
    /// `mjContact.includemargin`.
    pub margin: f64,
    pub condim: usize,
    /// MuJoCo's unpacked 5-vector `(slide, slide, spin, roll, roll)`.
    pub friction: [f64; 5],
    pub solref: SolRef,
    pub solimp: SolImp,
    /// The two geoms' bodies' inverse weights.
    pub invweight: [InvWeight; 2],
}

/// **`mj_instantiateContact`**: a contact's rows, in the shape its cone demands.
///
/// A frictionless contact is one row. A PYRAMIDAL one is `2(condim−1)` rows, a pair of opposing pyramid
/// edges `J_n ± μ_k J_tk` per friction direction — the cone written as a box, which is what lets a
/// box-constrained solver handle friction at all. An ELLIPTIC one is `condim` rows and is the cone itself.
///
/// ⛔ The regularisation is not per row: [`mujoco_cone_adjust`] rewrites a frictional contact's `R` so the
/// friction directions carry `impratio` times the normal's stiffness, and returns the `mu` of the
/// REGULARISED cone, which is what the solver's projection uses and is not `friction[0]`.
pub fn contact_rows(contacts: &[ContactSpec], nv: usize, qvel: &[f64], cone: Cone, impratio: f64, timestep: f64) -> Result<AssembledRows, String> {
    let mut out = AssembledRows { jac: DMatrix::zeros(0, nv), ..Default::default() };
    for c in contacts {
        if c.jac.ncols() != nv || c.jac.nrows() < c.condim.min(3) {
            return Err("contact Jacobian has the wrong shape".into());
        }
        let pyramidal = cone == Cone::Pyramidal && c.condim > 1;
        let rows = mujoco_cone_rows(if c.condim == 1 { Cone::Pyramidal } else { cone }, c.condim);
        let mut jac = DMatrix::zeros(rows, nv);
        if c.condim == 1 {
            jac.row_mut(0).copy_from(&c.jac.row(0));
        } else if pyramidal {
            let jn = c.jac.row(0);
            for k in 0..c.condim - 1 {
                let jt = c.jac.row(1 + k);
                jac.row_mut(2 * k).copy_from(&(jn + c.friction[k] * jt));
                jac.row_mut(2 * k + 1).copy_from(&(jn - c.friction[k] * jt));
            }
        } else {
            jac.view_mut((0, 0), (c.condim, nv)).copy_from(&c.jac.view((0, 0), (c.condim, nv)));
        }
        let da = mujoco_diag_approx(c.invweight[0], c.invweight[1], c.condim, &c.friction, pyramidal);
        let mut r: Vec<f64> = Vec::with_capacity(rows);
        let mut aref: Vec<f64> = Vec::with_capacity(rows);
        for i in 0..rows {
            let jvel: f64 = (0..nv).map(|k| jac[(i, k)] * qvel[k]).sum();
            // ⛔⛔ a row's POSITION is not the contact's depth. Under the PYRAMIDAL cone every row is
            // `J_n ± μ_k·J_tk` and carries the normal, so every row is at `dist` and switches on at
            // `includemargin`. Under the ELLIPTIC cone the friction rows are the tangents alone: MuJoCo gives
            // them `efc_pos = 0` and `efc_margin = 0`, so their reference acceleration is pure damping,
            // `−B·J·q̇`, with no spring pulling the contact out of penetration sideways.
            //
            // At rest the two readings are indistinguishable — both give `aref = 0` on a friction row — and
            // the regularisation hides it too, because [`mujoco_cone_adjust`] overwrites every friction row's
            // `R` from the normal's. It shows up only in a MOVING elliptic contact, and then it is the whole
            // tangential reference.
            let tangent_only = !pyramidal && c.condim > 1 && i > 0;
            let (pos, margin) = if tangent_only { (0.0, 0.0) } else { (c.dist, c.margin) };
            let (a, rr) = row_reference(&c.solref, &c.solimp, pos, margin, da[i.min(da.len() - 1)], jvel, timestep);
            aref.push(a);
            r.push(rr);
        }
        let mu = mujoco_cone_adjust(&mut r, if c.condim == 1 { Cone::Pyramidal } else { cone }, c.condim, &c.friction, impratio);
        let first = out.jac.nrows();
        let mut grown = DMatrix::zeros(first + rows, nv);
        grown.view_mut((0, 0), (first, nv)).copy_from(&out.jac);
        grown.view_mut((first, 0), (rows, nv)).copy_from(&jac);
        out.jac = grown;
        out.aref.extend(aref);
        out.d.extend(r.iter().map(|x| 1.0 / x.max(1e-15)));
        out.blocks.push(EfcBlock::Contact(ConeContact { cone: if c.condim == 1 { Cone::Pyramidal } else { cone }, condim: c.condim, mu, friction: c.friction }));
    }
    Ok(out)
}

/// **One constraint row's `efc_R` and `efc_aref`** — `mj_makeImpedance` followed by `mj_referenceConstraint`,
/// which is where every row type meets, whatever built it.
///
/// `pos` is the constraint's signed position (negative means violated), `margin` where it switches on,
/// `diag_a` the row's diagonal of `J M⁻¹ Jᵀ` at `qpos0` (`dof_invweight0` for a joint row,
/// [`mujoco_diag_approx`] for a contact), and `jvel` is `J·q̇`.
///
/// `R` is the regularisation `(1 − d)/d · Ā` and `aref = −B·J·q̇ − K·d·(pos − margin)`: the acceleration the
/// row is asked to reach, which is what makes the solve a spring-damper rather than a hard projection.
pub fn row_reference(solref: &SolRef, solimp: &SolImp, pos: f64, margin: f64, diag_a: f64, jvel: f64, timestep: f64) -> (f64, f64) {
    row_reference_at(solref, solimp, pos, pos, margin, diag_a, jvel, timestep)
}

/// [`row_reference`] with the impedance taken at a DIFFERENT position from the spring.
///
/// ⛔⛔ A multi-row equality is one constraint, and MuJoCo gives its rows ONE impedance, taken at the NORM
/// of the whole residual vector. A `connect` whose error is `(−0.085, 0, −0.123)` gives all three rows the
/// impedance of 0.1499, not of their own components — the middle row would otherwise be at `d0` while its
/// neighbours were at `d_width`, and `efc_D` would differ across rows of one ball joint. Measured on
/// MuJoCo 3.13.0 with a `width` wide enough for the impedance not to be saturated, which is the only
/// setting where the three readings (norm, own component, largest component) differ.
#[allow(clippy::too_many_arguments)]
pub fn row_reference_at(solref: &SolRef, solimp: &SolImp, imp_pos: f64, pos: f64, margin: f64, diag_a: f64, jvel: f64, timestep: f64) -> (f64, f64) {
    // ⛔ "integrator safety": in the standard `(timeconst, dampratio)` format MuJoCo floors the time constant
    // at TWO TIMESTEPS, because a reference stiffer than the integrator can follow is unstable. A model that
    // asks for 0.001 s at a 0.002 s timestep gets 0.004 s, and nothing in the file says so.
    let solref = if solref.0 > 0.0 && solref.1 > 0.0 { SolRef(solref.0.max(2.0 * timestep), solref.1) } else { *solref };
    let k = mujoco_kbip(&solref, solimp, imp_pos, margin);
    let r = ((1.0 - k[2]) * diag_a / k[2]).max(1e-15);
    (-k[1] * jvel - k[0] * k[2] * (pos - margin), r)
}

/// **The regularisation MuJoCo puts on a frictional contact's rows**, applied to `r` in place, returning the
/// contact's `mu` — the coefficient of the REGULARIZED cone, which is what the solver's projection uses and
/// what `mjContact.mu` reports. Not `friction[0]`: `<option impratio>` trades normal stiffness against
/// frictional stiffness by scaling the friction rows' `R` down, and `mu` follows so the cone the solver sees
/// is the one the model asked for.
///
/// In MuJoCo's order (`mj_makeImpedance`): the friction rows take `R[1] = R[0]/impratio`, then
/// `mu = friction[0]·√(R[1]/R[0])`; under the elliptic cone the remaining rows are scaled so that
/// `R[j]·μ_j² = R[1]·μ_1²`, which is what makes an anisotropic `friction` an ellipse rather than a circle;
/// under the pyramidal cone every row instead takes one common `R = 2μ²R[0]`, the value at which the
/// pyramid's friction impedance matches the ellipse's.
pub fn mujoco_cone_adjust(r: &mut [f64], cone: Cone, condim: usize, friction: &[f64], impratio: f64) -> f64 {
    if condim == 1 || r.len() < 2 {
        return 0.0;
    }
    r[1] = r[0] / impratio.max(MINVAL);
    let mu = friction[0] * (r[1] / r[0]).sqrt();
    match cone {
        Cone::Elliptic => {
            for j in 1..condim - 1 {
                r[j + 1] = r[1] * friction[0] * friction[0] / (friction[j] * friction[j]);
            }
        }
        Cone::Pyramidal => {
            let rpy = 2.0 * mu * mu * r[0];
            for v in r.iter_mut().take(2 * (condim - 1)) {
                *v = rpy;
            }
        }
    }
    mu
}

/// `mjtConstraintState`: the zone the constraint update put a row in. Every row of an elliptic contact
/// carries the same state, because the cone is one constraint on the whole contact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintState {
    /// The constraint is not active and exerts no force.
    Satisfied,
    /// Inside the quadratic region: `force = −D·jar`.
    Quadratic,
    /// At the negative friction-loss limit.
    LinearNeg,
    /// At the positive friction-loss limit.
    LinearPos,
    /// On the elliptic cone's surface, where the force is the projection onto the dual cone.
    Cone,
}

/// One contact as the constraint update sees it: how many rows it owns and, for the elliptic cone, the
/// regularized `mu` and the friction coefficients that shape the ellipse.
#[derive(Clone, Copy, Debug)]
pub struct ConeContact {
    pub cone: Cone,
    pub condim: usize,
    /// `mjContact.mu`, from [`mujoco_cone_adjust`].
    pub mu: f64,
    /// `mjContact.friction`, MuJoCo's unpacked 5-vector `(slide, slide, spin, roll, roll)`.
    pub friction: [f64; 5],
}

/// What [`mujoco_constraint_update`] produces.
#[derive(Clone, Debug)]
pub struct ConstraintUpdate {
    /// `efc_force`.
    pub force: Vec<f64>,
    /// `efc_state`.
    pub state: Vec<ConstraintState>,
    /// `s_hat(jar)`, the constraint part of the cost MuJoCo's solvers minimise over `qacc`.
    pub cost: f64,
}

/// **One block of `efc` rows**, in MuJoCo's own order — equality first, then friction, then limits and
/// contacts. The order is not cosmetic: `mj_constraintUpdate_impl` decides a row's law from its POSITION
/// (`i < ne`, then `i < ne + nf`), not from a per-row tag, so a block list is the faithful description.
#[derive(Clone, Debug)]
pub enum EfcBlock {
    /// `rows` equality rows: purely quadratic and unbounded, because an equality pushes as hard as it must.
    Equality { rows: usize },
    /// One friction row — `<joint frictionloss>` or a tendon's — bounded to ±`loss`. Between the bounds it
    /// is quadratic; outside, the force saturates and the cost becomes linear in `jar`.
    Friction { loss: f64 },
    /// One limit row, joint or tendon: the same one-sided law a frictionless contact row obeys.
    Limit,
    /// One contact, consuming [`mujoco_cone_rows`] rows.
    Contact(ConeContact),
}

impl EfcBlock {
    /// How many `efc` rows this block occupies.
    pub fn rows(&self) -> usize {
        match self {
            EfcBlock::Equality { rows } => *rows,
            EfcBlock::Friction { .. } | EfcBlock::Limit => 1,
            EfcBlock::Contact(c) => mujoco_cone_rows(c.cone, c.condim),
        }
    }
}

/// **`mj_constraintUpdate` for every row type MuJoCo has**, not only contacts: given `jar = J·qacc − a_ref`,
/// the force each row exerts, the zone it is in, and the cost.
///
/// `d` is `efc_D` (`1/R`) for every row, in order; `blocks` describes the same rows in the same order.
pub fn mujoco_constraint_update_blocks(blocks: &[EfcBlock], d: &[f64], jar: &[f64]) -> ConstraintUpdate {
    let n = d.len().min(jar.len());
    let mut force: Vec<f64> = (0..n).map(|i| -d[i] * jar[i]).collect();
    let mut state = vec![ConstraintState::Quadratic; n];
    let mut cost = 0.0;
    let mut i = 0;
    for b in blocks {
        let rows = b.rows();
        if i + rows > n {
            break;
        }
        match b {
            EfcBlock::Equality { .. } => {
                for k in i..i + rows {
                    cost += 0.5 * d[k] * jar[k] * jar[k];
                    state[k] = ConstraintState::Quadratic;
                }
            }
            EfcBlock::Friction { loss } => {
                // R is 1/D; the bounds live at |jar| = R·loss, where the quadratic meets the two lines
                let r = if d[i] > 0.0 { 1.0 / d[i] } else { 0.0 };
                if jar[i] <= -r * loss {
                    cost += -0.5 * r * loss * loss - loss * jar[i];
                    force[i] = *loss;
                    state[i] = ConstraintState::LinearNeg;
                } else if jar[i] >= r * loss {
                    cost += -0.5 * r * loss * loss + loss * jar[i];
                    force[i] = -loss;
                    state[i] = ConstraintState::LinearPos;
                } else {
                    cost += 0.5 * d[i] * jar[i] * jar[i];
                    state[i] = ConstraintState::Quadratic;
                }
            }
            EfcBlock::Limit => {
                if jar[i] >= 0.0 {
                    force[i] = 0.0;
                    state[i] = ConstraintState::Satisfied;
                } else {
                    cost += 0.5 * d[i] * jar[i] * jar[i];
                    state[i] = ConstraintState::Quadratic;
                }
            }
            EfcBlock::Contact(c) => contact_zone(c, d, jar, i, &mut force, &mut state, &mut cost),
        }
        i += rows;
    }
    ConstraintUpdate { force, state, cost }
}

/// The contact branch of [`mujoco_constraint_update_blocks`], split out so the block walker stays readable.
/// Writes this contact's rows of `force` and `state` and adds to `cost`.
///
/// ⛔ The rows of a PYRAMIDAL contact are independent constraints and their states are not one value — one
/// side of the pyramid can be satisfied while the opposite side is loaded. An elliptic contact is the
/// opposite: its rows are one cone and share a single zone.
fn contact_zone(c: &ConeContact, d: &[f64], jar: &[f64], i: usize, force: &mut [f64], state: &mut [ConstraintState], cost: &mut f64) {
    let rows = mujoco_cone_rows(c.cone, c.condim);
    if c.cone != Cone::Elliptic || c.condim == 1 {
        for k in i..i + rows {
            if jar[k] >= 0.0 {
                force[k] = 0.0;
                state[k] = ConstraintState::Satisfied;
            } else {
                *cost += 0.5 * d[k] * jar[k] * jar[k];
                state[k] = ConstraintState::Quadratic;
            }
        }
        return;
    }
    let dim = c.condim;
    let mu = c.mu;
    let mut u = [0.0f64; 6];
    u[0] = jar[i] * mu;
    for j in 1..dim {
        u[j] = jar[i + j] * c.friction[j - 1];
    }
    let nn = u[0];
    let t = u[1..dim].iter().map(|x| x * x).sum::<f64>().sqrt();
    let zone = if nn >= mu * t || (t <= 0.0 && nn >= 0.0) {
        for f in force.iter_mut().skip(i).take(dim) {
            *f = 0.0;
        }
        ConstraintState::Satisfied
    } else if mu * nn + t <= 0.0 || (t <= 0.0 && nn < 0.0) {
        for k in i..i + dim {
            *cost += 0.5 * d[k] * jar[k] * jar[k];
        }
        ConstraintState::Quadratic
    } else {
        let dm = d[i] / (mu * mu * mu.mul_add(mu, 1.0));
        let nmt = nn - mu * t;
        *cost += 0.5 * dm * nmt * nmt;
        force[i] = -dm * nmt * mu;
        for j in 1..dim {
            force[i + j] = -force[i] / t * u[j] * c.friction[j - 1];
        }
        ConstraintState::Cone
    };
    for st in state.iter_mut().skip(i).take(dim) {
        *st = zone;
    }
}

/// **`mj_constraintUpdate` for contact rows**: given `jar = J·qacc − a_ref`, the force each row exerts, the
/// zone it is in, and the cost. This is the function MuJoCo's own solvers evaluate at every trial `qacc`;
/// the pyramidal branch is a per-row clamp at zero, and the elliptic branch is the part a box-constrained
/// solver cannot express.
///
/// The elliptic branch maps the contact's `jar` into the regular dual cone with `diag(mu, friction)`, splits
/// it into normal `N` and tangential norm `T`, and takes one of three zones: above the cone (`N ≥ μT`) the
/// contact is satisfied and exerts nothing; below its polar (`μN + T ≤ 0`) it is fully quadratic; between
/// them the force is the projection onto the cone's surface, radial in the tangential plane.
///
/// `d` is `efc_D` (`1/R`) for every row, in order; `contacts` are in the same row order.
pub fn mujoco_constraint_update(contacts: &[ConeContact], d: &[f64], jar: &[f64]) -> ConstraintUpdate {
    let blocks: Vec<EfcBlock> = contacts.iter().cloned().map(EfcBlock::Contact).collect();
    mujoco_constraint_update_blocks(&blocks, d, jar)
}

/// **The constraint cost's Hessian in `jar`**, block-diagonal by contact, `nefc × nefc`.
///
/// A satisfied row contributes nothing and a quadratic row contributes `D`; an elliptic contact on the cone
/// contributes `mjContact.H` — MuJoCo's `flg_coneHessian` branch of `mj_constraintUpdate`, the second
/// derivative of the cone's radial projection, pre- and post-multiplied by `diag(mu, friction)`.
pub fn mujoco_constraint_hessian(contacts: &[ConeContact], d: &[f64], jar: &[f64]) -> DMatrix<f64> {
    let n = d.len().min(jar.len());
    let mut h = DMatrix::zeros(n, n);
    let u = mujoco_constraint_update(contacts, d, jar);
    let mut i = 0;
    for c in contacts {
        let rows = mujoco_cone_rows(c.cone, c.condim);
        if i + rows > n {
            break;
        }
        if c.cone != Cone::Elliptic || c.condim == 1 {
            for k in i..i + rows {
                if u.state[k] == ConstraintState::Quadratic {
                    h[(k, k)] = d[k];
                }
            }
            i += rows;
            continue;
        }
        let dim = c.condim;
        match u.state[i] {
            ConstraintState::Satisfied => {}
            ConstraintState::Cone => {
                let mu = c.mu;
                let mut uu = [0.0f64; 6];
                uu[0] = jar[i] * mu;
                for j in 1..dim {
                    uu[j] = jar[i + j] * c.friction[j - 1];
                }
                let (nn, t) = (uu[0], uu[1..dim].iter().map(|x| x * x).sum::<f64>().sqrt());
                let dm = d[i] / (mu * mu * mu.mul_add(mu, 1.0));
                let mut b = vec![0.0f64; dim * dim];
                // first row: (1, −mu/T · U)
                b[0] = 1.0;
                for j in 1..dim {
                    b[j] = -mu / t * uu[j];
                }
                // upper block: mu·N/T³ · UUᵀ, then (mu² − mu·N/T) on its diagonal
                let scl = mu * nn / (t * t * t);
                for k in 1..dim {
                    for j in k..dim {
                        b[k * dim + j] = scl * uu[j] * uu[k];
                    }
                }
                let scl = mu * mu - mu * nn / t;
                for j in 1..dim {
                    b[j * (dim + 1)] += scl;
                }
                // pre- and post-multiply by diag(mu, friction) and scale by Dm
                for k in 0..dim {
                    let sk = dm * if k == 0 { mu } else { c.friction[k - 1] };
                    for j in k..dim {
                        b[k * dim + j] *= sk * if j == 0 { mu } else { c.friction[j - 1] };
                    }
                }
                for k in 0..dim {
                    for j in k..dim {
                        h[(i + k, i + j)] = b[k * dim + j];
                        h[(i + j, i + k)] = b[k * dim + j];
                    }
                }
            }
            _ => {
                for k in i..i + dim {
                    h[(k, k)] = d[k];
                }
            }
        }
        i += dim;
    }
    h
}

/// **The constraint cost's Hessian in `jar`, for every row type**, block-diagonal, `nefc × nefc`.
///
/// ⛔ A SATURATED friction row has a linear cost, so it contributes nothing: the curvature vanishes the
/// moment the force reaches ±loss, and the Newton step there is a pure gradient step. An equality row is the
/// opposite — quadratic everywhere, so its second derivative is `D` and never anything else.
pub fn mujoco_constraint_hessian_blocks(blocks: &[EfcBlock], d: &[f64], jar: &[f64]) -> DMatrix<f64> {
    let n = d.len().min(jar.len());
    let mut h = DMatrix::zeros(n, n);
    for (i, sub) in hessian_block_list(blocks, d, jar) {
        let rows = sub.nrows();
        for k in 0..rows {
            for j in 0..rows {
                h[(i + k, i + j)] = sub[(k, j)];
            }
        }
    }
    h
}

/// The same Hessian as its diagonal BLOCKS, `(first row, block)` — every row is its own block but a
/// contact's, and a block that is all zero is left out. The Hessian of the constraint cost is block-diagonal
/// by construction, so this is all of it.
fn hessian_block_list(blocks: &[EfcBlock], d: &[f64], jar: &[f64]) -> Vec<(usize, DMatrix<f64>)> {
    let n = d.len().min(jar.len());
    let u = mujoco_constraint_update_blocks(blocks, d, jar);
    let mut out = Vec::with_capacity(blocks.len());
    let mut i = 0;
    for b in blocks {
        let rows = b.rows();
        if i + rows > n {
            break;
        }
        match b {
            EfcBlock::Equality { .. } => {
                for k in i..i + rows {
                    out.push((k, DMatrix::from_element(1, 1, d[k])));
                }
            }
            EfcBlock::Friction { .. } | EfcBlock::Limit => {
                if u.state[i] == ConstraintState::Quadratic {
                    out.push((i, DMatrix::from_element(1, 1, d[i])));
                }
            }
            EfcBlock::Contact(c) => {
                // the contact block on its own rows, by the routine the cone cases already pin
                let sub = mujoco_constraint_hessian(std::slice::from_ref(c), &d[i..i + rows], &jar[i..i + rows]);
                if sub.iter().any(|x| *x != 0.0) {
                    out.push((i, sub));
                }
            }
        }
        i += rows;
    }
    out
}

/// `Jᵀ·H·J` for the block-diagonal constraint Hessian, block by block: `Σ J_bᵀ·H_b·J_b`. The dense
/// product costs `nv·nefc²`, which on a scene with a few hundred contact rows is the whole of the solve;
/// this costs `nv²·Σ n_b`.
fn jt_h_j(blocks: &[EfcBlock], d: &[f64], jar: &[f64], jac: &DMatrix<f64>) -> DMatrix<f64> {
    let nv = jac.ncols();
    let mut out = DMatrix::zeros(nv, nv);
    for (i, h) in hessian_block_list(blocks, d, jar) {
        let jb = jac.rows(i, h.nrows());
        let hj = &h * jb;
        out.gemm_tr(1.0, &jb, &hj, 1.0);
    }
    out
}

/// What [`solve_constraints_newton`] produces.
#[derive(Clone, Debug)]
pub struct NewtonSolve {
    /// `qacc` at the minimum.
    pub qacc: DVector<f64>,
    /// `efc_force` there.
    pub force: Vec<f64>,
    /// `efc_state` there.
    pub state: Vec<ConstraintState>,
    /// The objective `½(a−a₀)ᵀM(a−a₀) + s_hat(jar)` at the minimum.
    pub cost: f64,
    pub iterations: usize,
    /// `‖M(a−a₀) − Jᵀf‖` at exit, MuJoCo's own measure of how solved the problem is.
    pub grad_norm: f64,
}

/// **Solve MuJoCo's constraint problem in the primal, over acceleration**, which is the only formulation
/// that expresses the elliptic cone.
///
/// Minimise `½(a − a_smooth)ᵀ M (a − a_smooth) + s_hat(J·a − a_ref)` by Newton with a backtracking line
/// search. `s_hat` is [`mujoco_constraint_update`]'s cost, so the pyramidal and frictionless rows are the
/// same box-shaped problem the dual solver handles and the elliptic rows are second-order cones.
///
/// The objective is strictly convex (`M` is positive definite and `s_hat` is convex), so its minimiser is
/// unique: a tightly converged solve reproduces MuJoCo's `qacc` whatever path it took to get there. That is
/// what this is checked against — MuJoCo's returned `qacc`, not its iteration sequence.
///
/// `jac` is `nefc × nv`, `d` is `efc_D` and `aref` is `efc_aref`, both in row order; `contacts` describes the
/// rows in the same order. Dense throughout, like the rest of this module.
pub fn solve_constraints_newton(
    m: &DMatrix<f64>,
    a_smooth: &DVector<f64>,
    jac: &DMatrix<f64>,
    aref: &[f64],
    d: &[f64],
    contacts: &[ConeContact],
    tol: f64,
    max_iter: usize,
) -> Result<NewtonSolve, String> {
    let blocks: Vec<EfcBlock> = contacts.iter().cloned().map(EfcBlock::Contact).collect();
    solve_constraints_newton_blocks(m, a_smooth, jac, aref, d, &blocks, tol, max_iter)
}

/// **`mj_solNoSlip`**, ported from MuJoCo 3.13.0's `solNoSlip` (monolithic): after the main solve, the
/// FRICTION forces are re-solved in the dual with their regularisation removed, so friction holds instead of
/// creeping. Only three kinds of row move: dof and tendon friction (a projected Gauss-Seidel step clamped to
/// ±`frictionloss`), each pair of opposing pyramid edges of a pyramidal contact (their sum, the normal share,
/// is kept and only the split moves), and the tangential rows of an elliptic contact (a QCQP against the
/// friction ellipsoid at the contact's normal force). Equality, limit and normal forces stay where the main
/// solve put them.
///
/// `m`, `jac`, `aref`, `d` and `blocks` are the problem as posed (MuJoCo's basis); `force` is `efc_force`
/// from the main solve and is updated in place. Returns the new `qacc = a₀ + M⁻¹·Jᵀ·f`.
///
/// `meaninertia` is MuJoCo's `stat.meaninertia` (the mean diagonal of `M` at `qpos0`): it only scales the
/// improvement that the `noslip_tolerance` test compares, and so decides WHEN the sweep stops.
#[allow(clippy::too_many_arguments)]
pub fn mujoco_noslip(m: &DMatrix<f64>, a_smooth: &DVector<f64>, jac: &DMatrix<f64>, aref: &[f64], d: &[f64], blocks: &[EfcBlock], force: &mut [f64], maxiter: usize, tolerance: f64, meaninertia: f64) -> Result<DVector<f64>, String> {
    const MINVAL: f64 = 1e-15;
    let (nv, nefc) = (m.nrows(), jac.nrows());
    let chol = m.clone().cholesky().ok_or("noslip: the mass matrix is not positive definite")?;
    let minv_jt = chol.solve(&jac.transpose());
    // `efc_AR` without `R`, since every noslip step runs with `flg_subR`: A = J·M⁻¹·Jᵀ
    let a = jac * &minv_jt;
    let r: Vec<f64> = d.iter().map(|x| 1.0 / x).collect();
    let b: Vec<f64> = (jac * a_smooth).iter().zip(aref).map(|(ja, ar)| ja - ar).collect();
    // `residual(.., flg_subR)`: b + (A + R)·f − R·f
    let residual = |f: &[f64], i: usize| -> f64 { b[i] + (0..nefc).map(|k| a[(i, k)] * f[k]).sum::<f64>() };
    // `extractBlock(.., flg_subR)`: the block of A, its diagonal clamped at 1e-10
    let block = |start: usize, n: usize| -> Vec<f64> {
        let mut ac = vec![0.0; n * n];
        for rr in 0..n {
            for cc in 0..n {
                ac[rr * n + cc] = a[(start + rr, start + cc)];
            }
            ac[rr * (n + 1)] = ac[rr * (n + 1)].max(1e-10);
        }
        ac
    };
    // `costChange`: a step that raises the cost is undone
    let cost_change = |ac: &[f64], f: &mut [f64], old: &[f64], res: &[f64], n: usize| -> f64 {
        let delta: Vec<f64> = (0..n).map(|k| f[k] - old[k]).collect();
        let change = if n == 1 {
            0.5 * delta[0] * delta[0] * ac[0] + delta[0] * res[0]
        } else {
            let quad: f64 = (0..n).map(|rr| delta[rr] * (0..n).map(|cc| ac[rr * n + cc] * delta[cc]).sum::<f64>()).sum();
            0.5 * quad + (0..n).map(|k| delta[k] * res[k]).sum::<f64>()
        };
        if change > 1e-10 {
            f[..n].copy_from_slice(&old[..n]);
            0.0
        } else {
            change
        }
    };
    let scale = 1.0 / (meaninertia * (nv.max(1) as f64));
    let mut iter = 0;
    while iter < maxiter {
        let mut improvement = 0.0;
        if iter == 0 {
            improvement += (0..nefc).map(|i| 0.5 * force[i] * force[i] * r[i]).sum::<f64>();
        }
        let mut row = 0usize;
        for bk in blocks {
            let rows = bk.rows();
            match bk {
                // dry friction: one row, a Gauss-Seidel step clamped to ±frictionloss
                EfcBlock::Friction { loss } => {
                    let i = row;
                    let res = residual(force, i);
                    let old = force[i];
                    let arinv = 1.0 / a[(i, i)].max(MINVAL);
                    force[i] = (force[i] - res * arinv).clamp(-loss, *loss);
                    let delta = force[i] - old;
                    improvement -= 0.5 * delta * delta / arinv + delta * res;
                }
                EfcBlock::Contact(c) if c.condim > 1 && c.cone == Cone::Pyramidal => {
                    // opposing pyramid edges in pairs: keep their sum, move their split
                    for j in (row..row + 2 * (c.condim - 1)).step_by(2) {
                        let res = [residual(force, j), residual(force, j + 1)];
                        let old = [force[j], force[j + 1]];
                        let ac = block(j, 2);
                        let bc = [res[0] - (ac[0] * old[0] + ac[1] * old[1]), res[1] - (ac[2] * old[0] + ac[3] * old[1])];
                        let mid = 0.5 * (force[j] + force[j + 1]);
                        let k1 = ac[0] + ac[3] - ac[1] - ac[2];
                        let k0 = mid * (ac[0] - ac[3]) + bc[0] - bc[1];
                        if k1 < MINVAL {
                            force[j] = mid;
                            force[j + 1] = mid;
                        } else {
                            let y = -k0 / k1;
                            if y < -mid {
                                force[j] = 0.0;
                                force[j + 1] = 2.0 * mid;
                            } else if y > mid {
                                force[j] = 2.0 * mid;
                                force[j + 1] = 0.0;
                            } else {
                                force[j] = mid + y;
                                force[j + 1] = mid - y;
                            }
                        }
                        improvement -= cost_change(&ac, &mut force[j..j + 2], &old, &res, 2);
                    }
                }
                EfcBlock::Contact(c) if c.condim > 1 && c.cone == Cone::Elliptic => {
                    // the tangential rows, against the friction ellipsoid at this normal force
                    let (i, n) = (row, c.condim - 1);
                    let res: Vec<f64> = (0..n).map(|k| residual(force, i + 1 + k)).collect();
                    let old: Vec<f64> = force[i + 1..i + 1 + n].to_vec();
                    let ac = block(i + 1, n);
                    let bc: Vec<f64> = (0..n).map(|rr| res[rr] - (0..n).map(|cc| ac[rr * n + cc] * old[cc]).sum::<f64>()).collect();
                    if force[i] < MINVAL {
                        force[i + 1..i + 1 + n].iter_mut().for_each(|x| *x = 0.0);
                    } else {
                        let mu = &c.friction[..n];
                        let (mut v, active) = qcqp(&ac, &bc, mu, force[i], n);
                        if active {
                            // put v on the ellipsoid, in case the QCQP is approximate
                            let s: f64 = (0..n).map(|k| v[k] * v[k] / (mu[k] * mu[k])).sum();
                            let scl = (force[i] * force[i] / s.max(MINVAL)).sqrt();
                            v.iter_mut().for_each(|x| *x *= scl);
                        }
                        force[i + 1..i + 1 + n].copy_from_slice(&v);
                    }
                    improvement -= cost_change(&ac, &mut force[i + 1..i + 1 + n], &old, &res, n);
                }
                _ => {}
            }
            row += rows;
        }
        improvement *= scale;
        iter += 1;
        if improvement < tolerance {
            break;
        }
    }
    // `mj_dualFinish`: back to joint space
    let f = DVector::from_row_slice(force);
    Ok(a_smooth + &minv_jt * f)
}

/// `mju_QCQP2`, line by line.
fn qcqp2(ain: &[f64], bin: &[f64], d: &[f64], r: f64) -> (Vec<f64>, bool) {
    let (b1, b2) = (bin[0] * d[0], bin[1] * d[1]);
    let (a11, a22, a12) = (ain[0] * d[0] * d[0], ain[3] * d[1] * d[1], ain[1] * d[0] * d[1]);
    let (mut la, mut v1, mut v2) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..20 {
        let det = (a11 + la) * (a22 + la) - a12 * a12;
        if det < 1e-10 {
            return (vec![0.0, 0.0], false);
        }
        let detinv = 1.0 / det;
        let (p11, p22, p12) = ((a22 + la) * detinv, (a11 + la) * detinv, -a12 * detinv);
        v1 = -p11 * b1 - p12 * b2;
        v2 = -p12 * b1 - p22 * b2;
        let val = v1 * v1 + v2 * v2 - r * r;
        if val < 1e-10 {
            break;
        }
        let deriv = -2.0 * (p11 * v1 * v1 + 2.0 * p12 * v1 * v2 + p22 * v2 * v2);
        let delta = -val / deriv;
        if delta < 1e-10 {
            break;
        }
        la += delta;
    }
    (vec![v1 * d[0], v2 * d[1]], la != 0.0)
}

/// `mju_QCQP3`, line by line.
fn qcqp3(ain: &[f64], bin: &[f64], d: &[f64], r: f64) -> (Vec<f64>, bool) {
    let (b1, b2, b3) = (bin[0] * d[0], bin[1] * d[1], bin[2] * d[2]);
    let (a11, a22, a33) = (ain[0] * d[0] * d[0], ain[4] * d[1] * d[1], ain[8] * d[2] * d[2]);
    let (a12, a13, a23) = (ain[1] * d[0] * d[1], ain[2] * d[0] * d[2], ain[5] * d[1] * d[2]);
    let (mut la, mut v1, mut v2, mut v3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for _ in 0..20 {
        let mut p11 = (a22 + la) * (a33 + la) - a23 * a23;
        let mut p22 = (a11 + la) * (a33 + la) - a13 * a13;
        let mut p33 = (a11 + la) * (a22 + la) - a12 * a12;
        let mut p12 = a13 * a23 - a12 * (a33 + la);
        let mut p13 = a12 * a23 - a13 * (a22 + la);
        let mut p23 = a12 * a13 - a23 * (a11 + la);
        let det = (a11 + la) * p11 + a12 * p12 + a13 * p13;
        if det < 1e-10 {
            return (vec![0.0, 0.0, 0.0], false);
        }
        let detinv = 1.0 / det;
        p11 *= detinv;
        p22 *= detinv;
        p33 *= detinv;
        p12 *= detinv;
        p13 *= detinv;
        p23 *= detinv;
        v1 = -p11 * b1 - p12 * b2 - p13 * b3;
        v2 = -p12 * b1 - p22 * b2 - p23 * b3;
        v3 = -p13 * b1 - p23 * b2 - p33 * b3;
        let val = v1 * v1 + v2 * v2 + v3 * v3 - r * r;
        if val < 1e-10 {
            break;
        }
        let deriv = -2.0 * (p11 * v1 * v1 + p22 * v2 * v2 + p33 * v3 * v3) - 4.0 * (p12 * v1 * v2 + p13 * v1 * v3 + p23 * v2 * v3);
        let delta = -val / deriv;
        if delta < 1e-10 {
            break;
        }
        la += delta;
    }
    (vec![v1 * d[0], v2 * d[1], v3 * d[2]], la != 0.0)
}

/// `mju_QCQP2`, `mju_QCQP3` and `mju_QCQP`: `min ½xᵀAx + bᵀx` subject to `Σ (xₖ/dₖ)² ≤ r²`, by Newton on the
/// multiplier after scaling the constraint to a ball. Returns the solution and whether the constraint was
/// active. Sizes 2 and 3 follow MuJoCo's closed forms line by line (their SPD test is on the determinant,
/// not on a Cholesky pivot); larger sizes follow the dense `mju_QCQP`.
fn qcqp(ain: &[f64], bin: &[f64], dsc: &[f64], r: f64, n: usize) -> (Vec<f64>, bool) {
    match n {
        2 => return qcqp2(ain, bin, dsc, r),
        3 => return qcqp3(ain, bin, dsc, r),
        _ => {}
    }
    let b: Vec<f64> = (0..n).map(|i| bin[i] * dsc[i]).collect();
    let a: Vec<f64> = (0..n * n).map(|k| ain[k] * dsc[k / n] * dsc[k % n]).collect();
    let mut la = 0.0;
    let mut res = vec![0.0; n];
    for _ in 0..20 {
        let ala = DMatrix::from_fn(n, n, |i, j| a[i * n + j] + if i == j { la } else { 0.0 });
        // factorize, with MuJoCo's rank check at 1e-10
        let Some(ch) = ala.clone().cholesky().filter(|ch| (0..n).all(|k| ch.l()[(k, k)] * ch.l()[(k, k)] >= 1e-10)) else {
            return (vec![0.0; n], false);
        };
        let x = -ch.solve(&DVector::from_row_slice(&b));
        res = x.iter().copied().collect();
        let val = x.dot(&x) - r * r;
        if val < 1e-10 {
            break;
        }
        let tmp = ch.solve(&x);
        let deriv = -2.0 * x.dot(&tmp);
        let delta = -val / deriv;
        if delta < 1e-10 {
            break;
        }
        la += delta;
    }
    ((0..n).map(|i| res[i] * dsc[i]).collect(), la != 0.0)
}

/// The same solve over MuJoCo's full row set — equality, friction and limit rows as well as contacts.
/// `blocks` describes the rows of `jac` in order, and their order is MuJoCo's: equality, friction, then
/// limits and contacts.
#[allow(clippy::too_many_arguments)]
pub fn solve_constraints_newton_blocks(
    m: &DMatrix<f64>,
    a_smooth: &DVector<f64>,
    jac: &DMatrix<f64>,
    aref: &[f64],
    d: &[f64],
    blocks: &[EfcBlock],
    tol: f64,
    max_iter: usize,
) -> Result<NewtonSolve, String> {
    let nv = m.nrows();
    let nefc = jac.nrows();
    if m.ncols() != nv || a_smooth.len() != nv || jac.ncols() != nv {
        return Err("mass matrix, a_smooth and Jacobian disagree on nv".into());
    }
    if aref.len() != nefc || d.len() != nefc {
        return Err("aref and D must have one entry per constraint row".into());
    }
    let arefv = DVector::from_row_slice(aref);
    let objective = |a: &DVector<f64>| -> (f64, ConstraintUpdate, Vec<f64>) {
        let jar: Vec<f64> = (jac * a - &arefv).iter().copied().collect();
        let u = mujoco_constraint_update_blocks(blocks, d, &jar);
        let da = a - a_smooth;
        (0.5 * (da.transpose() * m * &da)[(0, 0)] + u.cost, u, jar)
    };
    let mut a = a_smooth.clone();
    let (mut cost, mut u, mut jar) = objective(&a);
    let mut iterations = 0;
    let mut grad_norm = f64::INFINITY;
    for _ in 0..max_iter {
        let f = DVector::from_row_slice(&u.force);
        let grad = m * (&a - a_smooth) - jac.transpose() * &f;
        grad_norm = grad.norm();
        if grad_norm <= tol {
            break;
        }
        // the dense product while it is the cheaper one (`nv·nefc²` against `nv²·nefc`), block by block after
        let hess = if nefc <= nv { m + jac.transpose() * mujoco_constraint_hessian_blocks(blocks, d, &jar) * jac } else { m + jt_h_j(blocks, d, &jar, jac) };
        // a strictly convex objective has a positive-definite Hessian; a Cholesky that fails means the
        // active set has left it singular, so fall back to the steepest descent direction rather than stop
        let dir = match hess.cholesky() {
            Some(c) => -c.solve(&grad),
            None => -grad.clone(),
        };
        // backtracking line search; the direction is a descent direction, so some step decreases the cost
        let slope = grad.dot(&dir);
        if slope >= 0.0 {
            break;
        }
        let mut step = 1.0;
        let mut improved = false;
        let mut unchanged = false;
        for _ in 0..60 {
            let trial = &a + &dir * step;
            let (c2, u2, jar2) = objective(&trial);
            if c2 <= cost + 1e-4 * step * slope {
                // ⛔ the ROUNDING FLOOR. On a cost of 1e7 the Armijo test accepts a step that moves
                // nothing (`cost + 1e-4·step·slope` rounds back to `cost`, and `a + dir·step` back to `a`),
                // the gradient settles above the tolerance (aloha: 4e-4), and every later iteration repeats
                // this one exactly — 190 of aloha's 200. An accepted step that leaves `a` bit for bit
                // where it was is that fixed point, so stopping there changes no answer.
                unchanged = trial == a;
                a = trial;
                cost = c2;
                u = u2;
                jar = jar2;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        iterations += 1;
        if !improved || unchanged {
            break;
        }
    }
    Ok(NewtonSolve { qacc: a, force: u.force, state: u.state, cost, iterations, grad_norm })
}

/// One contact between two bodies, as MuJoCo's `mjContact` carries it into the constraint solver.
#[derive(Clone, Debug)]
pub struct MjContact {
    /// Rows `normal, t1, t2` of the contact Jacobian (relative velocity of body 2 with respect to body 1, in
    /// the contact frame, per generalised velocity), `3 × nv`; only the first row is used when `condim == 1`.
    pub jac: DMatrix<f64>,
    /// Signed distance, negative when penetrating (`mjContact.dist`).
    pub dist: f64,
    /// `mjContact.includemargin`: the margin at which the contact is active (`pos − margin` is what the
    /// impedance and reference see).
    pub margin: f64,
    /// `1` (frictionless) or `3` (sliding friction). Torsional and rolling rows (`4`, `6`) are not carried.
    pub condim: usize,
    /// `friction[0..2]` — the two tangential coefficients (MuJoCo copies `geom friction[0]` into both).
    pub friction: [f64; 2],
    pub solref: SolRef,
    pub solimp: SolImp,
    /// The two bodies' inverse weights, geom 1 then geom 2.
    pub invweight: [InvWeight; 2],
}

/// Everything MuJoCo derives per row and the force it solves for.
#[derive(Clone, Debug)]
pub struct MjContactSolve {
    /// Constraint Jacobian, `nefc × nv`.
    pub jac: DMatrix<f64>,
    /// `efc_R`.
    pub r: Vec<f64>,
    /// `efc_aref`.
    pub aref: Vec<f64>,
    /// `efc_KBIP` per row.
    pub kbip: Vec<[f64; 4]>,
    /// `efc_force` — the solved constraint forces (each row `≥ 0`).
    pub force: Vec<f64>,
    /// `qacc = a₀ + M⁻¹Jᵀf`.
    pub qacc: DVector<f64>,
    /// Largest complementarity residual of the projected Gauss–Seidel solve at exit.
    pub residual: f64,
}

/// **Assemble the rows and solve MuJoCo's dual problem** for a set of contacts on a system with mass matrix
/// `m` (`nv × nv`), unconstrained acceleration `a0` and `qvel`. `pyramidal` selects the cone (`true` is
/// MuJoCo's default `cone="pyramidal"`; the elliptic cone is not carried here) and `impratio` is
/// `<option impratio>` (default 1).
///
/// The solve is projected Gauss–Seidel on `min ½ fᵀ(A + R)f + fᵀ(J·a₀ − a_ref)`, `f ≥ 0`, which for
/// frictionless and pyramidal rows is exactly the box-constrained convex problem MuJoCo's own solvers converge
/// to; it runs until the complementarity residual is below `tol` or `max_iter` sweeps.
pub fn solve_contacts_mujoco(m: &DMatrix<f64>, a0: &DVector<f64>, qvel: &DVector<f64>, contacts: &[MjContact], pyramidal: bool, impratio: f64, tol: f64, max_iter: usize) -> Result<MjContactSolve, String> {
    let nv = m.nrows();
    if m.ncols() != nv || a0.len() != nv || qvel.len() != nv {
        return Err("mass matrix, a0 and qvel disagree on nv".into());
    }
    // --- rows: J, pos, margin, diagA, solver params
    let mut rows: Vec<DVector<f64>> = Vec::new();
    let mut pos = Vec::new();
    let mut margin = Vec::new();
    let mut diag_a = Vec::new();
    let mut params: Vec<(SolRef, SolImp)> = Vec::new();
    let mut groups: Vec<(usize, usize, bool, f64)> = Vec::new(); // (first row, nrows, pyramidal, mu)
    for c in contacts {
        if c.jac.ncols() != nv || c.jac.nrows() < if c.condim == 1 { 1 } else { 3 } {
            return Err("contact Jacobian has the wrong shape".into());
        }
        if c.condim != 1 && c.condim != 3 {
            return Err(format!("condim {} is not carried (1 or 3)", c.condim));
        }
        let jn = c.jac.row(0).transpose();
        let first = rows.len();
        let da = mujoco_diag_approx(c.invweight[0], c.invweight[1], c.condim, &c.friction, pyramidal);
        if c.condim == 1 {
            rows.push(jn);
            pos.push(c.dist);
            margin.push(c.margin);
            diag_a.push(da[0]);
            params.push((c.solref, c.solimp));
            groups.push((first, 1, false, 0.0));
        } else if pyramidal {
            // one pair of opposing pyramid edges per friction direction: J_n ± μ_k J_tk
            for k in 0..2 {
                let jt = c.jac.row(1 + k).transpose();
                rows.push(&jn + c.friction[k] * &jt);
                rows.push(&jn - c.friction[k] * &jt);
                for _ in 0..2 {
                    pos.push(c.dist);
                    margin.push(c.margin);
                    params.push((c.solref, c.solimp));
                }
                diag_a.push(da[2 * k]);
                diag_a.push(da[2 * k + 1]);
            }
            groups.push((first, 4, true, c.friction[0]));
        } else {
            return Err("the elliptic cone is not carried here; use pyramidal".into());
        }
    }
    let nefc = rows.len();
    // --- impedance, K, B, R (mj_makeImpedance), reference (mj_referenceConstraint)
    let mut kb = Vec::with_capacity(nefc);
    let mut r = Vec::with_capacity(nefc);
    for i in 0..nefc {
        let (solref, solimp) = params[i];
        kb.push(mujoco_kbip(&solref, &solimp, pos[i], margin[i]));
        r.push(row_reference(&solref, &solimp, pos[i], margin[i], diag_a[i], 0.0, 0.0).1);
    }
    for &(first, n, pyr, mu) in &groups {
        if pyr {
            // R[1] = R[0]/impratio; mu of the regularised cone = mu·sqrt(R[1]/R[0]); Rpy = 2·mu²·R[0] for all
            let r1 = r[first] / impratio.max(MINVAL);
            let mu_reg = mu * (r1 / r[first]).sqrt();
            let rpy = 2.0 * mu_reg * mu_reg * r[first];
            for i in first..first + n {
                r[i] = rpy;
            }
        }
    }
    let mut aref = Vec::with_capacity(nefc);
    let mut jac = DMatrix::zeros(nefc, nv);
    for i in 0..nefc {
        jac.row_mut(i).copy_from(&rows[i].transpose());
        let vel = rows[i].dot(qvel);
        aref.push(-kb[i][1] * vel - kb[i][0] * kb[i][2] * (pos[i] - margin[i]));
    }
    // --- dual problem: (A + R) f = a_ref − J a₀ with f ≥ 0, by projected Gauss–Seidel
    let minv = m.clone().cholesky().ok_or("mass matrix is not positive definite")?;
    let minv_jt = minv.solve(&jac.transpose()); // nv × nefc
    let a = &jac * &minv_jt; // nefc × nefc
    let rhs: Vec<f64> = (0..nefc).map(|i| aref[i] - jac.row(i).dot(&a0.transpose())).collect();
    let mut f = vec![0.0; nefc];
    let mut residual = f64::INFINITY;
    for _ in 0..max_iter {
        residual = 0.0f64;
        for i in 0..nefc {
            let mut s = rhs[i];
            for j in 0..nefc {
                if j != i {
                    s -= a[(i, j)] * f[j];
                }
            }
            let fi = (s / (a[(i, i)] + r[i])).max(0.0);
            residual = residual.max((fi - f[i]).abs());
            f[i] = fi;
        }
        if residual < tol {
            break;
        }
    }
    let force = DVector::from_vec(f.clone());
    let qacc = a0 + &minv_jt * &force;
    Ok(MjContactSolve { jac, r, aref, kbip: kb, force: f, qacc, residual })
}

#[cfg(test)]
mod cone_tests {
    //! Every number is MuJoCo 3.13.0's, from `scripts/mujoco_cone_probe.py`: `efc_R`, `efc_D`, `mjContact.mu`,
    //! and — at `jar = J·qacc − a_ref` for the `qacc` MuJoCo returned — its own `mj_constraintUpdate`'s
    //! `efc_force`, `efc_state` and cost, for a ball and a box on a plane under both cones at condim 1/3/4/6.
    //!
    //! ⛔ The `efc_state` left in `mjData` by `mj_forward` is STALE: it is the Newton solver's last internal
    //! evaluation, not the state at the `qacc` it returned, and it disagrees on 15 of the probe's 76 rows.
    //! A sliding elliptic contact is left reading `cone` while its own `efc_force` is exactly the quadratic
    //! branch's. These tests pin the re-evaluated state; pinning the leftover would have had this port
    //! "corrected" into disagreeing with MuJoCo's own function.
    use super::*;
    use crate::mujoco_cone_cases::CASES;
    use nalgebra::{DMatrix, DVector};

    fn mat(rows: &[&[f64]]) -> DMatrix<f64> {
        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        DMatrix::from_row_slice(rows.len(), rows[0].len(), &flat)
    }

    /// **Every case the cone probe recorded, solved from `qacc_smooth` and compared with MuJoCo's `qacc`.**
    /// The objective is strictly convex, so its minimiser is unique and this compares the answer rather than
    /// the path — which is the only fair comparison against a different solver (MuJoCo runs Newton with an
    /// exact line search on the cone cost; this runs Newton with a backtracking one).
    #[test]
    fn the_primal_solve_reaches_mujocos_qacc_on_every_probed_case() {
        let mut worst: f64 = 0.0;
        let mut worst_label = "";
        for case in CASES {
            let cone = if case.elliptic { Cone::Elliptic } else { Cone::Pyramidal };
            let contacts: Vec<ConeContact> = case
                .condim
                .iter()
                .zip(case.mu)
                .zip(case.friction)
                .map(|((&condim, &mu), &friction)| ConeContact { cone, condim, mu, friction })
                .collect();
            let rows: usize = contacts.iter().map(|c| mujoco_cone_rows(c.cone, c.condim)).sum();
            assert_eq!(rows, case.jac.len(), "{}: row count", case.label);
            let sol = solve_constraints_newton(
                &mat(case.m),
                &DVector::from_row_slice(case.qacc_smooth),
                &mat(case.jac),
                case.aref,
                case.d,
                &contacts,
                1e-10,
                200,
            )
            .expect("well-formed");
            let want = DVector::from_row_slice(case.qacc);
            let scale = want.norm().max(1.0);
            let err = (&sol.qacc - &want).norm() / scale;
            if err > worst {
                worst = err;
                worst_label = case.label;
            }
            if err >= 1e-6 {
                // A disagreement is only ours if our answer is WORSE. The objective is strictly convex, so
                // the lower cost is the more converged point, whichever solver found it.
                let cost_at = |a: &DVector<f64>| {
                    let jar: Vec<f64> = (mat(case.jac) * a - DVector::from_row_slice(case.aref)).iter().copied().collect();
                    let da = a - DVector::from_row_slice(case.qacc_smooth);
                    0.5 * (da.transpose() * mat(case.m) * &da)[(0, 0)] + mujoco_constraint_update(&contacts, case.d, &jar).cost
                };
                let (ours, theirs) = (cost_at(&sol.qacc), cost_at(&want));
                assert!(
                    ours <= theirs,
                    "{}: qacc off by {err:.2e} relative AND our cost {ours} is worse than MuJoCo's {theirs}\n  ours   {:?}\n  MuJoCo {:?}",
                    case.label,
                    sol.qacc.as_slice(),
                    want.as_slice()
                );
                eprintln!("  {}: {err:.2e} from MuJoCo's qacc, at a LOWER cost ({ours} vs {theirs}) — MuJoCo stopped first", case.label);
            }
        }
        assert!(CASES.len() >= 13, "the fixture should cover every probed case, found {}", CASES.len());
        eprintln!("worst relative qacc error {worst:.2e} ({worst_label}) over {} cases", CASES.len());
    }

    /// The cone Hessian is the second derivative of the cost, so a central difference of the gradient must
    /// reproduce it — the check that catches a transcription error in a formula no oracle exposes directly.
    #[test]
    fn the_cone_hessian_is_the_costs_second_derivative() {
        let c = ConeContact { cone: Cone::Elliptic, condim: 6, mu: 0.8, friction: [0.8, 0.8, 0.01, 0.002, 0.002] };
        let d = [79.58701389094136, 79.58701389094136, 79.58701389094136, 0.012435470920459588, 0.0004974188368183835, 0.0004974188368183835];
        let jar = [-0.3972220425962476, 0.7496758282083924, -1.3419316951191433, 69.46476424999835, 289.898727631373, 136.62338661644037];
        assert_eq!(mujoco_constraint_update(&[c], &d, &jar).state[0], ConstraintState::Cone, "the fixture must be on the cone");
        let h = mujoco_constraint_hessian(&[c], &d, &jar);
        // gradient of the cost in jar is −force
        let grad = |x: &[f64]| -> Vec<f64> { mujoco_constraint_update(&[c], &d, x).force.iter().map(|f| -f).collect() };
        for k in 0..6 {
            let step = 1e-6 * jar[k].abs().max(1.0);
            let (mut lo, mut hi) = (jar, jar);
            lo[k] -= step;
            hi[k] += step;
            let (gl, gh) = (grad(&lo), grad(&hi));
            for j in 0..6 {
                let fd = (gh[j] - gl[j]) / (2.0 * step);
                let want = h[(j, k)];
                assert!((fd - want).abs() < 1e-4 * want.abs().max(1e-3), "H[{j},{k}] = {want} but the finite difference says {fd}");
            }
        }
    }


    fn check(u: &ConstraintUpdate, force: &[f64], state: ConstraintState, cost: f64) {
        assert_eq!(u.state, vec![state; force.len()], "state");
        for (got, want) in u.force.iter().zip(force) {
            assert!((got - want).abs() < 1e-8 * want.abs().max(1.0), "force {got} vs MuJoCo {want}");
        }
        assert!((u.cost - cost).abs() < 1e-8 * cost.abs().max(1.0), "cost {} vs MuJoCo {cost}", u.cost);
    }

    #[test]
    fn the_cones_regularisation_and_mu_are_mujocos() {
        // elliptic, condim 6, anisotropic friction (0.8 slide, 0.01 spin, 0.002 roll), impratio 1
        let base = 0.012564863928307538;
        let fri = [0.8, 0.8, 0.01, 0.002, 0.002];
        let mut r = [base; 6];
        let mu = mujoco_cone_adjust(&mut r, Cone::Elliptic, 6, &fri, 1.0);
        assert_eq!(mu, 0.8);
        let want = [0.012564863928307538, 0.012564863928307538, 0.012564863928307536, 80.41512914116824, 2010.3782285292061, 2010.3782285292061];
        for (got, w) in r.iter().zip(want) {
            assert_eq!(*got, w, "R {got} vs MuJoCo {w}");
        }

        // elliptic, condim 3, impratio 3: the friction rows soften and mu follows
        let mut r = [base; 3];
        let mu = mujoco_cone_adjust(&mut r, Cone::Elliptic, 3, &[1.0, 1.0, 0.005, 0.0001, 0.0001], 3.0);
        assert_eq!(mu, 0.5773502691896258);
        assert_eq!(r, [0.012564863928307538, 0.004188287976102513, 0.004188287976102513]);

        // pyramidal, same model: every row takes one common R = 2μ²R₀, on a base that already carries the
        // pyramid's own diagApprox (here 2× the elliptic one, since friction[0] = 1)
        let mut r = [2.0 * base; 4];
        let mu = mujoco_cone_adjust(&mut r, Cone::Pyramidal, 3, &[1.0, 1.0, 0.005, 0.0001, 0.0001], 3.0);
        assert_eq!(mu, 0.5773502691896258);
        assert_eq!(r, [0.016753151904410055; 4]);
    }

    #[test]
    fn an_elliptic_contact_with_no_tangential_drive_is_quadratic() {
        // a ball resting on the floor: T = 0 and N < 0, the bottom zone, where the elliptic cone reduces to
        // the same quadratic a frictionless row would give
        let c = ConeContact { cone: Cone::Elliptic, condim: 3, mu: 1.0, friction: [1.0, 1.0, 0.005, 0.0001, 0.0001] };
        let jar = [-1.148394736842107, 0.0, 0.0];
        let u = mujoco_constraint_update(&[c], &[79.58701389094136; 3], &jar);
        check(&u, &[91.39730787333673, 0.0, 0.0], ConstraintState::Quadratic, 52.48009366163878);
    }

    #[test]
    fn a_sliding_elliptic_contact_stays_in_the_quadratic_zone_when_the_cone_is_not_reached() {
        // sliding at 0.4, 0.2 m/s: μN + T is still negative, so MuJoCo is in the bottom zone — and its own
        // efc_force is −D·jar, while the efc_state field left behind says `cone`
        let c = ConeContact { cone: Cone::Elliptic, condim: 3, mu: 1.0, friction: [1.0, 1.0, 0.005, 0.0001, 0.0001] };
        let jar = [-1.148394736842107, 0.3231176817316097, -0.6462353634632123];
        let u = mujoco_constraint_update(&[c], &[79.58701389094136; 3], &jar);
        check(&u, &[91.39730787333673, -25.715971424382392, 51.43194284876422], ConstraintState::Quadratic, 73.25330633694531);
    }

    #[test]
    fn condim_six_with_anisotropic_friction_lands_on_mujocos_ellipse() {
        // the case a pyramidal cone cannot express: five friction rows with three different coefficients, so
        // the cone's cross-section is an ellipsoid and every row's force is radial within it
        let c = ConeContact { cone: Cone::Elliptic, condim: 6, mu: 0.8, friction: [0.8, 0.8, 0.01, 0.002, 0.002] };
        let d = [79.58701389094136, 79.58701389094136, 79.58701389094136, 0.012435470920459588, 0.0004974188368183835, 0.0004974188368183835];
        let jar = [-0.3972220425962476, 0.7496758282083924, -1.3419316951191433, 69.46476424999835, 289.898727631373, 136.62338661644037];
        let u = mujoco_constraint_update(&[c], &d, &jar);
        check(
            &u,
            &[94.54386359548204, -29.24686952427194, 52.35236314265267, -0.4234386259260889, -0.0706858046443038, -0.03331278510643513],
            ConstraintState::Cone,
            92.09548391605598,
        );
    }

    #[test]
    fn a_pyramidal_contact_clamps_each_row_at_zero() {
        // the same ball, pyramidal, impratio 3: four rows that each stand alone, all pressed into contact
        let c = ConeContact { cone: Cone::Pyramidal, condim: 3, mu: 0.5773502691896258, friction: [1.0, 1.0, 0.005, 0.0001, 0.0001] };
        let u = mujoco_constraint_update(&[c], &[59.690260418206; 4], &[-0.39599818511796414; 4]);
        check(&u, &[23.637234794828228; 4], ConstraintState::Quadratic, 18.72060415991834);
    }

    #[test]
    fn a_contact_the_solver_has_satisfied_exerts_nothing() {
        // jar ≥ 0 on a pyramidal row, and N ≥ μT on an elliptic contact, are both "no force"
        let c = ConeContact { cone: Cone::Pyramidal, condim: 3, mu: 0.5, friction: [1.0, 1.0, 0.005, 0.0001, 0.0001] };
        let u = mujoco_constraint_update(&[c], &[10.0; 4], &[1.0, 2.0, 0.0, -1.0]);
        assert_eq!(u.state, vec![ConstraintState::Satisfied, ConstraintState::Satisfied, ConstraintState::Satisfied, ConstraintState::Quadratic]);
        assert_eq!(&u.force[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(u.force[3], 10.0);
        assert_eq!(u.cost, 5.0);

        let c = ConeContact { cone: Cone::Elliptic, condim: 3, mu: 1.0, friction: [1.0, 1.0, 0.005, 0.0001, 0.0001] };
        let u = mujoco_constraint_update(&[c], &[10.0; 3], &[5.0, 1.0, 1.0]);
        assert_eq!(u.state, vec![ConstraintState::Satisfied; 3]);
        assert_eq!(u.force, vec![0.0; 3]);
        assert_eq!(u.cost, 0.0);
    }

    #[test]
    fn the_row_count_follows_the_cone_and_the_condim() {
        assert_eq!(mujoco_cone_rows(Cone::Pyramidal, 1), 1);
        assert_eq!(mujoco_cone_rows(Cone::Elliptic, 1), 1);
        assert_eq!(mujoco_cone_rows(Cone::Pyramidal, 3), 4);
        assert_eq!(mujoco_cone_rows(Cone::Elliptic, 3), 3);
        assert_eq!(mujoco_cone_rows(Cone::Pyramidal, 4), 6);
        assert_eq!(mujoco_cone_rows(Cone::Elliptic, 4), 4);
        assert_eq!(mujoco_cone_rows(Cone::Pyramidal, 6), 10);
        assert_eq!(mujoco_cone_rows(Cone::Elliptic, 6), 6);
    }
}

#[cfg(test)]
mod block_tests {
    use super::*;

    /// ⛔⛔ **`solimp` is clipped where it is USED, not where it is stored** — and the clip reaches the
    /// STIFFNESS, not only the sigmoid. A model may write `0.99999`; MuJoCo reports it back unchanged and
    /// then works with `0.9999`, so `1 − d` is `1e-4` and not `1e-5`. Reading the stored value makes every
    /// regularisation on that row out by exactly TEN, and clipping only inside the sigmoid still leaves
    /// `aref` out by 1.8e-4 while `efc_D` looks perfect.
    #[test]
    fn an_impedance_past_the_limit_is_clipped_in_the_stiffness_as_well_as_the_sigmoid() {
        let past = SolImp { d0: 0.9999, d_width: 0.99999, width: 0.001, midpoint: 0.5, power: 2.0 };
        let clipped = SolImp { d_width: 0.9999, ..past };
        let r = SolRef(0.004, 1.0);
        // far outside the width: the impedance saturates, and at the clipped ceiling
        let (imp, _) = mujoco_impedance(&past, 0.03, 0.0);
        assert_eq!(imp, 0.9999, "the stored 0.99999 must not reach the answer");
        // K and B are built from d_width, so they must see the clipped value too
        let (a, b) = (mujoco_kbip(&r, &past, 0.03, 0.0), mujoco_kbip(&r, &clipped, 0.03, 0.0));
        assert_eq!(a, b, "the stiffness was computed from the unclipped impedance");
        // ⛔ and the difference is not decorative: 1e-5 against 1e-4 is a factor of ten in R
        let unclipped_r = (1.0 - 0.99999) * 25.7376 / 0.99999;
        let (_, r_real) = row_reference(&r, &past, 0.03, 0.0, 25.7376, 0.0, 0.002);
        // ten to within the difference between the two denominators, which is itself only 1e-4
        assert!((r_real / unclipped_r - 10.0).abs() < 1e-2, "R is {r_real}, ten times {unclipped_r} was expected");
    }

    /// ⛔ In the standard `(timeconst, dampratio)` format MuJoCo floors the time constant at TWO TIMESTEPS —
    /// a reference stiffer than the integrator can follow is unstable — and nothing in the file says so.
    #[test]
    fn a_reference_time_constant_is_floored_at_two_timesteps() {
        let s = SolImp::default();
        let stiff = SolRef(0.001, 1.0);
        let (a_fast, _) = row_reference(&stiff, &s, -0.01, 0.0, 1.0, 0.0, 0.002);
        let (a_floor, _) = row_reference(&SolRef(0.004, 1.0), &s, -0.01, 0.0, 1.0, 0.0, 0.002);
        assert!((a_fast - a_floor).abs() < 1e-12, "0.001 s at a 0.002 s step must become 0.004 s");
        // with a small enough timestep the request stands
        let (a_kept, _) = row_reference(&stiff, &s, -0.01, 0.0, 1.0, 0.0, 0.0001);
        assert!((a_kept - a_fast).abs() > 1e-6, "the floor was applied when it should not have been");
        // ⛔ and the DIRECT format (both negative: stiffness and damping) is never floored
        let direct = SolRef(-1000.0, -50.0);
        let (d1, _) = row_reference(&direct, &s, -0.01, 0.0, 1.0, 0.0, 0.002);
        let (d2, _) = row_reference(&direct, &s, -0.01, 0.0, 1.0, 0.0, 10.0);
        assert_eq!(d1, d2);
    }

    /// A friction row is a quadratic well with a flat floor on each side: inside `|jar| < R·loss` it behaves
    /// like any other row, outside it SATURATES at ±loss and the cost becomes linear. Both branches are
    /// checked against their closed forms, and the corner is checked from both sides so the two pieces are
    /// shown to meet.
    #[test]
    fn a_friction_row_saturates_at_plus_and_minus_its_loss() {
        let (d, loss) = (4.0, 3.0);
        let r = 1.0 / d;
        let block = [EfcBlock::Friction { loss }];
        // inside the band: force = −D·jar, cost = ½D·jar²
        let jar = [0.5 * r * loss];
        let u = mujoco_constraint_update_blocks(&block, &[d], &jar);
        assert_eq!(u.state[0], ConstraintState::Quadratic);
        assert!((u.force[0] + d * jar[0]).abs() < 1e-15);
        assert!((u.cost - 0.5 * d * jar[0] * jar[0]).abs() < 1e-15);
        // past the positive bound: the force stops at −loss whatever `jar` does
        for far in [1.0001, 2.0, 100.0] {
            let jar = [far * r * loss];
            let u = mujoco_constraint_update_blocks(&block, &[d], &jar);
            assert_eq!(u.state[0], ConstraintState::LinearPos);
            assert!((u.force[0] + loss).abs() < 1e-15, "force {} at {far}", u.force[0]);
            assert!((u.cost - (-0.5 * r * loss * loss + loss * jar[0])).abs() < 1e-13);
        }
        let jar = [-2.0 * r * loss];
        let u = mujoco_constraint_update_blocks(&block, &[d], &jar);
        assert_eq!(u.state[0], ConstraintState::LinearNeg);
        assert!((u.force[0] - loss).abs() < 1e-15);
        // ⛔ the two pieces meet: at the corner the quadratic and the line agree in value AND slope, which is
        // what makes the cost differentiable and the Newton solve well posed
        let eps = 1e-9;
        let c = |x: f64| mujoco_constraint_update_blocks(&block, &[d], &[x]).cost;
        let (lo, hi) = (r * loss - eps, r * loss + eps);
        assert!((c(hi) - c(lo)).abs() < 1e-8, "value jumps at the corner");
        let (dlo, dhi) = ((c(lo) - c(lo - eps)) / eps, (c(hi + eps) - c(hi)) / eps);
        assert!((dhi - dlo).abs() < 1e-5, "slope jumps at the corner: {dlo} then {dhi}");
        // and a saturated row has NO curvature, so it contributes nothing to the Hessian
        assert_eq!(mujoco_constraint_hessian_blocks(&block, &[d], &[2.0 * r * loss])[(0, 0)], 0.0);
        assert!((mujoco_constraint_hessian_blocks(&block, &[d], &[0.0])[(0, 0)] - d).abs() < 1e-15);
    }

    /// An equality row is quadratic everywhere and unbounded in both directions — it pushes as hard as it
    /// must, which is exactly what a contact row may not do.
    #[test]
    fn an_equality_row_is_never_satisfied_and_never_saturates() {
        let d = [7.0, 7.0];
        let block = [EfcBlock::Equality { rows: 2 }];
        for jar in [[-5.0, 3.0], [1e6, -1e6], [0.0, 0.0]] {
            let u = mujoco_constraint_update_blocks(&block, &d, &jar);
            assert!(u.state.iter().all(|s| *s == ConstraintState::Quadratic));
            for k in 0..2 {
                assert!((u.force[k] + d[k] * jar[k]).abs() < 1e-9 * d[k] * jar[k].abs().max(1.0));
                assert!((mujoco_constraint_hessian_blocks(&block, &d, &jar)[(k, k)] - d[k]).abs() < 1e-15);
            }
        }
        // ⛔ the same `jar` on a LIMIT row is one-sided: positive means satisfied, and it exerts nothing
        let limit = [EfcBlock::Limit, EfcBlock::Limit];
        let u = mujoco_constraint_update_blocks(&limit, &d, &[-5.0, 3.0]);
        assert_eq!(u.state, vec![ConstraintState::Quadratic, ConstraintState::Satisfied]);
        assert_eq!(u.force[1], 0.0);
    }

    /// ⛔ A row's law comes from its POSITION, not a tag: the same numbers read as an equality, a friction
    /// row or a limit depending only on where they sit. Getting `ne`/`nf` wrong is therefore silent.
    #[test]
    fn the_same_row_means_three_different_things_at_three_positions() {
        let (d, jar) = ([2.0], [4.0]);
        let e = mujoco_constraint_update_blocks(&[EfcBlock::Equality { rows: 1 }], &d, &jar);
        let f = mujoco_constraint_update_blocks(&[EfcBlock::Friction { loss: 0.5 }], &d, &jar);
        let l = mujoco_constraint_update_blocks(&[EfcBlock::Limit], &d, &jar);
        assert_eq!((e.force[0], f.force[0], l.force[0]), (-8.0, -0.5, 0.0));
        assert!(e.cost > 0.0 && f.cost > 0.0 && l.cost == 0.0);
    }
}

#[cfg(test)]
mod tests {
    //! Every expected number is MuJoCo 3.13.0's (`mj_forward` on a unit-mass, 0.1 m sphere on a plane, Euler,
    //! warmstart off, Newton solver at tolerance 1e-15) — `efc_R`, `efc_aref`, `efc_force`, `qacc`.
    use super::*;

    const R_BALL: f64 = 0.1;
    const M_BALL: f64 = 1.0;
    const I_BALL: f64 = 0.4 * M_BALL * R_BALL * R_BALL; // solid sphere

    /// The free-body system MuJoCo's free joint sees at identity orientation: qvel = [v_world; ω_body].
    fn ball() -> (DMatrix<f64>, DVector<f64>) {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![M_BALL, M_BALL, M_BALL, I_BALL, I_BALL, I_BALL]));
        let a0 = DVector::from_vec(vec![0.0, 0.0, -9.81, 0.0, 0.0, 0.0]);
        (m, a0)
    }

    /// Contact of the ball (geom 2) with the world plane (geom 1): normal +z, tangents +y and −x as MuJoCo's
    /// frame `[0 0 1, 0 1 0, -1 0 0]` gives them. **MuJoCo puts the contact point halfway between the two
    /// surfaces** (`mjContact.pos`), so the lever arm is `R + dist/2`, not `R` — with `R` the sliding facet
    /// force came out 13.2086 against MuJoCo's 13.2429, a 0.26% that was entirely this half-penetration.
    fn contact(z: f64, condim: usize, solref: SolRef, solimp: SolImp) -> MjContact {
        let dist = z - R_BALL;
        let arm = R_BALL + dist / 2.0;
        let mut jac = DMatrix::zeros(3, 6);
        // velocity of the contact point on the ball: v + ω × r with r = (0, 0, −arm)
        // normal (z): v_z
        jac[(0, 2)] = 1.0;
        // t1 = +y: v_y + (ω × r)_y, (ω × r) = (ω_y·r_z − ω_z·r_y, ω_z·r_x − ω_x·r_z, ω_x·r_y − ω_y·r_x)
        jac[(1, 1)] = 1.0;
        jac[(1, 3)] = arm; // ω_z·0 − ω_x·(−arm)
        // t2 = −x: −(v_x + ω_y·r_z) = −v_x + arm·ω_y
        jac[(2, 0)] = -1.0;
        jac[(2, 4)] = arm;
        MjContact {
            jac,
            dist,
            margin: 0.0,
            condim,
            friction: [1.0, 1.0],
            solref,
            solimp,
            invweight: [InvWeight::STATIC, InvWeight::free_body(M_BALL, [I_BALL; 3])],
        }
    }

    fn solve(z: f64, vz: f64, vx: f64, condim: usize, solref: SolRef, solimp: SolImp) -> MjContactSolve {
        let (m, a0) = ball();
        let qvel = DVector::from_vec(vec![vx, 0.0, vz, 0.0, 0.0, 0.0]);
        solve_contacts_mujoco(&m, &a0, &qvel, &[contact(z, condim, solref, solimp)], true, 1.0, 1e-16, 10_000).unwrap()
    }

    fn close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{what}: {a} vs MuJoCo {b}");
    }

    #[test]
    fn impedance_and_kbip_are_mujocos() {
        let (d, _) = mujoco_impedance(&SolImp::default(), -1e-4, 0.0);
        close(d, 0.901, 1e-12, "impedance at 0.1 width");
        let (d, _) = mujoco_impedance(&SolImp::default(), -5e-4, 0.0);
        close(d, 0.925, 1e-12, "impedance at the midpoint");
        let (d, _) = mujoco_impedance(&SolImp::default(), -5e-3, 0.0);
        close(d, 0.95, 1e-12, "impedance saturated");
        let k = mujoco_kbip(&SolRef::default(), &SolImp::default(), -5e-4, 0.0);
        close(k[0], 2770.083102493075, 1e-12, "K");
        close(k[1], 105.26315789473685, 1e-12, "B");
        close(k[3], -99.99999999999979, 1e-9, "impedance derivative");
        let custom = mujoco_kbip(&SolRef(0.05, 0.5), &SolImp { d0: 0.8, d_width: 0.99, width: 0.01, midpoint: 0.3, power: 3.0 }, -5e-4, 0.0);
        close(custom[0], 1632.4864809713292, 1e-12, "custom K");
        close(custom[1], 40.4040404040404, 1e-12, "custom B");
        close(custom[2], 0.8002638888888889, 1e-12, "custom impedance");
        let direct = mujoco_kbip(&SolRef(-1000.0, -50.0), &SolImp::default(), -5e-4, 0.0);
        close(direct[0], 1108.03324099723, 1e-12, "direct K");
        close(direct[1], 52.631578947368425, 1e-12, "direct B");
    }

    #[test]
    fn frictionless_sphere_on_plane_matches_mujoco_across_a_penetration_ladder() {
        // (z, R, aref, force, qacc_z)
        let cases = [
            (0.0999, 1.098779134295e-01, 2.495844875346e-01, 9.063685623269e+00, -0.746314376731295),
            (0.0995, 8.108108108108e-02, 1.281163434903e+00, 1.025932617729e+01, 0.44932617728531987),
            (0.099, 5.263157894737e-02, 2.631578947368e+00, 1.181950000000e+01, 2.009500000000002),
            (0.095, 5.263157894737e-02, 1.315789473684e+01, 2.181950000000e+01, 12.009500000000013),
        ];
        for (z, r, aref, force, qacc) in cases {
            let s = solve(z, 0.0, 0.0, 1, SolRef::default(), SolImp::default());
            close(s.r[0], r, 1e-11, "R");
            close(s.aref[0], aref, 1e-11, "aref");
            close(s.force[0], force, 1e-11, "force");
            close(s.qacc[2], qacc, 1e-11, "qacc_z");
        }
        // MuJoCo activates a contact only when dist < margin, so a ball exactly touching (dist = 0) has no
        // constraint row at all and falls at g; passed here it would push (rhs = 9.81 > 0). The activation
        // test is the caller's, upstream of this solver, as it is in MuJoCo.
    }

    #[test]
    fn closing_and_opening_velocities_enter_through_the_reference() {
        let s = solve(0.0995, -0.3, 0.0, 1, SolRef::default(), SolImp::default());
        close(s.aref[0], 3.286011080332e+01, 1e-11, "closing aref");
        close(s.force[0], 3.946985249307e+01, 1e-11, "closing force");
        close(s.qacc[2], 29.6598524930748, 1e-11, "closing qacc");
        let s = solve(0.0995, 0.3, 0.0, 1, SolRef::default(), SolImp::default());
        close(s.aref[0], -3.029778393352e+01, 1e-11, "opening aref");
        assert_eq!(s.force[0], 0.0, "an opening contact carries no force");
        close(s.qacc[2], -9.81, 1e-12, "opening qacc");
    }

    #[test]
    fn custom_and_direct_solref_match_mujoco() {
        let s = solve(0.0995, -0.1, 0.0, 1, SolRef(0.05, 0.5), SolImp { d0: 0.8, d_width: 0.99, width: 0.01, midpoint: 0.3, power: 3.0 });
        close(s.r[0], 2.495878095767e-01, 1e-11, "custom R");
        close(s.aref[0], 4.693614030314e+00, 1e-11, "custom aref");
        close(s.force[0], 1.160671856684e+01, 1e-11, "custom force");
        close(s.qacc[2], 1.7967185668428245, 1e-11, "custom qacc");
        let s = solve(0.0995, -0.1, 0.0, 1, SolRef(-1000.0, -50.0), SolImp::default());
        close(s.aref[0], 5.775623268698e+00, 1e-11, "direct aref");
        close(s.force[0], 1.441670152355e+01, 1e-11, "direct force");
        close(s.qacc[2], 4.606701523545706, 1e-11, "direct qacc");
    }

    #[test]
    fn pyramidal_cone_at_rest_and_sliding_matches_mujoco() {
        // at rest: four equal facet forces whose normal components sum to the frictionless answer
        let s = solve(0.0995, 0.0, 0.0, 3, SolRef::default(), SolImp::default());
        assert_eq!(s.r.len(), 4);
        for i in 0..4 {
            close(s.r[i], 3.243243243243e-01, 1e-11, "pyramidal R");
            close(s.aref[i], 1.281163434903e+00, 1e-11, "pyramidal aref");
            close(s.force[i], 2.564831544321e+00, 1e-10, "pyramidal facet force at rest");
        }
        close(s.qacc[2], 0.44932617728532215, 1e-10, "resting qacc_z");
        assert!(s.qacc[0].abs() < 1e-12 && s.qacc[4].abs() < 1e-10);
        // sliding at +0.5 m/s in x: only the facet opposing the slide carries force; friction = μ f
        let s = solve(0.0995, 0.0, 0.5, 3, SolRef::default(), SolImp::default());
        close(s.force[2], 1.324290563555e+01, 1e-10, "active facet force");
        assert_eq!(s.force[0], 0.0);
        assert_eq!(s.force[1], 0.0);
        assert_eq!(s.force[3], 0.0);
        close(s.qacc[0], -13.24290563554995, 1e-10, "sliding qacc_x");
        close(s.qacc[2], 3.4329056355499494, 1e-10, "sliding qacc_z");
        close(s.qacc[4], 330.2449592865268, 1e-9, "sliding spin-up ω_y");
    }

    /// **An ELLIPTIC contact's friction rows have no position reference.** Under the pyramidal cone every
    /// row is `J_n ± μ_k·J_tk` and carries the contact's depth, so every row's `aref` has the spring term
    /// `−K·d·(dist − margin)`. Under the elliptic cone the friction rows are the tangents alone and MuJoCo
    /// gives them `efc_pos = 0`: at rest their `aref` is exactly zero while the normal row's is not.
    ///
    /// The mutation this refuses: passing the contact's `dist` to every row. It is invisible in `efc_D`
    /// (`mujoco_cone_adjust` overwrites every friction row's `R` from the normal's) and invisible in any
    /// test that only ever looks at the normal row.
    #[test]
    fn an_elliptic_contacts_friction_rows_carry_no_depth() {
        let nv = 3;
        let jac = DMatrix::identity(3, nv);
        let spec = |condim: usize| ContactSpec {
            jac: jac.clone(),
            dist: -0.01,
            margin: 0.0,
            condim,
            friction: [1.0, 1.0, 0.005, 1e-4, 1e-4],
            solref: SolRef::default(),
            solimp: SolImp::default(),
            invweight: [InvWeight::STATIC, InvWeight { tran: 1.0, rot: 1.0 }],
        };
        let at_rest = vec![0.0; nv];
        let ell = contact_rows(&[spec(3)], nv, &at_rest, Cone::Elliptic, 1.0, 0.002).unwrap();
        assert_eq!(ell.aref.len(), 3);
        assert!(ell.aref[0].abs() > 1.0, "the normal row must still be pushed out: {}", ell.aref[0]);
        assert_eq!(ell.aref[1], 0.0, "an elliptic friction row at rest has no reference at all");
        assert_eq!(ell.aref[2], 0.0);
        // the contrast that makes the rule a rule: every pyramidal row carries the same depth
        let pyr = contact_rows(&[spec(3)], nv, &at_rest, Cone::Pyramidal, 1.0, 0.002).unwrap();
        assert_eq!(pyr.aref.len(), 4);
        for (i, a) in pyr.aref.iter().enumerate() {
            assert!((a - pyr.aref[0]).abs() < 1e-12, "pyramidal row {i}: {a} vs {}", pyr.aref[0]);
            assert!(a.abs() > 1.0, "pyramidal row {i} lost the depth: {a}");
        }
        // and moving, the elliptic friction row is pure damping: −B·J·q̇, with B from solref and solimp
        let moving = vec![0.0, 0.5, 0.0];
        let m = contact_rows(&[spec(3)], nv, &moving, Cone::Elliptic, 1.0, 0.002).unwrap();
        let b = 2.0 / (SolImp::default().d_width * SolRef::default().0);
        close(m.aref[1], -b * 0.5, 1e-12, "elliptic friction aref is -B·J·q̇");
        assert_eq!(m.aref[2], 0.0, "the untravelled tangent stays at zero");
    }

    #[test]
    fn diag_approx_is_mujocos_two_side_sum() {
        let ball = InvWeight::free_body(1.0, [I_BALL; 3]);
        assert_eq!(mujoco_diag_approx(InvWeight::STATIC, ball, 1, &[1.0, 1.0], true), vec![1.0]);
        // pyramidal: tran + μ²·tran for the two sliding directions
        assert_eq!(mujoco_diag_approx(InvWeight::STATIC, ball, 3, &[1.0, 1.0], true), vec![2.0, 2.0, 2.0, 2.0]);
        // elliptic: tran for the first three rows
        assert_eq!(mujoco_diag_approx(InvWeight::STATIC, ball, 3, &[1.0, 1.0], false), vec![1.0, 1.0, 1.0]);
    }
}
