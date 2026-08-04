//! **The compass-gait biped** — the smallest robot on which a hybrid-zero-dynamics certificate is a real
//! statement, built here so that certificate can be verified rather than assumed.
//!
//! Two rigid legs hinged at a hip, point feet, one actuator between them. Four states, one input: the robot
//! is **underactuated by one**, which is the whole difficulty of legged locomotion in its smallest form. There
//! is no ankle torque, so the stance leg's fall cannot be arrested by feedback at all; what stabilises a gait
//! is the *impact*, and that is exactly the structure hybrid zero dynamics is built around.
//!
//! # Why it is derived here rather than transcribed
//!
//! The published models of this robot differ in coordinate convention, and a sign error in the Coriolis term
//! or the impact map produces a simulation that walks convincingly and certifies nothing. So the dynamics are
//! derived from the Lagrangian in this file's own coordinates, and the impact is built from
//! [`plastic_impact`](ferromotion_core::plastic_impact) on the *extended* model rather than from a
//! transcribed matrix pair. That leaves three physics invariants available as independent checks, and the
//! tests use all three:
//!
//! * with no torque the continuous phase **conserves energy**;
//! * the impact **never increases energy** and **conserves angular momentum about the impacting foot**;
//! * the post-impact state satisfies the *new* pin constraint exactly, so the reduction back to two degrees
//!   of freedom is exact rather than approximate.
//!
//! # Coordinates
//!
//! `θ₁` is the stance leg's angle from vertical and `θ₂` the swing leg's, both positive forward, with the
//! stance foot at the origin:
//!
//! ```text
//! hip        = l·(sin θ₁,  cos θ₁)
//! swing foot = hip − l·(sin θ₂, cos θ₂)
//! ```
//!
//! so the swing foot touches down when `cos θ₂ = cos θ₁` with the legs apart, i.e. on the guard
//! `θ₁ + θ₂ = 0` with `θ₁ > 0`. A step runs `θ₁: −α → +α`, and the impact swaps the legs.

use ferromotion_core::plastic_impact;
use nalgebra::{DMatrix, DVector, Matrix2, Vector2};

/// A compass-gait biped: two legs of length `l = a + b`, leg mass `m` at distance `a` from the foot, and a
/// point mass `m_h` at the hip.
#[derive(Clone, Copy, Debug)]
pub struct CompassGait {
    /// Leg mass.
    pub m: f64,
    /// Hip mass.
    pub m_h: f64,
    /// Distance from foot to leg centre of mass.
    pub a: f64,
    /// Distance from leg centre of mass to hip.
    pub b: f64,
    pub g: f64,
    /// **Downhill ground slope, in radians.** Leg angles are measured from the surface normal, so the slope
    /// enters only through gravity — the kinematics, the guard and the impact are identical in the slope frame.
    ///
    /// This is not a decoration. On level ground the compass gait loses energy at every impact and the hip
    /// torque of a symmetric virtual constraint cannot put it back, so there is no periodic gait to certify at
    /// all; the first attempt at this model stalled mid-step and the constraint then extrapolated outside
    /// `|θ₁| ≤ α` into nonsense. A shallow descent is the canonical setting precisely because gravity supplies
    /// the step's energy budget, which is why the passive walker works and why a certificate has an orbit to
    /// attach to.
    pub slope: f64,
}

/// Stance and swing angles with their rates: `[θ₁, θ₂, θ̇₁, θ̇₂]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaitState {
    pub th1: f64,
    pub th2: f64,
    pub d1: f64,
    pub d2: f64,
}

impl GaitState {
    pub fn new(th1: f64, th2: f64, d1: f64, d2: f64) -> Self {
        GaitState { th1, th2, d1, d2 }
    }
    pub fn to_vec(self) -> DVector<f64> {
        DVector::from_row_slice(&[self.th1, self.th2, self.d1, self.d2])
    }
    pub fn from_vec(v: &DVector<f64>) -> Self {
        GaitState::new(v[0], v[1], v[2], v[3])
    }
}

impl Default for CompassGait {
    /// The parameters this model is usually studied with: 1 m legs, 5 kg legs, a 10 kg hip, and a shallow
    /// downhill slope so that a periodic gait exists.
    fn default() -> Self {
        CompassGait { m: 5.0, m_h: 10.0, a: 0.5, b: 0.5, g: 9.81, slope: 0.05 }
    }
}

impl CompassGait {
    /// Leg length.
    pub fn l(&self) -> f64 {
        self.a + self.b
    }

    /// The pinned mass matrix. Derived from
    /// `T = ½[m a² + (m_h + m)l²]θ̇₁² + ½ m b² θ̇₂² − m l b θ̇₁θ̇₂ cos(θ₁−θ₂)`.
    pub fn mass_matrix(&self, th1: f64, th2: f64) -> Matrix2<f64> {
        let l = self.l();
        let off = -self.m * l * self.b * (th1 - th2).cos();
        Matrix2::new(self.m * self.a * self.a + (self.m_h + self.m) * l * l, off, off, self.m * self.b * self.b)
    }

    /// Coriolis and centrifugal terms, as the vector added to `M q̈`.
    pub fn coriolis(&self, th1: f64, th2: f64, d1: f64, d2: f64) -> Vector2<f64> {
        let k = self.m * self.l() * self.b * (th1 - th2).sin();
        Vector2::new(-k * d2 * d2, k * d1 * d1)
    }

    /// Gravity terms, `∂V/∂q` for `V = g[(m a + m_h l + m l)cos(θ₁+γ) − m b cos(θ₂+γ)]`. The slope `γ` shifts
    /// each leg's angle to the true vertical without touching the mass matrix.
    pub fn gravity(&self, th1: f64, th2: f64) -> Vector2<f64> {
        let l = self.l();
        Vector2::new(-self.g * (self.m * self.a + self.m_h * l + self.m * l) * (th1 + self.slope).sin(), self.g * self.m * self.b * (th2 + self.slope).sin())
    }

    /// The input map: a single hip torque acts equally and oppositely on the two legs.
    pub fn input_map(&self) -> Vector2<f64> {
        Vector2::new(-1.0, 1.0)
    }

    /// Kinetic energy alone. This, not the total, is what an inelastic collision must not increase — and on a
    /// slope it is the only comparison available across an impact, because the pinned origin moves to the new
    /// foot, which is *downhill*, so the potential's reference shifts discontinuously.
    pub fn kinetic(&self, s: &GaitState) -> f64 {
        let v = Vector2::new(s.d1, s.d2);
        0.5 * (v.transpose() * self.mass_matrix(s.th1, s.th2) * v)[0]
    }

    /// Total mechanical energy. Conserved by the continuous phase with no torque, which is the check that
    /// the mass matrix, Coriolis and gravity terms are mutually consistent.
    pub fn energy(&self, s: &GaitState) -> f64 {
        let l = self.l();
        let m = self.mass_matrix(s.th1, s.th2);
        let v = Vector2::new(s.d1, s.d2);
        let kinetic = 0.5 * (v.transpose() * m * v)[0];
        let potential = self.g * ((self.m * self.a + self.m_h * l + self.m * l) * (s.th1 + self.slope).cos() - self.m * self.b * (s.th2 + self.slope).cos());
        kinetic + potential
    }

    /// Joint accelerations under hip torque `tau`.
    pub fn accel(&self, s: &GaitState, tau: f64) -> Vector2<f64> {
        let m = self.mass_matrix(s.th1, s.th2);
        let rhs = self.input_map() * tau - self.coriolis(s.th1, s.th2, s.d1, s.d2) - self.gravity(s.th1, s.th2);
        m.try_inverse().map(|mi| mi * rhs).unwrap_or_else(Vector2::zeros)
    }

    /// One explicit fourth-order Runge-Kutta step of the continuous phase.
    pub fn flow_step(&self, s: &GaitState, tau: f64, dt: f64) -> GaitState {
        let f = |st: &GaitState| {
            let acc = self.accel(st, tau);
            [st.d1, st.d2, acc[0], acc[1]]
        };
        let add = |st: &GaitState, k: &[f64; 4], h: f64| GaitState::new(st.th1 + h * k[0], st.th2 + h * k[1], st.d1 + h * k[2], st.d2 + h * k[3]);
        let k1 = f(s);
        let k2 = f(&add(s, &k1, dt / 2.0));
        let k3 = f(&add(s, &k2, dt / 2.0));
        let k4 = f(&add(s, &k3, dt));
        let avg = [
            (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        ];
        add(s, &avg, dt)
    }

    /// The guard function, zero at heel strike: `θ₁ + θ₂`. Strike requires the legs be apart, so a crossing
    /// only counts when `θ₁` is positive.
    pub fn guard(&self, s: &GaitState) -> f64 {
        s.th1 + s.th2
    }

    /// Height of the swing foot above the ground, `l(cos θ₁ − cos θ₂)`. The guard is where this vanishes with
    /// the legs apart; it is also what distinguishes a real strike from the legs merely passing.
    pub fn swing_foot_height(&self, s: &GaitState) -> f64 {
        self.l() * (s.th1.cos() - s.th2.cos())
    }

    /// The **extended** mass matrix in `[θ₁, θ₂, hip_x, hip_y]`, used only for the impact. Unpinning the base
    /// is what makes the impact expressible as a plastic contact rather than a transcribed matrix pair.
    fn extended_mass(&self, th1: f64, th2: f64) -> DMatrix<f64> {
        let (m, mh, b) = (self.m, self.m_h, self.b);
        let mut me = DMatrix::zeros(4, 4);
        me[(0, 0)] = m * b * b;
        me[(1, 1)] = m * b * b;
        me[(2, 2)] = 2.0 * m + mh;
        me[(3, 3)] = 2.0 * m + mh;
        me[(0, 2)] = -m * b * th1.cos();
        me[(2, 0)] = me[(0, 2)];
        me[(0, 3)] = m * b * th1.sin();
        me[(3, 0)] = me[(0, 3)];
        me[(1, 2)] = -m * b * th2.cos();
        me[(2, 1)] = me[(1, 2)];
        me[(1, 3)] = m * b * th2.sin();
        me[(3, 1)] = me[(1, 3)];
        me
    }

    /// Jacobian of the foot on leg `which` (0 = stance, 1 = swing) in extended coordinates.
    fn foot_jacobian(&self, th: f64, which: usize) -> DMatrix<f64> {
        let l = self.l();
        let mut j = DMatrix::zeros(2, 4);
        j[(0, which)] = -l * th.cos();
        j[(1, which)] = l * th.sin();
        j[(0, 2)] = 1.0;
        j[(1, 3)] = 1.0;
        j
    }

    /// **The impact**, as a plastic contact at the landing foot followed by relabelling the legs.
    ///
    /// The pre-impact extended velocity is reconstructed from the pinned state (the hip rides on the stance
    /// leg), the landing foot's velocity is projected to zero in the mass metric by
    /// [`plastic_impact`](ferromotion_core::plastic_impact), and the legs then swap roles. The projection is
    /// `M`-orthogonal, so it removes exactly the kinetic energy the collision absorbs and no more — which is
    /// why the energy check below is a real test and not a tautology.
    pub fn impact(&self, s: &GaitState) -> GaitState {
        let l = self.l();
        // extended pre-impact velocity: the hip is carried by the stance leg
        let qd = DVector::from_row_slice(&[s.d1, s.d2, l * s.d1 * s.th1.cos(), -l * s.d1 * s.th1.sin()]);
        let me = self.extended_mass(s.th1, s.th2);
        let j = self.foot_jacobian(s.th2, 1); // the swing foot is the one landing
        let qd_plus = plastic_impact(&me, &qd, &j);
        // relabel: the landing leg becomes the stance leg
        GaitState::new(s.th2, s.th1, qd_plus[1], qd_plus[0])
    }

    /// Angular momentum of the whole robot about a world point, in the extended description. Conserved
    /// through the impact about the impacting foot, because the impulse acts there and exerts no moment
    /// about itself.
    pub fn angular_momentum_about(&self, s: &GaitState, qd: &DVector<f64>, pivot: (f64, f64)) -> f64 {
        let l = self.l();
        let hip = (l * s.th1.sin(), l * s.th1.cos());
        // bodies: stance leg CoM, swing leg CoM, hip mass — positions and velocities in extended coordinates
        let bodies = [
            (self.m, (hip.0 - self.b * s.th1.sin(), hip.1 - self.b * s.th1.cos()), (qd[2] - self.b * qd[0] * s.th1.cos(), qd[3] + self.b * qd[0] * s.th1.sin())),
            (self.m, (hip.0 - self.b * s.th2.sin(), hip.1 - self.b * s.th2.cos()), (qd[2] - self.b * qd[1] * s.th2.cos(), qd[3] + self.b * qd[1] * s.th2.sin())),
            (self.m_h, hip, (qd[2], qd[3])),
        ];
        bodies.iter().map(|(mass, p, v)| mass * ((p.0 - pivot.0) * v.1 - (p.1 - pivot.1) * v.0)).sum()
    }

    /// Reconstruct the extended velocity from a pinned state, for use with
    /// [`angular_momentum_about`](Self::angular_momentum_about).
    pub fn extended_velocity(&self, s: &GaitState) -> DVector<f64> {
        let l = self.l();
        DVector::from_row_slice(&[s.d1, s.d2, l * s.d1 * s.th1.cos(), -l * s.d1 * s.th1.sin()])
    }
}

/// **A relative-degree-two virtual constraint** for the compass gait: prescribe the swing leg as a function
/// of the stance leg, `θ₂ = h_d(θ₁)`, and drive the difference to zero.
///
/// The shape is chosen so the constraint is *compatible with the geometry of a step*. Over a step `θ₁` runs
/// from `−α` to `+α`, and the impact swaps the legs, so the swing leg must travel from `+α` to `−α` in the
/// same time. Any `h_d` with `h_d(±α) = ∓α` does that; the simplest such family is
///
/// ```text
/// h_d(θ₁) = −θ₁ + c·(α² − θ₁²)
/// ```
///
/// The mirror is *degenerate* and it is worth seeing why: `θ₂ = −θ₁` is exactly the guard, so the
/// zero-dynamics manifold would coincide with the impact surface and the robot would strike the ground for the
/// whole step. The shape terms are what lift the swing leg off that surface.
///
/// **Two parameters, and both are needed.** The full family used here is
///
/// ```text
/// h_d(θ₁) = −θ₁ + (α² − θ₁²)·(c + e·θ₁)
/// ```
///
/// which still satisfies `h_d(±α) = ∓α` for any `c, e`. One parameter is consumed buying **hybrid
/// invariance** — a single scalar equation, so a single unknown answers it. The other is what remains to set
/// the **energy balance**: the compass gait loses energy at every impact and level-ground walking only closes
/// if the constraint injects it back, so a one-parameter family produces an invariant manifold carrying no
/// periodic gait at all. That failure is the reason the second parameter exists rather than a convenience.
#[derive(Clone, Copy, Debug)]
pub struct VirtualConstraint {
    /// Half the inter-leg angle at strike; the step ends at `θ₁ = +α`.
    pub alpha: f64,
    /// Symmetric shape parameter, solved for to obtain hybrid invariance.
    pub c: f64,
    /// Antisymmetric shape parameter, which sets the energy the constraint injects over a step and therefore
    /// the gait speed.
    pub e: f64,
}

impl VirtualConstraint {
    /// Desired swing angle and its first two derivatives with respect to `θ₁`.
    pub fn desired(&self, th1: f64) -> (f64, f64, f64) {
        let (a2, t) = (self.alpha * self.alpha, th1);
        let shape = self.c + self.e * t;
        (-t + (a2 - t * t) * shape, -1.0 - 2.0 * t * shape + (a2 - t * t) * self.e, -2.0 * self.c - 6.0 * self.e * t)
    }

    /// The output `y = θ₂ − h_d(θ₁)` and its rate `ẏ = θ̇₂ − h_d'(θ₁)θ̇₁`. The pair vanishes exactly on the
    /// zero-dynamics manifold `Z`.
    pub fn output(&self, s: &GaitState) -> (f64, f64) {
        let (hd, hd1, _) = self.desired(s.th1);
        (s.th2 - hd, s.d2 - hd1 * s.d1)
    }

    /// The state on `Z` with stance angle `th1` and rate `d1`: the swing coordinates are determined, which is
    /// what makes the restricted dynamics two-dimensional and the section one-dimensional.
    pub fn on_manifold(&self, th1: f64, d1: f64) -> GaitState {
        let (hd, hd1, _) = self.desired(th1);
        GaitState::new(th1, hd, d1, hd1 * d1)
    }
}

/// The decomposition of the output's second derivative, `ÿ = L_f²h + L_gL_fh · τ`.
#[derive(Clone, Copy, Debug)]
pub struct OutputDynamics {
    pub lf2h: f64,
    /// The decoupling term. Its vanishing is a loss of authority over the output, and the reason the
    /// feedback-linearising torque has to be guarded rather than assumed.
    pub lglfh: f64,
}

impl CompassGait {
    /// The output's second derivative split into drift and input terms.
    pub fn output_dynamics(&self, s: &GaitState, vc: &dyn SwingConstraint) -> OutputDynamics {
        let (_, hd1, hd2) = vc.desired(s.th1);
        let m = self.mass_matrix(s.th1, s.th2);
        let Some(mi) = m.try_inverse() else { return OutputDynamics { lf2h: 0.0, lglfh: 0.0 } };
        let row = Vector2::new(-hd1, 1.0); // ∂(ÿ)/∂q̈
        let drift = -self.coriolis(s.th1, s.th2, s.d1, s.d2) - self.gravity(s.th1, s.th2);
        OutputDynamics { lf2h: (row.transpose() * mi * drift)[0] - hd2 * s.d1 * s.d1, lglfh: (row.transpose() * mi * self.input_map())[0] }
    }

    /// The **hybrid-zero-dynamics torque**: feedback-linearise the output and then stabilise it.
    ///
    /// `τ = −(L_gL_fh)⁻¹ (L_f²h − v)` where `v` is whatever second derivative the output should have. Passing
    /// `v` from a [`ResClf`](crate::ResClf) is the Ames construction, and it is what makes the convergence
    /// rate a design parameter rather than an outcome. Returns `None` when the decoupling term vanishes.
    pub fn hzd_torque(&self, s: &GaitState, vc: &dyn SwingConstraint, v: f64) -> Option<f64> {
        let od = self.output_dynamics(s, vc);
        if od.lglfh.abs() < 1e-9 {
            return None; // no authority over the output
        }
        Some(-(od.lf2h - v) / od.lglfh)
    }

    /// Integrate one step: flow to the guard, then apply the impact. `control` supplies the torque from the
    /// current state. Returns the post-impact state and the step duration, or `None` if the guard is not
    /// reached within `max_time` (the robot stalled or fell).
    ///
    /// The guard is bracketed to a bisection tolerance rather than a timestep, because the linearisation of a
    /// return map is only as good as the section it is taken on.
    pub fn step_to_guard(&self, start: &GaitState, control: &dyn Fn(&GaitState) -> f64, dt: f64, max_time: f64) -> Option<(GaitState, f64)> {
        let mut s = *start;
        let mut t = 0.0;
        // The guard is zero at the start of a step (the legs are apart but symmetric), so a crossing only
        // counts once the stance leg has passed vertical.
        while t < max_time {
            let prev = s;
            s = self.flow_step(&s, control(&s), dt);
            t += dt;
            if !s.th1.is_finite() || s.d1.abs() > 1e3 {
                return None;
            }
            // A stall is a failure to complete the step, and it has to be caught here: outside `|θ₁| ≤ α` the
            // virtual constraint extrapolates and the integration runs away, so a stalled step would otherwise
            // be reported as a blow-up rather than as the robot stopping.
            if s.d1 <= 0.0 {
                return None;
            }
            if prev.th1 > 0.0 && self.guard(&prev) > 0.0 && self.guard(&s) <= 0.0 {
                // bisect the last step onto the guard
                let (mut lo, mut hi) = (0.0, dt);
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if self.guard(&self.flow_step(&prev, control(&prev), mid)) > 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let at_guard = self.flow_step(&prev, control(&prev), 0.5 * (lo + hi));
                return Some((self.impact(&at_guard), t - dt + 0.5 * (lo + hi)));
            }
        }
        None
    }
}

/// The coefficients of the **restricted (zero) dynamics** at a point on `Z`, written in momentum coordinates.
///
/// On `Z` the swing leg is a function of the stance leg, so the robot has one degree of freedom. Taking the
/// combination of the equations of motion that annihilates the input map (`σ = (1,1)`, since a hip torque acts
/// as `(−1, +1)`) removes the torque entirely and leaves one second-order equation in `θ₁`. Substituting
/// `θ̈₂ = h'θ̈₁ + h''θ̇₁²` gives
///
/// ```text
/// D(θ₁)·θ̈₁ = A(θ₁)·θ̇₁² + B(θ₁)
/// ```
///
/// and then the useful step: with `ζ = θ̇₁²`, `dζ/dθ₁ = 2θ̈₁`, so
///
/// ```text
/// dζ/dθ₁ = 2A/D · ζ + 2B/D
/// ```
///
/// which is **linear in `ζ`**. That linearity is not an approximation and it is the reason the restricted
/// return map is affine — `ρ(ζ) = δ²ζ − V` is a *consequence* of the zero dynamics having one degree of
/// freedom, not a fitted form. It also means `δ²` and `V` follow from two quadratures instead of a
/// simulation.
#[derive(Clone, Copy, Debug)]
pub struct RestrictedCoeffs {
    pub a: f64,
    pub b: f64,
    pub d: f64,
}

impl RestrictedCoeffs {
    /// `dζ/dθ₁ = p·ζ + q`. `None` where `D` vanishes, which is a loss of the reduction itself.
    pub fn linear_form(&self) -> Option<(f64, f64)> {
        if self.d.abs() < 1e-12 {
            return None;
        }
        Some((2.0 * self.a / self.d, 2.0 * self.b / self.d))
    }
}

/// The affine restricted return map, its region of attraction, and the evidence behind them.
#[derive(Clone, Debug)]
pub struct RestrictedMap {
    /// Contraction on `ζ` from the flow alone, `exp(∫ p dθ₁)`.
    pub flow_gain: f64,
    /// Contraction on `ζ` from the impact, `(θ̇₁⁺/θ̇₁⁻)²`. Independent of speed, because the impact is linear in
    /// velocity.
    pub impact_gain: f64,
    /// `δ² = flow_gain · impact_gain`: the full step's multiplier on `ζ`.
    pub delta_sq: f64,
    /// The affine offset, so the map is `ρ(ζ) = δ²ζ − v_zero`.
    pub v_zero: f64,
}

impl RestrictedMap {
    pub fn apply(&self, zeta: f64) -> f64 {
        self.delta_sq * zeta - self.v_zero
    }

    /// The periodic gait: the fixed point `ζ* = −V/(1 − δ²)`, when one exists and is a positive squared rate.
    pub fn gait(&self) -> Option<f64> {
        let d = 1.0 - self.delta_sq;
        if d.abs() < 1e-12 {
            return None;
        }
        let z = -self.v_zero / d;
        (z > 0.0).then_some(z)
    }

    /// Whether the gait is exponentially stable: `0 < δ² < 1` with a positive fixed point.
    pub fn stable(&self) -> bool {
        self.delta_sq > 0.0 && self.delta_sq < 1.0 && self.gait().is_some()
    }

    /// The **certified region of attraction on the section**, as an interval of `ζ`, given the stall threshold
    /// from [`CompassGait::stall_threshold`](crate::CompassGait::stall_threshold).
    ///
    /// For an affine contraction every `ζ` converges to the fixed point, so the boundary is not the map's — it is
    /// the *physics*: below the stall threshold the robot stops before reaching the guard. The interval
    /// `[stall, ∞)` is a genuine region of attraction when it is **forward invariant**, which for this map is one
    /// inequality: `ρ(stall) ≥ stall`. Monotonicity (`δ² > 0`) then carries the whole interval, so this is a
    /// statement about a continuum and not about sampled initial conditions.
    ///
    /// Returns `(lo, hi)` with `lo` the stall threshold, or `None` if the interval is not invariant or holds no
    /// gait. `hi` is supplied by the caller, since the upper limit is where the *model* stops being credible
    /// rather than anything the map decides.
    pub fn certified_basin(&self, stall: f64, ceiling: f64) -> Option<(f64, f64)> {
        let g = self.gait()?;
        let invariant = self.apply(stall) >= stall - 1e-12;
        (invariant && stall < g && g < ceiling && self.stable()).then_some((stall, ceiling))
    }
}

/// Anything that can serve as a relative-degree-two virtual constraint: a desired swing angle and its first
/// two derivatives with respect to the stance angle, plus the step's half-angle.
pub trait SwingConstraint {
    /// `(h_d, h_d', h_d'')` at `th1`.
    fn desired(&self, th1: f64) -> (f64, f64, f64);
    /// Half the inter-leg angle at strike.
    fn alpha(&self) -> f64;

    /// The state on `Z` with stance angle `th1` and rate `d1`.
    fn on_manifold(&self, th1: f64, d1: f64) -> GaitState {
        let (hd, hd1, _) = self.desired(th1);
        GaitState::new(th1, hd, d1, hd1 * d1)
    }
    /// The output `(y, ẏ)`, zero exactly on `Z`.
    fn output(&self, s: &GaitState) -> (f64, f64) {
        let (hd, hd1, _) = self.desired(s.th1);
        (s.th2 - hd, s.d2 - hd1 * s.d1)
    }
}

impl SwingConstraint for VirtualConstraint {
    fn desired(&self, th1: f64) -> (f64, f64, f64) {
        VirtualConstraint::desired(self, th1)
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl CompassGait {
    /// Coefficients of the restricted dynamics at stance angle `th1` under constraint `vc`.
    pub fn restricted_coeffs(&self, vc: &dyn SwingConstraint, th1: f64) -> RestrictedCoeffs {
        let (hd, hd1, hd2) = vc.desired(th1);
        let m = self.mass_matrix(th1, hd);
        let g = self.gravity(th1, hd);
        // sigma = (1,1) annihilates the input map (-1, 1)
        let s1 = m[(0, 0)] + m[(1, 0)];
        let s2 = m[(0, 1)] + m[(1, 1)];
        // Coriolis sum: h1 + h2 = k(θ̇₁² − θ̇₂²) = k(1 − h'²)ζ
        let k = self.m * self.l() * self.b * (th1 - hd).sin();
        RestrictedCoeffs { a: -(s2 * hd2 + k * (1.0 - hd1 * hd1)), b: -(g[0] + g[1]), d: s1 + s2 * hd1 }
    }

    /// The **smallest `ζ` at the start of a step that completes it**, i.e. the stall threshold.
    ///
    /// This is exact rather than searched, and the reason is the linearity of the `ζ` equation. Propagating
    /// `ζ(θ) = G(θ)·ζ₀ + off(θ)` with `G(θ) > 0` always, the robot survives to `θ` exactly when
    /// `ζ₀ > −off(θ)/G(θ)`, so the binding constraint is the maximum of that ratio over the step — one sweep,
    /// no bisection, and valid for **every** initial `ζ` at once rather than for the ones sampled.
    ///
    /// That is what makes the region of attraction certifiable here: a one-dimensional linear map lets a
    /// property be checked over a continuum, which is precisely the leverage a reduced-order certificate is
    /// supposed to provide.
    pub fn stall_threshold(&self, vc: &dyn SwingConstraint, steps: usize) -> Option<f64> {
        let alpha = vc.alpha();
        let n = steps.max(16);
        let h = 2.0 * alpha / n as f64;
        let (mut gain, mut off) = (1.0f64, 0.0f64);
        let mut worst = f64::NEG_INFINITY;
        for i in 0..n {
            let th = -alpha + (i as f64 + 0.5) * h;
            let (p, q) = self.restricted_coeffs(vc, th).linear_form()?;
            let step_gain = (p * h).exp();
            gain *= step_gain;
            off = off * step_gain + q * h * (0.5 * p * h).exp();
            if gain <= 0.0 {
                return None;
            }
            worst = worst.max(-off / gain);
        }
        Some(worst.max(0.0))
    }

    /// The **restricted return map, by quadrature**: integrate the linear `ζ` equation across the step, then
    /// apply the impact's velocity ratio.
    ///
    /// `steps` is the quadrature resolution. Because the `ζ` equation is linear, the flow's contribution
    /// separates into a multiplier and an offset exactly, so this is a pair of one-dimensional integrals — no
    /// simulation, no root-finding, and nothing fitted.
    pub fn restricted_map(&self, vc: &dyn SwingConstraint, steps: usize) -> Option<RestrictedMap> {
        let alpha = vc.alpha();
        let n = steps.max(16);
        let h = 2.0 * alpha / n as f64;
        // Integrate ζ' = pζ + q with the multiplier method, tracking the homogeneous gain and the particular
        // part together: ζ(α) = G·ζ(−α) + off, where G = exp(∫p) and off = ∫ exp(∫_s^α p) q(s) ds.
        let mut gain = 1.0f64;
        let mut off = 0.0f64;
        for i in 0..n {
            // midpoint rule on each cell, which is second-order and needs no derivative of p or q
            let th = -alpha + (i as f64 + 0.5) * h;
            let (p, q) = self.restricted_coeffs(vc, th).linear_form()?;
            let step_gain = (p * h).exp();
            // propagate: zeta -> step_gain*zeta + q*h*step_gain^(1/2) (midpoint placement of the source)
            gain *= step_gain;
            off = off * step_gain + q * h * (0.5 * p * h).exp();
        }
        // the impact, which is linear in velocity so its gain on ζ is a pure ratio
        let pre = vc.on_manifold(alpha, 1.0);
        let post = self.impact(&pre);
        let impact_gain = (post.d1 / pre.d1).powi(2);
        let delta_sq = gain * impact_gain;
        Some(RestrictedMap { flow_gain: gain, impact_gain, delta_sq, v_zero: -off * impact_gain })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The first invariant: energy.** With no torque the continuous phase is conservative, so any error in
    /// the mass matrix, the Coriolis vector, or the gravity vector shows up as drift. This is the check that
    /// the three were derived from the same Lagrangian.
    #[test]
    fn the_continuous_phase_conserves_energy_without_torque() {
        let r = CompassGait::default();
        for &(th1, th2, d1, d2) in &[(-0.3f64, 0.3f64, 1.2f64, -0.4f64), (0.15, -0.25, -0.8, 0.9), (0.05, 0.4, 0.3, 0.3)] {
            let mut s = GaitState::new(th1, th2, d1, d2);
            let e0 = r.energy(&s);
            for _ in 0..200_000 {
                s = r.flow_step(&s, 0.0, 1e-5);
            }
            let drift = (r.energy(&s) - e0).abs() / e0.abs().max(1e-12);
            eprintln!("from ({th1}, {th2}, {d1}, {d2}): energy {e0:.6} -> {:.6}, relative drift {drift:.2e}", r.energy(&s));
            assert!(drift < 1e-9, "energy drifted by {drift:.2e}, so the derived terms are inconsistent");
        }
    }

    /// Gravity and the mass matrix must also agree with the energy function itself — checked by
    /// finite-differencing the potential and the kinetic form directly, which catches a sign error the
    /// conservation test could mask if two errors cancelled.
    #[test]
    fn the_gravity_vector_is_the_gradient_of_the_potential_in_the_energy() {
        let r = CompassGait::default();
        let potential = |th1: f64, th2: f64| r.energy(&GaitState::new(th1, th2, 0.0, 0.0));
        let eps = 1e-7;
        for &(th1, th2) in &[(0.2f64, -0.3f64), (-0.4, 0.1), (0.05, 0.45)] {
            let g_fd = Vector2::new((potential(th1 + eps, th2) - potential(th1 - eps, th2)) / (2.0 * eps), (potential(th1, th2 + eps) - potential(th1, th2 - eps)) / (2.0 * eps));
            let g = r.gravity(th1, th2);
            assert!((g - g_fd).norm() < 1e-6, "gravity disagrees with the potential gradient at ({th1}, {th2}): {g:?} vs {g_fd:?}");
        }
    }

    /// **The second invariant: the impact conserves angular momentum about the impacting foot**, and never
    /// adds energy. Both are independent of how the impact was built, which is what makes them a test of it.
    #[test]
    fn the_impact_conserves_angular_momentum_and_never_adds_energy() {
        let r = CompassGait::default();
        let l = r.l();
        for &alpha in &[0.2f64, 0.3, 0.4] {
            for &d1 in &[-1.5f64, -0.9, -0.4] {
                // on the guard: th2 = -th1
                let pre = GaitState::new(alpha, -alpha, d1, 0.6 * d1.abs());
                let post = r.impact(&pre);

                // the pivot is the landing foot, in the pre-impact frame
                let hip = (l * pre.th1.sin(), l * pre.th1.cos());
                let foot = (hip.0 - l * pre.th2.sin(), hip.1 - l * pre.th2.cos());
                assert!(foot.1.abs() < 1e-12, "the landing foot must be on the ground, height {}", foot.1);

                let am_pre = r.angular_momentum_about(&pre, &r.extended_velocity(&pre), foot);
                // post-impact, in the *new* frame the pivot is the origin; rebuild the pre-impact labelling to
                // compare like with like
                let as_pre_labels = GaitState::new(pre.th1, pre.th2, post.d2, post.d1);
                let qd_post = DVector::from_row_slice(&[post.d2, post.d1, l * post.d1 * pre.th2.cos(), -l * post.d1 * pre.th2.sin()]);
                let am_post = r.angular_momentum_about(&as_pre_labels, &qd_post, foot);
                let rel = (am_post - am_pre).abs() / am_pre.abs().max(1e-12);
                eprintln!("alpha {alpha}, d1 {d1}: angular momentum about the landing foot {am_pre:.8} -> {am_post:.8} ({rel:.2e})");
                assert!(rel < 1e-9, "angular momentum about the impacting foot must be conserved, off by {rel:.2e}");

                // The collision is dissipative — in *kinetic* energy. The total is not comparable across the
                // impact on a slope, because the pinned origin jumps to the new foot and that foot is downhill,
                // so the potential's reference moves. Comparing totals here reads as the impact "adding"
                // energy, which is a bookkeeping artefact rather than physics.
                let (k_pre, k_post) = (r.kinetic(&pre), r.kinetic(&post));
                assert!(k_post <= k_pre + 1e-9, "the impact must not add kinetic energy: {k_pre} -> {k_post}");
                assert!(k_post < k_pre, "a real collision loses some: {k_pre} -> {k_post}");
            }
        }
    }

    /// **Hybrid invariance is solvable, exactly, and it is a measure-zero condition.**
    ///
    /// The regression test for the certificate pipeline's one algebraic step. On `Z` the pre-impact velocity
    /// ratio is fixed by the constraint; the impact is linear in velocity at a fixed configuration, so it maps
    /// that ratio to a definite post-impact ratio; and landing back on `Z` requires that ratio to equal
    /// `h_d'(−α)`. One equation, one unknown — so `c` is *solved*, not searched. `y` needs no work at all,
    /// because `h_d(∓α) = ±α` holds for every parameter by construction.
    ///
    /// The full pipeline, including the periodic gait and the Morris-Grizzle reduction, is in the
    /// `compass_hzd_certificate` example; this keeps the algebra honest without paying for the simulation.
    #[test]
    fn hybrid_invariance_is_solvable_exactly_and_is_measure_zero() {
        let r = CompassGait::default();
        for &alpha in &[0.15f64, 0.22, 0.30] {
            for &e in &[-2.0f64, 0.0, 3.5] {
                // defect: the post-impact velocity ratio against the one Z demands
                let defect = |c: f64| {
                    let vc = VirtualConstraint { alpha, c, e };
                    let post = r.impact(&vc.on_manifold(alpha, 1.0)); // linear in velocity, so the scale is free
                    post.d2 / post.d1 - vc.desired(-alpha).1
                };
                // bracket then bisect
                let mut bracket = None;
                let mut prev = (-6.0f64, defect(-6.0));
                for k in 1..=800 {
                    let c = -6.0 + k as f64 * 0.02;
                    if prev.1 * defect(c) < 0.0 {
                        bracket = Some((prev.0, c));
                        break;
                    }
                    prev = (c, defect(c));
                }
                let (mut lo, mut hi) = bracket.expect("a root must exist in this range");
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if defect(lo) * defect(mid) <= 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let c = 0.5 * (lo + hi);
                let vc = VirtualConstraint { alpha, c, e };

                // measured on the impact itself, not on the root-find
                let post = r.impact(&vc.on_manifold(alpha, 1.0));
                let (y, yd) = vc.output(&post);
                let dist = (y * y + yd * yd).sqrt();
                eprintln!("alpha {alpha}, e {e:>5}: c = {c:.8} gives distance to Z = {dist:.2e}");
                assert!(y.abs() < 1e-14, "y is automatic by construction, got {y:.2e}");
                assert!(dist < 1e-9, "the impact must land back on Z, distance {dist:.2e}");

                // and it is measure zero: perturbing c leaves Z immediately
                let off = VirtualConstraint { alpha, c: c * 1.1, e };
                let po = r.impact(&off.on_manifold(alpha, 1.0));
                let (oy, oyd) = off.output(&po);
                assert!((oy * oy + oyd * oyd).sqrt() > 1e-3, "10% off c must leave Z, so invariance is not generic");
            }
        }
    }

    /// **The analytic restricted map against the full four-dimensional simulation.**
    ///
    /// This is the checkpoint the whole reduced-order story rests on. `restricted_map` computes `δ²` and `V` by
    /// quadrature on a one-degree-of-freedom equation, never touching the full model or the feedback
    /// controller. The `compass_hzd_certificate` example measures the same numbers by simulating the full
    /// four-state robot with a RES-CLF holding it on `Z` and fitting the return map. They have almost nothing
    /// in common computationally, so agreement is real evidence — and it is what licenses using the cheap
    /// analytic map for design and training.
    #[test]
    fn the_quadrature_restricted_map_matches_the_full_simulation() {
        let r = CompassGait::default();
        // the M0 gait: alpha = 0.22 with c solved for hybrid invariance at e = 3.5
        let vc = VirtualConstraint { alpha: 0.22, c: 5.42474849, e: 3.5 };
        let map = r.restricted_map(&vc, 4000).expect("the reduction should exist");
        let gait = map.gait().expect("a periodic gait");

        eprintln!("quadrature: flow gain {:.6}, impact gain {:.6}, delta^2 {:.6}, V {:.6}", map.flow_gain, map.impact_gain, map.delta_sq, map.v_zero);
        eprintln!("            zeta* {:.6} -> stance rate {:.6} /s", gait, gait.sqrt());
        // measured in the example by simulating the full 4-state model with a RES-CLF: delta^2 = 0.914102,
        // zeta* = 4.698605, stance rate 2.16762653
        assert!((map.delta_sq - 0.914102).abs() < 2e-4, "delta^2 must match the full simulation's 0.914102, got {:.6}", map.delta_sq);
        assert!((gait - 4.698605).abs() < 2e-3, "the fixed point must match the simulation's 4.698605, got {gait:.6}");
        assert!((gait.sqrt() - 2.16762653).abs() < 1e-3, "stance rate must match 2.167627, got {:.6}", gait.sqrt());
        assert!(map.stable(), "and the gait must be certified stable by the reduced map");

        // The quadrature must converge, or the agreement above is a coincidence at one resolution.
        let coarse = r.restricted_map(&vc, 200).unwrap();
        let fine = r.restricted_map(&vc, 16000).unwrap();
        let drift = (coarse.delta_sq - fine.delta_sq).abs();
        eprintln!("            resolution 200 vs 16000: delta^2 differs by {drift:.2e}");
        assert!(drift < 1e-4, "the quadrature must be converged, {drift:.2e} apart");
    }

    /// The impact's gain on `ζ` does not depend on the gait speed, because the impact is linear in velocity.
    /// That is what makes `δ²` a property of the *constraint and geometry* rather than of the operating point,
    /// and it is why the return map is affine rather than merely smooth.
    #[test]
    fn the_impact_gain_is_independent_of_speed() {
        let r = CompassGait::default();
        let vc = VirtualConstraint { alpha: 0.22, c: 5.42474849, e: 3.5 };
        let gains: Vec<f64> = [0.5f64, 1.0, 2.5, 7.0].iter().map(|&d1| {
            let pre = vc.on_manifold(0.22, d1);
            (r.impact(&pre).d1 / pre.d1).powi(2)
        }).collect();
        eprintln!("impact gain on zeta at stance rates 0.5/1/2.5/7: {:?}", gains.iter().map(|g| format!("{g:.10}")).collect::<Vec<_>>());
        for g in &gains {
            assert!((g - gains[0]).abs() < 1e-12, "the impact gain must not depend on speed: {gains:?}");
        }
    }

    /// **The third invariant: the reduction is exact.** After the impact the landing foot must be
    /// stationary, so the post-impact velocity is consistent with the *new* pin — which is what licenses
    /// dropping back to two degrees of freedom rather than carrying the extended model forward.
    #[test]
    fn the_post_impact_state_satisfies_the_new_pin_exactly() {
        let r = CompassGait::default();
        let l = r.l();
        for &alpha in &[0.15f64, 0.3, 0.45] {
            let pre = GaitState::new(alpha, -alpha, -1.1, 0.5);
            let qd = r.extended_velocity(&pre);
            let me = r.extended_mass(pre.th1, pre.th2);
            let j = r.foot_jacobian(pre.th2, 1);
            let qd_plus = plastic_impact(&me, &qd, &j);

            // the landing foot is stationary
            let residual = (&j * &qd_plus).norm();
            assert!(residual < 1e-10, "the landing foot must be stationary after the impact, residual {residual:.2e}");

            // and the hip velocity is exactly what the new stance leg's rate implies, so the pinned state
            // carries all the information
            let post = r.impact(&pre);
            let implied = (l * post.d1 * post.th1.cos(), -l * post.d1 * post.th1.sin());
            let actual = (qd_plus[2] - (-l * qd_plus[1] * pre.th2.cos()) * 0.0, qd_plus[3]);
            eprintln!("alpha {alpha}: hip velocity after impact ({:.8}, {:.8}), implied by the new stance rate ({:.8}, {:.8})", actual.0, actual.1, implied.0, implied.1);
            assert!((actual.0 - implied.0).abs() < 1e-9 && (actual.1 - implied.1).abs() < 1e-9, "the pinned reduction is not exact");
        }
    }
}
