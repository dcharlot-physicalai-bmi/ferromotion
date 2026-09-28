//! **MuJoCo's actuators**: the force an actuator produces and where it lands in joint space.
//!
//! An MJCF `<position kp="80"/>` is sugar. MuJoCo resolves every shortcut tag into the same four fields
//! before anything runs — a gain type, a bias type, and their parameter vectors — and `mj_fwdActuation` then
//! evaluates one law for all of them:
//!
//! ```text
//! force = gain·input + bias,   gain = prm₀ (fixed) or prm₀ + prm₁·length + prm₂·velocity (affine)
//!                              bias = 0 (none)     or prm₀ + prm₁·length + prm₂·velocity (affine)
//! ```
//!
//! so a position servo is a fixed gain `kp` against an affine bias `(0, −kp, −kv)`, a velocity servo is a
//! fixed gain `kv` against `(0, 0, −kv)`, and a motor is a unit gain with no bias. Reading it that way is the
//! difference between porting three special cases and porting one equation.
//!
//! ⛔ `dampratio` is NOT a parameter of this law. MuJoCo stores it in `biasprm[2]` with a POSITIVE sign as a
//! marker, and `mj_setConst` later rewrites it as a damping — `−dampratio·2·√(kp·mass)`, where `mass` is the
//! transmission-reflected inertia `Σ diag(M)/gear²` at `qpos0`. It cannot be resolved while reading the file,
//! because it needs the mass matrix; [`resolve_dampratio`] is applied by the loader once the tree exists.
//!
//! Carried here: fixed, affine and muscle gains; no, affine and muscle biases; the activation laws
//! `integrator`, `filter`, `filterexact` and `muscle` with `actrange` and `actearly`; and the `mujoco.pid`
//! actuator plugin ([`Pid`]) — every actuator in Menagerie. The transmissions (joint, tendon, site, body) live with
//! the model in [`crate::MjcfTree`]. What is not carried — a `user` or `dcmotor` law, or a shortcut not yet
//! resolved (`<position timeconst>`, `<intvelocity>`, `<cylinder>`, `<damper>`) — is refused by the loader
//! rather than approximated.

/// `mjtDyn`: whether an actuator carries an activation state, and under what law (`mj_fwdActuation`).
/// Every one but `None` makes the ACTIVATION the gain's input in place of the control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActDyn {
    None,
    /// `act' = ctrl`.
    Integrator,
    /// `act' = (ctrl − act) / τ`, `τ = dynprm[0]`, integrated like any other state.
    Filter,
    /// The same filter, integrated EXACTLY over the step: `act += act'·τ·(1 − e^(−h/τ))`.
    FilterExact,
    /// `act' = mju_muscleDynamics(ctrl, act, dynprm)` — see [`muscle_dynamics`].
    Muscle,
}

/// The first nine of a ten-wide parameter vector, which is what every muscle curve takes.
fn prm9(p: &[f64; 10]) -> [f64; 9] {
    let mut out = [0.0; 9];
    out.copy_from_slice(&p[..9]);
    out
}

/// `mjtGain`, restricted to what a stateless joint actuator can be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActGain {
    /// `gain = gainprm[0]`.
    Fixed,
    /// `gain = gainprm[0] + gainprm[1]·length + gainprm[2]·velocity`.
    Affine,
    /// `gain = mju_muscleGain(length, velocity, lengthrange, acc0, gainprm[0..9])` — see [`muscle_gain`].
    Muscle,
}

/// `mjtBias`, likewise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActBias {
    None,
    /// `bias = biasprm[0] + biasprm[1]·length + biasprm[2]·velocity`.
    Affine,
    /// `bias = mju_muscleBias(length, lengthrange, acc0, biasprm[0..9])` — see [`muscle_bias`].
    Muscle,
}

/// A transmission whose moment is NOT fixed when the file is read — it turns with the model, so the tree
/// rebuilds it at every state ([`crate::MjcfTree::actuator_state`]).
#[derive(Clone, Debug)]
pub enum DynTransmission {
    /// `<general site="..." gear="fx fy fz tx ty tz">`: a wrench applied at a site, in that site's frame.
    /// ⛔ Its `actuator_length` is ZERO however the model is posed; its velocity is not.
    Site { site: String, gear: [f64; 6] },
    /// A SPATIAL tendon, by index into [`crate::MjcfTree::tendons`]: `length = gear·L(q)` along the path.
    SpatialTendon { index: usize },
    /// `<adhesion body="...">` (`mjTRN_BODY`): the moment is MINUS the AVERAGE of the normal Jacobians of
    /// every contact touching the body — active ones and those found in the gap band alike — so a positive
    /// control pulls the surfaces together. It is a function of the CONTACT SET, so it is rebuilt at every
    /// state; the length is always zero and `gear` is not applied.
    Body { body: String },
}

/// **The `mujoco.pid` actuator plugin** (MuJoCo 3.13.0, `plugin/actuator/pid.cc`): a PID servo on the
/// transmission length, `force = kp·e + kd·ė + ki·∫e`, with an optional clamp on the integral and an
/// optional slew limit on the setpoint, both carried as ACTIVATIONS — the error integral first (when
/// `ki ≠ 0`), then the previous setpoint (when `slewmax` is set).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pid {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    /// the integral's clamp, `imax / ki` — the file states it as a FORCE
    pub i_max: Option<f64>,
    /// the setpoint's rate limit
    pub slew_max: Option<f64>,
}

impl Pid {
    /// `Pid::ActDim`: one activation for the integral, one for the slew limiter.
    pub fn actdim(&self) -> usize {
        usize::from(self.ki != 0.0) + usize::from(self.slew_max.is_some())
    }

    /// `Pid::GetCtrl`: the control clamped to `ctrlrange`, then rate-limited against the previous one —
    /// but only once time has started (`d->time > 0`): at `t = 0` there is no previous setpoint.
    fn setpoint(&self, a: &Actuator, ctrl: f64, act: &[f64], time: f64, h: f64) -> f64 {
        let mut u = match a.ctrlrange {
            Some([lo, hi]) => ctrl.clamp(lo, hi),
            None => ctrl,
        };
        if let Some(slew) = self.slew_max.filter(|_| time > 0.0) {
            let prev = act[usize::from(self.ki != 0.0)];
            // contracted as the C compiles: `prev ∓ slew·h` is one fused multiply-add each way
            u = u.clamp((-slew).mul_add(h, prev), slew.mul_add(h, prev));
        }
        u
    }

    /// The error integral one step on, clamped: `∫e + e·h`.
    fn integral(&self, act: &[f64], error: f64, h: f64) -> f64 {
        if self.ki == 0.0 {
            return 0.0;
        }
        let i = error.mul_add(h, act[0]);
        match self.i_max {
            Some(m) => i.clamp(-m, m),
            None => i,
        }
    }

    /// `Pid::Compute`: the force, before `forcerange`. `act` is this actuator's own activations.
    pub fn force(&self, a: &Actuator, length: f64, velocity: f64, ctrl: f64, act: &[f64], time: f64, h: f64) -> f64 {
        let error = self.setpoint(a, ctrl, act, time, h) - length;
        // `ctrl_dot` is zero for a stateless setpoint, so the error's rate is the transmission's
        let error_dot = 0.0 - velocity;
        self.ki.mul_add(self.integral(act, error, h), self.kd.mul_add(error_dot, self.kp * error))
    }

    /// `Pid::ActDot`: rates that land each activation exactly on its next value after one Euler step —
    /// `(∫e_next − ∫e)/h` and `(setpoint − previous)/h`.
    pub fn act_dot(&self, a: &Actuator, length: f64, ctrl: f64, act: &[f64], time: f64, h: f64) -> Vec<f64> {
        let u = self.setpoint(a, ctrl, act, time, h);
        let error = u - length;
        let mut out = Vec::with_capacity(2);
        if self.ki != 0.0 {
            out.push((self.integral(act, error, h) - act[0]) / h);
        }
        if self.slew_max.is_some() {
            out.push((u - act[out.len()]) / h);
        }
        out
    }
}

/// One actuator's transmission evaluated at a state: what MuJoCo calls `actuator_moment`,
/// `actuator_length` and `actuator_velocity`.
#[derive(Clone, Debug, Default)]
pub struct ActState {
    pub moment: Vec<(usize, f64)>,
    pub length: f64,
    pub velocity: f64,
}

/// One actuator, with its parameters resolved as MuJoCo's compiler resolves them.
#[derive(Clone, Debug)]
pub struct Actuator {
    pub name: String,
    /// **`actuator_moment`**: which dofs this actuator pulls on, and how hard — `(dof, moment)` for every
    /// non-zero entry, with `gear` already folded in. A JOINT transmission is the single entry
    /// `(joint, gear)`; a FIXED TENDON is one entry per joint the tendon names, `(dof, gear·coef)`. The
    /// whole of MuJoCo's transmission lives here: `length = m·q`, `velocity = m·q̇`, and the same `m` carries
    /// the force back.
    pub moment: Vec<(usize, f64)>,
    /// A transmission whose moment moves with the model — a site, or a spatial tendon. `moment` is empty
    /// here and the live values come from [`crate::MjcfTree::actuator_state`].
    pub dynamic: Option<DynTransmission>,
    /// `actuator_gear[0]`: the transmission ratio. `length = gear·q`, `velocity = gear·q̇`, and the moment
    /// that carries the force back to the joint is the same `gear`.
    pub gear: f64,
    pub gain: ActGain,
    /// `mjNGAIN` wide, because a muscle's gain reads nine of them. An affine gain uses the first three and
    /// leaves the rest at zero, which is what MuJoCo stores too.
    pub gainprm: [f64; 10],
    pub bias: ActBias,
    /// ⛔ A POSITIVE `biasprm[2]` is an unresolved `dampratio`, not a damping — see [`resolve_dampratio`].
    pub biasprm: [f64; 10],
    /// `mjtDyn`: whether this actuator carries an activation state, and under what law.
    pub dynamics: ActDyn,
    /// `actuator_dynprm`, the first three — a muscle's `(tau_act, tau_deact, tausmooth)`.
    pub dynprm: [f64; 3],
    /// **`actuator_lengthrange`**: the transmission length at the two ends of the muscle's operating
    /// range, which is what turns a length in metres into the normalised length every muscle curve is
    /// written in. Stated in the file, or computed by MuJoCo's `mj_setLengthRange` when it is not.
    pub lengthrange: [f64; 2],
    /// **`actuator_acc0`**: `‖M⁻¹·moment‖` at `qpos0` — the acceleration a unit force on this transmission
    /// produces. Only a muscle with `force < 0` reads it, and for those it IS the peak force, as
    /// `scale/acc0`. Computed once by the compiler, like every other `*0`.
    pub acc0: f64,
    /// `ctrlrange` when `ctrllimited`; the control is clamped into it before anything else.
    pub ctrlrange: Option<[f64; 2]>,
    /// `forcerange` when `forcelimited`; the force is clamped into it after gain and bias.
    pub forcerange: Option<[f64; 2]>,
    /// `actrange` when `actlimited`: the activation is clamped into it each time it is INTEGRATED
    /// ([`Actuator::next_activation`]), never when it is read.
    pub actrange: Option<[f64; 2]>,
    /// `actearly`: the force reads the activation at the END of the step, `next_activation(act, act')`,
    /// instead of the one it starts with — removing the one-step delay between a control and its force.
    pub actearly: bool,
    /// A `mujoco.pid` plugin computes this actuator's force and activations in place of gain and bias.
    pub pid: Option<Pid>,
    /// The FIXED tendon a tendon transmission drives, by index into [`crate::MjcfTree::tendons`] — what
    /// MuJoCo's own pipeline reads (`length = gear·ten_length`, `moment = gear·ten_J`); `moment` above is
    /// the same row folded at load.
    pub tendon: Option<usize>,
}

impl Actuator {
    /// `actuator_length` for a joint transmission.
    pub fn length(&self, q: &[f64]) -> f64 {
        if self.dynamic.is_some() {
            return 0.0;
        }
        self.moment.iter().map(|(d, m)| m * q[*d]).sum()
    }

    /// `actuator_velocity`.
    pub fn velocity(&self, qd: &[f64]) -> f64 {
        self.moment.iter().map(|(d, m)| m * qd[*d]).sum()
    }

    /// `actuator_actnum`: how many activations this actuator carries.
    pub fn actnum(&self) -> usize {
        match &self.pid {
            Some(p) => p.actdim(),
            None => usize::from(self.dynamics != ActDyn::None),
        }
    }

    /// **`act_dot`**, as `mj_fwdActuation` computes it: from the control clamped to `ctrlrange`, and zero for
    /// an actuator with no state.
    pub fn act_dot(&self, ctrl: f64, act: f64) -> f64 {
        let ctrl = match self.ctrlrange {
            Some([lo, hi]) => ctrl.clamp(lo, hi),
            None => ctrl,
        };
        match self.dynamics {
            ActDyn::None => 0.0,
            ActDyn::Integrator => ctrl,
            ActDyn::Filter | ActDyn::FilterExact => (ctrl - act) / self.dynprm[0].max(MINVAL),
            ActDyn::Muscle => muscle_dynamics(ctrl, act, &self.dynprm),
        }
    }

    /// **`mj_nextActivation`**: the activation one step of `h` later — `act + h·act'`, or for `FilterExact`
    /// the exact solution `act + act'·τ·(1 − e^(−h/τ))` — then clamped to `actrange` where it is limited.
    /// This is both how `mj_advance` integrates the state and what an `actearly` force reads.
    pub fn next_activation(&self, act: f64, act_dot: f64, h: f64) -> f64 {
        let next = if self.dynamics == ActDyn::FilterExact {
            let tau = self.dynprm[0].max(MINVAL);
            (act_dot * tau).mul_add(1.0 - (-h / tau).exp(), act)
        } else {
            act_dot.mul_add(h, act)
        };
        match self.actrange {
            Some([lo, hi]) => next.clamp(lo, hi),
            None => next,
        }
    }

    /// `actuator_force`: the scalar force, before the transmission carries it to the joint.
    ///
    /// ⛔ The input to the gain is the ACTIVATION where the actuator has one, and the control only where it
    /// does not — `mj_fwdActuation` reads `act[actadr]` in place of `ctrl[i]` the moment `dyntype` is not
    /// `none`. A muscle driven by its control instead of its activation responds with no delay at all,
    /// which is the one thing the activation state exists to prevent.
    pub fn force(&self, length: f64, velocity: f64, ctrl: f64, act: f64) -> f64 {
        let ctrl = match self.ctrlrange {
            Some([lo, hi]) => ctrl.clamp(lo, hi),
            None => ctrl,
        };
        let input = match self.dynamics {
            ActDyn::None => ctrl,
            _ => act,
        };
        let gain = match self.gain {
            ActGain::Fixed => self.gainprm[0],
            ActGain::Affine => self.gainprm[0] + self.gainprm[1] * length + self.gainprm[2] * velocity,
            ActGain::Muscle => muscle_gain(length, velocity, &self.lengthrange, self.acc0, &prm9(&self.gainprm)),
        };
        let bias = match self.bias {
            ActBias::None => 0.0,
            ActBias::Affine => self.biasprm[0] + self.biasprm[1] * length + self.biasprm[2] * velocity,
            ActBias::Muscle => muscle_bias(length, &self.lengthrange, self.acc0, &prm9(&self.biasprm)),
        };
        let f = gain * input + bias;
        match self.forcerange {
            Some([lo, hi]) => f.clamp(lo, hi),
            None => f,
        }
    }
}

/// **`qfrc_actuator`**: every actuator's force carried through its transmission into joint space, then
/// clamped PER JOINT.
///
/// ⛔ That last clamp belongs to the JOINT, not the actuator: `<joint actuatorfrcrange="-87 87">` caps the
/// total actuator force on that degree of freedom, after every actuator driving it has been summed.
/// `franka_fr3` sets no `forcerange` on any actuator and relies entirely on this — its servos ask for 531 N·m
/// where the joint passes 87 — so a port that reads only the actuator's own limits is out by a factor of six
/// with every actuator parameter matching exactly.
///
/// `dof_force_range` is one entry per degree of freedom; `None` where the joint is unlimited.
pub fn qfrc_actuator(acts: &[Actuator], q: &[f64], qd: &[f64], ctrl: &[f64], dof_force_range: &[Option<[f64; 2]>]) -> Vec<f64> {
    let state: Vec<ActState> = acts.iter().map(|a| ActState { moment: a.moment.clone(), length: a.length(q), velocity: a.velocity(qd) }).collect();
    qfrc_actuator_with(acts, &state, ctrl, &[], dof_force_range)
}

/// [`qfrc_actuator`] with each transmission already evaluated, which a site or a spatial tendon needs: their
/// moments turn with the model and cannot be resolved when the file is read.
/// `act` is one entry per actuator — the activation of the ones that carry a state, ignored for the rest.
/// An EMPTY slice means every activation is zero, which is what `mj_resetData` leaves behind and therefore
/// what a state sampled by `mj_forward` on a fresh `mjData` actually has.
pub fn qfrc_actuator_with(acts: &[Actuator], state: &[ActState], ctrl: &[f64], act: &[f64], dof_force_range: &[Option<[f64; 2]>]) -> Vec<f64> {
    let forces: Vec<f64> = acts.iter().enumerate().map(|(i, a)| state.get(i).map_or(0.0, |st| a.force(st.length, st.velocity, ctrl.get(i).copied().unwrap_or(0.0), act.get(i).copied().unwrap_or(0.0)))).collect();
    qfrc_from_forces(state, &forces, dof_force_range)
}

/// `moment'·force`, then each joint's `actuatorfrcrange` — [`qfrc_actuator_with`] once the forces are known.
pub fn qfrc_from_forces(state: &[ActState], forces: &[f64], dof_force_range: &[Option<[f64; 2]>]) -> Vec<f64> {
    let mut out = vec![0.0; dof_force_range.len()];
    for (st, f) in state.iter().zip(forces) {
        for (d, m) in &st.moment {
            out[*d] += m * f;
        }
    }
    for (o, r) in out.iter_mut().zip(dof_force_range) {
        if let Some([lo, hi]) = r {
            *o = o.clamp(*lo, *hi);
        }
    }
    out
}

/// **Turn a `dampratio` into a damping**, as `mj_setConst` does once the mass matrix exists.
///
/// MuJoCo marks an unresolved `dampratio` by leaving `biasprm[2]` POSITIVE (a real damping is stored
/// negative), and only for a position-like actuator — one whose gain and bias agree that `gainprm[0] ==
/// −biasprm[1]`, i.e. the same `kp` on both sides. `dof_m0` is MuJoCo's `dof_M0`: each dof's composite
/// inertia at `qpos0`, armature included.
///
/// ⛔ The reflected mass is `Σ dof_M0 / trn²` over the transmission's entries — MuJoCo SUMS the masses.
/// This used to take `1 / Σ (trn² / M)`, which is the same number for a joint transmission (to the last bit
/// or two) and a DIFFERENT one for a fixed tendon over several dofs.
pub fn resolve_dampratio(acts: &mut [Actuator], dof_m0: &[f64]) {
    for a in acts.iter_mut() {
        if a.gainprm[0] != -a.biasprm[1] || a.biasprm[2] <= 0.0 {
            continue;
        }
        let mut mass = 0.0;
        for (d, t) in &a.moment {
            let trn = t.abs();
            let trn2 = trn * trn;
            if trn2 > MINVAL {
                mass += dof_m0[*d] / trn2;
            }
        }
        let damping = a.biasprm[2] * 2.0 * (a.gainprm[0] * mass).sqrt();
        a.biasprm[2] = -damping;
    }
}

// ---------------------------------------------------------------------------------------------------
// MUSCLES
// ---------------------------------------------------------------------------------------------------

/// MuJoCo's `mjMINVAL`. Every division in the muscle curves is floored by it, and the floor is not
/// decoration: `ms_human_700` ships muscles whose `range` is a single point, which makes `L0` a division
/// by zero in exact arithmetic and a finite number here.
const MINVAL: f64 = 1e-15;

/// `mju_sigmoid`: the quintic smoothstep `6x⁵ − 15x⁴ + 10x³`, clamped to `[0, 1]` outside the unit
/// interval. Twice continuously differentiable at both ends, which is why the activation filter can use it
/// to blend two timescales without putting a kink in the derivative an integrator has to swallow.
pub fn sigmoid(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)
}

/// The active force-length curve, on the NORMALISED length: zero outside `[lmin, lmax]`, and four
/// quadratic pieces that meet at `(lmin+1)/2`, `1` and `(1+lmax)/2` with value `0.5`, `1`, `0.5`.
fn muscle_gain_length(length: f64, lmin: f64, lmax: f64) -> f64 {
    if length <= lmin || length >= lmax {
        return 0.0;
    }
    let (a, b) = (0.5 * (lmin + 1.0), 0.5 * (1.0 + lmax));
    if length <= a {
        let x = (length - lmin) / (a - lmin).max(MINVAL);
        0.5 * x * x
    } else if length <= 1.0 {
        let x = (1.0 - length) / (1.0 - a).max(MINVAL);
        1.0 - 0.5 * x * x
    } else if length <= b {
        let x = (length - 1.0) / (b - 1.0).max(MINVAL);
        1.0 - 0.5 * x * x
    } else {
        let x = (lmax - length) / (lmax - b).max(MINVAL);
        0.5 * x * x
    }
}

/// Normalise a transmission length onto the muscle's operating range, and the scale that does it.
/// `L0` is the muscle's optimal fibre length in transmission units: the whole curve set is written in
/// units of it, which is how one set of shape parameters describes every muscle in a body.
fn muscle_normalise(len: f64, lengthrange: &[f64; 2], prm: &[f64; 9]) -> (f64, f64) {
    let l0 = (lengthrange[1] - lengthrange[0]) / (prm[1] - prm[0]).max(MINVAL);
    (prm[0] + (len - lengthrange[0]) / l0.max(MINVAL), l0)
}

/// **`mju_muscleGain`** — the ACTIVE force a muscle can produce at this length and shortening velocity,
/// per unit activation. `prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax)`.
///
/// ⛔ Negative on purpose. A muscle pulls, so its force opposes increasing length, and MuJoCo returns the
/// gain already signed — a port that takes the magnitude gets a muscle that pushes.
///
/// ⛔ `force < 0` is not an error, it is MuJoCo's "scale me": the peak force becomes `scale / acc0`, where
/// `acc0` is the acceleration a unit force on this transmission produces at `qpos0`. It is how a model
/// states muscle strength in units of the body it moves rather than in newtons, and it is the DEFAULT.
pub fn muscle_gain(len: f64, vel: f64, lengthrange: &[f64; 2], acc0: f64, prm: &[f64; 9]) -> f64 {
    let force = if prm[2] < 0.0 { prm[3] / acc0.max(MINVAL) } else { prm[2] };
    let (l, l0) = muscle_normalise(len, lengthrange, prm);
    let v = vel / (l0 * prm[6]).max(MINVAL);
    let fl = muscle_gain_length(l, prm[4], prm[5]);
    // the force-velocity curve: zero at and beyond the maximum shortening rate, quadratic to 1 at rest,
    // and a second quadratic up to the eccentric plateau `fvmax`
    let y = prm[8] - 1.0;
    let fv = if v <= -1.0 {
        0.0
    } else if v <= 0.0 {
        (v + 1.0) * (v + 1.0)
    } else if v <= y {
        prm[8] - (y - v) * (y - v) / y.max(MINVAL)
    } else {
        prm[8]
    };
    -force * fl * fv
}

/// **`mjd_muscleGain_vel`**: `∂ muscle_gain / ∂ velocity`, the one velocity derivative of a muscle's force
/// that an implicit integrator carries (the passive force has no velocity term). The same curve as
/// [`muscle_gain`], differentiated piece by piece.
pub fn muscle_gain_vel(len: f64, vel: f64, lengthrange: &[f64; 2], acc0: f64, prm: &[f64; 9]) -> f64 {
    let force = if prm[2] < 0.0 { prm[3] / acc0.max(MINVAL) } else { prm[2] };
    let (l, l0) = muscle_normalise(len, lengthrange, prm);
    let v = vel / (l0 * prm[6]).max(MINVAL);
    let fl = muscle_gain_length(l, prm[4], prm[5]);
    let y = prm[8] - 1.0;
    let dfv = if v <= -1.0 {
        0.0
    } else if v <= 0.0 {
        2.0 * v + 2.0
    } else if v <= y {
        (-2.0 * v + 2.0 * y) / y.max(MINVAL)
    } else {
        0.0
    };
    -force * fl * dfv / (l0 * prm[6]).max(MINVAL)
}

/// **`mju_muscleBias`** — the PASSIVE force, from the tissue being stretched past its optimum. Zero up to
/// the optimal length, half-quadratic to the knee at `(1+lmax)/2`, and LINEAR beyond it, so a muscle
/// stretched far past its range keeps resisting instead of saturating.
pub fn muscle_bias(len: f64, lengthrange: &[f64; 2], acc0: f64, prm: &[f64; 9]) -> f64 {
    let force = if prm[2] < 0.0 { prm[3] / acc0.max(MINVAL) } else { prm[2] };
    let (l, _) = muscle_normalise(len, lengthrange, prm);
    let b = 0.5 * (1.0 + prm[5]);
    if l <= 1.0 {
        0.0
    } else if l <= b {
        let x = (l - 1.0) / (b - 1.0).max(MINVAL);
        -force * prm[7] * 0.5 * x * x
    } else {
        let x = (l - b) / (b - 1.0).max(MINVAL);
        -force * prm[7] * (0.5 + x)
    }
}

/// **`mju_muscleDynamics`** — `acṫ` for a muscle's activation state, a first-order filter whose time
/// constant depends on which way it is moving (Millard et al. 2013): activating is fast, deactivating is
/// slow, and both scale with the current activation.
///
/// ⛔ The two timescales are computed from the CLAMPED activation, but the excess excitation `dctrl` is
/// measured against the RAW one. A port that clamps both agrees everywhere `act ∈ [0,1]` — which is
/// everywhere a simulation that started in range ever goes — and disagrees exactly where a caller hands in
/// an activation from outside, which is where a test would look.
pub fn muscle_dynamics(ctrl: f64, act: f64, prm: &[f64; 3]) -> f64 {
    let ctrlclamp = ctrl.clamp(0.0, 1.0);
    let actclamp = act.clamp(0.0, 1.0);
    let tau_act = prm[0] * (0.5 + 1.5 * actclamp);
    let tau_deact = prm[1] / (0.5 + 1.5 * actclamp);
    let dctrl = ctrlclamp - act;
    let tau = if prm[2] < MINVAL {
        if dctrl > 0.0 {
            tau_act
        } else {
            tau_deact
        }
    } else {
        tau_deact + (tau_act - tau_deact) * sigmoid(dctrl / prm[2] + 0.5)
    };
    dctrl / tau.max(MINVAL)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⛔ Every number here came out of MuJoCo 3.13.0's own `mju_muscleGain`, `mju_muscleBias` and
    /// `mju_muscleDynamics`, on `ms_human_700`'s `addbrev_r` parameters. The four lengths are chosen to
    /// land in four different pieces of the force-length curve and the last one past the passive knee,
    /// and the three velocities on both sides of the isometric point.
    #[test]
    fn the_muscle_curves_are_mujocos_own() {
        let prm = [0.75, 1.05, 625.82, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2];
        let lr = [0.086661, 0.189083];
        #[rustfmt::skip]
        let want = [
            (0.09, -0.5, -0.18839969715207558, 0.0),
            (0.09, 0.0, -336.9135104901837, 0.0),
            (0.09, 0.3, -404.2962125882204, 0.0),
            (0.13, -0.5, -0.3075589024828487, 0.0),
            (0.13, 0.0, -550.0048624513569, 0.0),
            (0.13, 0.3, -660.0058349416282, 0.0),
            (0.17, -0.5, -0.3498569039751974, 0.0),
            (0.17, 0.0, -625.6460040504492, 0.0),
            (0.17, 0.3, -750.775204860539, 0.0),
            (0.2, -0.5, -0.33688895045050105, -30.373814160035547),
            (0.2, 0.0, -602.4555275692035, -30.373814160035547),
            (0.2, 0.3, -722.9466330830442, -30.373814160035547),
        ];
        for (len, vel, g, b) in want {
            let (mg, mb) = (muscle_gain(len, vel, &lr, 1.0, &prm), muscle_bias(len, &lr, 1.0, &prm));
            assert!((mg - g).abs() <= 1e-12 * g.abs().max(1.0), "gain at len {len} vel {vel}: {mg} vs {g}");
            assert!((mb - b).abs() <= 1e-12 * b.abs().max(1.0), "bias at len {len}: {mb} vs {b}");
        }
        // ⛔ the passive curve is flat below the optimum and the active one is not: a port that mixed the
        // two would pass on the stretched samples alone
        assert_eq!(muscle_bias(0.13, &lr, 1.0, &prm), 0.0);
        assert!(muscle_gain(0.13, 0.0, &lr, 1.0, &prm) < -500.0);
        // `force < 0` means "scale me by `scale/acc0`", so halving acc0 doubles every force
        let scaled = [0.75, 1.05, -1.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2];
        let (a, b) = (muscle_gain(0.13, 0.0, &lr, 2.0, &scaled), muscle_gain(0.13, 0.0, &lr, 1.0, &scaled));
        assert!((2.0 * a - b).abs() < 1e-9, "{a} {b}");
    }

    /// The activation filter, again against MuJoCo's own values. ⛔ The last pair is the one that matters:
    /// `ctrl = 1.3` is clamped to 1 and `act = -0.1` is NOT, so `dctrl` is 1.1 and the timescales are
    /// taken at `act = 0`. Clamping both would give 1.0 and a different answer.
    #[test]
    fn the_activation_filter_clamps_the_control_and_not_the_activation() {
        #[rustfmt::skip]
        let want = [
            (0.8, 0.2, [0.01, 0.04, 0.0], 75.00000000000001),
            (0.8, 0.2, [0.01, 0.04, 0.2], 75.00000000000001),
            (0.2, 0.8, [0.01, 0.04, 0.0], -25.500000000000004),
            (0.2, 0.8, [0.01, 0.04, 0.2], -25.500000000000004),
            (1.3, -0.1, [0.01, 0.04, 0.0], 220.0),
            (1.3, -0.1, [0.01, 0.04, 0.2], 219.99999999999983),
        ];
        for (ctrl, act, prm, a) in want {
            let mine = muscle_dynamics(ctrl, act, &prm);
            assert!((mine - a).abs() <= 1e-12 * a.abs().max(1.0), "ctrl {ctrl} act {act} prm {prm:?}: {mine} vs {a}");
        }
        // clamping the activation too would make `dctrl` 1.0 instead of 1.1, so the rate would be 200
        assert!((muscle_dynamics(1.3, -0.1, &[0.01, 0.04, 0.0]) - 220.0).abs() < 1e-9);
        // the quintic smoothstep, at the three points that pin it
        assert_eq!(sigmoid(-1.0), 0.0);
        assert_eq!(sigmoid(2.0), 1.0);
        assert!((sigmoid(0.5) - 0.5).abs() < 1e-15);
    }

    fn act(gain: ActGain, gainprm3: [f64; 3], bias: ActBias, biasprm3: [f64; 3]) -> Actuator {
        let (mut gainprm, mut biasprm) = ([0.0; 10], [0.0; 10]);
        gainprm[..3].copy_from_slice(&gainprm3);
        biasprm[..3].copy_from_slice(&biasprm3);
        Actuator {
            name: "a".into(), moment: vec![(0, 1.0)], dynamic: None, gear: 1.0, gain, gainprm, bias, biasprm,
            dynamics: ActDyn::None, dynprm: [0.0; 3], lengthrange: [0.0; 2], acc0: 0.0, ctrlrange: None, forcerange: None,
            actrange: None, actearly: false, pid: None, tendon: None,
        }
    }

    /// The three shortcut tags are the SAME law with different parameters, and that is the whole point of
    /// resolving them: a motor, a position servo and a velocity servo differ only in `gainprm`/`biasprm`.
    #[test]
    fn a_motor_a_position_servo_and_a_velocity_servo_are_one_equation() {
        let motor = act(ActGain::Fixed, [1.0, 0.0, 0.0], ActBias::None, [0.0; 3]);
        assert_eq!(motor.force(0.3, 1.7, 2.0, 0.0), 2.0, "a motor passes its control through");
        let (kp, kv) = (80.0, 5.0);
        let position = act(ActGain::Fixed, [kp, 0.0, 0.0], ActBias::Affine, [0.0, -kp, -kv]);
        // kp·(ctrl − q) − kv·q̇, which is the servo law written the way a controls text would
        let (q, qd, ctrl) = (0.3, 1.7, 0.5);
        assert!((position.force(q, qd, ctrl, 0.0) - (kp * (ctrl - q) - kv * qd)).abs() < 1e-12);
        let velocity = act(ActGain::Fixed, [kv, 0.0, 0.0], ActBias::Affine, [0.0, 0.0, -kv]);
        assert!((velocity.force(q, qd, ctrl, 0.0) - kv * (ctrl - qd)).abs() < 1e-12);
    }

    /// The control is clamped BEFORE the gain sees it and the force AFTER the bias is added — the order
    /// matters, because a clamped control still leaves the bias free to drive the force out of range.
    #[test]
    fn the_control_is_clamped_going_in_and_the_force_going_out() {
        let mut a = act(ActGain::Fixed, [10.0, 0.0, 0.0], ActBias::Affine, [0.0, -10.0, 0.0]);
        a.ctrlrange = Some([-1.0, 1.0]);
        // ctrl 5 is clamped to 1, so force = 10·1 − 10·q
        assert!((a.force(0.0, 0.0, 5.0, 0.0) - 10.0).abs() < 1e-12);
        a.forcerange = Some([-4.0, 4.0]);
        assert!((a.force(0.0, 0.0, 5.0, 0.0) - 4.0).abs() < 1e-12, "the force limit applies after the bias");
        // and the bias alone can hit the limit with no control at all
        assert!((a.force(10.0, 0.0, 0.0, 0.0) + 4.0).abs() < 1e-12);
    }

    /// ⛔ The last clamp is the JOINT's, applied to the SUM — two actuators on one dof are capped together,
    /// and an actuator with no `forcerange` of its own is still capped.
    #[test]
    fn the_joint_caps_the_total_actuator_force_on_its_dof() {
        let mut a = act(ActGain::Fixed, [1.0, 0.0, 0.0], ActBias::None, [0.0; 3]);
        a.gear = 2.0;
        a.moment = vec![(0, 2.0)];
        let b = a.clone();
        let q = [0.0];
        // two actuators, gear 2, controls 10 and 5: 2·10 + 2·5 = 30 before the joint has its say
        assert_eq!(qfrc_actuator(&[a.clone(), b.clone()], &q, &q, &[10.0, 5.0], &[None]), vec![30.0]);
        assert_eq!(qfrc_actuator(&[a, b], &q, &q, &[10.0, 5.0], &[Some([-7.0, 7.0])]), vec![7.0]);
    }

    /// ⛔ `dampratio` is a marker, not a damping: positive means unresolved, and it becomes
    /// `−dampratio·2·√(kp·mass)` with the transmission-reflected inertia. Only position-like actuators —
    /// `gainprm[0] == −biasprm[1]` — are touched.
    #[test]
    fn dampratio_becomes_a_damping_once_the_mass_is_known() {
        let kp = 100.0;
        let mut acts = vec![
            act(ActGain::Fixed, [kp, 0.0, 0.0], ActBias::Affine, [0.0, -kp, 2.0]),
            // a velocity servo: gainprm[0] != −biasprm[1], so its biasprm[2] is left alone
            act(ActGain::Fixed, [5.0, 0.0, 0.0], ActBias::Affine, [0.0, 0.0, 3.0]),
        ];
        acts[0].gear = 2.0;
        acts[0].moment = vec![(0, 2.0)];
        resolve_dampratio(&mut acts, &[0.8]);
        // mass = 0.8/4 = 0.2, damping = 2·2·sqrt(100·0.2)
        assert!((acts[0].biasprm[2] + 2.0 * 2.0 * (kp * 0.2f64).sqrt()).abs() < 1e-12, "{}", acts[0].biasprm[2]);
        assert_eq!(acts[1].biasprm[2], 3.0, "a non-position actuator's biasprm[2] is not a dampratio");
        // and resolving twice does not resolve again
        let once = acts[0].biasprm[2];
        resolve_dampratio(&mut acts, &[0.8]);
        assert_eq!(acts[0].biasprm[2], once);
    }
}
