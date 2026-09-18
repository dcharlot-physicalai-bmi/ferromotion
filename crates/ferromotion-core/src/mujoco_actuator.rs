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
//! Carried here: a **joint** transmission (hinge or slide) with a **fixed** gain, **no** or an **affine**
//! bias, and no activation state — which is 2,827 of Menagerie's 4,693 actuators, and every `motor`,
//! `position` and `velocity` among them. Tendon transmissions, muscles, adhesion and stateful dynamics are
//! refused by the loader rather than approximated.

/// `mjtGain`, restricted to what a stateless joint actuator can be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActGain {
    /// `gain = gainprm[0]`.
    Fixed,
    /// `gain = gainprm[0] + gainprm[1]·length + gainprm[2]·velocity`.
    Affine,
}

/// `mjtBias`, likewise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActBias {
    None,
    /// `bias = biasprm[0] + biasprm[1]·length + biasprm[2]·velocity`.
    Affine,
}

/// One actuator, with its parameters resolved as MuJoCo's compiler resolves them.
#[derive(Clone, Debug)]
pub struct Actuator {
    pub name: String,
    /// The tree joint (one degree of freedom) this drives.
    pub joint: usize,
    /// `actuator_gear[0]`: the transmission ratio. `length = gear·q`, `velocity = gear·q̇`, and the moment
    /// that carries the force back to the joint is the same `gear`.
    pub gear: f64,
    pub gain: ActGain,
    pub gainprm: [f64; 3],
    pub bias: ActBias,
    /// ⛔ A POSITIVE `biasprm[2]` is an unresolved `dampratio`, not a damping — see [`resolve_dampratio`].
    pub biasprm: [f64; 3],
    /// `ctrlrange` when `ctrllimited`; the control is clamped into it before anything else.
    pub ctrlrange: Option<[f64; 2]>,
    /// `forcerange` when `forcelimited`; the force is clamped into it after gain and bias.
    pub forcerange: Option<[f64; 2]>,
}

impl Actuator {
    /// `actuator_length` for a joint transmission.
    pub fn length(&self, q: &[f64]) -> f64 {
        self.gear * q[self.joint]
    }

    /// `actuator_velocity`.
    pub fn velocity(&self, qd: &[f64]) -> f64 {
        self.gear * qd[self.joint]
    }

    /// `actuator_force`: the scalar force, before the transmission carries it to the joint.
    pub fn force(&self, length: f64, velocity: f64, ctrl: f64) -> f64 {
        let ctrl = match self.ctrlrange {
            Some([lo, hi]) => ctrl.clamp(lo, hi),
            None => ctrl,
        };
        let gain = match self.gain {
            ActGain::Fixed => self.gainprm[0],
            ActGain::Affine => self.gainprm[0] + self.gainprm[1] * length + self.gainprm[2] * velocity,
        };
        let bias = match self.bias {
            ActBias::None => 0.0,
            ActBias::Affine => self.biasprm[0] + self.biasprm[1] * length + self.biasprm[2] * velocity,
        };
        let f = gain * ctrl + bias;
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
    let mut out = vec![0.0; dof_force_range.len()];
    for (i, a) in acts.iter().enumerate() {
        let f = a.force(a.length(q), a.velocity(qd), ctrl.get(i).copied().unwrap_or(0.0));
        out[a.joint] += a.gear * f;
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
/// −biasprm[1]`, i.e. the same `kp` on both sides. `m0_diag` is `diag(M)` at `qpos0`.
pub fn resolve_dampratio(acts: &mut [Actuator], m0_diag: &[f64]) {
    for a in acts.iter_mut() {
        if a.gainprm[0] != -a.biasprm[1] || a.biasprm[2] <= 0.0 {
            continue;
        }
        // the inertia the transmission reflects: diag(M)/gear² summed over the moment's non-zeros, which for
        // a joint transmission is the one dof it drives
        let trn2 = a.gear * a.gear;
        let mass = if trn2 > 1e-15 { m0_diag[a.joint] / trn2 } else { 0.0 };
        a.biasprm[2] = -(a.biasprm[2] * 2.0 * (a.gainprm[0] * mass).sqrt());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn act(gain: ActGain, gainprm: [f64; 3], bias: ActBias, biasprm: [f64; 3]) -> Actuator {
        Actuator { name: "a".into(), joint: 0, gear: 1.0, gain, gainprm, bias, biasprm, ctrlrange: None, forcerange: None }
    }

    /// The three shortcut tags are the SAME law with different parameters, and that is the whole point of
    /// resolving them: a motor, a position servo and a velocity servo differ only in `gainprm`/`biasprm`.
    #[test]
    fn a_motor_a_position_servo_and_a_velocity_servo_are_one_equation() {
        let motor = act(ActGain::Fixed, [1.0, 0.0, 0.0], ActBias::None, [0.0; 3]);
        assert_eq!(motor.force(0.3, 1.7, 2.0), 2.0, "a motor passes its control through");
        let (kp, kv) = (80.0, 5.0);
        let position = act(ActGain::Fixed, [kp, 0.0, 0.0], ActBias::Affine, [0.0, -kp, -kv]);
        // kp·(ctrl − q) − kv·q̇, which is the servo law written the way a controls text would
        let (q, qd, ctrl) = (0.3, 1.7, 0.5);
        assert!((position.force(q, qd, ctrl) - (kp * (ctrl - q) - kv * qd)).abs() < 1e-12);
        let velocity = act(ActGain::Fixed, [kv, 0.0, 0.0], ActBias::Affine, [0.0, 0.0, -kv]);
        assert!((velocity.force(q, qd, ctrl) - kv * (ctrl - qd)).abs() < 1e-12);
    }

    /// The control is clamped BEFORE the gain sees it and the force AFTER the bias is added — the order
    /// matters, because a clamped control still leaves the bias free to drive the force out of range.
    #[test]
    fn the_control_is_clamped_going_in_and_the_force_going_out() {
        let mut a = act(ActGain::Fixed, [10.0, 0.0, 0.0], ActBias::Affine, [0.0, -10.0, 0.0]);
        a.ctrlrange = Some([-1.0, 1.0]);
        // ctrl 5 is clamped to 1, so force = 10·1 − 10·q
        assert!((a.force(0.0, 0.0, 5.0) - 10.0).abs() < 1e-12);
        a.forcerange = Some([-4.0, 4.0]);
        assert!((a.force(0.0, 0.0, 5.0) - 4.0).abs() < 1e-12, "the force limit applies after the bias");
        // and the bias alone can hit the limit with no control at all
        assert!((a.force(10.0, 0.0, 0.0) + 4.0).abs() < 1e-12);
    }

    /// ⛔ The last clamp is the JOINT's, applied to the SUM — two actuators on one dof are capped together,
    /// and an actuator with no `forcerange` of its own is still capped.
    #[test]
    fn the_joint_caps_the_total_actuator_force_on_its_dof() {
        let mut a = act(ActGain::Fixed, [1.0, 0.0, 0.0], ActBias::None, [0.0; 3]);
        a.gear = 2.0;
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
        acts[1].joint = 0;
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
