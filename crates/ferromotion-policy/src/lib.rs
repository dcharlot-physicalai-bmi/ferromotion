//! ferromotion-policy — an on-device runner for exported *learned* policies (RL / VLA MLPs).
//!
//! This is the interop tier of the program: we **run** trained weights, we don't rewrite training.
//! The common deployable policy is a small feed-forward net (obs → action), so this is a pure-Rust
//! MLP inference engine with the pieces real policies need — observation normalization, a tanh
//! squash, and action scaling — plus a JSON loader for an exported checkpoint. No training, no
//! heavyweight framework; compiles clean to `wasm32` (the same on-device path as our released
//! checkpoint and in-browser VLA). Large transformer VLAs stay an ONNX/`candle` concern; this
//! handles the MLP-policy case that covers most RL control.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Elementwise nonlinearity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    Identity,
    Relu,
    Tanh,
}

impl Activation {
    fn apply(self, v: &mut DVector<f64>) {
        match self {
            Activation::Identity => {}
            Activation::Relu => v.apply(|x| *x = x.max(0.0)),
            Activation::Tanh => v.apply(|x| *x = x.tanh()),
        }
    }

    fn parse(s: &str) -> Activation {
        match s.to_ascii_lowercase().as_str() {
            "relu" => Activation::Relu,
            "tanh" => Activation::Tanh,
            _ => Activation::Identity,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Activation::Identity => "identity",
            Activation::Relu => "relu",
            Activation::Tanh => "tanh",
        }
    }
}

/// One dense layer: `activation(W·x + b)`, `W` is `out×in`.
#[derive(Clone, Debug)]
pub struct Layer {
    pub w: DMatrix<f64>,
    pub b: DVector<f64>,
    pub act: Activation,
}

/// A multilayer perceptron.
#[derive(Clone, Debug)]
pub struct Mlp {
    pub layers: Vec<Layer>,
}

impl Mlp {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    /// Forward pass: observation → raw network output.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut v = DVector::from_row_slice(x);
        for layer in &self.layers {
            let mut y = &layer.w * &v + &layer.b;
            layer.act.apply(&mut y);
            v = y;
        }
        v.as_slice().to_vec()
    }
}

/// A deployable policy: normalize the observation, run the net, optionally tanh-squash, then scale
/// into the action range — the standard export-time inference pipeline for RL policies.
#[derive(Clone, Debug)]
pub struct Policy {
    pub obs_mean: Vec<f64>,
    pub obs_std: Vec<f64>,
    pub net: Mlp,
    pub squash: bool,
    pub act_scale: Vec<f64>,
    pub act_bias: Vec<f64>,
}

impl Policy {
    /// Deterministic action for an observation (the mean action — what you deploy).
    pub fn act(&self, obs: &[f64]) -> Vec<f64> {
        let norm: Vec<f64> = obs
            .iter()
            .enumerate()
            .map(|(i, &o)| {
                let m = self.obs_mean.get(i).copied().unwrap_or(0.0);
                let s = self.obs_std.get(i).copied().unwrap_or(1.0);
                if s.abs() < 1e-12 { o - m } else { (o - m) / s }
            })
            .collect();
        let mut a = self.net.forward(&norm);
        for (i, ai) in a.iter_mut().enumerate() {
            if self.squash {
                *ai = ai.tanh();
            }
            let scale = self.act_scale.get(i).copied().unwrap_or(1.0);
            let bias = self.act_bias.get(i).copied().unwrap_or(0.0);
            *ai = *ai * scale + bias;
        }
        a
    }

    pub fn from_json(s: &str) -> Result<Policy, String> {
        let p: PolicyJson = serde_json::from_str(s).map_err(|e| format!("policy JSON: {e}"))?;
        p.into_policy()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(&PolicyJson::from_policy(self)).expect("serialize policy")
    }
}

// --- JSON schema (plain vectors, so nalgebra needs no serde feature) ---

#[derive(Serialize, Deserialize)]
struct LayerJson {
    w: Vec<Vec<f64>>, // out×in
    b: Vec<f64>,
    act: String,
}

#[derive(Serialize, Deserialize)]
struct PolicyJson {
    #[serde(default)]
    obs_mean: Vec<f64>,
    #[serde(default)]
    obs_std: Vec<f64>,
    layers: Vec<LayerJson>,
    #[serde(default)]
    squash: bool,
    #[serde(default)]
    act_scale: Vec<f64>,
    #[serde(default)]
    act_bias: Vec<f64>,
}

impl PolicyJson {
    fn into_policy(self) -> Result<Policy, String> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for (li, l) in self.layers.iter().enumerate() {
            let out = l.w.len();
            let inn = l.w.first().map(|r| r.len()).unwrap_or(0);
            if l.w.iter().any(|r| r.len() != inn) {
                return Err(format!("layer {li}: ragged weight matrix"));
            }
            if l.b.len() != out {
                return Err(format!("layer {li}: bias len {} != rows {out}", l.b.len()));
            }
            let w = DMatrix::from_fn(out, inn, |r, c| l.w[r][c]);
            layers.push(Layer { w, b: DVector::from_row_slice(&l.b), act: Activation::parse(&l.act) });
        }
        Ok(Policy {
            obs_mean: self.obs_mean,
            obs_std: self.obs_std,
            net: Mlp::new(layers),
            squash: self.squash,
            act_scale: self.act_scale,
            act_bias: self.act_bias,
        })
    }

    fn from_policy(p: &Policy) -> PolicyJson {
        let layers = p
            .net
            .layers
            .iter()
            .map(|l| LayerJson {
                w: (0..l.w.nrows()).map(|r| (0..l.w.ncols()).map(|c| l.w[(r, c)]).collect()).collect(),
                b: l.b.as_slice().to_vec(),
                act: l.act.name().to_string(),
            })
            .collect();
        PolicyJson {
            obs_mean: p.obs_mean.clone(),
            obs_std: p.obs_std.clone(),
            layers,
            squash: p.squash,
            act_scale: p.act_scale.clone(),
            act_bias: p.act_bias.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(w: &[&[f64]], b: &[f64], act: Activation) -> Layer {
        let out = w.len();
        let inn = w[0].len();
        Layer { w: DMatrix::from_fn(out, inn, |r, c| w[r][c]), b: DVector::from_row_slice(b), act }
    }

    #[test]
    fn mlp_forward_matches_hand_computation() {
        // 2→2 tanh, then 2→1 identity.
        let net = Mlp::new(vec![
            layer(&[&[1.0, -1.0], &[0.5, 2.0]], &[0.1, -0.2], Activation::Tanh),
            layer(&[&[1.0, -3.0]], &[0.0], Activation::Identity),
        ]);
        let x = [0.7, -0.4];
        let h0 = (1.0 * 0.7 + -1.0 * -0.4 + 0.1_f64).tanh();
        let h1 = (0.5 * 0.7 + 2.0 * -0.4 + -0.2_f64).tanh();
        let expect = 1.0 * h0 + -3.0 * h1;
        assert!((net.forward(&x)[0] - expect).abs() < 1e-12);
    }

    #[test]
    fn normalization_squash_and_scaling() {
        // Identity net; check the surrounding pipeline math.
        let pol = Policy {
            obs_mean: vec![1.0, 2.0],
            obs_std: vec![2.0, 4.0],
            net: Mlp::new(vec![layer(&[&[1.0, 0.0], &[0.0, 1.0]], &[0.0, 0.0], Activation::Identity)]),
            squash: true,
            act_scale: vec![3.0, 5.0],
            act_bias: vec![0.5, -1.0],
        };
        // obs -> normalized [ (3-1)/2, (10-2)/4 ] = [1, 2] -> tanh -> *scale + bias.
        let a = pol.act(&[3.0, 10.0]);
        assert!((a[0] - (1.0_f64.tanh() * 3.0 + 0.5)).abs() < 1e-12);
        assert!((a[1] - (2.0_f64.tanh() * 5.0 - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn json_round_trips() {
        let json = r#"{"obs_mean":[0,0],"obs_std":[1,1],
            "layers":[{"w":[[-4.0,-4.0]],"b":[0.0],"act":"identity"}],
            "squash":false,"act_scale":[1.0],"act_bias":[0.0]}"#;
        let pol = Policy::from_json(json).unwrap();
        let a = pol.act(&[0.5, 0.25]);
        assert!((a[0] - (-4.0 * 0.5 - 4.0 * 0.25)).abs() < 1e-12);
        // Serialize back and re-parse: same output.
        let pol2 = Policy::from_json(&pol.to_json()).unwrap();
        assert!((pol2.act(&[0.5, 0.25])[0] - a[0]).abs() < 1e-12);
    }

    #[test]
    fn linear_policy_executed_by_runner_stabilizes_double_integrator() {
        // A (degenerate-MLP) linear feedback policy u = -K·[x,v] loaded as an exported net; the runner
        // executes it in closed loop and must regulate ẍ = u to the origin. Proves end-to-end policy
        // execution (a multi-layer net runs through the identical forward path).
        let json = r#"{"layers":[{"w":[[-6.0,-5.0]],"b":[0.0],"act":"identity"}]}"#;
        let pol = Policy::from_json(json).unwrap();
        let (mut x, mut v, dt) = (1.0, 0.0, 0.01);
        for _ in 0..2000 {
            let u = pol.act(&[x, v])[0];
            v += u * dt;
            x += v * dt;
        }
        assert!(x.abs() < 1e-2 && v.abs() < 1e-2, "policy did not regulate: x={x}, v={v}");
    }

    // The runner plugs into the ferromotion ecosystem: a policy that outputs joint torques, executed against
    // ferromotion-core's forward_dynamics, controls a real robot.
    #[test]
    fn torque_policy_regulates_the_arm_via_forward_dynamics() {
        use ferromotion_core::{forward_dynamics, from_urdf_full};
        const ARM2: &str = r#"<robot name="a2"><link name="base"/><link name="l1"><inertial><origin xyz="0.3 0 0" rpy="0 0 0"/><mass value="1.5"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link><link name="l2"><inertial><origin xyz="0.25 0 0" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link><link name="tool"/><joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.6 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="0.5 0 0" rpy="0 0 0"/></joint></robot>"#;
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = nalgebra::Vector3::new(0.0, 0.0, -9.81);
        let (kp, kd) = (100.0, 20.0);
        // obs = [q0-qd0, q1-qd1, qd0, qd1]; action = [-kp·e0 - kd·qd0, -kp·e1 - kd·qd1] (a PD net).
        let net = Mlp::new(vec![layer(
            &[&[-kp, 0.0, -kd, 0.0], &[0.0, -kp, 0.0, -kd]],
            &[0.0, 0.0],
            Activation::Identity,
        )]);
        let pol = Policy { obs_mean: vec![], obs_std: vec![], net, squash: false, act_scale: vec![], act_bias: vec![] };
        let q_des = [0.6, -0.8];
        let (mut q, mut qd, dt) = (vec![0.0, 0.0], vec![0.0, 0.0], 1e-3);
        for _ in 0..4000 {
            let obs = [q[0] - q_des[0], q[1] - q_des[1], qd[0], qd[1]];
            let tau = pol.act(&obs);
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
        }
        let e = ((q[0] - q_des[0]).powi(2) + (q[1] - q_des[1]).powi(2)).sqrt();
        assert!(e < 1e-3, "policy-driven arm did not regulate: err {e}, q={q:?}");
    }
}

pub mod flow;
mod rtc;
pub use flow::{sample_field, sample_mlp, Integrator};
pub use rtc::{guided_sample, rtc_mask, sample_rtc};
