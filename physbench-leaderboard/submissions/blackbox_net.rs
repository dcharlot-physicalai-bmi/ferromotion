// Learned submission (BLACK-BOX): a trained neural net maps the whole state (θ,ω) to the next-step delta.
// Accurate step to step, but with no physics structure it PUMPS energy over a rollout and FAILS — the exact
// failure the benchmark is built to catch. Weights are from src/bin/train_nets.rs (gradient-checked).
use crate::bench::{Meta, Model};
use serde::Deserialize;
use std::sync::OnceLock;

pub const META: Meta = Meta { name: "black-box-net", author: "reference (trained)", kind: "learned" };

#[derive(Deserialize)]
struct W { w1: Vec<Vec<f64>>, b1: Vec<f64>, w2: Vec<Vec<f64>>, b2: Vec<f64>, w3: Vec<Vec<f64>>, b3: Vec<f64> }
fn weights() -> &'static W {
    static WEIGHTS: OnceLock<W> = OnceLock::new();
    WEIGHTS.get_or_init(|| serde_json::from_str(include_str!("blackbox_net.weights.json")).unwrap())
}
fn forward(x: &[f64]) -> Vec<f64> {
    let w = weights();
    let lin = |m: &[Vec<f64>], b: &[f64], x: &[f64]| -> Vec<f64> {
        m.iter().zip(b).map(|(row, bi)| row.iter().zip(x).map(|(a, c)| a * c).sum::<f64>() + bi).collect() };
    let h1: Vec<f64> = lin(&w.w1, &w.b1, x).iter().map(|v| v.tanh()).collect();
    let h2: Vec<f64> = lin(&w.w2, &w.b2, &h1).iter().map(|v| v.tanh()).collect();
    lin(&w.w3, &w.b3, &h2)
}

pub struct M;
impl Model for M {
    fn step(&self, th: f64, w: f64, _dt: f64) -> (f64, f64) {
        // Predicts the delta over the trained step; a black-box next-state map has no notion of the metric,
        // so it drifts on a rollout and fails to reverse in time.
        let d = forward(&[th, w]);
        (th + d[0], w + d[1])
    }
}
