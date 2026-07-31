//! Trains the two learned submissions on the frictionless pendulum and writes their weights as JSON next to
//! the submission files. Provenance for the neural-net entries on the leaderboard — run `cargo run --release
//! --bin train_nets`. Pure std + serde (no nalgebra), gradient-checked so the training is verifiably real.
//!
//!   STRUCTURED — a net learns only the FORCE f(θ), then a symplectic step carries the conservation law.
//!                Conservative by construction; PASSES.
//!   BLACK-BOX  — a net maps the whole state (θ,ω) to the next-step delta. No structure; PUMPS energy, FAILS.
use serde::Serialize;

const G: f64 = 9.81;
const DT: f64 = 0.02;
const H: usize = 16;

fn hash(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u01(i: u32) -> f64 { (hash(i) % 1_000_000) as f64 / 1_000_000.0 }
fn randn(i: u32) -> f64 { (0..12).map(|k| u01(i * 13 + k)).sum::<f64>() - 6.0 }
fn accel(th: f64) -> f64 { -G * th.sin() }
fn rk4(th: f64, w: f64, dt: f64) -> (f64, f64) {
    let (mut t, mut v) = (th, w); let h = dt / 20.0;
    for _ in 0..20 {
        let (k1t, k1v) = (v, accel(t));
        let (k2t, k2v) = (v + 0.5 * h * k1v, accel(t + 0.5 * h * k1t));
        let (k3t, k3v) = (v + 0.5 * h * k2v, accel(t + 0.5 * h * k2t));
        let (k4t, k4v) = (v + h * k3v, accel(t + h * k3t));
        t += h / 6.0 * (k1t + 2.0 * k2t + 2.0 * k3t + k4t);
        v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    }
    (t, v)
}

#[derive(Serialize, Clone)]
struct Mlp { w1: Vec<Vec<f64>>, b1: Vec<f64>, w2: Vec<Vec<f64>>, b2: Vec<f64>, w3: Vec<Vec<f64>>, b3: Vec<f64> }
fn mat(r: usize, c: usize, s: u32) -> Vec<Vec<f64>> {
    (0..r).map(|i| (0..c).map(|j| randn(s + (i * 131 + j) as u32) * (2.0 / c as f64).sqrt()).collect()).collect()
}
impl Mlp {
    fn new(nin: usize, nout: usize, s: u32) -> Self {
        Mlp { w1: mat(H, nin, s + 1), b1: vec![0.0; H], w2: mat(H, H, s + 2), b2: vec![0.0; H], w3: mat(nout, H, s + 3), b3: vec![0.0; nout] }
    }
    fn zeros_like(&self) -> Mlp {
        Mlp { w1: self.w1.iter().map(|r| vec![0.0; r.len()]).collect(), b1: vec![0.0; self.b1.len()],
              w2: self.w2.iter().map(|r| vec![0.0; r.len()]).collect(), b2: vec![0.0; self.b2.len()],
              w3: self.w3.iter().map(|r| vec![0.0; r.len()]).collect(), b3: vec![0.0; self.b3.len()] }
    }
    fn fwd(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let lin = |w: &Vec<Vec<f64>>, b: &Vec<f64>, x: &[f64]| -> Vec<f64> {
            w.iter().zip(b).map(|(row, bi)| row.iter().zip(x).map(|(wij, xj)| wij * xj).sum::<f64>() + bi).collect() };
        let h1: Vec<f64> = lin(&self.w1, &self.b1, x).iter().map(|v| v.tanh()).collect();
        let h2: Vec<f64> = lin(&self.w2, &self.b2, &h1).iter().map(|v| v.tanh()).collect();
        let y = lin(&self.w3, &self.b3, &h2);
        (y, h1, h2)
    }
    // accumulate the gradient of ½‖y−tgt‖² into g; returns the loss.
    fn bwd(&self, x: &[f64], tgt: &[f64], g: &mut Mlp) -> f64 {
        let (y, h1, h2) = self.fwd(x);
        let dy: Vec<f64> = y.iter().zip(tgt).map(|(a, b)| a - b).collect();
        let loss = 0.5 * dy.iter().map(|d| d * d).sum::<f64>();
        for i in 0..dy.len() { for j in 0..H { g.w3[i][j] += dy[i] * h2[j]; } g.b3[i] += dy[i]; }
        let mut dh2 = vec![0.0; H];
        for j in 0..H { let mut s = 0.0; for i in 0..dy.len() { s += self.w3[i][j] * dy[i]; } dh2[j] = s * (1.0 - h2[j] * h2[j]); }
        for i in 0..H { for j in 0..H { g.w2[i][j] += dh2[i] * h1[j]; } g.b2[i] += dh2[i]; }
        let mut dh1 = vec![0.0; H];
        for j in 0..H { let mut s = 0.0; for i in 0..H { s += self.w2[i][j] * dh2[i]; } dh1[j] = s * (1.0 - h1[j] * h1[j]); }
        for i in 0..H { for j in 0..x.len() { g.w1[i][j] += dh1[i] * x[j]; } g.b1[i] += dh1[i]; }
        loss
    }
}

struct Adam { m: Mlp, v: Mlp, t: f64 }
impl Adam {
    fn new(p: &Mlp) -> Self { Adam { m: p.zeros_like(), v: p.zeros_like(), t: 0.0 } }
    fn step(&mut self, p: &mut Mlp, g: &Mlp, lr: f64) {
        self.t += 1.0; let (b1, b2, e) = (0.9, 0.999, 1e-8);
        macro_rules! upd2 { ($pf:ident) => { for i in 0..p.$pf.len() { for j in 0..p.$pf[i].len() {
            self.m.$pf[i][j] = b1*self.m.$pf[i][j] + 0.1*g.$pf[i][j]; self.v.$pf[i][j] = b2*self.v.$pf[i][j] + 0.001*g.$pf[i][j]*g.$pf[i][j];
            p.$pf[i][j] -= lr*(self.m.$pf[i][j]/(1.0-b1.powf(self.t)))/((self.v.$pf[i][j]/(1.0-b2.powf(self.t))).sqrt()+e); } } } }
        macro_rules! upd1 { ($pf:ident) => { for i in 0..p.$pf.len() {
            self.m.$pf[i] = b1*self.m.$pf[i] + 0.1*g.$pf[i]; self.v.$pf[i] = b2*self.v.$pf[i] + 0.001*g.$pf[i]*g.$pf[i];
            p.$pf[i] -= lr*(self.m.$pf[i]/(1.0-b1.powf(self.t)))/((self.v.$pf[i]/(1.0-b2.powf(self.t))).sqrt()+e); } } }
        upd2!(w1); upd1!(b1); upd2!(w2); upd1!(b2); upd2!(w3); upd1!(b3);
    }
}

fn grad_check(net: &Mlp, x: &[f64], tgt: &[f64]) -> (f64, f64) {
    let mut g = net.zeros_like(); net.bwd(x, tgt, &mut g);
    let analytic = g.w2[3][2];
    let mut np = net.clone(); let d = 1e-6;
    np.w2[3][2] += d; let lp = { let (y, _, _) = np.fwd(x); 0.5 * y.iter().zip(tgt).map(|(a, b)| (a - b).powi(2)).sum::<f64>() };
    np.w2[3][2] -= 2.0 * d; let lm = { let (y, _, _) = np.fwd(x); 0.5 * y.iter().zip(tgt).map(|(a, b)| (a - b).powi(2)).sum::<f64>() };
    (analytic, (lp - lm) / (2.0 * d))
}

fn main() {
    let steps = 40_000u32;
    // STRUCTURED: force net θ -> f(θ), fit to the true force -g sinθ.
    let mut sf = Mlp::new(1, 1, 10); let mut sfo = Adam::new(&sf);
    // BLACK-BOX: (θ,ω) -> Δ(θ,ω) over one dt.
    let mut bb = Mlp::new(2, 2, 20); let mut bbo = Adam::new(&bb);

    let (sa, sd) = grad_check(&sf, &[0.4], &[accel(0.4)]);
    println!("structured grad-check: analytic {:+.3e} fd {:+.3e} -> {}", sa, sd, if (sa - sd).abs() < 1e-5 { "MATCH" } else { "MISMATCH" });
    let (ba, bd) = grad_check(&bb, &[0.4, 1.0], &[0.1, 0.2]);
    println!("black-box  grad-check: analytic {:+.3e} fd {:+.3e} -> {}", ba, bd, if (ba - bd).abs() < 1e-5 { "MATCH" } else { "MISMATCH" });

    for k in 0..steps {
        let th = -3.3 + 6.6 * u01(k * 5 + 1);
        let mut g = sf.zeros_like(); sf.bwd(&[th], &[accel(th)], &mut g); sfo.step(&mut sf, &g, 1e-3);
        let w = -5.0 + 10.0 * u01(k * 5 + 2);
        let (tn, wn) = rk4(th, w, DT);
        let mut gb = bb.zeros_like(); bb.bwd(&[th, w], &[tn - th, wn - w], &mut gb); bbo.step(&mut bb, &gb, 1e-3);
    }

    // report fit
    let (mut sfmse, mut bbmse, n) = (0.0f64, 0.0f64, 500u32);
    for k in 0..n { let th = -3.3 + 6.6 * u01(900_000 + k * 3);
        sfmse += (sf.fwd(&[th]).0[0] - accel(th)).powi(2);
        let w = -5.0 + 10.0 * u01(900_001 + k * 3); let (tn, wn) = rk4(th, w, DT); let d = bb.fwd(&[th, w]).0;
        bbmse += ((th + d[0] - tn).powi(2) + (w + d[1] - wn).powi(2)) / 2.0; }
    println!("structured force MSE {:.2e} · black-box one-step MSE {:.2e}", sfmse / n as f64, bbmse / n as f64);

    std::fs::write("submissions/structured_net.weights.json", serde_json::to_string(&sf).unwrap()).unwrap();
    std::fs::write("submissions/blackbox_net.weights.json", serde_json::to_string(&bb).unwrap()).unwrap();
    println!("wrote submissions/structured_net.weights.json and submissions/blackbox_net.weights.json");
}
