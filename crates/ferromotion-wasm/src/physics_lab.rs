//! **Physics bench** — the wasm rigs surfacing the rest of the new capabilities live: a volumetric
//! Neo-Hookean **soft body** (ferromotion-fem), **granular** media (ferromotion-dem), and a
//! **batched-RL** cartpole swing-up (ferromotion-control). Each carries its verified receipt — the
//! soft body's elastic energy, the pile's kinetic energy decaying to rest, the pole reaching upright
//! by parallel random shooting.

use ferromotion_control::{batch_rollout, Cartpole};
use ferromotion_core::{quadruped, quadruped_trot_tau, tree_floating_contact_step, FootContact, Joint, LinkInertia};
use ferromotion_dem::{DemSim, Grain};
use ferromotion_fem::FemSim;
use nalgebra::{Isometry3, Matrix3, Point3, Rotation3, Vector3, Vector6};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------------------------
// Soft body — volumetric tetrahedral Neo-Hookean FEM (a jelly cube pinned at top, wobbling).
// ---------------------------------------------------------------------------------------------
#[wasm_bindgen]
pub struct FemLab {
    sim: FemSim,
    edges: Vec<[usize; 2]>,
}

#[wasm_bindgen]
impl FemLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> FemLab {
        let mut sim = FemSim::box_grid(3, 3, 3, 0.28, 0.5, 3.0e3, 1.5e3, 2e-4);
        sim.damping_rate = 20.080_321_285_140_6; // old per-step 0.004 at dt = 2e-4
        sim.gravity = Vector3::new(0.0, 0.0, -9.81);
        sim.floor = Some(0.0);
        sim.k_contact = 3.0e4;
        // lift the block above the floor and give it a slight tumble, then drop it
        for i in 0..sim.n_verts() {
            sim.x[i].z += 0.55;
            sim.v[i] = Vector3::new(0.6, 0.0, 0.0);
        }
        // unique tet edges for the wireframe
        use std::collections::HashSet;
        let mut set = HashSet::new();
        // reconstruct edges from a fresh box_grid tets list is not exposed; derive from proximity
        // (cube vertices a fixed grid): connect verts within ~1.05·spacing.
        let h = 0.28;
        for a in 0..sim.n_verts() {
            for b in (a + 1)..sim.n_verts() {
                if (sim.x[a] - sim.x[b]).norm() < 1.05 * h {
                    set.insert((a, b));
                }
            }
        }
        let edges = set.into_iter().map(|(a, b)| [a, b]).collect();
        FemLab { sim, edges }
    }

    pub fn step(&mut self, k: usize) {
        for _ in 0..k {
            self.sim.step();
        }
    }

    /// Vertex positions as flat `[x, z, …]` (front x–z view), in world units.
    pub fn verts(&self) -> Vec<f64> {
        self.sim.x.iter().flat_map(|p| [p.x, p.z]).collect()
    }

    /// Edges as flat `[a, b, …]` vertex-index pairs (the wireframe).
    pub fn edges(&self) -> Vec<u32> {
        self.edges.iter().flat_map(|e| [e[0] as u32, e[1] as u32]).collect()
    }

    pub fn energy(&self) -> f64 {
        self.sim.energy()
    }
    pub fn n_verts(&self) -> usize {
        self.sim.n_verts()
    }
    pub fn pinned(&self) -> Vec<u8> {
        self.sim.pinned.iter().map(|&p| p as u8).collect()
    }
}

impl Default for FemLab {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------------------
// Granular — DEM grains pouring into a pile on the floor.
// ---------------------------------------------------------------------------------------------
#[wasm_bindgen]
pub struct DemLab {
    sim: DemSim,
}

#[wasm_bindgen]
impl DemLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> DemLab {
        // a loose column of grains above the floor, slightly staggered so it topples into a pile
        let mut grains = Vec::new();
        let r = 0.09;
        for k in 0..10 {
            for i in 0..3 {
                let jitter = ((k * 7 + i * 13) % 5) as f64 * 0.01;
                grains.push(Grain {
                    x: Vector3::new(-0.2 + i as f64 * 0.19 + jitter, 0.0, 0.4 + k as f64 * 0.2),
                    v: Vector3::zeros(),
                    r,
                    m: 0.4,
                });
            }
        }
        let sim = DemSim::new(grains, 4.0e4, 80.0, 0.5, 2e-4);
        DemLab { sim }
    }

    pub fn step(&mut self, k: usize) {
        for _ in 0..k {
            self.sim.step();
        }
    }

    /// Grain centers as flat `[x, z, …]` (front x–z view).
    pub fn positions(&self) -> Vec<f64> {
        self.sim.grains.iter().flat_map(|g| [g.x.x, g.x.z]).collect()
    }
    pub fn radius(&self) -> f64 {
        self.sim.grains.first().map(|g| g.r).unwrap_or(0.1)
    }
    pub fn kinetic_energy(&self) -> f64 {
        self.sim.kinetic_energy()
    }
    pub fn n(&self) -> usize {
        self.sim.grains.len()
    }
}

impl Default for DemLab {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------------------
// Batched-RL — cartpole swing-up by parallel random shooting (on-device RL in the browser).
// ---------------------------------------------------------------------------------------------
#[wasm_bindgen]
pub struct CartpoleLab {
    cp: Cartpole,
    state: [f64; 4],
    dt: f64,
    t: u64,
}

#[wasm_bindgen]
impl CartpoleLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CartpoleLab {
        CartpoleLab { cp: Cartpole::default(), state: [0.0, 0.0, std::f64::consts::PI, 0.0], dt: 0.02, t: 0 }
    }

    /// One control step: sample `b` action sequences, roll them all out in parallel (the batched
    /// primitive), execute the best action's first step. Returns the chosen force.
    pub fn control_step(&mut self, b: usize, horizon: usize) -> f64 {
        // deterministic candidate actions (splitmix64), reseeded per step
        let mut s = 0x1000u64.wrapping_add(self.t);
        let acts: Vec<f64> = (0..b)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                12.0 * ((((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0)
            })
            .collect();
        let x0: Vec<f64> = (0..b).flat_map(|_| self.state).collect();
        let finals = batch_rollout(&self.cp, &x0, &acts, horizon, self.dt);
        let (mut bi, mut bc) = (0usize, f64::INFINITY);
        for i in 0..b {
            let f = &finals[4 * i..4 * i + 4];
            let c = (1.0 - f[2].cos()) * 5.0 + 0.1 * f[0] * f[0] + 0.02 * f[3] * f[3];
            if c < bc {
                bc = c;
                bi = i;
            }
        }
        self.state = self.cp.step(self.state, acts[bi], self.dt);
        self.t += 1;
        acts[bi]
    }

    pub fn cart_x(&self) -> f64 {
        self.state[0]
    }
    pub fn theta(&self) -> f64 {
        self.state[2]
    }
    /// Upright error `1 − cos θ` (0 = balanced) — the receipt.
    pub fn upright_error(&self) -> f64 {
        1.0 - self.state[2].cos()
    }
    pub fn reset(&mut self) {
        self.state = [0.0, 0.0, std::f64::consts::PI, 0.0];
        self.t = 0;
    }
}

impl Default for CartpoleLab {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------------------
// Legged locomotion — a QUADRUPED (torso + four 2-joint legs) walking forward on the ground by a
// scripted static crawl gait, integrated with tree-structured floating-base dynamics + penalty
// contact. All on your own device, no cloud: the same tree ABA the RL benches use, run per-frame.
// ---------------------------------------------------------------------------------------------
#[wasm_bindgen]
pub struct QuadrupedLab {
    joints: Vec<Joint>,
    inertia: Vec<LinkInertia>,
    parent: Vec<isize>,
    contacts: Vec<FootContact>,
    base_inertia: LinkInertia,
    base: Isometry3<f64>,
    v0: Vector6<f64>,
    q: Vec<f64>,
    qd: Vec<f64>,
    dt: f64,
    freq: f64,
    t: u64,
}

// leg attachment corners (front-left, front-right, back-left, back-right) and the 0.3 m segments,
// mirroring `ferromotion_core::quadruped`.
const QCORNERS: [(f64, f64); 4] = [(0.15, 0.1), (0.15, -0.1), (-0.15, 0.1), (-0.15, -0.1)];
const QSEG: f64 = 0.3;

#[wasm_bindgen]
impl QuadrupedLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> QuadrupedLab {
        let (joints, inertia, parent, contacts) = quadruped();
        let n = joints.len();
        let base_inertia = LinkInertia {
            mass: 8.0,
            com: Vector3::zeros(),
            inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)),
        };
        QuadrupedLab {
            joints,
            inertia,
            parent,
            contacts,
            base_inertia,
            base: Isometry3::translation(0.0, 0.0, 0.62),
            v0: Vector6::zeros(),
            q: vec![0.0; n],
            qd: vec![0.0; n],
            dt: 2e-4,
            freq: 1.0,
            t: 0,
        }
    }

    /// Advance the walk by `k` physics substeps (dt = 0.2 ms each). The gait clock drives a scripted
    /// crawl; the tree floating-base dynamics + penalty ground contact do the rest.
    pub fn step(&mut self, k: usize) {
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (floor, kn, kd) = (0.0, 1.5e4, 120.0);
        for _ in 0..k {
            let phase = std::f64::consts::TAU * self.freq * self.t as f64 * self.dt;
            let tau = quadruped_trot_tau(&self.q, &self.qd, phase);
            let (b, v, qn, qdn) = tree_floating_contact_step(
                &self.joints,
                &self.inertia,
                &self.parent,
                &self.base_inertia,
                self.base,
                self.v0,
                &self.q,
                &self.qd,
                &tau,
                &self.contacts,
                floor,
                kn,
                kd,
                self.dt,
                g,
            );
            self.base = b;
            self.v0 = v;
            self.q = qn;
            self.qd = qdn;
            self.t += 1;
        }
    }

    /// World-frame joint positions, flat `[hip.xyz, knee.xyz, foot.xyz]` per leg (4 legs → 36 floats).
    /// The island projects these to the screen (a slight y-shear gives the near/far legs depth).
    pub fn joints_world(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(36);
        for (leg, &(cx, cy)) in QCORNERS.iter().enumerate() {
            let (qh, qk) = (self.q[leg * 2], self.q[leg * 2 + 1]);
            let hip = Point3::new(cx, cy, 0.0);
            let rh = Rotation3::from_axis_angle(&Vector3::y_axis(), qh);
            let knee = hip + rh * Vector3::new(0.0, 0.0, -QSEG);
            let rk = Rotation3::from_axis_angle(&Vector3::y_axis(), qh + qk);
            let foot = knee + rk * Vector3::new(0.0, 0.0, -QSEG);
            for p in [hip, knee, foot] {
                let w = self.base.transform_point(&p);
                out.extend_from_slice(&[w.x, w.y, w.z]);
            }
        }
        out
    }

    /// Torso centre `[x, y, z]` (world).
    pub fn base_pos(&self) -> Vec<f64> {
        let t = self.base.translation.vector;
        vec![t.x, t.y, t.z]
    }
    /// Forward distance travelled (metres) — the receipt.
    pub fn forward_x(&self) -> f64 {
        self.base.translation.x
    }
    pub fn base_z(&self) -> f64 {
        self.base.translation.z
    }
    /// Upright alignment (world-up · body-up); 1.0 = perfectly level.
    pub fn up_alignment(&self) -> f64 {
        self.base.rotation.to_rotation_matrix().matrix()[(2, 2)]
    }
    pub fn reset(&mut self) {
        let n = self.joints.len();
        self.base = Isometry3::translation(0.0, 0.0, 0.62);
        self.v0 = Vector6::zeros();
        self.q = vec![0.0; n];
        self.qd = vec![0.0; n];
        self.t = 0;
    }
}

impl Default for QuadrupedLab {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------------------
// Multiphysics — two-way FEM<->DEM: grains raining onto a compliant soft slab.
// ---------------------------------------------------------------------------------------------
#[wasm_bindgen]
pub struct CoupledLab {
    sim: ferromotion_coupled::CoupledFemDem,
    edges: Vec<[usize; 2]>,
}

#[wasm_bindgen]
impl CoupledLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CoupledLab {
        use ferromotion_dem::{DemSim, Grain};
        use ferromotion_fem::FemSim;
        let fem = FemSim::box_grid(4, 1, 1, 0.2, 0.5, 3.0e3, 1.5e3, 2e-4);
        // grains dropped above the slab
        let grains: Vec<Grain> = (0..12)
            .map(|k| Grain {
                x: Vector3::new(0.05 + (k % 4) as f64 * 0.19 + ((k * 7) % 3) as f64 * 0.02, 0.1, 0.7 + (k / 4) as f64 * 0.22),
                v: Vector3::zeros(),
                r: 0.075,
                m: 0.2,
            })
            .collect();
        let dem = DemSim::new(grains, 4.0e4, 70.0, 0.5, 2e-4);
        let mut sim = ferromotion_coupled::CoupledFemDem::new(fem, dem, 0.08, 4.0e4);
        sim.floor = Some(0.0);
        sim.fem.damping_rate = 154.639_175_257_732_0; // old per-step 0.03 at dt = 2e-4
        // pin the slab's base so it acts as a compliant mat the grains land on
        let zmin = sim.fem.x.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
        for i in 0..sim.fem.n_verts() {
            if sim.fem.x[i].z < zmin + 1e-6 {
                sim.fem.pinned[i] = true;
            }
        }
        // FEM wireframe edges (grid proximity)
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for a in 0..sim.fem.n_verts() {
            for b in (a + 1)..sim.fem.n_verts() {
                if (sim.fem.x[a] - sim.fem.x[b]).norm() < 1.05 * 0.2 {
                    set.insert((a, b));
                }
            }
        }
        let edges = set.into_iter().map(|(a, b)| [a, b]).collect();
        CoupledLab { sim, edges }
    }

    pub fn step(&mut self, k: usize) {
        for _ in 0..k {
            self.sim.step();
        }
    }

    /// FEM vertices as flat `[x, z, …]` (front x–z view).
    pub fn fem_verts(&self) -> Vec<f64> {
        self.sim.fem.x.iter().flat_map(|p| [p.x, p.z]).collect()
    }
    pub fn fem_edges(&self) -> Vec<u32> {
        self.edges.iter().flat_map(|e| [e[0] as u32, e[1] as u32]).collect()
    }
    /// Grain centers as flat `[x, z, …]`.
    pub fn grains(&self) -> Vec<f64> {
        self.sim.dem.grains.iter().flat_map(|g| [g.x.x, g.x.z]).collect()
    }
    pub fn grain_radius(&self) -> f64 {
        self.sim.dem.grains.first().map(|g| g.r).unwrap_or(0.08)
    }
    pub fn kinetic_energy(&self) -> f64 {
        self.sim.kinetic_energy()
    }
}

impl Default for CoupledLab {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------------------
// Grown circuits — a base case plus a recursive rule grow an adder of any width, the graph is
// lowered to transistors, and the analog voltages decide whether it computes. All on your own
// device: the growth, the lowering and the nonlinear solve run here, in this page.
// ---------------------------------------------------------------------------------------------
#[wasm_bindgen]
pub struct MorphoLab {
    grown: ferromotion_circuit::morpho::Grown,
    tech: ferromotion_circuit::morpho::Tech,
    solved: Option<ferromotion_circuit::morpho::Solved>,
    bits: usize,
    last_x: u32,
    last_y: u32,
}

#[wasm_bindgen]
impl MorphoLab {
    /// Grow an adder of `bits` width. The recursion is the entire description.
    #[wasm_bindgen(constructor)]
    pub fn new(bits: usize) -> MorphoLab {
        let bits = bits.clamp(1, 8);
        MorphoLab {
            grown: ferromotion_circuit::morpho::grow_adder(bits),
            tech: ferromotion_circuit::morpho::Tech::default(),
            solved: None,
            bits,
            last_x: 0,
            last_y: 0,
        }
    }

    /// Drive `x + y` through the grown transistors and solve. Returns true when the analog voltages
    /// read back as the right number.
    pub fn solve(&mut self, x: u32, y: u32) -> bool {
        let steps = ferromotion_circuit::morpho::recommended_steps(self.gate_count());
        // warm-start from the previous operating point: changing an input is a small perturbation, so
        // this converges in a few iterations instead of walking the supply up again
        let warm = self.solved.as_ref().map(|s| s.state.clone());
        let s = self.grown.evaluate_warm(x, y, self.tech, steps, warm.as_ref());
        let ok = s.value == x + y;
        self.solved = Some(s);
        self.last_x = x;
        self.last_y = y;
        ok
    }

    pub fn bits(&self) -> usize {
        self.bits
    }
    /// Gates the rule grew (two transistors each).
    pub fn gate_count(&self) -> usize {
        self.grown.netlist.gates().len()
    }
    /// Output voltage of every grown gate, in growth order: the circuit lit by its own solution.
    pub fn gate_voltages(&self) -> Vec<f64> {
        self.solved.as_ref().map(|s| s.gate_v.clone()).unwrap_or_default()
    }
    /// Voltage of each sum bit, then the carry out.
    pub fn output_voltages(&self) -> Vec<f64> {
        self.solved.as_ref().map(|s| s.out_v.clone()).unwrap_or_default()
    }
    /// The number read off the output nodes.
    pub fn value(&self) -> u32 {
        self.solved.as_ref().map(|s| s.value).unwrap_or(0)
    }
    pub fn worst_high(&self) -> f64 {
        self.solved.as_ref().map(|s| s.worst_high).unwrap_or(f64::NAN)
    }
    pub fn worst_low(&self) -> f64 {
        self.solved.as_ref().map(|s| s.worst_low).unwrap_or(f64::NAN)
    }
    /// How well the nonlinear solve converged. Reported next to the answer on purpose.
    pub fn residual(&self) -> f64 {
        self.solved.as_ref().map(|s| s.residual).unwrap_or(f64::NAN)
    }
    pub fn vdd(&self) -> f64 {
        self.tech.vdd
    }
    pub fn vth(&self) -> f64 {
        self.tech.vth
    }
    /// The analytic logic-low a conducting gate should produce: the resistor divider.
    pub fn predicted_low(&self) -> f64 {
        self.tech.predicted_low()
    }

    // --- topology, so the page can draw the graph that grew ---

    /// Every gate's wiring, flat `[in_a, in_b, out, …]` node ids in growth order. Reading these in
    /// order is watching the recursion unfold: each gate appears only after the gates it listens to.
    pub fn gate_wires(&self) -> Vec<u32> {
        self.grown.netlist.gates().iter().flat_map(|&(a, b, out)| [a as u32, b as u32, out as u32]).collect()
    }
    /// Total node count, ground and rail included.
    pub fn node_count(&self) -> usize {
        self.grown.netlist.n_nodes()
    }
    /// The primary input nodes: the `a` bits then the `b` bits, least significant first.
    pub fn input_nodes(&self) -> Vec<u32> {
        self.grown.a.iter().chain(self.grown.b.iter()).map(|&n| n as u32).collect()
    }
    /// The output nodes: the sum bits least significant first, then the carry out.
    pub fn output_nodes(&self) -> Vec<u32> {
        self.grown.sum.iter().chain(std::iter::once(&self.grown.cout)).map(|&n| n as u32).collect()
    }
    /// Solved voltage of a single node, so the drawing can colour inputs and wires too.
    pub fn node_voltage(&self, node: usize) -> f64 {
        if node == 0 {
            return 0.0;
        }
        if node == 1 {
            return self.tech.vdd;
        }
        // primary inputs are driven, gates are solved; find whichever this is
        if let Some(i) = self.grown.a.iter().position(|&n| n == node) {
            return if (self.last_x >> i) & 1 == 1 { self.tech.vdd } else { 0.0 };
        }
        if let Some(i) = self.grown.b.iter().position(|&n| n == node) {
            return if (self.last_y >> i) & 1 == 1 { self.tech.vdd } else { 0.0 };
        }
        self.grown
            .netlist
            .gates()
            .iter()
            .position(|&(_, _, out)| out == node)
            .and_then(|g| self.solved.as_ref().map(|s| s.gate_v[g]))
            .unwrap_or(0.0)
    }
}
