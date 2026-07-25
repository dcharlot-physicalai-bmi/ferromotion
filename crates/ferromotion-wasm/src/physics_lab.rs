//! **Physics bench** — the wasm rigs surfacing the rest of the new capabilities live: a volumetric
//! Neo-Hookean **soft body** (ferromotion-fem), **granular** media (ferromotion-dem), and a
//! **batched-RL** cartpole swing-up (ferromotion-control). Each carries its verified receipt — the
//! soft body's elastic energy, the pile's kinetic energy decaying to rest, the pole reaching upright
//! by parallel random shooting.

use ferromotion_control::{batch_rollout, Cartpole};
use ferromotion_dem::{DemSim, Grain};
use ferromotion_fem::FemSim;
use nalgebra::Vector3;
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
        sim.damping = 0.004;
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
