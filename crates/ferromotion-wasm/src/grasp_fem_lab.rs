//! **Rigid↔FEM grasp bench** — a parallel-jaw gripper picks up a soft (FEM) block and sets it down.
//! The jaws squeeze the deformable cube, friction (`2·μ·N ≳ m·g`) carries it up as the gripper lifts,
//! and opening the jaws releases it back to the floor — a full pick-and-place on a *deformable*
//! object, all pure Rust → WebAssembly. This is [`ferromotion_coupled::GraspFemSim`] live.

use ferromotion_coupled::GraspFemSim;
use ferromotion_fem::FemSim;
use nalgebra::Vector3;
use std::collections::HashSet;
use wasm_bindgen::prelude::*;

const H: f64 = 0.07;
const SQUEEZE: f64 = 0.02;
const GAP: f64 = 0.03;
const LIFT_V: f64 = 0.5;

fn build() -> (GraspFemSim, Vec<[usize; 2]>, f64, f64) {
    let mut fem = FemSim::box_grid(3, 3, 3, H, 0.02, 1.0e4, 6.0e3, 2.0e-4);
    fem.damping = 0.02;
    let xs: Vec<f64> = fem.x.iter().map(|p| p.x).collect();
    let (xmin, xmax) = (xs.iter().cloned().fold(f64::INFINITY, f64::min), xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let zmin = fem.x.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
    let ymid = fem.x.iter().map(|p| p.y).sum::<f64>() / fem.n_verts() as f64;
    let zmid = fem.x.iter().map(|p| p.z).sum::<f64>() / fem.n_verts() as f64;
    let half_x = 0.5 * (xmax - xmin);
    // unique tet edges for the wireframe (grid proximity)
    let mut set = HashSet::new();
    for a in 0..fem.n_verts() {
        for b in (a + 1)..fem.n_verts() {
            if (fem.x[a] - fem.x[b]).norm() < 1.05 * H {
                set.insert((a, b));
            }
        }
    }
    let edges: Vec<[usize; 2]> = set.into_iter().map(|(a, b)| [a, b]).collect();
    let center = Vector3::new(0.5 * (xmin + xmax), ymid, zmid);
    let mut sim = GraspFemSim::new(fem, center, half_x + GAP, 5.0e4, 0.9); // start open
    sim.floor = Some(zmin); // the block rests on the floor until it is lifted
    (sim, edges, half_x, zmin)
}

#[wasm_bindgen]
pub struct GraspFemLab {
    sim: GraspFemSim,
    edges: Vec<[usize; 2]>,
    half_x: f64,
    floor_z: f64,
    phase: u8, // 0 squeeze, 1 lift, 2 hold, 3 lower, 4 release
    t: u32,
}

// phase durations, in frames
const T_SQUEEZE: u32 = 55;
const T_LIFT: u32 = 60;
const T_HOLD: u32 = 25;
const T_LOWER: u32 = 60;
const T_RELEASE: u32 = 35;

#[wasm_bindgen]
impl GraspFemLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GraspFemLab {
        let (sim, edges, half_x, floor_z) = build();
        GraspFemLab { sim, edges, half_x, floor_z, phase: 0, t: 0 }
    }

    /// Advance one animation frame: `sub` physics steps under the current phase, then advance the
    /// pick-and-place state machine.
    pub fn step(&mut self, sub: usize) {
        // set the gripper command for this phase
        match self.phase {
            0 => {
                // squeeze: close the jaws from open to a firm grip
                let a = (self.t as f64 / T_SQUEEZE as f64).min(1.0);
                self.sim.half_width = (self.half_x + GAP) * (1.0 - a) + (self.half_x - SQUEEZE) * a;
                self.sim.jaw_vel = Vector3::zeros();
            }
            1 => self.sim.jaw_vel = Vector3::new(0.0, 0.0, LIFT_V), // lift
            2 => self.sim.jaw_vel = Vector3::zeros(),               // hold at the top
            3 => self.sim.jaw_vel = Vector3::new(0.0, 0.0, -LIFT_V), // lower back down
            _ => {
                // release: open the jaws, let the block rest on the floor
                let a = (self.t as f64 / T_RELEASE as f64).min(1.0);
                self.sim.half_width = (self.half_x - SQUEEZE) * (1.0 - a) + (self.half_x + GAP) * a;
                self.sim.jaw_vel = Vector3::zeros();
            }
        }
        for _ in 0..sub {
            self.sim.step();
        }
        self.t += 1;
        let dur = [T_SQUEEZE, T_LIFT, T_HOLD, T_LOWER, T_RELEASE][self.phase as usize];
        if self.t >= dur {
            self.t = 0;
            if self.phase >= 4 {
                // fresh block for the next cycle (keeps the demo drift-free)
                let (sim, edges, half_x, floor_z) = build();
                self.sim = sim;
                self.edges = edges;
                self.half_x = half_x;
                self.floor_z = floor_z;
                self.phase = 0;
            } else {
                self.phase += 1;
            }
        }
    }

    /// Soft-body vertices as flat `[x, z, …]` (front x–z view).
    pub fn verts(&self) -> Vec<f64> {
        self.sim.fem.x.iter().flat_map(|p| [p.x, p.z]).collect()
    }
    pub fn edges(&self) -> Vec<u32> {
        self.edges.iter().flat_map(|e| [e[0] as u32, e[1] as u32]).collect()
    }
    /// Jaw draw data: `[left_x, right_x, center_z, pad_half]` — two vertical bars in the x–z view.
    pub fn jaws(&self) -> Vec<f64> {
        vec![self.sim.center.x - self.sim.half_width, self.sim.center.x + self.sim.half_width, self.sim.center.z, 1.6 * self.half_x]
    }
    pub fn floor_z(&self) -> f64 {
        self.floor_z
    }
    pub fn centroid_xz(&self) -> Vec<f64> {
        let c = self.sim.object_centroid();
        vec![c.x, c.z]
    }
    /// Height of the block's centroid above the floor (how far it has been lifted).
    pub fn lift_height(&self) -> f64 {
        self.sim.object_centroid().z - self.floor_z
    }
    pub fn phase(&self) -> u8 {
        self.phase
    }
    /// Whether the block is currently held clear of the floor.
    pub fn held(&self) -> bool {
        self.lift_height() > 1.2 * self.half_x + 0.01
    }
    pub fn n_verts(&self) -> usize {
        self.sim.fem.n_verts()
    }
}

impl Default for GraspFemLab {
    fn default() -> Self {
        Self::new()
    }
}
