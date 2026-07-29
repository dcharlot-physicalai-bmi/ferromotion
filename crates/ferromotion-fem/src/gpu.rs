//! **The wgpu GPU path for the volumetric FEM** (the *optimize* leg): the same per-tet elastic force
//! and per-vertex semi-implicit-Euler integration as the verified CPU reference ([`FemSim::step`]),
//! as two WGSL compute dispatches per step. WebGPU/WGSL has no f32 atomics, so the tet-force scatter
//! is a precomputed per-vertex GATHER over each vertex's incident tets (CSR adjacency). wgpu-portable
//! by construction: Metal / Vulkan / DX12 native and WebGPU in the browser — the hardware-open GPU
//! soft body. Feature `gpu`. f32 on the GPU (verification tolerances account for it vs the f64 core).

use crate::FemSim;
use nalgebra::Vector3;
use wgpu::util::DeviceExt;

const WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> TETS: array<u32>;        // 4 vertex indices per tet
@group(0) @binding(1) var<storage, read> GEO: array<f32>;         // 10 per tet: 9 = Dm^-1 (col-major), 1 = |rest vol|
@group(0) @binding(2) var<storage, read_write> X: array<f32>;     // 3 per vertex
@group(0) @binding(3) var<storage, read_write> V: array<f32>;     // 3 per vertex
@group(0) @binding(4) var<storage, read_write> TF: array<f32>;    // 12 per tet: 4 nodal forces
@group(0) @binding(5) var<storage, read> ADJ: array<u32>;         // [nV+1 header of offsets past the header][packed (tet<<2|corner)]
@group(0) @binding(6) var<storage, read> PIN: array<u32>;         // 1 = pinned
@group(0) @binding(7) var<storage, read> PRM: array<f32>;         // see FemGpu::params

fn gx(i: u32) -> vec3<f32> { return vec3<f32>(X[i*3u], X[i*3u+1u], X[i*3u+2u]); }
fn gv(i: u32) -> vec3<f32> { return vec3<f32>(V[i*3u], V[i*3u+1u], V[i*3u+2u]); }

fn invT3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> mat3x3<f32> {
  let det = dot(a, cross(b, c));
  return mat3x3<f32>(cross(b, c), cross(c, a), cross(a, b)) * (1.0 / det);
}

@compute @workgroup_size(64)
fn force_kernel(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let e = g.y * (nwg.x * 64u) + g.x;
  let nT = u32(PRM[8]);
  if (e >= nT) { return; }
  let mu = PRM[3]; let lambda = PRM[4];
  let i0 = TETS[e*4u]; let i1 = TETS[e*4u+1u]; let i2 = TETS[e*4u+2u]; let i3 = TETS[e*4u+3u];
  let x0 = gx(i0); let x1 = gx(i1); let x2 = gx(i2); let x3 = gx(i3);
  let ds = mat3x3<f32>(x1 - x0, x2 - x0, x3 - x0);
  let b = e*10u;
  let dm = mat3x3<f32>(
    vec3<f32>(GEO[b],     GEO[b+1u], GEO[b+2u]),
    vec3<f32>(GEO[b+3u],  GEO[b+4u], GEO[b+5u]),
    vec3<f32>(GEO[b+6u],  GEO[b+7u], GEO[b+8u]));
  let vol = GEO[b+9u];
  let f = ds * dm;
  let jval = determinant(f);
  var f0 = vec3<f32>(0.0); var f1 = vec3<f32>(0.0); var f2 = vec3<f32>(0.0); var f3 = vec3<f32>(0.0);
  if (jval > 0.0) {
    let fit = invT3(f[0], f[1], f[2]);                 // F^-T
    let p = mu * (f - fit) + (lambda * log(jval)) * fit;
    let h = (-vol) * (p * transpose(dm));              // H = -V0 * P * Dm^-T, columns are nodal forces
    f1 = h[0]; f2 = h[1]; f3 = h[2];
    f0 = -(f1 + f2 + f3);
  }
  let o = e*12u;
  TF[o]      = f0.x; TF[o+1u]  = f0.y; TF[o+2u]  = f0.z;
  TF[o+3u]   = f1.x; TF[o+4u]  = f1.y; TF[o+5u]  = f1.z;
  TF[o+6u]   = f2.x; TF[o+7u]  = f2.y; TF[o+8u]  = f2.z;
  TF[o+9u]   = f3.x; TF[o+10u] = f3.y; TF[o+11u] = f3.z;
}

@compute @workgroup_size(64)
fn integrate_kernel(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let v = g.y * (nwg.x * 64u) + g.x;
  let nV = u32(PRM[9]);
  if (v >= nV) { return; }
  if (PIN[v] == 1u) { V[v*3u] = 0.0; V[v*3u+1u] = 0.0; V[v*3u+2u] = 0.0; return; }
  let mass = PRM[0]; let dt = PRM[1]; let damping = PRM[2];
  let grav = vec3<f32>(PRM[5], PRM[6], PRM[7]);
  let xx0 = gx(v); let vv0 = gv(v);
  // gather the elastic force from every incident tet (atomic-free scatter)
  var fsum = vec3<f32>(0.0);
  let s = ADJ[v]; let e = ADJ[v+1u];
  for (var k = s; k < e; k = k + 1u) {
    let packed = ADJ[k];
    let o = (packed >> 2u)*12u + (packed & 3u)*3u;
    fsum = fsum + vec3<f32>(TF[o], TF[o+1u], TF[o+2u]);
  }
  // optional penalty floor at z = PRM[10] (spring-dashpot), matching FemSim::step
  if (PRM[13] > 0.5) {
    let pen = PRM[10] - xx0.z;
    if (pen > 0.0) {
      let vn = min(vv0.z, 0.0);
      fsum.z = fsum.z + PRM[12] * pen - PRM[11] * vn;   // k_contact*pen - gamma*vn
    }
  }
  let a = fsum * (1.0 / mass) + grav;
  let vv = (vv0 + dt * a) * (1.0 - damping);
  let xx = xx0 + dt * vv;
  V[v*3u] = vv.x; V[v*3u+1u] = vv.y; V[v*3u+2u] = vv.z;
  X[v*3u] = xx.x; X[v*3u+1u] = xx.y; X[v*3u+2u] = xx.z;
}
"#;

/// A tetrahedral Neo-Hookean soft body stepped on the GPU — the same scheme as [`FemSim`].
pub struct FemGpu {
    n_verts: usize,
    n_tets: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso_force: wgpu::ComputePipeline,
    pso_integrate: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    x_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl FemGpu {
    /// Build a GPU stepper mirroring `sim`'s current state, mesh, material, and (optional) floor.
    /// `None` when the platform has no usable GPU.
    pub fn from_sim(sim: &FemSim) -> Option<Self> {
        let n_verts = sim.n_verts();
        let n_tets = sim.n_tets();

        let x0: Vec<f32> = sim.x.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
        let v0: Vec<f32> = sim.v.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
        let tets: Vec<u32> =
            sim.tets().iter().flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32, t[3] as u32]).collect();
        let mut geo = Vec::with_capacity(n_tets * 10);
        for (m, vol) in sim.dm_inv().iter().zip(sim.vol()) {
            geo.extend(m.as_slice().iter().map(|&v| v as f32)); // 9, column-major
            geo.push(vol.abs() as f32);
        }
        let pin: Vec<u32> = sim.pinned.iter().map(|&p| p as u32).collect();

        // per-vertex CSR adjacency of incident (tet, corner), packed into one u32 buffer:
        // [nV+1 header of offsets past the header][packed (tet<<2 | corner)]
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n_verts];
        for (e, t) in sim.tets().iter().enumerate() {
            for (corner, &vtx) in t.iter().enumerate() {
                buckets[vtx].push(((e as u32) << 2) | corner as u32);
            }
        }
        let header = (n_verts + 1) as u32;
        let mut adj: Vec<u32> = vec![0; n_verts + 1];
        let mut flat: Vec<u32> = Vec::new();
        for (v, b) in buckets.iter().enumerate() {
            adj[v] = header + flat.len() as u32;
            flat.extend_from_slice(b);
        }
        adj[n_verts] = header + flat.len() as u32;
        adj.extend_from_slice(&flat);

        let gamma = 0.7 * (sim.k_contact * sim.mass).sqrt();
        let (floor_z, has_floor) = match sim.floor {
            Some(z) => (z as f32, 1.0f32),
            None => (0.0, 0.0),
        };
        let prm: [f32; 14] = [
            sim.mass as f32, sim.dt as f32, sim.damping as f32, sim.mu as f32, sim.lambda as f32,
            sim.gravity.x as f32, sim.gravity.y as f32, sim.gravity.z as f32,
            n_tets as f32, n_verts as f32,
            floor_z, gamma as f32, sim.k_contact as f32, has_floor,
        ];

        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;

        let storage_init = |label, data: &[u8], extra: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE | extra,
            })
        };
        let none = wgpu::BufferUsages::empty();
        let tets_buf = storage_init("fem-tets", bytemuck::cast_slice(&tets), none);
        let geo_buf = storage_init("fem-geo", bytemuck::cast_slice(&geo), none);
        let x_buf = storage_init("fem-x", bytemuck::cast_slice(&x0), wgpu::BufferUsages::COPY_SRC);
        let v_buf = storage_init("fem-v", bytemuck::cast_slice(&v0), none);
        let tf_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fem-tf"),
            size: (n_tets * 12 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let adj_buf = storage_init("fem-adj", bytemuck::cast_slice(&adj), none);
        let pin_buf = storage_init("fem-pin", bytemuck::cast_slice(&pin), none);
        let prm_buf = storage_init("fem-prm", bytemuck::cast_slice(&prm), none);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fem-staging"),
            size: (n_verts * 3 * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fem"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        // Explicit shared layout: `force_kernel` never touches V/ADJ/PIN and `integrate_kernel` never
        // touches TETS/GEO, so an auto layout would derive a *different* group per entry point — one
        // 8-binding layout keeps a single bind group valid for both passes.
        let rw = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fem-bgl"),
            entries: &[ro(0), ro(1), rw(2), rw(3), rw(4), ro(5), ro(6), ro(7)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fem-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pso_force = mk("force_kernel");
        let pso_integrate = mk("integrate_kernel");
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fem-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: geo_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: tf_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: adj_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pin_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: prm_buf.as_entire_binding() },
            ],
        });

        Some(Self { n_verts, n_tets, device, queue, pso_force, pso_integrate, bind, x_buf, staging })
    }

    /// Advance `steps` collide-free FEM steps (force then integrate). Encoders are chunked so long
    /// runs don't build one giant command buffer.
    pub fn run(&self, steps: usize) {
        let f_groups = (self.n_tets as u32).div_ceil(64);
        let i_groups = (self.n_verts as u32).div_ceil(64);
        let dims = |groups: u32| {
            let gy = groups.div_ceil(65535);
            (groups.div_ceil(gy), gy)
        };
        let (fx, fy) = dims(f_groups);
        let (ix, iy) = dims(i_groups);
        let mut done = 0;
        while done < steps {
            let chunk = (steps - done).min(512);
            let mut enc = self.device.create_command_encoder(&Default::default());
            for _ in 0..chunk {
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.pso_force);
                    pass.set_bind_group(0, &self.bind, &[]);
                    pass.dispatch_workgroups(fx, fy, 1);
                }
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.pso_integrate);
                    pass.set_bind_group(0, &self.bind, &[]);
                    pass.dispatch_workgroups(ix, iy, 1);
                }
            }
            self.queue.submit([enc.finish()]);
            done += chunk;
        }
    }

    /// Read back the vertex positions (blocking).
    pub fn positions(&self) -> Vec<Vector3<f64>> {
        let n = self.n_verts * 3;
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.x_buf, 0, &self.staging, 0, (n * 4) as u64);
        self.queue.submit([enc.finish()]);
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let view = slice.get_mapped_range().expect("mapped range");
        let data: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        self.staging.unmap();
        data.chunks(3).map(|c| Vector3::new(c[0] as f64, c[1] as f64, c[2] as f64)).collect()
    }
}

#[cfg(test)]
mod verification {
    use super::*;

    /// The GPU stepper reproduces the verified CPU `FemSim::step` trajectory (f32 vs f64) for a
    /// pinned-top jelly cube kicked sideways — the cross-oracle at f32 tolerance.
    #[test]
    fn gpu_matches_cpu_pinned_cube() {
        let mut sim = FemSim::box_grid(3, 3, 3, 0.1, 0.02, 3.0e3, 1.5e3, 2.0e-4);
        sim.damping = 0.01;
        sim.gravity = Vector3::new(0.0, 0.0, -9.81);
        sim.floor = None;
        let zmax = sim.x.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max);
        for i in 0..sim.n_verts() {
            if sim.x[i].z > zmax - 1e-6 {
                sim.pinned[i] = true;
            } else {
                sim.v[i] = Vector3::new(0.9, 0.3, 0.0);
            }
        }
        let Some(g) = FemGpu::from_sim(&sim) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let steps = 200;
        g.run(steps);
        let gpu = g.positions();
        for _ in 0..steps {
            sim.step();
        }
        let mut worst = 0.0f64;
        for i in 0..sim.n_verts() {
            worst = worst.max((gpu[i] - sim.x[i]).norm());
        }
        eprintln!("GPU vs CPU FEM (pinned cube, {steps} steps): worst vertex {worst:.3e} m");
        assert!(worst < 1e-3, "GPU FEM diverged from the CPU reference: {worst}");
    }

    /// The GPU stepper with the penalty floor: a cube dropped onto the ground lands and settles the
    /// same as the CPU reference.
    #[test]
    fn gpu_matches_cpu_dropped_cube_with_floor() {
        let mut sim = FemSim::box_grid(3, 3, 3, 0.08, 0.02, 2.0e3, 1.0e3, 1.5e-4);
        sim.damping = 0.02;
        sim.gravity = Vector3::new(0.0, 0.0, -9.81);
        sim.floor = Some(0.0);
        sim.k_contact = 4.0e3;
        for p in sim.x.iter_mut() {
            p.z += 0.05; // lift above the floor, then drop
        }
        let Some(g) = FemGpu::from_sim(&sim) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let steps = 300;
        g.run(steps);
        let gpu = g.positions();
        for _ in 0..steps {
            sim.step();
        }
        let mut worst = 0.0f64;
        for i in 0..sim.n_verts() {
            worst = worst.max((gpu[i] - sim.x[i]).norm());
        }
        eprintln!("GPU vs CPU FEM (dropped cube + floor, {steps} steps): worst vertex {worst:.3e} m");
        assert!(worst < 2e-3, "GPU FEM (floor) diverged from the CPU reference: {worst}");
    }
}
