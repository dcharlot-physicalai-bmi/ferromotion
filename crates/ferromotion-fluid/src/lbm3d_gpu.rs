//! **The fabric GPU path for the D3Q19 lattice** (Honest Fluids — the 2-D GPU path into three
//! dimensions). Same collide-then-stream scheme as the verified [`crate::lbm3d`] CPU reference, as
//! two WGSL compute dispatches per step over ping-ponged f32 buffers on the 19-velocity 3-D
//! stencil — so the CPU solver is the oracle and the check is ordering-exact. wgpu-portable
//! (Metal/Vulkan/DX12 native, WebGPU in the browser), feature `gpu`. Periodic boundaries.

use wgpu::util::DeviceExt;

const WGSL: &str = r#"
struct Params { nx: u32, ny: u32, nz: u32, omega: f32 }

@group(0) @binding(0) var<storage, read_write> fa: array<f32>;
@group(0) @binding(1) var<storage, read_write> fb: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

const CX = array<i32,19>(0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0);
const CY = array<i32,19>(0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1);
const CZ = array<i32,19>(0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1);
const W = array<f32,19>(
    0.3333333333333333,
    0.05555555555555555, 0.05555555555555555, 0.05555555555555555,
    0.05555555555555555, 0.05555555555555555, 0.05555555555555555,
    0.027777777777777776, 0.027777777777777776, 0.027777777777777776, 0.027777777777777776,
    0.027777777777777776, 0.027777777777777776, 0.027777777777777776, 0.027777777777777776,
    0.027777777777777776, 0.027777777777777776, 0.027777777777777776, 0.027777777777777776);

fn feq(k: u32, rho: f32, ux: f32, uy: f32, uz: f32) -> f32 {
    let cu = f32(CX[k]) * ux + f32(CY[k]) * uy + f32(CZ[k]) * uz;
    let uu = ux * ux + uy * uy + uz * uz;
    return W[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu);
}

@compute @workgroup_size(64)
fn collide(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let cell = gid.y * (nwg.x * 64u) + gid.x;
    if (cell >= p.nx * p.ny * p.nz) { return; }
    let base = cell * 19u;
    var rho = 0.0; var mx = 0.0; var my = 0.0; var mz = 0.0;
    for (var k = 0u; k < 19u; k++) {
        let fk = fa[base + k];
        rho += fk; mx += fk * f32(CX[k]); my += fk * f32(CY[k]); mz += fk * f32(CZ[k]);
    }
    let ux = mx / rho; let uy = my / rho; let uz = mz / rho;
    for (var k = 0u; k < 19u; k++) {
        fa[base + k] += p.omega * (feq(k, rho, ux, uy, uz) - fa[base + k]);
    }
}

@compute @workgroup_size(64)
fn stream(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let cell = gid.y * (nwg.x * 64u) + gid.x;
    if (cell >= p.nx * p.ny * p.nz) { return; }
    let x = i32(cell / (p.ny * p.nz));
    let y = i32((cell / p.nz) % p.ny);
    let z = i32(cell % p.nz);
    let base = cell * 19u;
    let nxi = i32(p.nx); let nyi = i32(p.ny); let nzi = i32(p.nz);
    for (var k = 0u; k < 19u; k++) {
        let xn = u32((x + CX[k] + nxi) % nxi);
        let yn = u32((y + CY[k] + nyi) % nyi);
        let zn = u32((z + CZ[k] + nzi) % nzi);
        fb[((xn * p.ny + yn) * p.nz + zn) * 19u + k] = fa[base + k];
    }
}
"#;

/// A D3Q19 BGK lattice on the GPU — same scheme + periodic boundaries as [`crate::lbm3d::LbmD3Q19`].
pub struct LbmGpu3 {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso_collide: wgpu::ComputePipeline,
    pso_stream: wgpu::ComputePipeline,
    binds: [wgpu::BindGroup; 2],
    bufs: [wgpu::Buffer; 2],
    _params: wgpu::Buffer,
    staging: wgpu::Buffer,
    parity: std::cell::Cell<usize>,
}

impl LbmGpu3 {
    /// Build on the first available adapter; `None` when the platform has no usable GPU.
    pub fn new(nx: usize, ny: usize, nz: usize, tau: f64) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let desc = wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
                max_buffer_size: adapter.limits().max_buffer_size,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc)).ok()?;

        let n = nx * ny * nz * 19;
        let w = crate::lbm3d::W;
        let mut init = vec![0.0f32; n];
        for c in init.chunks_mut(19) {
            for (k, v) in c.iter_mut().enumerate() {
                *v = w[k] as f32;
            }
        }
        let mk_buf = |label, data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            })
        };
        let buf_a = mk_buf("lbm3-a", &init);
        let buf_b = mk_buf("lbm3-b", &init);
        let params: [u32; 4] = [nx as u32, ny as u32, nz as u32, (1.0f32 / tau as f32).to_bits()];
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lbm3-params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lbm3-staging"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lbm3"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lbm3-bgl"),
            entries: &[
                storage(0),
                storage(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lbm3-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let mk_pso = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pso_collide = mk_pso("collide");
        let pso_stream = mk_pso("stream");
        let mk_bind = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lbm3-bind"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                ],
            })
        };
        let binds = [mk_bind(&buf_a, &buf_b), mk_bind(&buf_b, &buf_a)];
        Some(Self { nx, ny, nz, device, queue, pso_collide, pso_stream, binds, bufs: [buf_a, buf_b], _params: params, staging, parity: std::cell::Cell::new(0) })
    }

    /// Upload a velocity field (equilibrium at unit density), mirroring the CPU `set_velocity`.
    pub fn set_velocity(&self, fu: impl Fn(f64, f64, f64) -> (f64, f64, f64)) {
        let mut host = vec![0.0f32; self.nx * self.ny * self.nz * 19];
        for x in 0..self.nx {
            for y in 0..self.ny {
                for z in 0..self.nz {
                    let (ux, uy, uz) = fu(x as f64, y as f64, z as f64);
                    let base = ((x * self.ny + y) * self.nz + z) * 19;
                    for k in 0..19 {
                        host[base + k] = crate::lbm3d::feq3::<f64>(k, 1.0, ux, uy, uz) as f32;
                    }
                }
            }
        }
        self.queue.write_buffer(&self.bufs[0], 0, bytemuck::cast_slice(&host));
        self.parity.set(0);
    }

    /// Run `steps` collide+stream cycles (true ping-pong, chunked submits, 2-D dispatch past the cap).
    pub fn run(&self, steps: usize) {
        let cells = (self.nx * self.ny * self.nz) as u32;
        let groups = cells.div_ceil(64);
        let gy = groups.div_ceil(65535);
        let gx = groups.div_ceil(gy);
        let mut done = 0;
        while done < steps {
            let chunk = (steps - done).min(256);
            let mut enc = self.device.create_command_encoder(&Default::default());
            for i in 0..chunk {
                let bind = &self.binds[(self.parity.get() + i) % 2];
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pso_collide);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
                pass.set_pipeline(&self.pso_stream);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            self.queue.submit([enc.finish()]);
            self.parity.set((self.parity.get() + chunk) % 2);
            done += chunk;
        }
    }

    /// Read back the distributions (blocking) and return macroscopic velocity per cell (row-major).
    pub fn velocities(&self) -> Vec<(f64, f64, f64)> {
        let n = self.nx * self.ny * self.nz * 19;
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.bufs[self.parity.get()], 0, &self.staging, 0, (n * 4) as u64);
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
        data.chunks(19)
            .map(|c| {
                let (mut rho, mut mx, mut my, mut mz) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for (k, &f) in c.iter().enumerate() {
                    rho += f as f64;
                    mx += f as f64 * crate::lbm3d::CX[k] as f64;
                    my += f as f64 * crate::lbm3d::CY[k] as f64;
                    mz += f as f64 * crate::lbm3d::CZ[k] as f64;
                }
                (mx / rho, my / rho, mz / rho)
            })
            .collect()
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use crate::lbm3d::{GenLbm3, Lbm3Bc};
    use std::f64::consts::PI;

    /// The GPU D3Q19 lattice against the analytic shear-wave decay AND the CPU reference trajectory
    /// (same scheme, same ordering) — the cross-oracle at f32 tolerance.
    #[test]
    fn gpu_matches_analytic_and_cpu_reference() {
        let n = 24;
        let tau = 0.8;
        let Some(g) = LbmGpu3::new(n, n, n, tau) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let (u0, k) = (0.04, 2.0 * PI / n as f64);
        let init = |_x: f64, _y: f64, z: f64| (u0 * (k * z).sin(), 0.0, 0.0);
        g.set_velocity(init);
        let steps = 150;
        g.run(steps);
        let vel = g.velocities();

        let mut c = GenLbm3::new(n, n, n, tau, Lbm3Bc::Periodic);
        c.set_velocity(|_, _, z| (u0 * (k * z).sin(), 0.0, 0.0));
        for _ in 0..steps {
            c.step();
        }
        let nu = (tau - 0.5) / 3.0;
        let decay = (-nu * k * k * steps as f64).exp();
        let (mut err_ana, mut err_cpu) = (0.0f64, 0.0f64);
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let gx = vel[(x * n + y) * n + z].0;
                    let want = u0 * (k * z as f64).sin() * decay;
                    err_ana = err_ana.max((gx - want).abs());
                    let (_, cx, _, _) = c.macroscopic(x, y, z);
                    err_cpu = err_cpu.max((gx - cx).abs());
                }
            }
        }
        eprintln!("GPU D3Q19 shear wave: vs analytic {:.2e}, vs CPU reference {:.2e} (rel)", err_ana / u0, err_cpu / u0);
        assert!(err_ana / u0 < 0.05, "GPU vs analytic: {}", err_ana / u0);
        assert!(err_cpu / u0 < 1e-4, "GPU vs CPU reference: {}", err_cpu / u0);
    }
}
