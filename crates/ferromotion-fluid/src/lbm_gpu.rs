//! **The fabric GPU path for the D2Q9 lattice** (Honest Fluids, the *optimize* leg): the same
//! collide-then-stream scheme as the verified CPU reference, as two WGSL compute dispatches per
//! step over ping-ponged f32 buffers — so the CPU solver is the oracle and the comparison is
//! ordering-exact, not merely statistical. wgpu-portable by construction: Metal/Vulkan/DX12
//! native and WebGPU in the browser — the hardware-open answer to OpenCL-bound, license-closed
//! GPU LBM incumbents. Feature `gpu`.
//!
//! f32 on the lattice (the GPU baseline everywhere); verification tolerances account for it.
//! Boundary handling matches the CPU reference: fully periodic, or bounce-back cavity walls with
//! the momentum-corrected moving lid.

use crate::lbm::LbmBc;
use wgpu::util::DeviceExt;

const WGSL: &str = r#"
struct Params {
    nx: u32,
    ny: u32,
    omega: f32,
    lid_u: f32,
    cavity: u32, // 0 = periodic, 1 = cavity
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> fa: array<f32>; // collide in place
@group(0) @binding(1) var<storage, read_write> fb: array<f32>; // stream target
@group(0) @binding(2) var<uniform> p: Params;

const CX = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const CY = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
const W = array<f32, 9>(0.4444444444444444, 0.1111111111111111, 0.1111111111111111,
    0.1111111111111111, 0.1111111111111111, 0.0277777777777778, 0.0277777777777778,
    0.0277777777777778, 0.0277777777777778);
const OPP = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);

fn feq(k: u32, rho: f32, ux: f32, uy: f32) -> f32 {
    let cu = f32(CX[k]) * ux + f32(CY[k]) * uy;
    let uu = ux * ux + uy * uy;
    return W[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu);
}

@compute @workgroup_size(64)
fn collide(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let cell = gid.y * (nwg.x * 64u) + gid.x;
    if (cell >= p.nx * p.ny) { return; }
    let base = cell * 9u;
    var rho = 0.0;
    var mx = 0.0;
    var my = 0.0;
    for (var k = 0u; k < 9u; k++) {
        let fk = fa[base + k];
        rho += fk;
        mx += fk * f32(CX[k]);
        my += fk * f32(CY[k]);
    }
    let ux = mx / rho;
    let uy = my / rho;
    for (var k = 0u; k < 9u; k++) {
        fa[base + k] += p.omega * (feq(k, rho, ux, uy) - fa[base + k]);
    }
}

@compute @workgroup_size(64)
fn stream(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(num_workgroups) nwg: vec3<u32>) {
    let cell = gid.y * (nwg.x * 64u) + gid.x;
    if (cell >= p.nx * p.ny) { return; }
    let x = i32(cell / p.ny);
    let y = i32(cell % p.ny);
    let base = cell * 9u;
    let nxi = i32(p.nx);
    let nyi = i32(p.ny);
    for (var k = 0u; k < 9u; k++) {
        var xn = x + CX[k];
        var yn = y + CY[k];
        if (p.cavity == 0u) {
            xn = (xn + nxi) % nxi;
            yn = (yn + nyi) % nyi;
            fb[(u32(xn) * p.ny + u32(yn)) * 9u + k] = fa[base + k];
        } else {
            if (xn < 0 || xn >= nxi || yn < 0 || yn >= nyi) {
                var v = fa[base + k];
                if (yn >= nyi) {
                    v -= 6.0 * W[k] * f32(CX[k]) * p.lid_u;
                }
                fb[base + OPP[k]] = v;
            } else {
                fb[(u32(xn) * p.ny + u32(yn)) * 9u + k] = fa[base + k];
            }
        }
    }
}
"#;

/// A D2Q9 BGK lattice on the GPU — same scheme, same boundaries as [`crate::lbm::LbmD2Q9`].
pub struct LbmGpu {
    pub nx: usize,
    pub ny: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso_collide: wgpu::ComputePipeline,
    pso_stream: wgpu::ComputePipeline,
    binds: [wgpu::BindGroup; 2],
    bufs: [wgpu::Buffer; 2],
    _params: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// Which buffer currently holds the state (flips every step — true ping-pong, no copies).
    parity: std::cell::Cell<usize>,
}

impl LbmGpu {
    /// Build on the first available adapter; `None` when the platform has no usable GPU.
    pub fn new(nx: usize, ny: usize, tau: f64, bc: LbmBc) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        // Ask for the adapter's real storage-binding ceiling — the default 128MiB caps the
        // lattice at ~1930² (nx·ny·9·4 bytes in one binding).
        let desc = wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
                max_buffer_size: adapter.limits().max_buffer_size,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc)).ok()?;

        let n = nx * ny * 9;
        let mut init = vec![0.0f32; n];
        for c in init.chunks_mut(9) {
            for (k, v) in c.iter_mut().enumerate() {
                *v = crate::lbm::W[k] as f32;
            }
        }
        let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lbm-a"),
            contents: bytemuck::cast_slice(&init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lbm-b"),
            contents: bytemuck::cast_slice(&init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let (cavity, lid_u) = match bc {
            LbmBc::Periodic => (0u32, 0.0f32),
            LbmBc::Cavity { lid_u } => (1u32, lid_u as f32),
        };
        let params: [u32; 6] = [
            nx as u32,
            ny as u32,
            (1.0f32 / tau as f32).to_bits(),
            lid_u.to_bits(),
            cavity,
            0,
        ];
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lbm-params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lbm-staging"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lbm"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        // Explicit layout: `collide` never touches fb, so auto-layout would derive a 2-binding
        // group for it — one shared 3-binding layout keeps a single bind group for both passes.
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lbm-bgl"),
            entries: &[
                storage(0),
                storage(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lbm-layout"),
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
                label: Some("lbm-bind"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                ],
            })
        };
        let binds = [mk_bind(&buf_a, &buf_b), mk_bind(&buf_b, &buf_a)];
        Some(Self {
            nx,
            ny,
            device,
            queue,
            pso_collide,
            pso_stream,
            binds,
            bufs: [buf_a, buf_b],
            _params: params,
            staging,
            parity: std::cell::Cell::new(0),
        })
    }

    /// Upload a velocity field (equilibrium at unit density), mirroring the CPU `set_velocity`.
    pub fn set_velocity(&self, fu: impl Fn(f64, f64) -> (f64, f64)) {
        let mut host = vec![0.0f32; self.nx * self.ny * 9];
        for x in 0..self.nx {
            for y in 0..self.ny {
                let (ux, uy) = fu(x as f64, y as f64);
                let base = (x * self.ny + y) * 9;
                for k in 0..9 {
                    host[base + k] = crate::lbm::feq::<f64>(k, 1.0, ux, uy) as f32;
                }
            }
        }
        self.queue.write_buffer(&self.bufs[0], 0, bytemuck::cast_slice(&host));
        self.parity.set(0);
    }

    /// Run `steps` collide+stream cycles. True ping-pong: collide mutates the source buffer in
    /// place, stream scatters into the other, the bind groups swap — zero copies per step.
    /// Encoders are chunked so long runs (steady-state cavity: tens of thousands of steps)
    /// don't build one giant command buffer.
    pub fn run(&self, steps: usize) {
        let n_cells = (self.nx * self.ny) as u32;
        let groups = n_cells.div_ceil(64);
        // 2-D dispatch: the per-dimension workgroup cap is 65535 (hit at 2048^2 with 1-D).
        let gy = groups.div_ceil(65535);
        let gx = groups.div_ceil(gy);
        let mut done = 0;
        while done < steps {
            let chunk = (steps - done).min(512);
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

    /// Read back the full distribution set (blocking) and return macroscopic velocities per cell.
    pub fn velocities(&self) -> Vec<(f64, f64)> {
        let n = self.nx * self.ny * 9;
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
        data.chunks(9)
            .map(|c| {
                let (mut rho, mut mx, mut my) = (0.0f64, 0.0f64, 0.0f64);
                for (k, &f) in c.iter().enumerate() {
                    rho += f as f64;
                    mx += f as f64 * crate::lbm::CX[k] as f64;
                    my += f as f64 * crate::lbm::CY[k] as f64;
                }
                (mx / rho, my / rho)
            })
            .collect()
    }

    /// Mid-x vertical profile of u/lid, mirroring the CPU `centerline_u`.
    pub fn centerline_u(&self, lid: f64) -> Vec<(f64, f64)> {
        let vel = self.velocities();
        let x = self.nx / 2;
        (0..self.ny)
            .map(|y| ((y as f64 + 0.5) / self.ny as f64, vel[x * self.ny + y].0 / lid))
            .collect()
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use std::f64::consts::PI;

    /// The GPU lattice against the analytic Taylor–Green decay AND against the CPU reference
    /// trajectory (same scheme, same ordering) — the cross-oracle at f32 tolerance.
    #[test]
    fn gpu_matches_analytic_and_cpu_reference() {
        let Some(g) = LbmGpu::new(32, 32, 0.8, LbmBc::Periodic) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let n = 32usize;
        let (u0, k) = (0.05, 2.0 * PI / n as f64);
        let init = |x: f64, y: f64| (u0 * (k * x).sin() * (k * y).cos(), -u0 * (k * x).cos() * (k * y).sin());
        g.set_velocity(init);
        let steps = 200;
        g.run(steps);
        let vel = g.velocities();

        // CPU reference, identical setup
        let mut c = crate::lbm::LbmD2Q9::new(n, n, 0.8, LbmBc::Periodic);
        c.set_velocity(init);
        for _ in 0..steps {
            c.step();
        }
        let nu = (0.8 - 0.5) / 3.0;
        let decay = (-2.0 * nu * k * k * steps as f64).exp();
        let (mut err_ana, mut err_cpu) = (0.0f64, 0.0f64);
        for x in 0..n {
            for y in 0..n {
                let (gx, gy) = vel[x * n + y];
                let want_x = u0 * (k * x as f64).sin() * (k * y as f64).cos() * decay;
                let want_y = -u0 * (k * x as f64).cos() * (k * y as f64).sin() * decay;
                err_ana = err_ana.max((gx - want_x).abs().max((gy - want_y).abs()));
                let (_, cx, cy) = c.macroscopic(x, y);
                err_cpu = err_cpu.max((gx - cx).abs().max((gy - cy).abs()));
            }
        }
        assert!(err_ana / u0 < 0.05, "GPU vs analytic: rel {err_ana}");
        assert!(err_cpu / u0 < 1e-4, "GPU vs CPU reference: rel {}", err_cpu / u0);
        eprintln!("GPU TG: vs analytic {:.2e}, vs CPU reference {:.2e} (rel)", err_ana / u0, err_cpu / u0);
    }

    /// The GPU cavity against the Ghia et al. (1982) Re=100 table — the same cross-oracle the
    /// CPU reference passes, at the same grid and Mach number, plus f32 headroom.
    #[test]
    fn gpu_cavity_re100_matches_ghia() {
        const GHIA: &[(f64, f64)] = &[
            (0.0547, -0.03717),
            (0.1016, -0.06434),
            (0.2813, -0.15662),
            (0.4531, -0.21090),
            (0.5000, -0.20581),
            (0.6172, -0.13641),
            (0.7344, 0.00332),
            (0.8516, 0.23151),
            (0.9531, 0.68717),
            (0.9766, 0.84123),
        ];
        let n = 96;
        let lid = 0.1;
        let tau = 3.0 * (lid * n as f64 / 100.0) + 0.5;
        let Some(g) = LbmGpu::new(n, n, tau, LbmBc::Cavity { lid_u: lid }) else {
            eprintln!("no GPU — skipping");
            return;
        };
        g.run((40.0 * n as f64 / lid) as usize);
        let profile = g.centerline_u(lid);
        let interp = |yq: f64| -> f64 {
            for w in profile.windows(2) {
                let ((y0, u0), (y1, u1)) = (w[0], w[1]);
                if (y0..=y1).contains(&yq) {
                    return u0 + (u1 - u0) * (yq - y0) / (y1 - y0);
                }
            }
            profile.last().unwrap().1
        };
        let mut worst = 0.0f64;
        for &(y, u_ref) in GHIA {
            worst = worst.max((interp(y) - u_ref).abs());
        }
        assert!(worst < 0.03, "GPU cavity vs Ghia: worst deviation {worst:.4}");
        eprintln!("GPU cavity Re=100 vs Ghia: worst centerline deviation {worst:.4}");
    }
}
