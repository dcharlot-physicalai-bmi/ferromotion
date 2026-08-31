//! **The wgpu GPU path for batched rigid-body rollouts** — the on-device-RL axis (Brax/MJX/Newton
//! simulate thousands of environment instances in parallel). [`CartpoleGpu`] rolls a whole batch of
//! candidate action sequences through the cartpole dynamics in one WGSL dispatch: one GPU thread per
//! candidate future, each an independent lane (no cross-lane dependence), reproducing
//! [`crate::batch_rollout`] — the verified CPU reference — on native Metal/Vulkan/DX12 (and WebGPU). The
//! batched primitive behind gradient-free policy search (random shooting / MPPI). Feature `gpu`.

use crate::Cartpole;

const WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> ACTS: array<f32>;      // b candidate forces
@group(0) @binding(1) var<storage, read> PRM: array<f32>;       // [mc,mp,l,g,dt,horizon,b, x,xd,th,thd]
@group(0) @binding(2) var<storage, read_write> FIN: array<f32>; // b × 4 final states
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.y * (nwg.x * 64u) + g.x;
  let mc=PRM[0]; let mp=PRM[1]; let l=PRM[2]; let gg=PRM[3]; let dt=PRM[4];
  let horizon=u32(PRM[5]); let b=u32(PRM[6]);
  if (i >= b) { return; }
  let u = ACTS[i];
  var s0=PRM[7]; var s1=PRM[8]; var s2=PRM[9]; var s3=PRM[10];   // this candidate's own rollout state
  let total = mc + mp;
  for (var k=0u; k<horizon; k=k+1u) {                            // semi-implicit Euler, matches Cartpole::step
    let st=sin(s2); let ct=cos(s2);
    let temp=(u + mp*l*s3*s3*st)/total;
    let thdd=(gg*st - ct*temp)/(l*(4.0/3.0 - mp*ct*ct/total));
    let xdd=temp - mp*l*thdd*ct/total;
    let xd=s1+dt*xdd; let thd=s3+dt*thdd;
    s0=s0+dt*xd; s1=xd; s2=s2+dt*thd; s3=thd;
  }
  let o=i*4u; FIN[o]=s0; FIN[o+1u]=s1; FIN[o+2u]=s2; FIN[o+3u]=s3;
}
"#;

/// A batched cartpole rollout on the GPU — the same scheme as [`crate::batch_rollout`], one thread per
/// candidate. Sized for exactly `b` candidates rolled out `horizon` steps.
pub struct CartpoleGpu {
    cp: Cartpole,
    b: usize,
    horizon: usize,
    dt: f64,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    acts_buf: wgpu::Buffer,
    prm_buf: wgpu::Buffer,
    fin_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl CartpoleGpu {
    /// Build a rollout kernel for `cp`, sized for batches of `b` candidates × `horizon` steps at
    /// timestep `dt`. `None` when there is no usable GPU.
    pub fn new(cp: Cartpole, b: usize, horizon: usize, dt: f64) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;

        let acts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cp-acts"),
            size: (b * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let prm_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cp-prm"),
            size: 11 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fin_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cp-fin"),
            size: (b * 4 * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cp-staging"),
            size: (b * 4 * 4).max(4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cartpole"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let ro = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let mut rw = ro(2);
        rw.ty = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cp-bgl"),
            entries: &[ro(0), ro(1), rw],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cp-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pso = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cartpole"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cp-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: acts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: prm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: fin_buf.as_entire_binding() },
            ],
        });

        Some(Self { cp, b, horizon, dt, device, queue, pso, bind, acts_buf, prm_buf, fin_buf, staging })
    }

    /// Roll every candidate out from `state` under its constant force `acts[i]` for `horizon` steps;
    /// return the packed final states `[x,ẋ,θ,θ̇, …]` (`4·b`). Matches [`crate::batch_rollout`].
    pub fn rollout(&self, state: [f64; 4], acts: &[f64]) -> Vec<f64> {
        assert_eq!(acts.len(), self.b, "action batch size mismatch");
        let acts32: Vec<f32> = acts.iter().map(|&v| v as f32).collect();
        self.queue.write_buffer(&self.acts_buf, 0, bytemuck::cast_slice(&acts32));
        let prm: [f32; 11] = [
            self.cp.mc as f32, self.cp.mp as f32, self.cp.l as f32, self.cp.g as f32, self.dt as f32,
            self.horizon as f32, self.b as f32,
            state[0] as f32, state[1] as f32, state[2] as f32, state[3] as f32,
        ];
        self.queue.write_buffer(&self.prm_buf, 0, bytemuck::cast_slice(&prm));

        let groups = (self.b as u32).div_ceil(64);
        let gy = groups.div_ceil(65535);
        let gx = groups.div_ceil(gy);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pso);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&self.fin_buf, 0, &self.staging, 0, (self.b * 4 * 4) as u64);
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
        data.iter().map(|&v| v as f64).collect()
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use crate::batch_rollout;

    /// The GPU batched rollout reproduces the CPU `batch_rollout` per candidate (f32 vs f64), for a
    /// batch of random constant-force candidates from a hanging-down start — the cross-oracle.
    #[test]
    fn gpu_rollout_matches_cpu() {
        let cp = Cartpole::default();
        let (b, horizon, dt) = (2048usize, 25usize, 0.02f64);
        let state = [0.0, 0.0, std::f64::consts::PI, 0.0];

        // deterministic candidate forces (splitmix64)
        let mut s = 0x1234u64;
        let acts: Vec<f64> = (0..b)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                12.0 * ((((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0)
            })
            .collect();

        let Some(g) = CartpoleGpu::new(cp, b, horizon, dt) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let gpu = g.rollout(state, &acts);

        let x0: Vec<f64> = (0..b).flat_map(|_| state).collect();
        let cpu = batch_rollout(&cp, &x0, &acts, horizon, dt);

        let mut worst = 0.0f64;
        for i in 0..b * 4 {
            worst = worst.max((gpu[i] - cpu[i]).abs());
        }
        eprintln!("GPU vs CPU batch_rollout ({b} candidates × {horizon} steps): worst {worst:.3e}");
        assert!(worst < 1e-3, "GPU rollout diverged from the CPU reference: {worst}");
    }
}
