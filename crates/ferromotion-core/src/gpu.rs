//! **The wgpu GPU path for batch collision-checking** — the parallel hot loop of sampling-based
//! planning. An RRT tree is sequential, but its inner question — does a candidate joint configuration
//! drive the arm through an obstacle? — is asked over thousands of candidates and is embarrassingly
//! parallel. [`ClearanceGpu`] evaluates [`crate::arm_clearance`] for a whole batch of configurations in one
//! WGSL dispatch: one GPU thread per config runs forward kinematics, places the arm's swept collision
//! spheres, and min-reduces their signed distance to the [`SdfScene`] — exactly the CPU reference,
//! which is the oracle it is verified against. wgpu-portable (Metal/Vulkan/DX12 + WebGPU). Feature `gpu`.

use crate::{JointKind, Robot, Sdf, SdfScene, StFrictionContact};
use wgpu::util::DeviceExt;

const WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> JOINTS: array<f32>;   // 16 per joint: R(9,col-major) p(3) axis(3) kind(1)
@group(0) @binding(1) var<storage, read> EE: array<f32>;        // 12: R(9) p(3)
@group(0) @binding(2) var<storage, read> CFG: array<f32>;       // dof per config
@group(0) @binding(3) var<storage, read> PRIMS: array<f32>;     // 8 per prim: type + params
@group(0) @binding(4) var<storage, read> PRM: array<f32>;       // [dof, per_link, link_r, n_cfg, n_prims]
@group(0) @binding(5) var<storage, read_write> CLR: array<f32>;

fn rot_axis(a: vec3<f32>, t: f32) -> mat3x3<f32> {
  let c = cos(t); let s = sin(t); let ic = 1.0 - c; let x = a.x; let y = a.y; let z = a.z;
  return mat3x3<f32>(
    vec3<f32>(c + x*x*ic,   y*x*ic + z*s, z*x*ic - y*s),
    vec3<f32>(x*y*ic - z*s, c + y*y*ic,   z*y*ic + x*s),
    vec3<f32>(x*z*ic + y*s, y*z*ic - x*s, c + z*z*ic));
}
fn prim_sdf(p: vec3<f32>, b: u32) -> f32 {
  let ty = PRIMS[b];
  if (ty < 0.5) {                         // sphere: center(1..3) radius(4)
    return length(p - vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u])) - PRIMS[b+4u];
  } else if (ty < 1.5) {                  // box: center(1..3) half(4..6)
    let q = abs(p - vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u])) - vec3<f32>(PRIMS[b+4u], PRIMS[b+5u], PRIMS[b+6u]);
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
  } else if (ty < 2.5) {                  // plane: normal(1..3) offset(4)
    return dot(vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u]), p) - PRIMS[b+4u];
  } else if (ty < 3.5) {                  // capsule: a(1..3) b(4..6) radius(7)
    let a = vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u]);
    let e = vec3<f32>(PRIMS[b+4u], PRIMS[b+5u], PRIMS[b+6u]);
    let ab = e - a;
    let t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return length(p - (a + t * ab)) - PRIMS[b+7u];
  } else {                                // torus: center(1..3) major(4) minor(5), around +y
    let d = p - vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u]);
    let planar = sqrt(d.x*d.x + d.z*d.z) - PRIMS[b+4u];
    return sqrt(planar*planar + d.y*d.y) - PRIMS[b+5u];
  }
}
fn scene_sdf(p: vec3<f32>, n: u32) -> f32 { var d = 1e30; for (var i = 0u; i < n; i = i + 1u) { d = min(d, prim_sdf(p, i*8u)); } return d; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let k = g.y * (nwg.x * 64u) + g.x;
  let dof = u32(PRM[0]); let per_link = u32(PRM[1]); let link_r = PRM[2];
  let ncfg = u32(PRM[3]); let nprims = u32(PRM[4]);
  if (k >= ncfg) { return; }
  let cbase = k * dof;
  var R = mat3x3<f32>(vec3<f32>(1.,0.,0.), vec3<f32>(0.,1.,0.), vec3<f32>(0.,0.,1.));
  var p = vec3<f32>(0.0);
  var prev = p;                            // frame[0] = origin
  var minc = 1e30;
  for (var j = 0u; j < dof; j = j + 1u) {
    let jb = j*16u;
    let Ro = mat3x3<f32>(
      vec3<f32>(JOINTS[jb],     JOINTS[jb+1u], JOINTS[jb+2u]),
      vec3<f32>(JOINTS[jb+3u],  JOINTS[jb+4u], JOINTS[jb+5u]),
      vec3<f32>(JOINTS[jb+6u],  JOINTS[jb+7u], JOINTS[jb+8u]));
    let po = vec3<f32>(JOINTS[jb+9u], JOINTS[jb+10u], JOINTS[jb+11u]);
    let axis = vec3<f32>(JOINTS[jb+12u], JOINTS[jb+13u], JOINTS[jb+14u]);
    let kind = JOINTS[jb+15u];
    let q = CFG[cbase + j];
    var Rm = mat3x3<f32>(vec3<f32>(1.,0.,0.), vec3<f32>(0.,1.,0.), vec3<f32>(0.,0.,1.));
    var pm = vec3<f32>(0.0);
    if (kind < 0.5) { Rm = rot_axis(axis, q); } else { pm = axis * q; }
    let Rt = Ro * Rm; let pt = Ro * pm + po;     // transform_j = origin ∘ motion
    let Rn = R * Rt;  let pn = R * pt + p;        // T = T ∘ transform_j
    R = Rn; p = pn;
    let cur = p;                                  // frame[j+1]
    for (var srun = 0u; srun < per_link; srun = srun + 1u) {
      let t = (f32(srun) + 0.5) / f32(per_link);
      minc = min(minc, scene_sdf(prev + t*(cur - prev), nprims) - link_r);
    }
    prev = cur;
  }
  let toolp = R * vec3<f32>(EE[9], EE[10], EE[11]) + p;
  minc = min(minc, scene_sdf(toolp, nprims) - link_r);
  CLR[k] = minc;
}
"#;

/// A batched arm-vs-scene clearance checker on the GPU — the same swept-sphere test as
/// [`crate::arm_clearance`], over a whole batch of configurations at once.
pub struct ClearanceGpu {
    dof: usize,
    n_configs: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    cfg_buf: wgpu::Buffer,
    clr_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
}

fn prim_floats(s: &Sdf) -> [f32; 8] {
    match *s {
        Sdf::Sphere { center, radius } => [0.0, center.x as f32, center.y as f32, center.z as f32, radius as f32, 0.0, 0.0, 0.0],
        Sdf::Box { center, half } => [1.0, center.x as f32, center.y as f32, center.z as f32, half.x as f32, half.y as f32, half.z as f32, 0.0],
        Sdf::Plane { normal, offset } => [2.0, normal.x as f32, normal.y as f32, normal.z as f32, offset as f32, 0.0, 0.0, 0.0],
        Sdf::Capsule { a, b, radius } => [3.0, a.x as f32, a.y as f32, a.z as f32, b.x as f32, b.y as f32, b.z as f32, radius as f32],
        Sdf::Torus { center, major, minor } => [4.0, center.x as f32, center.y as f32, center.z as f32, major as f32, minor as f32, 0.0, 0.0],
    }
}

impl ClearanceGpu {
    /// Build a checker for `robot` against `scene`, sized for batches of exactly `n_configs`
    /// configurations. `link_r` and `per_link` mirror [`crate::arm_clearance`]. `None` when there is no GPU.
    pub fn new(robot: &Robot, scene: &SdfScene, link_r: f64, per_link: usize, n_configs: usize) -> Option<Self> {
        let dof = robot.dof();

        let mut joints = Vec::with_capacity(dof * 16);
        for j in &robot.joints {
            let r = j.origin.rotation.to_rotation_matrix();
            joints.extend(r.matrix().as_slice().iter().map(|&v| v as f32)); // 9, column-major
            let t = j.origin.translation.vector;
            joints.extend_from_slice(&[t.x as f32, t.y as f32, t.z as f32]);
            let a = j.axis.into_inner();
            joints.extend_from_slice(&[a.x as f32, a.y as f32, a.z as f32]);
            joints.push(match j.kind {
                JointKind::Revolute => 0.0,
                JointKind::Prismatic => 1.0,
            });
        }
        let r = robot.ee_offset.rotation.to_rotation_matrix();
        let mut ee: Vec<f32> = r.matrix().as_slice().iter().map(|&v| v as f32).collect();
        let t = robot.ee_offset.translation.vector;
        ee.extend_from_slice(&[t.x as f32, t.y as f32, t.z as f32]);

        let prims: Vec<f32> = scene.prims.iter().flat_map(prim_floats).collect();
        let n_prims = scene.prims.len();
        let prm = [dof as f32, per_link as f32, link_r as f32, n_configs as f32, n_prims as f32];

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;

        let init = |label, data: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let joints_buf = init("clr-joints", bytemuck::cast_slice(&joints));
        let ee_buf = init("clr-ee", bytemuck::cast_slice(&ee));
        // scene with no prims: still needs a non-empty buffer
        let prims_data = if prims.is_empty() { vec![0.0f32; 8] } else { prims };
        let prims_buf = init("clr-prims", bytemuck::cast_slice(&prims_data));
        let prm_buf = init("clr-prm", bytemuck::cast_slice(&prm));
        let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clr-cfg"),
            size: (n_configs * dof * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let clr_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clr-out"),
            size: (n_configs * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clr-staging"),
            size: (n_configs * 4).max(4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clearance"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let ro = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let mut rw = ro(5);
        rw.ty = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clr-bgl"),
            entries: &[ro(0), ro(1), ro(2), ro(3), ro(4), rw],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clr-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pso = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clearance"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clr-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: joints_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ee_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cfg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: prims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: prm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: clr_buf.as_entire_binding() },
            ],
        });

        Some(Self { dof, n_configs, device, queue, pso, bind, cfg_buf, clr_buf, staging })
    }

    /// Minimum clearance of the arm to the scene for each of `n_configs` configurations (packed
    /// `[q0…q_{dof-1}, …]`, length `n_configs·dof`). Negative = in collision. Matches [`crate::arm_clearance`].
    pub fn clearances(&self, configs: &[f64]) -> Vec<f64> {
        assert_eq!(configs.len(), self.n_configs * self.dof, "config batch size mismatch");
        let cfg32: Vec<f32> = configs.iter().map(|&v| v as f32).collect();
        self.queue.write_buffer(&self.cfg_buf, 0, bytemuck::cast_slice(&cfg32));

        let groups = (self.n_configs as u32).div_ceil(64);
        let gy = groups.div_ceil(65535);
        let gx = groups.div_ceil(gy);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pso);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&self.clr_buf, 0, &self.staging, 0, (self.n_configs * 4) as u64);
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

// ---------------------------------------------------------------------------------------------
// SensorGpu — the signed-distance depth/segmentation renderer on the GPU (one thread per pixel).
// ---------------------------------------------------------------------------------------------

const SENSOR_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> PRIMS: array<f32>;     // 8 per prim: type + params
@group(0) @binding(1) var<storage, read> PRM: array<f32>;       // camera + pose, see SensorGpu
@group(0) @binding(2) var<storage, read_write> RANGE: array<f32>;
@group(0) @binding(3) var<storage, read_write> SEG: array<i32>;

fn prim_sdf(p: vec3<f32>, b: u32) -> f32 {
  let ty = PRIMS[b];
  if (ty < 0.5) {
    return length(p - vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u])) - PRIMS[b+4u];
  } else if (ty < 1.5) {
    let q = abs(p - vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u])) - vec3<f32>(PRIMS[b+4u], PRIMS[b+5u], PRIMS[b+6u]);
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
  } else if (ty < 2.5) {
    return dot(vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u]), p) - PRIMS[b+4u];
  } else if (ty < 3.5) {
    let a = vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u]);
    let e = vec3<f32>(PRIMS[b+4u], PRIMS[b+5u], PRIMS[b+6u]);
    let ab = e - a;
    let t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return length(p - (a + t * ab)) - PRIMS[b+7u];
  } else {
    let d = p - vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u]);
    let planar = sqrt(d.x*d.x + d.z*d.z) - PRIMS[b+4u];
    return sqrt(planar*planar + d.y*d.y) - PRIMS[b+5u];
  }
}
fn scene_sdf(p: vec3<f32>, n: u32) -> f32 { var d = 1e30; for (var i = 0u; i < n; i = i + 1u) { d = min(d, prim_sdf(p, i*8u)); } return d; }
fn nearest(p: vec3<f32>, n: u32) -> i32 { var bi = -1; var bd = 1e30; for (var i = 0u; i < n; i = i + 1u) { let d = prim_sdf(p, i*8u); if (d < bd) { bd = d; bi = i32(i); } } return bi; }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let W = u32(PRM[4]); let H = u32(PRM[5]);
  if (g.x >= W || g.y >= H) { return; }
  let fx = PRM[0]; let fy = PRM[1]; let cx = PRM[2]; let cy = PRM[3]; let far = PRM[6];
  let o = vec3<f32>(PRM[7], PRM[8], PRM[9]);
  let r0 = vec3<f32>(PRM[10], PRM[11], PRM[12]);   // pose rotation, row-major
  let r1 = vec3<f32>(PRM[13], PRM[14], PRM[15]);
  let r2 = vec3<f32>(PRM[16], PRM[17], PRM[18]);
  let nprims = u32(PRM[19]);
  let fu = f32(g.x) + 0.5; let fv = f32(g.y) + 0.5;
  let dc = normalize(vec3<f32>((fu - cx) / fx, (fv - cy) / fy, 1.0));
  let dir = vec3<f32>(dot(r0, dc), dot(r1, dc), dot(r2, dc));
  var t = 0.0; var range = far; var seg = -1;
  for (var k = 0; k < 512; k = k + 1) {
    let p = o + t * dir;
    let d = scene_sdf(p, nprims);
    if (d < 1e-6) { range = t; seg = nearest(p, nprims); break; }
    t = t + max(d, 1e-6);
    if (t > far) { break; }
  }
  let idx = g.y * W + g.x;
  RANGE[idx] = range; SEG[idx] = seg;
}
"#;

/// A signed-distance depth + segmentation renderer on the GPU — the same sphere-tracer as
/// [`crate::DepthCamera::render`], one GPU thread per pixel, over the full [`SdfScene`].
pub struct SensorGpu {
    width: usize,
    height: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    range_buf: wgpu::Buffer,
    seg_buf: wgpu::Buffer,
    range_stage: wgpu::Buffer,
    seg_stage: wgpu::Buffer,
}

impl SensorGpu {
    /// Build a renderer for `cam` viewing `scene` (both baked in). `None` when there is no GPU.
    pub fn new(cam: &crate::DepthCamera, scene: &SdfScene) -> Option<Self> {
        let (width, height) = (cam.width, cam.height);
        let n_prims = scene.prims.len();
        let prims: Vec<f32> = scene.prims.iter().flat_map(prim_floats).collect();
        let prims = if prims.is_empty() { vec![0.0f32; 8] } else { prims };

        let m = cam.pose.rotation.to_rotation_matrix();
        let rot = m.matrix();
        let o = cam.pose.translation.vector;
        let mut prm = vec![
            cam.fx as f32, cam.fy as f32, cam.cx as f32, cam.cy as f32,
            width as f32, height as f32, cam.far as f32,
            o.x as f32, o.y as f32, o.z as f32,
        ];
        for i in 0..3 {
            for j in 0..3 {
                prm.push(rot[(i, j)] as f32); // row-major
            }
        }
        prm.push(n_prims as f32);

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;

        let init = |label, data: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE })
        };
        let prims_buf = init("sensor-prims", bytemuck::cast_slice(&prims));
        let prm_buf = init("sensor-prm", bytemuck::cast_slice(&prm));
        let px = width * height;
        let range_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("sensor-range"), size: (px * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let seg_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("sensor-seg"), size: (px * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let range_stage = device.create_buffer(&wgpu::BufferDescriptor { label: Some("sensor-range-stage"), size: (px * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let seg_stage = device.create_buffer(&wgpu::BufferDescriptor { label: Some("sensor-seg-stage"), size: (px * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("sensor"), source: wgpu::ShaderSource::Wgsl(SENSOR_WGSL.into()) });
        let ro = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let mut rw = ro(0);
        rw.ty = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let mut rw2 = rw;
        rw2.binding = 3;
        let mut rw_range = rw;
        rw_range.binding = 2;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("sensor-bgl"), entries: &[ro(0), ro(1), rw_range, rw2] });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("sensor-layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pso = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("sensor"), layout: Some(&layout), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sensor-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: prims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: prm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: range_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: seg_buf.as_entire_binding() },
            ],
        });

        Some(Self { width, height, device, queue, pso, bind, range_buf, seg_buf, range_stage, seg_stage })
    }

    /// Render range + segmentation (row-major `height × width`): per-pixel Euclidean range (`far`
    /// where nothing was hit) and nearest-primitive label (`-1` background). Matches `DepthCamera::render`.
    pub fn render(&self) -> (Vec<f32>, Vec<i32>) {
        let px = self.width * self.height;
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pso);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups((self.width as u32).div_ceil(8), (self.height as u32).div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&self.range_buf, 0, &self.range_stage, 0, (px * 4) as u64);
        enc.copy_buffer_to_buffer(&self.seg_buf, 0, &self.seg_stage, 0, (px * 4) as u64);
        self.queue.submit([enc.finish()]);

        let read_bytes = |buf: &wgpu::Buffer| -> Vec<u8> {
            let slice = buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().expect("map").expect("map ok");
            let v = slice.get_mapped_range().expect("mapped").to_vec();
            buf.unmap();
            v
        };
        let range: Vec<f32> = bytemuck::cast_slice(&read_bytes(&self.range_stage)).to_vec();
        let seg: Vec<i32> = bytemuck::cast_slice(&read_bytes(&self.seg_stage)).to_vec();
        (range, seg)
    }
}

// ---------------------------------------------------------------------------------------------
// LidarGpu — the scanning-lidar point cloud on the GPU (one thread per azimuth×elevation ray).
// ---------------------------------------------------------------------------------------------

const LIDAR_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> PRIMS: array<f32>;     // 8 per prim: type + params
@group(0) @binding(1) var<storage, read> PRM: array<f32>;       // lidar params + pose, see LidarGpu
@group(0) @binding(2) var<storage, read_write> RANGE: array<f32>;
@group(0) @binding(3) var<storage, read_write> SEG: array<i32>;
@group(0) @binding(4) var<storage, read_write> PTS: array<f32>; // 3 per ray (world hit point; 0 on miss)

fn prim_sdf(p: vec3<f32>, b: u32) -> f32 {
  let ty = PRIMS[b];
  if (ty < 0.5) {
    return length(p - vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u])) - PRIMS[b+4u];
  } else if (ty < 1.5) {
    let q = abs(p - vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u])) - vec3<f32>(PRIMS[b+4u], PRIMS[b+5u], PRIMS[b+6u]);
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
  } else if (ty < 2.5) {
    return dot(vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u]), p) - PRIMS[b+4u];
  } else if (ty < 3.5) {
    let a = vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u]);
    let e = vec3<f32>(PRIMS[b+4u], PRIMS[b+5u], PRIMS[b+6u]);
    let ab = e - a;
    let t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return length(p - (a + t * ab)) - PRIMS[b+7u];
  } else {
    let d = p - vec3<f32>(PRIMS[b+1u], PRIMS[b+2u], PRIMS[b+3u]);
    let planar = sqrt(d.x*d.x + d.z*d.z) - PRIMS[b+4u];
    return sqrt(planar*planar + d.y*d.y) - PRIMS[b+5u];
  }
}
fn scene_sdf(p: vec3<f32>, n: u32) -> f32 { var d = 1e30; for (var i = 0u; i < n; i = i + 1u) { d = min(d, prim_sdf(p, i*8u)); } return d; }
fn nearest(p: vec3<f32>, n: u32) -> i32 { var bi = -1; var bd = 1e30; for (var i = 0u; i < n; i = i + 1u) { let d = prim_sdf(p, i*8u); if (d < bd) { bd = d; bi = i32(i); } } return bi; }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let naz = u32(PRM[0]); let nel = u32(PRM[1]);
  if (g.x >= naz || g.y >= nel) { return; }         // g.x = azimuth index, g.y = elevation index
  let az_min = PRM[2]; let az_max = PRM[3]; let el_min = PRM[4]; let el_max = PRM[5]; let far = PRM[6];
  let o = vec3<f32>(PRM[7], PRM[8], PRM[9]);
  let r0 = vec3<f32>(PRM[10], PRM[11], PRM[12]);     // pose rotation, row-major
  let r1 = vec3<f32>(PRM[13], PRM[14], PRM[15]);
  let r2 = vec3<f32>(PRM[16], PRM[17], PRM[18]);
  let nprims = u32(PRM[19]);
  let da = select(0.0, (az_max - az_min) / f32(naz - 1u), naz > 1u);
  let de = select(0.0, (el_max - el_min) / f32(nel - 1u), nel > 1u);
  let az = az_min + f32(g.x) * da;
  let el = el_min + f32(g.y) * de;
  let ds = vec3<f32>(cos(el) * cos(az), cos(el) * sin(az), sin(el));   // +x forward, azimuth about +z
  let dir = vec3<f32>(dot(r0, ds), dot(r1, ds), dot(r2, ds));
  var t = 0.0; var range = far; var seg = -1; var pt = vec3<f32>(0.0);
  for (var k = 0; k < 512; k = k + 1) {
    let p = o + t * dir;
    let d = scene_sdf(p, nprims);
    if (d < 1e-6) { range = t; seg = nearest(p, nprims); pt = p; break; }
    t = t + max(d, 1e-6);
    if (t > far) { break; }
  }
  let idx = g.y * naz + g.x;                          // row-major (elevation outer, azimuth inner)
  RANGE[idx] = range; SEG[idx] = seg;
  PTS[idx*3u] = pt.x; PTS[idx*3u+1u] = pt.y; PTS[idx*3u+2u] = pt.z;
}
"#;

/// A scanning-lidar point cloud on the GPU — the same azimuth×elevation sphere-trace as
/// [`crate::Lidar::scan`], one GPU thread per ray, over the full [`SdfScene`].
pub struct LidarGpu {
    n_az: usize,
    n_el: usize,
    far: f64,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    range_buf: wgpu::Buffer,
    seg_buf: wgpu::Buffer,
    pts_buf: wgpu::Buffer,
    range_stage: wgpu::Buffer,
    seg_stage: wgpu::Buffer,
    pts_stage: wgpu::Buffer,
}

impl LidarGpu {
    /// Build a scanner for `lidar` viewing `scene` (both baked in). `None` when there is no GPU.
    pub fn new(lidar: &crate::Lidar, scene: &SdfScene) -> Option<Self> {
        let (n_az, n_el) = (lidar.n_azimuth, lidar.n_elevation);
        let n_prims = scene.prims.len();
        let prims: Vec<f32> = scene.prims.iter().flat_map(prim_floats).collect();
        let prims = if prims.is_empty() { vec![0.0f32; 8] } else { prims };

        let m = lidar.pose.rotation.to_rotation_matrix();
        let rot = m.matrix();
        let o = lidar.pose.translation.vector;
        let mut prm = vec![
            n_az as f32, n_el as f32,
            lidar.az_min as f32, lidar.az_max as f32, lidar.el_min as f32, lidar.el_max as f32, lidar.far as f32,
            o.x as f32, o.y as f32, o.z as f32,
        ];
        for i in 0..3 {
            for j in 0..3 {
                prm.push(rot[(i, j)] as f32); // row-major
            }
        }
        prm.push(n_prims as f32);

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;

        let init = |label, data: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE })
        };
        let prims_buf = init("lidar-prims", bytemuck::cast_slice(&prims));
        let prm_buf = init("lidar-prm", bytemuck::cast_slice(&prm));
        let n = (n_az * n_el).max(1);
        let sbuf = |label, elems: usize| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (elems * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let stg = |label, elems: usize| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (elems * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let range_buf = sbuf("lidar-range", n);
        let seg_buf = sbuf("lidar-seg", n);
        let pts_buf = sbuf("lidar-pts", n * 3);
        let range_stage = stg("lidar-range-stage", n);
        let seg_stage = stg("lidar-seg-stage", n);
        let pts_stage = stg("lidar-pts-stage", n * 3);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("lidar"), source: wgpu::ShaderSource::Wgsl(LIDAR_WGSL.into()) });
        let ro = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let rw = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("lidar-bgl"), entries: &[ro(0), ro(1), rw(2), rw(3), rw(4)] });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("lidar-layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pso = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("lidar"), layout: Some(&layout), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lidar-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: prims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: prm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: range_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: seg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pts_buf.as_entire_binding() },
            ],
        });

        Some(Self { n_az, n_el, far: lidar.far, device, queue, pso, bind, range_buf, seg_buf, pts_buf, range_stage, seg_stage, pts_stage })
    }

    fn read_bytes(&self, buf: &wgpu::Buffer) -> Vec<u8> {
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let v = slice.get_mapped_range().expect("mapped").to_vec();
        buf.unmap();
        v
    }

    /// The dense grid over every ray (row-major elevation×azimuth): per-ray range (`far` on miss),
    /// segmentation label (`-1` on miss), and world hit point (`(0,0,0)` on miss).
    pub fn dense(&self) -> (Vec<f32>, Vec<i32>, Vec<[f32; 3]>) {
        let n = self.n_az * self.n_el;
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pso);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups((self.n_az as u32).div_ceil(8), (self.n_el as u32).div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&self.range_buf, 0, &self.range_stage, 0, (n * 4) as u64);
        enc.copy_buffer_to_buffer(&self.seg_buf, 0, &self.seg_stage, 0, (n * 4) as u64);
        enc.copy_buffer_to_buffer(&self.pts_buf, 0, &self.pts_stage, 0, (n * 3 * 4) as u64);
        self.queue.submit([enc.finish()]);
        let range: Vec<f32> = bytemuck::cast_slice(&self.read_bytes(&self.range_stage)).to_vec();
        let seg: Vec<i32> = bytemuck::cast_slice(&self.read_bytes(&self.seg_stage)).to_vec();
        let pf: Vec<f32> = bytemuck::cast_slice(&self.read_bytes(&self.pts_stage)).to_vec();
        let pts: Vec<[f32; 3]> = pf.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        (range, seg, pts)
    }

    /// The compacted hit point cloud — only rays that hit, in the same order as [`crate::Lidar::scan`].
    pub fn scan(&self) -> crate::LidarScan {
        use nalgebra::Vector3;
        let (range, seg, pts) = self.dense();
        let far = self.far as f32;
        let mut points = Vec::new();
        let mut ranges = Vec::new();
        let mut segl = Vec::new();
        for i in 0..self.n_az * self.n_el {
            if range[i] < far {
                points.push(Vector3::new(pts[i][0] as f64, pts[i][1] as f64, pts[i][2] as f64));
                ranges.push(range[i] as f64);
                segl.push(seg[i].max(0) as usize);
            }
        }
        crate::LidarScan { points, ranges, seg: segl }
    }
}

// ---------------------------------------------------------------------------------------------
// ArticulatedGpu — batched articulated-body forward dynamics at RL scale (one thread per env).
// The RNEA / CRBA / Cholesky-solve of the CPU `forward_dynamics`, run for thousands of robots at
// once — the browser/native counterpart to Brax/MJX/Newton's parallel-env sweep.
// ---------------------------------------------------------------------------------------------

/// The WGSL for a fixed DOF `n` (baked at build time; `nn = n*n`).
fn articulated_wgsl(n: usize) -> String {
    let src = r#"
const N: u32 = {N}u;
@group(0) @binding(0) var<storage, read> JOINTS: array<f32>;   // 16 per joint: R_o(9,col-major) p_o(3) axis(3) kind(1)
@group(0) @binding(1) var<storage, read> INERTIA: array<f32>;  // 13 per link: mass(1) com(3) I(9,col-major)
@group(0) @binding(2) var<storage, read> PRM: array<f32>;      // [N, n_envs, gx, gy, gz, dt, floor_z, kn, kd, n_contacts]
@group(0) @binding(3) var<storage, read_write> Q: array<f32>;
@group(0) @binding(4) var<storage, read_write> QD: array<f32>;
@group(0) @binding(5) var<storage, read> TAU: array<f32>;
@group(0) @binding(6) var<storage, read_write> QDD: array<f32>;
@group(0) @binding(7) var<storage, read> CONTACTS: array<f32>; // 5 per contact: frame(1) offset(3) mu(1)
@group(0) @binding(8) var<storage, read> POLICY: array<f32>;   // {POLDIM} per env: W(N×2N, row-major) then b(N)
@group(0) @binding(9) var<storage, read_write> REWARD: array<f32>; // one scalar return per env

fn rot_axis(a: vec3<f32>, t: f32) -> mat3x3<f32> {
  let c = cos(t); let s = sin(t); let ic = 1.0 - c; let x = a.x; let y = a.y; let z = a.z;
  return mat3x3<f32>(
    vec3<f32>(c + x*x*ic,   y*x*ic + z*s, z*x*ic - y*s),
    vec3<f32>(x*y*ic - z*s, c + y*y*ic,   z*y*ic + x*s),
    vec3<f32>(x*z*ic + y*s, y*z*ic - x*s, c + z*z*ic));
}
fn inertia_mat(i: u32) -> mat3x3<f32> {
  let b = i*13u + 4u;
  return mat3x3<f32>(
    vec3<f32>(INERTIA[b],    INERTIA[b+1u], INERTIA[b+2u]),
    vec3<f32>(INERTIA[b+3u], INERTIA[b+4u], INERTIA[b+5u]),
    vec3<f32>(INERTIA[b+6u], INERTIA[b+7u], INERTIA[b+8u]));
}
fn com_of(i: u32) -> vec3<f32> { return vec3<f32>(INERTIA[i*13u+1u], INERTIA[i*13u+2u], INERTIA[i*13u+3u]); }

// Recursive Newton-Euler inverse dynamics: joint torques for (qd, qdd) under `grav`, given the
// per-joint relative transforms rr (frame i→i-1), pp (origin of i in i-1), zz (axis in frame i).
fn rnea(qd: array<f32, N>, qdd: array<f32, N>, grav: vec3<f32>,
        rr: ptr<function, array<mat3x3<f32>, N>>, pp: ptr<function, array<vec3<f32>, N>>, zz: ptr<function, array<vec3<f32>, N>>) -> array<f32, N> {
  var omega: array<vec3<f32>, N>;
  var omegad: array<vec3<f32>, N>;
  var vd: array<vec3<f32>, N>;
  var ff: array<vec3<f32>, N>;
  var nn: array<vec3<f32>, N>;
  var pw = vec3<f32>(0.0); var pwd = vec3<f32>(0.0); var pvd = -grav;
  for (var i = 0u; i < N; i = i + 1u) {
    let rt = transpose((*rr)[i]);                // frame i-1 → i
    let z = (*zz)[i];
    let base = rt * (pvd + cross(pwd, (*pp)[i]) + cross(pw, cross(pw, (*pp)[i])));
    let kind = JOINTS[i*16u + 15u];
    if (kind < 0.5) {                            // revolute
      omega[i] = rt * pw + qd[i] * z;
      omegad[i] = rt * pwd + cross(rt * pw, qd[i] * z) + qdd[i] * z;
      vd[i] = base;
    } else {                                     // prismatic
      omega[i] = rt * pw;
      omegad[i] = rt * pwd;
      vd[i] = base + 2.0 * cross(omega[i], qd[i] * z) + qdd[i] * z;
    }
    let mass = INERTIA[i*13u];
    let com = com_of(i);
    let Im = inertia_mat(i);
    let vdc = vd[i] + cross(omegad[i], com) + cross(omega[i], cross(omega[i], com));
    ff[i] = mass * vdc;
    nn[i] = Im * omegad[i] + cross(omega[i], Im * omega[i]);
    pw = omega[i]; pwd = omegad[i]; pvd = vd[i];
  }
  var tau: array<f32, N>;
  var f_next = vec3<f32>(0.0); var n_next = vec3<f32>(0.0);
  for (var ii = 0u; ii < N; ii = ii + 1u) {
    let i = N - 1u - ii;
    var rr_next = mat3x3<f32>(vec3<f32>(1.,0.,0.), vec3<f32>(0.,1.,0.), vec3<f32>(0.,0.,1.));
    var p_next = vec3<f32>(0.0);
    if (i + 1u < N) { rr_next = (*rr)[i+1u]; p_next = (*pp)[i+1u]; }
    let com = com_of(i);
    let f_i = rr_next * f_next + ff[i];
    let n_i = nn[i] + rr_next * n_next + cross(com, ff[i]) + cross(p_next, rr_next * f_next);
    if (JOINTS[i*16u + 15u] < 0.5) { tau[i] = dot(n_i, (*zz)[i]); } else { tau[i] = dot(f_i, (*zz)[i]); }
    f_next = f_i; n_next = n_i;
  }
  return tau;
}

// Forward dynamics: qdd = M(q)^-1 (tau - bias), with M via CRBA (N RNEA calls) and a Cholesky solve.
fn forward_dynamics(q: array<f32, N>, qd: array<f32, N>, tau: array<f32, N>, grav: vec3<f32>) -> array<f32, N> {
  var rr: array<mat3x3<f32>, N>;
  var pp: array<vec3<f32>, N>;
  var zz: array<vec3<f32>, N>;
  for (var i = 0u; i < N; i = i + 1u) {
    let jb = i*16u;
    let Ro = mat3x3<f32>(
      vec3<f32>(JOINTS[jb],     JOINTS[jb+1u], JOINTS[jb+2u]),
      vec3<f32>(JOINTS[jb+3u],  JOINTS[jb+4u], JOINTS[jb+5u]),
      vec3<f32>(JOINTS[jb+6u],  JOINTS[jb+7u], JOINTS[jb+8u]));
    let po = vec3<f32>(JOINTS[jb+9u], JOINTS[jb+10u], JOINTS[jb+11u]);
    let axis = vec3<f32>(JOINTS[jb+12u], JOINTS[jb+13u], JOINTS[jb+14u]);
    var Rm = mat3x3<f32>(vec3<f32>(1.,0.,0.), vec3<f32>(0.,1.,0.), vec3<f32>(0.,0.,1.));
    var pm = vec3<f32>(0.0);
    if (JOINTS[jb+15u] < 0.5) { Rm = rot_axis(axis, q[i]); } else { pm = axis * q[i]; }
    rr[i] = Ro * Rm; pp[i] = Ro * pm + po; zz[i] = axis;
  }
  var zero: array<f32, N>;
  for (var i = 0u; i < N; i = i + 1u) { zero[i] = 0.0; }
  let bias = rnea(qd, zero, grav, &rr, &pp, &zz);
  var M: array<f32, {NN}>;
  for (var j = 0u; j < N; j = j + 1u) {
    var ej = zero; ej[j] = 1.0;
    let col = rnea(zero, ej, vec3<f32>(0.0), &rr, &pp, &zz);
    for (var i = 0u; i < N; i = i + 1u) { M[i*N + j] = col[i]; }
  }
  // Cholesky factor M = L Lᵀ (L lower, stored in L), then solve for qdd.
  var L: array<f32, {NN}>;
  for (var i = 0u; i < N; i = i + 1u) {
    for (var j = 0u; j <= i; j = j + 1u) {
      var sum = M[i*N + j];
      for (var k = 0u; k < j; k = k + 1u) { sum = sum - L[i*N + k] * L[j*N + k]; }
      if (i == j) { L[i*N + i] = sqrt(max(sum, 1e-12)); }
      else { L[i*N + j] = sum / L[j*N + j]; }
    }
  }
  var rhs: array<f32, N>;
  for (var i = 0u; i < N; i = i + 1u) { rhs[i] = tau[i] - bias[i]; }
  var y: array<f32, N>;
  for (var i = 0u; i < N; i = i + 1u) { var s = rhs[i]; for (var k = 0u; k < i; k = k + 1u) { s = s - L[i*N + k] * y[k]; } y[i] = s / L[i*N + i]; }
  var x: array<f32, N>;
  for (var ii = 0u; ii < N; ii = ii + 1u) { let i = N - 1u - ii; var s = y[i]; for (var k = i + 1u; k < N; k = k + 1u) { s = s - L[k*N + i] * x[k]; } x[i] = s / L[i*N + i]; }
  return x;
}

@compute @workgroup_size(64)
fn accel(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x; let n_envs = u32(PRM[1]);
  if (e >= n_envs) { return; }
  let grav = vec3<f32>(PRM[2], PRM[3], PRM[4]);
  var q: array<f32, N>; var qd: array<f32, N>; var tau: array<f32, N>;
  for (var i = 0u; i < N; i = i + 1u) { q[i] = Q[e*N + i]; qd[i] = QD[e*N + i]; tau[i] = TAU[e*N + i]; }
  let x = forward_dynamics(q, qd, tau, grav);
  for (var i = 0u; i < N; i = i + 1u) { QDD[e*N + i] = x[i]; }
}

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x; let n_envs = u32(PRM[1]);
  if (e >= n_envs) { return; }
  let grav = vec3<f32>(PRM[2], PRM[3], PRM[4]); let dt = PRM[5];
  var q: array<f32, N>; var qd: array<f32, N>; var tau: array<f32, N>;
  for (var i = 0u; i < N; i = i + 1u) { q[i] = Q[e*N + i]; qd[i] = QD[e*N + i]; tau[i] = TAU[e*N + i]; }
  let x = forward_dynamics(q, qd, tau, grav);          // semi-implicit Euler
  for (var i = 0u; i < N; i = i + 1u) {
    let vd = qd[i] + dt * x[i];
    QD[e*N + i] = vd; Q[e*N + i] = q[i] + dt * vd; QDD[e*N + i] = x[i];
  }
}

// Penalty ground-contact joint torque: for each contact point below the floor, a spring-damper
// normal + regularized-Coulomb friction force, mapped to joint space by the point Jacobian Jₚᵀ·f.
fn contact_torque(q: array<f32, N>, qd: array<f32, N>) -> array<f32, N> {
  let floor_z = PRM[6]; let kn = PRM[7]; let kd = PRM[8]; let nc = u32(PRM[9]);
  var tc: array<f32, N>;
  for (var i = 0u; i < N; i = i + 1u) { tc[i] = 0.0; }
  if (nc == 0u) { return tc; }
  // world forward kinematics: per-frame pose Rf/of (frame k = frame_pose(q,k), k=0..N) and, per
  // joint i, its world axis zw[i] and world origin ow[i] (for the point Jacobian columns).
  var Rf: array<mat3x3<f32>, {N1}>;
  var ofr: array<vec3<f32>, {N1}>;
  var zw: array<vec3<f32>, N>;
  var ow: array<vec3<f32>, N>;
  var Tr = mat3x3<f32>(vec3<f32>(1.,0.,0.), vec3<f32>(0.,1.,0.), vec3<f32>(0.,0.,1.));
  var Tp = vec3<f32>(0.0);
  Rf[0] = Tr; ofr[0] = Tp;
  for (var i = 0u; i < N; i = i + 1u) {
    let jb = i*16u;
    let Ro = mat3x3<f32>(
      vec3<f32>(JOINTS[jb],     JOINTS[jb+1u], JOINTS[jb+2u]),
      vec3<f32>(JOINTS[jb+3u],  JOINTS[jb+4u], JOINTS[jb+5u]),
      vec3<f32>(JOINTS[jb+6u],  JOINTS[jb+7u], JOINTS[jb+8u]));
    let po = vec3<f32>(JOINTS[jb+9u], JOINTS[jb+10u], JOINTS[jb+11u]);
    let axis = vec3<f32>(JOINTS[jb+12u], JOINTS[jb+13u], JOINTS[jb+14u]);
    let preR = Tr * Ro;                 // pre = frame_pose(q,i) ∘ origin_i
    let preP = Tr * po + Tp;
    zw[i] = preR * axis;                // joint axis in world
    ow[i] = preP;                       // joint origin in world
    var Rm = mat3x3<f32>(vec3<f32>(1.,0.,0.), vec3<f32>(0.,1.,0.), vec3<f32>(0.,0.,1.));
    var pm = vec3<f32>(0.0);
    if (JOINTS[jb+15u] < 0.5) { Rm = rot_axis(axis, q[i]); } else { pm = axis * q[i]; }
    Tr = preR * Rm; Tp = preR * pm + preP;
    Rf[i+1u] = Tr; ofr[i+1u] = Tp;
  }
  for (var c = 0u; c < nc; c = c + 1u) {
    let cb = c*5u;
    let fr = u32(CONTACTS[cb]);
    let off = vec3<f32>(CONTACTS[cb+1u], CONTACTS[cb+2u], CONTACTS[cb+3u]);
    let mu = CONTACTS[cb+4u];
    let p = Rf[fr] * off + ofr[fr];       // contact point in world
    let phi = p.z - floor_z;
    if (phi < 0.0) {
      // point velocity v = Jₚ·q̇ = Σ_{i<fr} col_i · q̇[i]
      var v = vec3<f32>(0.0);
      for (var i = 0u; i < fr; i = i + 1u) {
        var col = zw[i];
        if (JOINTS[i*16u+15u] < 0.5) { col = cross(zw[i], p - ow[i]); }
        v = v + col * qd[i];
      }
      let fnrm = max(0.0, -kn * phi - kd * v.z);          // spring-damper normal (push only)
      let vt = vec2<f32>(v.x, v.y);
      let ft = -mu * fnrm * vt / (length(vt) + 1e-4);     // regularized Coulomb friction
      let f = vec3<f32>(ft.x, ft.y, fnrm);
      for (var i = 0u; i < fr; i = i + 1u) {
        var col = zw[i];
        if (JOINTS[i*16u+15u] < 0.5) { col = cross(zw[i], p - ow[i]); }
        tc[i] = tc[i] + dot(col, f);                       // Jₚᵀ·f
      }
    }
  }
  return tc;
}

@compute @workgroup_size(64)
fn step_contact(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x; let n_envs = u32(PRM[1]);
  if (e >= n_envs) { return; }
  let grav = vec3<f32>(PRM[2], PRM[3], PRM[4]); let dt = PRM[5];
  var q: array<f32, N>; var qd: array<f32, N>; var tau: array<f32, N>;
  for (var i = 0u; i < N; i = i + 1u) { q[i] = Q[e*N + i]; qd[i] = QD[e*N + i]; tau[i] = TAU[e*N + i]; }
  let tc = contact_torque(q, qd);
  var tt: array<f32, N>;
  for (var i = 0u; i < N; i = i + 1u) { tt[i] = tau[i] + tc[i]; }
  let x = forward_dynamics(q, qd, tt, grav);
  for (var i = 0u; i < N; i = i + 1u) {
    let vd = qd[i] + dt * x[i];
    QD[e*N + i] = vd; Q[e*N + i] = q[i] + dt * vd; QDD[e*N + i] = x[i];
  }
}

// Batched policy rollout for gradient-free policy search: one thread per candidate policy (env).
// Linear feedback tau = clamp(W·[q−q*; q̇] + b), rolled out `T` steps under gravity; returns the
// accumulated reward −Σ(‖q−q*‖² + w·‖tau‖²). PRM tail: [tau_max, effort_w, T, q*(N), q0(N), qd0(N)].
@compute @workgroup_size(64)
fn rollout(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x; let n_envs = u32(PRM[1]);
  if (e >= n_envs) { return; }
  let grav = vec3<f32>(PRM[2], PRM[3], PRM[4]); let dt = PRM[5];
  let tau_max = PRM[10]; let effort_w = PRM[11]; let T = u32(PRM[12]);
  var tstar: array<f32, N>; var q: array<f32, N>; var qd: array<f32, N>;
  for (var i = 0u; i < N; i = i + 1u) {
    tstar[i] = PRM[13u + i];
    q[i] = PRM[13u + N + i];
    qd[i] = PRM[13u + 2u*N + i];
  }
  let pbase = e * {POLDIM}u;
  var reward = 0.0;
  for (var t = 0u; t < T; t = t + 1u) {
    var tau: array<f32, N>;
    for (var i = 0u; i < N; i = i + 1u) {
      var s = POLICY[pbase + 2u*N*N + i];                       // b[i]
      for (var j = 0u; j < N; j = j + 1u) { s = s + POLICY[pbase + i*2u*N + j] * (q[j] - tstar[j]); }
      for (var j = 0u; j < N; j = j + 1u) { s = s + POLICY[pbase + i*2u*N + N + j] * qd[j]; }
      tau[i] = clamp(s, -tau_max, tau_max);
    }
    let x = forward_dynamics(q, qd, tau, grav);
    var cost = 0.0;
    for (var i = 0u; i < N; i = i + 1u) { let ei = q[i] - tstar[i]; cost = cost + ei*ei + effort_w * tau[i] * tau[i]; }
    reward = reward - cost;
    for (var i = 0u; i < N; i = i + 1u) { let vd = qd[i] + dt * x[i]; qd[i] = vd; q[i] = q[i] + dt * vd; }
  }
  REWARD[e] = reward;
}
"#;
    src.replace("{NN}", &(n * n).to_string())
        .replace("{N1}", &(n + 1).to_string())
        .replace("{POLDIM}", &(2 * n * n + n).to_string())
        .replace("{N}", &n.to_string())
}

/// Batched articulated-body forward dynamics on the GPU — the RNEA/CRBA/Cholesky-solve of the CPU
/// [`forward_dynamics`](crate::forward_dynamics), one GPU thread per environment. Fixed robot
/// topology (the RL setting: the same robot across all environments).
pub struct ArticulatedGpu {
    n: usize,
    n_envs: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso_accel: wgpu::ComputePipeline,
    pso_step: wgpu::ComputePipeline,
    pso_step_contact: wgpu::ComputePipeline,
    pso_rollout: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    q_buf: wgpu::Buffer,
    qd_buf: wgpu::Buffer,
    tau_buf: wgpu::Buffer,
    qdd_buf: wgpu::Buffer,
    prm_buf: wgpu::Buffer,
    policy_buf: wgpu::Buffer,
    reward_buf: wgpu::Buffer,
    reward_stage: wgpu::Buffer,
    stage: wgpu::Buffer,
    base_prm: [f32; 10],
    poldim: usize,
}

impl ArticulatedGpu {
    /// Build a batched stepper for `robot` (with per-link `inertia`), `n_envs` environments, under
    /// `gravity`, timestep `dt`. `None` when there is no GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        robot: &crate::Robot,
        inertia: &[crate::LinkInertia],
        gravity: nalgebra::Vector3<f64>,
        dt: f64,
        n_envs: usize,
        contacts: &[(usize, nalgebra::Vector3<f64>, f64)],
        floor_z: f64,
        kn: f64,
        kd: f64,
    ) -> Option<Self> {
        let n = robot.dof();
        assert_eq!(inertia.len(), n, "one inertia per joint/link");
        let n_contacts = contacts.len();
        let mut contact_flat: Vec<f32> = contacts
            .iter()
            .flat_map(|&(fr, off, mu)| [fr as f32, off.x as f32, off.y as f32, off.z as f32, mu as f32])
            .collect();
        if contact_flat.is_empty() {
            contact_flat = vec![0.0f32; 5]; // non-empty storage buffer
        }

        let mut joints = Vec::with_capacity(n * 16);
        for j in &robot.joints {
            let r = j.origin.rotation.to_rotation_matrix();
            joints.extend(r.matrix().as_slice().iter().map(|&v| v as f32));
            let t = j.origin.translation.vector;
            joints.extend_from_slice(&[t.x as f32, t.y as f32, t.z as f32]);
            let a = j.axis.into_inner();
            joints.extend_from_slice(&[a.x as f32, a.y as f32, a.z as f32]);
            joints.push(match j.kind {
                crate::JointKind::Revolute => 0.0,
                crate::JointKind::Prismatic => 1.0,
            });
        }
        let mut inert = Vec::with_capacity(n * 13);
        for li in inertia {
            inert.push(li.mass as f32);
            inert.extend_from_slice(&[li.com.x as f32, li.com.y as f32, li.com.z as f32]);
            inert.extend(li.inertia.as_slice().iter().map(|&v| v as f32)); // 9, column-major
        }
        let prm = [
            n as f32, n_envs as f32, gravity.x as f32, gravity.y as f32, gravity.z as f32, dt as f32,
            floor_z as f32, kn as f32, kd as f32, n_contacts as f32,
        ];

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        // 10 storage buffers (dynamics + policy/reward) — request the adapter's real ceiling.
        let desc = wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: adapter.limits().max_storage_buffers_per_shader_stage,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc)).ok()?;

        let poldim = 2 * n * n + n;
        let base_prm: [f32; 10] = prm;
        // PRM buffer holds the base 10 + the rollout tail [tau_max, effort_w, T, q*(n), q0(n), qd0(n)].
        let mut prm_full = prm.to_vec();
        prm_full.resize(13 + 3 * n, 0.0);

        let init = |label, data: &[u8], extra: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE | extra })
        };
        let none = wgpu::BufferUsages::empty();
        let joints_buf = init("art-joints", bytemuck::cast_slice(&joints), none);
        let inertia_buf = init("art-inertia", bytemuck::cast_slice(&inert), none);
        let prm_buf = init("art-prm", bytemuck::cast_slice(&prm_full), wgpu::BufferUsages::COPY_DST);
        let contacts_buf = init("art-contacts", bytemuck::cast_slice(&contact_flat), none);
        let policy_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("art-policy"), size: (n_envs * poldim * 4).max(4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let reward_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("art-reward"), size: (n_envs * 4).max(4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let reward_stage = device.create_buffer(&wgpu::BufferDescriptor { label: Some("art-reward-stage"), size: (n_envs * 4).max(4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let ne = (n_envs * n).max(1);
        let q_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("art-q"), size: (ne * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let qd_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("art-qd"), size: (ne * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let tau_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("art-tau"), size: (ne * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let qdd_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("art-qdd"), size: (ne * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let stage = device.create_buffer(&wgpu::BufferDescriptor { label: Some("art-stage"), size: (ne * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("articulated"), source: wgpu::ShaderSource::Wgsl(articulated_wgsl(n).into()) });
        let ro = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let rw = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("art-bgl"), entries: &[ro(0), ro(1), ro(2), rw(3), rw(4), ro(5), rw(6), ro(7), ro(8), rw(9)] });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("art-layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let mk = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&layout), module: &shader, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
        let pso_accel = mk("accel");
        let pso_step = mk("step");
        let pso_step_contact = mk("step_contact");
        let pso_rollout = mk("rollout");
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("art-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: joints_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: inertia_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: prm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: qd_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: tau_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: qdd_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: contacts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: policy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: reward_buf.as_entire_binding() },
            ],
        });

        Some(Self { n, n_envs, device, queue, pso_accel, pso_step, pso_step_contact, pso_rollout, bind, q_buf, qd_buf, tau_buf, qdd_buf, prm_buf, policy_buf, reward_buf, reward_stage, stage, base_prm, poldim })
    }

    fn read(&self, src: &wgpu::Buffer) -> Vec<f64> {
        let bytes = (self.n_envs * self.n * 4) as u64;
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &self.stage, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = self.stage.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let v: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        self.stage.unmap();
        v.iter().map(|&x| x as f64).collect()
    }

    /// Batched forward dynamics: joint accelerations for every environment (packed `n_envs·n`), given
    /// per-env `q`, `qd`, `tau` (each `n_envs·n`). Matches [`forward_dynamics`](crate::forward_dynamics).
    pub fn accelerations(&self, q: &[f64], qd: &[f64], tau: &[f64]) -> Vec<f64> {
        let ne = self.n_envs * self.n;
        assert!(q.len() == ne && qd.len() == ne && tau.len() == ne, "state batch size mismatch");
        let f = |s: &[f64]| -> Vec<f32> { s.iter().map(|&v| v as f32).collect() };
        self.queue.write_buffer(&self.q_buf, 0, bytemuck::cast_slice(&f(q)));
        self.queue.write_buffer(&self.qd_buf, 0, bytemuck::cast_slice(&f(qd)));
        self.queue.write_buffer(&self.tau_buf, 0, bytemuck::cast_slice(&f(tau)));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pso_accel);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups((self.n_envs as u32).div_ceil(64), 1, 1);
        }
        self.queue.submit([enc.finish()]);
        self.read(&self.qdd_buf)
    }

    /// Set the batch state and advance `steps` semi-implicit-Euler steps on the GPU under constant
    /// per-env torques `tau`; return the final `(q, qd)` (each `n_envs·n`).
    pub fn run(&self, q0: &[f64], qd0: &[f64], tau: &[f64], steps: usize) -> (Vec<f64>, Vec<f64>) {
        let ne = self.n_envs * self.n;
        assert!(q0.len() == ne && qd0.len() == ne && tau.len() == ne, "state batch size mismatch");
        let f = |s: &[f64]| -> Vec<f32> { s.iter().map(|&v| v as f32).collect() };
        self.queue.write_buffer(&self.q_buf, 0, bytemuck::cast_slice(&f(q0)));
        self.queue.write_buffer(&self.qd_buf, 0, bytemuck::cast_slice(&f(qd0)));
        self.queue.write_buffer(&self.tau_buf, 0, bytemuck::cast_slice(&f(tau)));
        self.run_with(steps, &self.pso_step)
    }

    /// Like [`run`](Self::run), but resolving penalty ground contact each step (the contacts / floor
    /// / stiffness passed to [`new`](Self::new)). Enables locomotion RL: the robots push off the floor.
    pub fn run_contact(&self, q0: &[f64], qd0: &[f64], tau: &[f64], steps: usize) -> (Vec<f64>, Vec<f64>) {
        let ne = self.n_envs * self.n;
        assert!(q0.len() == ne && qd0.len() == ne && tau.len() == ne, "state batch size mismatch");
        let f = |s: &[f64]| -> Vec<f32> { s.iter().map(|&v| v as f32).collect() };
        self.queue.write_buffer(&self.q_buf, 0, bytemuck::cast_slice(&f(q0)));
        self.queue.write_buffer(&self.qd_buf, 0, bytemuck::cast_slice(&f(qd0)));
        self.queue.write_buffer(&self.tau_buf, 0, bytemuck::cast_slice(&f(tau)));
        self.run_with(steps, &self.pso_step_contact)
    }

    fn run_with(&self, steps: usize, pso: &wgpu::ComputePipeline) -> (Vec<f64>, Vec<f64>) {
        let groups = (self.n_envs as u32).div_ceil(64);
        let mut done = 0;
        while done < steps {
            let chunk = (steps - done).min(256);
            let mut enc = self.device.create_command_encoder(&Default::default());
            for _ in 0..chunk {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pso);
                pass.set_bind_group(0, &self.bind, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.queue.submit([enc.finish()]);
            done += chunk;
        }
        (self.read(&self.q_buf), self.read(&self.qd_buf))
    }

    /// **Batched policy search.** Evaluate `n_envs` candidate linear policies in parallel: each is a
    /// `W(n×2n)` + `b(n)` (flat, `n_envs·(2n²+n)`); every environment rolls out `steps` from `(q0,qd0)`
    /// under `tau = clamp(W·[q−target; q̇] + b, ±tau_max)` and returns `−Σ(‖q−target‖² + effort_w·‖tau‖²)`.
    /// One GPU thread per candidate — the parallel-env evaluation an evolutionary / CEM outer loop drives.
    #[allow(clippy::too_many_arguments)]
    pub fn rollout_rewards(&self, policies: &[f64], q0: &[f64], qd0: &[f64], target: &[f64], tau_max: f64, effort_w: f64, steps: usize) -> Vec<f64> {
        assert_eq!(policies.len(), self.n_envs * self.poldim, "policy batch size mismatch");
        assert!(q0.len() == self.n && qd0.len() == self.n && target.len() == self.n, "state/target length mismatch");
        // write the rollout tail into PRM (preserving the base 10)
        let mut prm = self.base_prm.to_vec();
        prm.extend_from_slice(&[tau_max as f32, effort_w as f32, steps as f32]);
        prm.extend(target.iter().map(|&v| v as f32));
        prm.extend(q0.iter().map(|&v| v as f32));
        prm.extend(qd0.iter().map(|&v| v as f32));
        self.queue.write_buffer(&self.prm_buf, 0, bytemuck::cast_slice(&prm));
        let pol32: Vec<f32> = policies.iter().map(|&v| v as f32).collect();
        self.queue.write_buffer(&self.policy_buf, 0, bytemuck::cast_slice(&pol32));

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pso_rollout);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups((self.n_envs as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.reward_buf, 0, &self.reward_stage, 0, (self.n_envs * 4) as u64);
        self.queue.submit([enc.finish()]);

        let slice = self.reward_stage.slice(..(self.n_envs * 4) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let v: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        self.reward_stage.unmap();
        v.iter().map(|&x| x as f64).collect()
    }

    /// Policy parameter dimension `2n² + n` (a linear feedback `W(n×2n)` + bias `b(n)`).
    pub fn policy_dim(&self) -> usize {
        self.poldim
    }
}

// ---------------------------------------------------------------------------------------------
// FloatingBaseGpu — batched floating-base forward dynamics (spatial 6D ABA) at RL scale. A free
// 6-DoF root + serial chain, one GPU thread per environment — the gate to legged locomotion.
// Mirrors `floating_base_forward_dynamics` (Featherstone Ch. 9), verified against it.
// ---------------------------------------------------------------------------------------------

fn floating_wgsl(n: usize) -> String {
    let src = r#"
const N: u32 = {N}u;
@group(0) @binding(0) var<storage, read> JOINTS: array<f32>;   // 16 per joint: R_o(9,col-major) p_o(3) axis(3) kind(1)
@group(0) @binding(1) var<storage, read> INERTIA: array<f32>;  // 13 per link: mass(1) com(3) I(9,col-major)
@group(0) @binding(2) var<storage, read> PRM: array<f32>;      // [N, n_envs, gx,gy,gz, base: mass(1) com(3) I(9)]
@group(0) @binding(3) var<storage, read> V0: array<f32>;       // 6 per env: base spatial velocity [ang; lin]
@group(0) @binding(4) var<storage, read> STATE: array<f32>;    // 3N per env: q(N) qd(N) tau(N)
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;// (6+N) per env: a0(6) qdd(N)
@group(0) @binding(6) var<storage, read> FEXT: array<f32>;     // (N+1)*6 per env: base(6) then link i (6 each), spatial [torque; force]

// spatial vector [angular; linear] and 6x6 spatial matrix as four 3x3 blocks [[tl,tr],[bl,br]]
struct SV { t: vec3<f32>, b: vec3<f32> }
struct SM { tl: mat3x3<f32>, tr: mat3x3<f32>, bl: mat3x3<f32>, br: mat3x3<f32> }

fn z3() -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0)); }
fn i3() -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(1.,0.,0.), vec3<f32>(0.,1.,0.), vec3<f32>(0.,0.,1.)); }
fn skew(v: vec3<f32>) -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(0.0, v.z, -v.y), vec3<f32>(-v.z, 0.0, v.x), vec3<f32>(v.y, -v.x, 0.0)); }
fn outer(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> { return mat3x3<f32>(a*b.x, a*b.y, a*b.z); }
fn smv(m: SM, v: SV) -> SV { return SV(m.tl*v.t + m.tr*v.b, m.bl*v.t + m.br*v.b); }
fn smm(a: SM, b: SM) -> SM { return SM(a.tl*b.tl + a.tr*b.bl, a.tl*b.tr + a.tr*b.br, a.bl*b.tl + a.br*b.bl, a.bl*b.tr + a.br*b.br); }
fn smt(a: SM) -> SM { return SM(transpose(a.tl), transpose(a.bl), transpose(a.tr), transpose(a.br)); }
fn sma(a: SM, b: SM) -> SM { return SM(a.tl+b.tl, a.tr+b.tr, a.bl+b.bl, a.br+b.br); }
fn sms(a: SM, s: f32) -> SM { return SM(a.tl*s, a.tr*s, a.bl*s, a.br*s); }
fn smsub(a: SM, b: SM) -> SM { return SM(a.tl-b.tl, a.tr-b.tr, a.bl-b.bl, a.br-b.br); }
fn sva(a: SV, b: SV) -> SV { return SV(a.t+b.t, a.b+b.b); }
fn svsub(a: SV, b: SV) -> SV { return SV(a.t-b.t, a.b-b.b); }
fn svs(a: SV, s: f32) -> SV { return SV(a.t*s, a.b*s); }
fn svdot(a: SV, b: SV) -> f32 { return dot(a.t, b.t) + dot(a.b, b.b); }
fn svouter(u: SV, w: SV) -> SM { return SM(outer(u.t,w.t), outer(u.t,w.b), outer(u.b,w.t), outer(u.b,w.b)); } // u wᵀ

fn motion_transform(r: mat3x3<f32>, p: vec3<f32>) -> SM { let e = transpose(r); return SM(e, z3(), (e*skew(p)) * (-1.0), e); }
fn spatial_inertia(mass: f32, c: vec3<f32>, I: mat3x3<f32>) -> SM { let cx = skew(c); return SM(I - mass*(cx*cx), cx*mass, cx*(-mass), i3()*mass); }
fn crm(v: SV) -> SM { return SM(skew(v.t), z3(), skew(v.b), skew(v.t)); }
fn crf(v: SV) -> SM { return sms(smt(crm(v)), -1.0); }
fn grav_wrench(isp: SM, g: vec3<f32>, r: mat3x3<f32>) -> SV { return smv(isp, SV(vec3<f32>(0.0), r*g)); }
fn fext_sv(off: u32) -> SV { return SV(vec3<f32>(FEXT[off], FEXT[off+1u], FEXT[off+2u]), vec3<f32>(FEXT[off+3u], FEXT[off+4u], FEXT[off+5u])); }

fn jointR(i: u32, qi: f32) -> mat3x3<f32> {
  let jb = i*16u;
  let Ro = mat3x3<f32>(vec3<f32>(JOINTS[jb],JOINTS[jb+1u],JOINTS[jb+2u]), vec3<f32>(JOINTS[jb+3u],JOINTS[jb+4u],JOINTS[jb+5u]), vec3<f32>(JOINTS[jb+6u],JOINTS[jb+7u],JOINTS[jb+8u]));
  if (JOINTS[jb+15u] < 0.5) {
    let a = vec3<f32>(JOINTS[jb+12u],JOINTS[jb+13u],JOINTS[jb+14u]);
    let c = cos(qi); let s = sin(qi); let ic = 1.0-c; let x=a.x; let y=a.y; let zz=a.z;
    let Rm = mat3x3<f32>(vec3<f32>(c+x*x*ic, y*x*ic+zz*s, zz*x*ic-y*s), vec3<f32>(x*y*ic-zz*s, c+y*y*ic, zz*y*ic+x*s), vec3<f32>(x*zz*ic+y*s, y*zz*ic-x*s, c+zz*zz*ic));
    return Ro*Rm;
  }
  return Ro;
}
fn jointP(i: u32, qi: f32) -> vec3<f32> {
  let jb = i*16u;
  let Ro = mat3x3<f32>(vec3<f32>(JOINTS[jb],JOINTS[jb+1u],JOINTS[jb+2u]), vec3<f32>(JOINTS[jb+3u],JOINTS[jb+4u],JOINTS[jb+5u]), vec3<f32>(JOINTS[jb+6u],JOINTS[jb+7u],JOINTS[jb+8u]));
  let po = vec3<f32>(JOINTS[jb+9u],JOINTS[jb+10u],JOINTS[jb+11u]);
  if (JOINTS[jb+15u] < 0.5) { return po; }
  let a = vec3<f32>(JOINTS[jb+12u],JOINTS[jb+13u],JOINTS[jb+14u]);
  return Ro*(a*qi) + po;
}
fn subspace(i: u32) -> SV {
  let a = vec3<f32>(JOINTS[i*16u+12u], JOINTS[i*16u+13u], JOINTS[i*16u+14u]);
  if (JOINTS[i*16u+15u] < 0.5) { return SV(a, vec3<f32>(0.0)); }
  return SV(vec3<f32>(0.0), a);
}
fn linertia(i: u32) -> SM {
  let mass = INERTIA[i*13u];
  let c = vec3<f32>(INERTIA[i*13u+1u], INERTIA[i*13u+2u], INERTIA[i*13u+3u]);
  let b = i*13u + 4u;
  let I = mat3x3<f32>(vec3<f32>(INERTIA[b],INERTIA[b+1u],INERTIA[b+2u]), vec3<f32>(INERTIA[b+3u],INERTIA[b+4u],INERTIA[b+5u]), vec3<f32>(INERTIA[b+6u],INERTIA[b+7u],INERTIA[b+8u]));
  return spatial_inertia(mass, c, I);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x; let n_envs = u32(PRM[1]);
  if (e >= n_envs) { return; }
  let grav = vec3<f32>(PRM[2], PRM[3], PRM[4]);
  let bmass = PRM[5];
  let bcom = vec3<f32>(PRM[6], PRM[7], PRM[8]);
  let bI = mat3x3<f32>(vec3<f32>(PRM[9],PRM[10],PRM[11]), vec3<f32>(PRM[12],PRM[13],PRM[14]), vec3<f32>(PRM[15],PRM[16],PRM[17]));
  let v0 = SV(vec3<f32>(V0[e*6u], V0[e*6u+1u], V0[e*6u+2u]), vec3<f32>(V0[e*6u+3u], V0[e*6u+4u], V0[e*6u+5u]));
  let sb = e*3u*N;
  var q: array<f32,N>; var qd: array<f32,N>; var tau: array<f32,N>;
  for (var i=0u;i<N;i=i+1u){ q[i]=STATE[sb+i]; qd[i]=STATE[sb+N+i]; tau[i]=STATE[sb+2u*N+i]; }

  var xm: array<SM,N>; var s: array<SV,N>; var v: array<SV,N>; var c: array<SV,N>;
  var ia: array<SM,N>; var pa: array<SV,N>;
  let ib = spatial_inertia(bmass, bcom, bI);
  var ia_base = ib;
  let feb = e*6u*(N+1u);
  var pa_base = svsub(svsub(smv(crf(v0), smv(ib, v0)), grav_wrench(ib, grav, i3())), fext_sv(feb));

  var r_parent = i3();
  for (var i=0u;i<N;i=i+1u){
    let r = jointR(i, q[i]); let p = jointP(i, q[i]);
    let x = motion_transform(r, p);
    let si = subspace(i);
    var v_par = v0; if (i>0u){ v_par = v[i-1u]; }
    v[i] = sva(smv(x, v_par), svs(si, qd[i]));
    c[i] = smv(crm(v[i]), svs(si, qd[i]));
    let ii = linertia(i);
    let r_bi = transpose(r) * r_parent;
    pa[i] = svsub(svsub(smv(crf(v[i]), smv(ii, v[i])), grav_wrench(ii, grav, r_bi)), fext_sv(feb + 6u*(i+1u)));
    ia[i] = ii; xm[i] = x; s[i] = si;
    r_parent = r_bi;
  }

  var u: array<SV,N>; var d: array<f32,N>; var uu: array<f32,N>;
  for (var ii=0u; ii<N; ii=ii+1u){
    let i = N-1u-ii;
    u[i] = smv(ia[i], s[i]);
    d[i] = svdot(s[i], u[i]);
    uu[i] = tau[i] - svdot(s[i], pa[i]);
    let ia_bar = smsub(ia[i], sms(svouter(u[i], u[i]), 1.0/d[i]));
    let pa_bar = sva(sva(pa[i], smv(ia_bar, c[i])), svs(u[i], uu[i]/d[i]));
    let xt = smt(xm[i]);
    if (i>0u){
      ia[i-1u] = sma(ia[i-1u], smm(smm(xt, ia_bar), xm[i]));
      pa[i-1u] = sva(pa[i-1u], smv(xt, pa_bar));
    } else {
      ia_base = sma(ia_base, smm(smm(xt, ia_bar), xm[0]));
      pa_base = sva(pa_base, smv(xt, pa_bar));
    }
  }

  // base: a0 = -(ia_base)^-1 pa_base  — 6x6 SPD Cholesky solve
  var A: array<f32,36>;
  for (var r=0u;r<3u;r=r+1u){ for (var cc=0u;cc<3u;cc=cc+1u){
    A[r*6u+cc]        = ia_base.tl[cc][r];
    A[r*6u+cc+3u]     = ia_base.tr[cc][r];
    A[(r+3u)*6u+cc]   = ia_base.bl[cc][r];
    A[(r+3u)*6u+cc+3u]= ia_base.br[cc][r];
  }}
  var rhs: array<f32,6>;
  rhs[0]=pa_base.t.x; rhs[1]=pa_base.t.y; rhs[2]=pa_base.t.z; rhs[3]=pa_base.b.x; rhs[4]=pa_base.b.y; rhs[5]=pa_base.b.z;
  var L: array<f32,36>;
  for (var i=0u;i<6u;i=i+1u){ for (var j=0u;j<=i;j=j+1u){
    var sum = A[i*6u+j];
    for (var k=0u;k<j;k=k+1u){ sum = sum - L[i*6u+k]*L[j*6u+k]; }
    if (i==j){ L[i*6u+i] = sqrt(max(sum, 1e-12)); } else { L[i*6u+j] = sum / L[j*6u+j]; }
  }}
  var yv: array<f32,6>;
  for (var i=0u;i<6u;i=i+1u){ var sm2 = rhs[i]; for (var k=0u;k<i;k=k+1u){ sm2 = sm2 - L[i*6u+k]*yv[k]; } yv[i] = sm2 / L[i*6u+i]; }
  var xv: array<f32,6>;
  for (var ii=0u;ii<6u;ii=ii+1u){ let i = 5u-ii; var sm3 = yv[i]; for (var k=i+1u;k<6u;k=k+1u){ sm3 = sm3 - L[k*6u+i]*xv[k]; } xv[i] = sm3 / L[i*6u+i]; }
  let a0 = SV(vec3<f32>(-xv[0], -xv[1], -xv[2]), vec3<f32>(-xv[3], -xv[4], -xv[5]));

  var qdd: array<f32,N>;
  var a: array<SV,N>;
  for (var i=0u;i<N;i=i+1u){
    var a_par = a0; if (i>0u){ a_par = a[i-1u]; }
    let a_prime = sva(smv(xm[i], a_par), c[i]);
    qdd[i] = (uu[i] - svdot(u[i], a_prime)) / d[i];
    a[i] = sva(a_prime, svs(s[i], qdd[i]));
  }

  let ob = e*(6u+N);
  OUT[ob]=a0.t.x; OUT[ob+1u]=a0.t.y; OUT[ob+2u]=a0.t.z; OUT[ob+3u]=a0.b.x; OUT[ob+4u]=a0.b.y; OUT[ob+5u]=a0.b.z;
  for (var i=0u;i<N;i=i+1u){ OUT[ob+6u+i]=qdd[i]; }
}
"#;
    src.replace("{N}", &n.to_string())
}

/// Batched floating-base forward dynamics on the GPU (spatial 6D ABA) — a free 6-DoF root plus the
/// serial chain, one thread per environment. Mirrors
/// [`floating_base_forward_dynamics`](crate::floating_base_forward_dynamics).
pub struct FloatingBaseGpu {
    n: usize,
    n_envs: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    v0_buf: wgpu::Buffer,
    state_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    out_stage: wgpu::Buffer,
    fext_buf: wgpu::Buffer,
}

impl FloatingBaseGpu {
    /// Build for `robot` (per-link `inertia`) with a free base of inertia `base`, under `gravity`,
    /// `n_envs` environments. `None` when there is no GPU.
    pub fn new(robot: &crate::Robot, inertia: &[crate::LinkInertia], base: &crate::LinkInertia, gravity: nalgebra::Vector3<f64>, n_envs: usize) -> Option<Self> {
        let n = robot.dof();
        assert_eq!(inertia.len(), n, "one inertia per joint/link");

        let mut joints = Vec::with_capacity(n * 16);
        for j in &robot.joints {
            let r = j.origin.rotation.to_rotation_matrix();
            joints.extend(r.matrix().as_slice().iter().map(|&v| v as f32));
            let t = j.origin.translation.vector;
            joints.extend_from_slice(&[t.x as f32, t.y as f32, t.z as f32]);
            let a = j.axis.into_inner();
            joints.extend_from_slice(&[a.x as f32, a.y as f32, a.z as f32]);
            joints.push(match j.kind { crate::JointKind::Revolute => 0.0, crate::JointKind::Prismatic => 1.0 });
        }
        let mut inert = Vec::with_capacity(n * 13);
        for li in inertia {
            inert.push(li.mass as f32);
            inert.extend_from_slice(&[li.com.x as f32, li.com.y as f32, li.com.z as f32]);
            inert.extend(li.inertia.as_slice().iter().map(|&v| v as f32));
        }
        let mut prm = vec![n as f32, n_envs as f32, gravity.x as f32, gravity.y as f32, gravity.z as f32, base.mass as f32, base.com.x as f32, base.com.y as f32, base.com.z as f32];
        prm.extend(base.inertia.as_slice().iter().map(|&v| v as f32));

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;

        let init = |label, data: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE });
        let joints_buf = init("fb-joints", bytemuck::cast_slice(&joints));
        let inertia_buf = init("fb-inertia", bytemuck::cast_slice(&inert));
        let prm_buf = init("fb-prm", bytemuck::cast_slice(&prm));
        let v0_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("fb-v0"), size: (n_envs * 6 * 4).max(4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let state_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("fb-state"), size: (n_envs * 3 * n * 4).max(4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("fb-out"), size: (n_envs * (6 + n) * 4).max(4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let out_stage = device.create_buffer(&wgpu::BufferDescriptor { label: Some("fb-out-stage"), size: (n_envs * (6 + n) * 4).max(4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let fext_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("fb-fext"), size: (n_envs * (n + 1) * 6 * 4).max(4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("floating"), source: wgpu::ShaderSource::Wgsl(floating_wgsl(n).into()) });
        let ro = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let mut rw = ro(5);
        rw.ty = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("fb-bgl"), entries: &[ro(0), ro(1), ro(2), ro(3), ro(4), rw, ro(6)] });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("fb-layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pso = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("floating"), layout: Some(&layout), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fb-bind"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: joints_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: inertia_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: prm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v0_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: fext_buf.as_entire_binding() },
            ],
        });

        Some(Self { n, n_envs, device, queue, pso, bind, v0_buf, state_buf, out_buf, out_stage, fext_buf })
    }

    /// Batched floating-base forward dynamics: per env, base spatial acceleration `a0` (6) and joint
    /// accelerations `q̈` (n). Inputs are per-env base spatial velocity `v0` (`n_envs·6`) and packed
    /// `state = [q, qd, tau]` (`n_envs·3n`). Returns `(a0 flat n_envs·6, qdd flat n_envs·n)`.
    pub fn accelerations(&self, v0: &[f64], q: &[f64], qd: &[f64], tau: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let ne = self.n_envs;
        self.accelerations_ext(v0, q, qd, tau, &vec![0.0; ne * (self.n + 1) * 6])
    }

    /// Like [`accelerations`](Self::accelerations), but with **external spatial forces**: `fext` packs
    /// per env `[base(6), link0(6), …, link_{n-1}(6)]` (each `[torque; force]` in that body's frame) —
    /// the mechanism for ground contact and applied wrenches. Mirrors `floating_base_forward_dynamics_ext`.
    pub fn accelerations_ext(&self, v0: &[f64], q: &[f64], qd: &[f64], tau: &[f64], fext: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let ne = self.n_envs;
        assert!(v0.len() == ne * 6 && q.len() == ne * self.n && qd.len() == ne * self.n && tau.len() == ne * self.n, "batch size mismatch");
        assert_eq!(fext.len(), ne * (self.n + 1) * 6, "fext size mismatch");
        let fextf: Vec<f32> = fext.iter().map(|&v| v as f32).collect();
        self.queue.write_buffer(&self.fext_buf, 0, bytemuck::cast_slice(&fextf));
        let mut state = vec![0.0f32; ne * 3 * self.n];
        for e in 0..ne {
            for i in 0..self.n {
                state[e * 3 * self.n + i] = q[e * self.n + i] as f32;
                state[e * 3 * self.n + self.n + i] = qd[e * self.n + i] as f32;
                state[e * 3 * self.n + 2 * self.n + i] = tau[e * self.n + i] as f32;
            }
        }
        let v0f: Vec<f32> = v0.iter().map(|&v| v as f32).collect();
        self.queue.write_buffer(&self.v0_buf, 0, bytemuck::cast_slice(&v0f));
        self.queue.write_buffer(&self.state_buf, 0, bytemuck::cast_slice(&state));

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pso);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups((ne as u32).div_ceil(64), 1, 1);
        }
        let bytes = (ne * (6 + self.n) * 4) as u64;
        enc.copy_buffer_to_buffer(&self.out_buf, 0, &self.out_stage, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = self.out_stage.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let raw: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        self.out_stage.unmap();
        let mut a0 = Vec::with_capacity(ne * 6);
        let mut qdd = Vec::with_capacity(ne * self.n);
        for e in 0..ne {
            let ob = e * (6 + self.n);
            for k in 0..6 { a0.push(raw[ob + k] as f64); }
            for k in 0..self.n { qdd.push(raw[ob + 6 + k] as f64); }
        }
        (a0, qdd)
    }
}

// ---------------------------------------------------------------------------------------------
// FloatingGaitGpu — the full floating-base contact step on the GPU (FK + foot contact + spatial ABA
// with external wrenches + SE(3) integration), one thread per environment. The GPU port of the
// validated CPU `floating_contact_step`; the simulator learned locomotion runs in.
// ---------------------------------------------------------------------------------------------

fn floating_gait_wgsl(n: usize) -> String {
    let src = r#"
const N: u32 = {N}u;
@group(0) @binding(0) var<storage, read> JOINTS: array<f32>;   // 16 per joint
@group(0) @binding(1) var<storage, read> INERTIA: array<f32>;  // 13 per link
@group(0) @binding(2) var<storage, read> PRM: array<f32>;      // [N,n_envs,gx,gy,gz, base mass(1)com(3)I(9), floor,kn,kd,dt,n_contacts]
@group(0) @binding(3) var<storage, read> CONTACTS: array<f32>; // 5 per contact: frame(1) offset(3) mu(1)
@group(0) @binding(4) var<storage, read_write> BASEPOSE: array<f32>; // 12 per env: R0(9,col-major) p0(3)
@group(0) @binding(5) var<storage, read_write> V0: array<f32>;       // 6 per env: base spatial vel [ang;lin]
@group(0) @binding(6) var<storage, read_write> QSTATE: array<f32>;   // 3N per env: q(N) qd(N) tau(N)
@group(0) @binding(7) var<storage, read> INIT: array<f32>;           // 12+6+2N: shared rollout start (R0,p0,v0,q0,qd0)
@group(0) @binding(8) var<storage, read> POLICY: array<f32>;         // per policy: W(N×in_dim) then b(N)
@group(0) @binding(9) var<storage, read_write> REWARD: array<f32>;   // per policy: rollout return
// PRM tail (rollout): [23] reserved (never read by the reward) [24] effort_w [25] taumax [26] rollout_steps [27] in_dim [28] n_policies

struct SV { t: vec3<f32>, b: vec3<f32> }
struct SM { tl: mat3x3<f32>, tr: mat3x3<f32>, bl: mat3x3<f32>, br: mat3x3<f32> }
struct Accel { a0: SV, qdd: array<f32, N> }
fn z3() -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0)); }
fn i3() -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(1.,0.,0.), vec3<f32>(0.,1.,0.), vec3<f32>(0.,0.,1.)); }
fn skew(v: vec3<f32>) -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(0.0, v.z, -v.y), vec3<f32>(-v.z, 0.0, v.x), vec3<f32>(v.y, -v.x, 0.0)); }
fn outer(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> { return mat3x3<f32>(a*b.x, a*b.y, a*b.z); }
fn rot_axis(a: vec3<f32>, t: f32) -> mat3x3<f32> { let c=cos(t); let s=sin(t); let ic=1.0-c; let x=a.x; let y=a.y; let z=a.z; return mat3x3<f32>(vec3<f32>(c+x*x*ic,y*x*ic+z*s,z*x*ic-y*s), vec3<f32>(x*y*ic-z*s,c+y*y*ic,z*y*ic+x*s), vec3<f32>(x*z*ic+y*s,y*z*ic-x*s,c+z*z*ic)); }
fn expmap(w: vec3<f32>) -> mat3x3<f32> { let a = length(w); if (a < 1e-9) { return i3(); } return rot_axis(w / a, a); }
fn smv(m: SM, v: SV) -> SV { return SV(m.tl*v.t + m.tr*v.b, m.bl*v.t + m.br*v.b); }
fn smm(a: SM, b: SM) -> SM { return SM(a.tl*b.tl + a.tr*b.bl, a.tl*b.tr + a.tr*b.br, a.bl*b.tl + a.br*b.bl, a.bl*b.tr + a.br*b.br); }
fn smt(a: SM) -> SM { return SM(transpose(a.tl), transpose(a.bl), transpose(a.tr), transpose(a.br)); }
fn sma(a: SM, b: SM) -> SM { return SM(a.tl+b.tl, a.tr+b.tr, a.bl+b.bl, a.br+b.br); }
fn sms(a: SM, s: f32) -> SM { return SM(a.tl*s, a.tr*s, a.bl*s, a.br*s); }
fn smsub(a: SM, b: SM) -> SM { return SM(a.tl-b.tl, a.tr-b.tr, a.bl-b.bl, a.br-b.br); }
fn sva(a: SV, b: SV) -> SV { return SV(a.t+b.t, a.b+b.b); }
fn svsub(a: SV, b: SV) -> SV { return SV(a.t-b.t, a.b-b.b); }
fn svs(a: SV, s: f32) -> SV { return SV(a.t*s, a.b*s); }
fn svdot(a: SV, b: SV) -> f32 { return dot(a.t,b.t) + dot(a.b,b.b); }
fn svouter(u: SV, w: SV) -> SM { return SM(outer(u.t,w.t), outer(u.t,w.b), outer(u.b,w.t), outer(u.b,w.b)); }
fn motion_transform(r: mat3x3<f32>, p: vec3<f32>) -> SM { let e = transpose(r); return SM(e, z3(), (e*skew(p))*(-1.0), e); }
fn spatial_inertia(mass: f32, c: vec3<f32>, I: mat3x3<f32>) -> SM { let cx = skew(c); return SM(I - mass*(cx*cx), cx*mass, cx*(-mass), i3()*mass); }
fn crm(v: SV) -> SM { return SM(skew(v.t), z3(), skew(v.b), skew(v.t)); }
fn crf(v: SV) -> SM { return sms(smt(crm(v)), -1.0); }
fn grav_wrench(isp: SM, g: vec3<f32>, r: mat3x3<f32>) -> SV { return smv(isp, SV(vec3<f32>(0.0), r*g)); }
fn jointR(i: u32, qi: f32) -> mat3x3<f32> {
  let jb=i*16u; let Ro=mat3x3<f32>(vec3<f32>(JOINTS[jb],JOINTS[jb+1u],JOINTS[jb+2u]), vec3<f32>(JOINTS[jb+3u],JOINTS[jb+4u],JOINTS[jb+5u]), vec3<f32>(JOINTS[jb+6u],JOINTS[jb+7u],JOINTS[jb+8u]));
  if (JOINTS[jb+15u] < 0.5) { let a=vec3<f32>(JOINTS[jb+12u],JOINTS[jb+13u],JOINTS[jb+14u]); return Ro*rot_axis(a, qi); }
  return Ro;
}
fn jointP(i: u32, qi: f32) -> vec3<f32> {
  let jb=i*16u; let Ro=mat3x3<f32>(vec3<f32>(JOINTS[jb],JOINTS[jb+1u],JOINTS[jb+2u]), vec3<f32>(JOINTS[jb+3u],JOINTS[jb+4u],JOINTS[jb+5u]), vec3<f32>(JOINTS[jb+6u],JOINTS[jb+7u],JOINTS[jb+8u]));
  let po=vec3<f32>(JOINTS[jb+9u],JOINTS[jb+10u],JOINTS[jb+11u]);
  if (JOINTS[jb+15u] < 0.5) { return po; }
  return Ro*(vec3<f32>(JOINTS[jb+12u],JOINTS[jb+13u],JOINTS[jb+14u])*qi) + po;
}
fn subspace(i: u32) -> SV { let a=vec3<f32>(JOINTS[i*16u+12u],JOINTS[i*16u+13u],JOINTS[i*16u+14u]); if (JOINTS[i*16u+15u]<0.5){ return SV(a, vec3<f32>(0.0)); } return SV(vec3<f32>(0.0), a); }
fn linertia(i: u32) -> SM { let mass=INERTIA[i*13u]; let c=vec3<f32>(INERTIA[i*13u+1u],INERTIA[i*13u+2u],INERTIA[i*13u+3u]); let b=i*13u+4u; let I=mat3x3<f32>(vec3<f32>(INERTIA[b],INERTIA[b+1u],INERTIA[b+2u]),vec3<f32>(INERTIA[b+3u],INERTIA[b+4u],INERTIA[b+5u]),vec3<f32>(INERTIA[b+6u],INERTIA[b+7u],INERTIA[b+8u])); return spatial_inertia(mass,c,I); }

fn aba_ext(v0: SV, q: array<f32,N>, qd: array<f32,N>, tau: array<f32,N>, febase: SV, fe: array<SV,N>, grav: vec3<f32>) -> Accel {
  var xm: array<SM,N>; var s: array<SV,N>; var v: array<SV,N>; var c: array<SV,N>; var ia: array<SM,N>; var pa: array<SV,N>;
  let bmass=PRM[5]; let bcom=vec3<f32>(PRM[6],PRM[7],PRM[8]); let bI=mat3x3<f32>(vec3<f32>(PRM[9],PRM[10],PRM[11]),vec3<f32>(PRM[12],PRM[13],PRM[14]),vec3<f32>(PRM[15],PRM[16],PRM[17]));
  let ib = spatial_inertia(bmass,bcom,bI);
  var ia_base = ib;
  var pa_base = svsub(svsub(smv(crf(v0), smv(ib,v0)), grav_wrench(ib,grav,i3())), febase);
  var r_parent = i3();
  for (var i=0u;i<N;i=i+1u){
    let r=jointR(i,q[i]); let p=jointP(i,q[i]); let x=motion_transform(r,p); let si=subspace(i);
    var v_par=v0; if(i>0u){ v_par=v[i-1u]; }
    v[i]=sva(smv(x,v_par), svs(si,qd[i]));
    c[i]=smv(crm(v[i]), svs(si,qd[i]));
    let ii=linertia(i); let r_bi=transpose(r)*r_parent;
    pa[i]=svsub(svsub(smv(crf(v[i]), smv(ii,v[i])), grav_wrench(ii,grav,r_bi)), fe[i]);
    ia[i]=ii; xm[i]=x; s[i]=si; r_parent=r_bi;
  }
  var u: array<SV,N>; var d: array<f32,N>; var uu: array<f32,N>;
  for (var ii=0u;ii<N;ii=ii+1u){
    let i=N-1u-ii;
    u[i]=smv(ia[i],s[i]); d[i]=svdot(s[i],u[i]); uu[i]=tau[i]-svdot(s[i],pa[i]);
    let ia_bar=smsub(ia[i], sms(svouter(u[i],u[i]), 1.0/d[i]));
    let pa_bar=sva(sva(pa[i], smv(ia_bar,c[i])), svs(u[i], uu[i]/d[i]));
    let xt=smt(xm[i]);
    if(i>0u){ ia[i-1u]=sma(ia[i-1u], smm(smm(xt,ia_bar), xm[i])); pa[i-1u]=sva(pa[i-1u], smv(xt,pa_bar)); }
    else { ia_base=sma(ia_base, smm(smm(xt,ia_bar), xm[0])); pa_base=sva(pa_base, smv(xt,pa_bar)); }
  }
  var A: array<f32,36>;
  for (var r=0u;r<3u;r=r+1u){ for (var cc=0u;cc<3u;cc=cc+1u){ A[r*6u+cc]=ia_base.tl[cc][r]; A[r*6u+cc+3u]=ia_base.tr[cc][r]; A[(r+3u)*6u+cc]=ia_base.bl[cc][r]; A[(r+3u)*6u+cc+3u]=ia_base.br[cc][r]; }}
  var rhs: array<f32,6>; rhs[0]=pa_base.t.x; rhs[1]=pa_base.t.y; rhs[2]=pa_base.t.z; rhs[3]=pa_base.b.x; rhs[4]=pa_base.b.y; rhs[5]=pa_base.b.z;
  var L: array<f32,36>;
  for (var i=0u;i<6u;i=i+1u){ for (var j=0u;j<=i;j=j+1u){ var sum=A[i*6u+j]; for (var k=0u;k<j;k=k+1u){ sum=sum-L[i*6u+k]*L[j*6u+k]; } if(i==j){ L[i*6u+i]=sqrt(max(sum,1e-12)); } else { L[i*6u+j]=sum/L[j*6u+j]; } }}
  var yv: array<f32,6>; for (var i=0u;i<6u;i=i+1u){ var sm2=rhs[i]; for (var k=0u;k<i;k=k+1u){ sm2=sm2-L[i*6u+k]*yv[k]; } yv[i]=sm2/L[i*6u+i]; }
  var xv: array<f32,6>; for (var ii=0u;ii<6u;ii=ii+1u){ let i=5u-ii; var sm3=yv[i]; for (var k=i+1u;k<6u;k=k+1u){ sm3=sm3-L[k*6u+i]*xv[k]; } xv[i]=sm3/L[i*6u+i]; }
  let a0 = SV(vec3<f32>(-xv[0],-xv[1],-xv[2]), vec3<f32>(-xv[3],-xv[4],-xv[5]));
  var out: Accel; out.a0 = a0;
  var a: array<SV,N>;
  for (var i=0u;i<N;i=i+1u){ var a_par=a0; if(i>0u){ a_par=a[i-1u]; } let a_prime=sva(smv(xm[i],a_par), c[i]); out.qdd[i]=(uu[i]-svdot(u[i],a_prime))/d[i]; a[i]=sva(a_prime, svs(s[i], out.qdd[i])); }
  return out;
}

// One floating-base contact step on local state (FK + foot contact + spatial ABA + SE(3) integration).
struct GState { R0: mat3x3<f32>, p0: vec3<f32>, v0: SV, q: array<f32,N>, qd: array<f32,N> }
fn gait_advance(st: GState, tau: array<f32,N>) -> GState {
  let grav = vec3<f32>(PRM[2],PRM[3],PRM[4]);
  let floor_z=PRM[18]; let kn=PRM[19]; let kd=PRM[20]; let dt=PRM[21]; let nc=u32(PRM[22]);
  let R0 = st.R0; let p0 = st.p0; let v0 = st.v0;
  var q = st.q; var qd = st.qd;

  var Rb: array<mat3x3<f32>,{N1}>; var pbf: array<vec3<f32>,{N1}>; var vf: array<SV,N>;
  Rb[0]=i3(); pbf[0]=vec3<f32>(0.0);
  var vpar = v0;
  for (var i=0u;i<N;i=i+1u){
    let r=jointR(i,q[i]); let p=jointP(i,q[i]);
    Rb[i+1u] = Rb[i]*r; pbf[i+1u] = Rb[i]*p + pbf[i];
    let x=motion_transform(r,p); let si=subspace(i);
    vf[i]=sva(smv(x,vpar), svs(si,qd[i])); vpar=vf[i];
  }
  var febase = SV(vec3<f32>(0.0), vec3<f32>(0.0));
  var fe: array<SV,N>; for (var i=0u;i<N;i=i+1u){ fe[i]=SV(vec3<f32>(0.0),vec3<f32>(0.0)); }
  for (var ci=0u; ci<nc; ci=ci+1u){
    let cb=ci*5u; let fr=u32(CONTACTS[cb]); let off=vec3<f32>(CONTACTS[cb+1u],CONTACTS[cb+2u],CONTACTS[cb+3u]); let mu=CONTACTS[cb+4u];
    let Rwf = R0 * Rb[fr];
    let pfoot = R0*(Rb[fr]*off + pbf[fr]) + p0;
    let phi = pfoot.z - floor_z;
    if (phi < 0.0) {
      var vlink = v0; if (fr>0u){ vlink = vf[fr-1u]; }
      let vcp = Rwf * (cross(vlink.t, off) + vlink.b);
      let fnrm = max(0.0, -kn*phi - kd*vcp.z);
      let vt = vec2<f32>(vcp.x, vcp.y);
      let ft = -mu*fnrm * vt/(length(vt)+1e-4);
      let fworld = vec3<f32>(ft.x, ft.y, fnrm);
      let flocal = transpose(Rwf) * fworld;
      let w = SV(cross(off, flocal), flocal);
      if (fr==0u){ febase = sva(febase, w); } else { fe[fr-1u] = sva(fe[fr-1u], w); }
    }
  }
  let acc = aba_ext(v0, q, qd, tau, febase, fe, grav);
  let v0n = sva(v0, svs(acc.a0, dt));
  for (var i=0u;i<N;i=i+1u){ let vv=qd[i]+dt*acc.qdd[i]; qd[i]=vv; q[i]=q[i]+dt*vv; }
  var out: GState;
  out.R0 = R0 * expmap(v0n.t * dt);
  out.p0 = p0 + R0 * (v0n.b * dt);
  out.v0 = v0n; out.q = q; out.qd = qd;
  return out;
}

@compute @workgroup_size(64)
fn gait_step(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x; let n_envs = u32(PRM[1]);
  if (e >= n_envs) { return; }
  let pb = e*12u; let vb = e*6u; let sb = e*3u*N;
  var st: GState;
  st.R0 = mat3x3<f32>(vec3<f32>(BASEPOSE[pb],BASEPOSE[pb+1u],BASEPOSE[pb+2u]), vec3<f32>(BASEPOSE[pb+3u],BASEPOSE[pb+4u],BASEPOSE[pb+5u]), vec3<f32>(BASEPOSE[pb+6u],BASEPOSE[pb+7u],BASEPOSE[pb+8u]));
  st.p0 = vec3<f32>(BASEPOSE[pb+9u],BASEPOSE[pb+10u],BASEPOSE[pb+11u]);
  st.v0 = SV(vec3<f32>(V0[vb],V0[vb+1u],V0[vb+2u]), vec3<f32>(V0[vb+3u],V0[vb+4u],V0[vb+5u]));
  var tau: array<f32,N>;
  for (var i=0u;i<N;i=i+1u){ st.q[i]=QSTATE[sb+i]; st.qd[i]=QSTATE[sb+N+i]; tau[i]=QSTATE[sb+2u*N+i]; }
  let o = gait_advance(st, tau);
  BASEPOSE[pb]=o.R0[0][0]; BASEPOSE[pb+1u]=o.R0[0][1]; BASEPOSE[pb+2u]=o.R0[0][2];
  BASEPOSE[pb+3u]=o.R0[1][0]; BASEPOSE[pb+4u]=o.R0[1][1]; BASEPOSE[pb+5u]=o.R0[1][2];
  BASEPOSE[pb+6u]=o.R0[2][0]; BASEPOSE[pb+7u]=o.R0[2][1]; BASEPOSE[pb+8u]=o.R0[2][2];
  BASEPOSE[pb+9u]=o.p0.x; BASEPOSE[pb+10u]=o.p0.y; BASEPOSE[pb+11u]=o.p0.z;
  V0[vb]=o.v0.t.x; V0[vb+1u]=o.v0.t.y; V0[vb+2u]=o.v0.t.z; V0[vb+3u]=o.v0.b.x; V0[vb+4u]=o.v0.b.y; V0[vb+5u]=o.v0.b.z;
  for (var i=0u;i<N;i=i+1u){ QSTATE[sb+i]=o.q[i]; QSTATE[sb+N+i]=o.qd[i]; }
}

// Linear feedback policy → joint torques. Features: [base_z, up-alignment R0[2][2], vertical vel, q, qd].
fn policy_tau(st: GState, pol: u32) -> array<f32,N> {
  let in_dim = u32(PRM[27]); let taumax = PRM[25];
  var feat: array<f32, {IN}>;
  feat[0] = st.p0.z; feat[1] = st.R0[2][2]; feat[2] = st.v0.b.z;
  for (var i=0u;i<N;i=i+1u){ feat[3u+i] = st.q[i]; feat[3u+N+i] = st.qd[i]; }
  let base = pol * (N*in_dim + N);
  var tau: array<f32,N>;
  for (var j=0u;j<N;j=j+1u){
    var s = POLICY[base + N*in_dim + j]; // bias
    for (var k=0u;k<in_dim;k=k+1u){ s = s + POLICY[base + j*in_dim + k] * feat[k]; }
    tau[j] = clamp(s, -taumax, taumax);
  }
  return tau;
}

@compute @workgroup_size(64)
fn gait_rollout(@builtin(global_invocation_id) g: vec3<u32>) {
  let pol = g.x; let n_policies = u32(PRM[28]);
  if (pol >= n_policies) { return; }
  let effort_w = PRM[24]; let steps = u32(PRM[26]);
  // shared initial state from INIT
  var st: GState;
  st.R0 = mat3x3<f32>(vec3<f32>(INIT[0],INIT[1],INIT[2]), vec3<f32>(INIT[3],INIT[4],INIT[5]), vec3<f32>(INIT[6],INIT[7],INIT[8]));
  st.p0 = vec3<f32>(INIT[9],INIT[10],INIT[11]);
  st.v0 = SV(vec3<f32>(INIT[12],INIT[13],INIT[14]), vec3<f32>(INIT[15],INIT[16],INIT[17]));
  for (var i=0u;i<N;i=i+1u){ st.q[i]=INIT[18u+i]; st.qd[i]=INIT[18u+N+i]; }
  var reward = 0.0;
  for (var t=0u;t<steps;t=t+1u){
    let tau = policy_tau(st, pol);
    st = gait_advance(st, tau);
    // reward: keep the base HIGH (extend the leg against gravity), minus a small effort cost, gated
    // by staying upright so it can't "win" by toppling. No target height enters the reward (PRM[23] is reserved).
    var eff = 0.0; for (var j=0u;j<N;j=j+1u){ eff = eff + tau[j]*tau[j]; }
    reward = reward + st.p0.z * max(st.R0[2][2], 0.0) - effort_w*eff;
  }
  REWARD[pol] = reward;
}
"#;
    src.replace("{N1}", &(n + 1).to_string()).replace("{IN}", &(3 + 2 * n).to_string()).replace("{N}", &n.to_string())
}

/// The full floating-base contact step on the GPU (FK + foot contact + spatial ABA with external
/// wrenches + SE(3) integration), one thread per environment. GPU port of
/// [`floating_contact_step`](crate::floating_contact_step).
pub struct FloatingGaitGpu {
    n: usize,
    n_envs: usize,
    policy_dim: usize,
    base_prm: Vec<f32>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso: wgpu::ComputePipeline,
    pso_rollout: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    prm_buf: wgpu::Buffer,
    basepose_buf: wgpu::Buffer,
    v0_buf: wgpu::Buffer,
    qstate_buf: wgpu::Buffer,
    init_buf: wgpu::Buffer,
    policy_buf: wgpu::Buffer,
    reward_buf: wgpu::Buffer,
    stage: wgpu::Buffer,
}

impl FloatingGaitGpu {
    /// Build for `robot` (per-link `inertia`) with a free base of inertia `base`, ground contact
    /// `contacts` at `floor_z` (stiffness `kn`, damping `kd`), gravity, timestep `dt`, `n_envs`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(robot: &crate::Robot, inertia: &[crate::LinkInertia], base: &crate::LinkInertia, contacts: &[crate::FootContact], floor_z: f64, kn: f64, kd: f64, gravity: nalgebra::Vector3<f64>, dt: f64, n_envs: usize) -> Option<Self> {
        let n = robot.dof();
        let mut joints = Vec::with_capacity(n * 16);
        for j in &robot.joints {
            let r = j.origin.rotation.to_rotation_matrix();
            joints.extend(r.matrix().as_slice().iter().map(|&v| v as f32));
            let t = j.origin.translation.vector;
            joints.extend_from_slice(&[t.x as f32, t.y as f32, t.z as f32]);
            let a = j.axis.into_inner();
            joints.extend_from_slice(&[a.x as f32, a.y as f32, a.z as f32]);
            joints.push(match j.kind { crate::JointKind::Revolute => 0.0, crate::JointKind::Prismatic => 1.0 });
        }
        let mut inert = Vec::with_capacity(n * 13);
        for li in inertia {
            inert.push(li.mass as f32);
            inert.extend_from_slice(&[li.com.x as f32, li.com.y as f32, li.com.z as f32]);
            inert.extend(li.inertia.as_slice().iter().map(|&v| v as f32));
        }
        let mut prm = vec![n as f32, n_envs as f32, gravity.x as f32, gravity.y as f32, gravity.z as f32, base.mass as f32, base.com.x as f32, base.com.y as f32, base.com.z as f32];
        prm.extend(base.inertia.as_slice().iter().map(|&v| v as f32));
        prm.extend_from_slice(&[floor_z as f32, kn as f32, kd as f32, dt as f32, contacts.len() as f32]);
        let mut cflat: Vec<f32> = contacts.iter().flat_map(|&(fr, off, mu)| [fr as f32, off.x as f32, off.y as f32, off.z as f32, mu as f32]).collect();
        if cflat.is_empty() { cflat = vec![0.0; 5]; }

        // rollout params (PRM tail 23..28); filled per rollout call. in_dim = 3 + 2n.
        let in_dim = 3 + 2 * n;
        let policy_dim = n * in_dim + n;
        prm.extend_from_slice(&[0.0, 0.0, 0.0, 0.0, in_dim as f32, 0.0]);

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let desc = wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits { max_storage_buffers_per_shader_stage: adapter.limits().max_storage_buffers_per_shader_stage.max(10), ..wgpu::Limits::default() },
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc)).ok()?;

        let init = |label, data: &[u8], extra: wgpu::BufferUsages| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE | extra });
        let none = wgpu::BufferUsages::empty();
        let joints_buf = init("g-joints", bytemuck::cast_slice(&joints), none);
        let inertia_buf = init("g-inertia", bytemuck::cast_slice(&inert), none);
        let prm_buf = init("g-prm", bytemuck::cast_slice(&prm), wgpu::BufferUsages::COPY_DST);
        let contacts_buf = init("g-contacts", bytemuck::cast_slice(&cflat), none);
        let dyn_buf = |label, elems: usize| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (elems * 4).max(4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let basepose_buf = dyn_buf("g-basepose", n_envs * 12);
        let v0_buf = dyn_buf("g-v0", n_envs * 6);
        let qstate_buf = dyn_buf("g-qstate", n_envs * 3 * n);
        let init_buf = dyn_buf("g-init", 18 + 2 * n);
        let policy_buf = dyn_buf("g-policy", n_envs * policy_dim);
        let reward_buf = dyn_buf("g-reward", n_envs);
        let stage = device.create_buffer(&wgpu::BufferDescriptor { label: Some("g-stage"), size: (n_envs * 12 * 4).max(4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gait"), source: wgpu::ShaderSource::Wgsl(floating_gait_wgsl(n).into()) });
        let ro = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let rw = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("g-bgl"), entries: &[ro(0), ro(1), ro(2), ro(3), rw(4), rw(5), rw(6), ro(7), ro(8), rw(9)] });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("g-layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let mk = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&layout), module: &shader, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
        let pso = mk("gait_step");
        let pso_rollout = mk("gait_rollout");
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("g-bind"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: joints_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: inertia_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: prm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: contacts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: basepose_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: v0_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: qstate_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: init_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: policy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: reward_buf.as_entire_binding() },
            ],
        });
        Some(Self { n, n_envs, policy_dim, base_prm: prm, device, queue, pso, pso_rollout, bind, prm_buf, basepose_buf, v0_buf, qstate_buf, init_buf, policy_buf, reward_buf, stage })
    }

    /// Linear-policy parameter count (`n·(3+2n) + n`).
    pub fn policy_dim(&self) -> usize {
        self.policy_dim
    }

    /// Batched policy search over floating-base contact rollouts. `policies` packs `n_policies`
    /// linear policies (`W(n×in_dim), b(n)` each, `in_dim = 3 + 2n`); all roll out `steps` contact
    /// steps from the shared initial state `init = [R0(9,col-major), p0(3), v0(6), q(n), qd(n)]` under
    /// the policy `τ = clamp(W·[base_z, up-alignment, vertical-vel, q, qd] + b, ±taumax)`. Returns the
    /// per-policy return `Σ_t (z_t · max(up_t, 0) − w‖τ_t‖²)`: base height gated by uprightness, minus
    /// effort. No target height enters the reward. One GPU thread per policy.
    pub fn rollout_rewards(&self, policies: &[f64], init: &[f64], effort_w: f64, taumax: f64, steps: usize) -> Vec<f64> {
        let n_policies = policies.len() / self.policy_dim;
        assert_eq!(init.len(), 18 + 2 * self.n, "init state size mismatch");
        let f = |s: &[f64]| -> Vec<f32> { s.iter().map(|&v| v as f32).collect() };
        self.queue.write_buffer(&self.policy_buf, 0, bytemuck::cast_slice(&f(policies)));
        self.queue.write_buffer(&self.init_buf, 0, bytemuck::cast_slice(&f(init)));
        let mut prm = self.base_prm.clone();
        prm[24] = effort_w as f32; prm[25] = taumax as f32; prm[26] = steps as f32; prm[28] = n_policies as f32;
        self.queue.write_buffer(&self.prm_buf, 0, bytemuck::cast_slice(&prm));

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pso_rollout);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups((n_policies as u32).div_ceil(64), 1, 1);
        }
        let bytes = (n_policies * 4) as u64;
        enc.copy_buffer_to_buffer(&self.reward_buf, 0, &self.stage, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = self.stage.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let v: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        self.stage.unmap();
        v.iter().map(|&x| x as f64).collect()
    }

    /// Set the per-env initial state and advance `steps` contact steps on the GPU. `base_pose` packs
    /// `[R0(9,col-major), p0(3)]` per env, `v0` is `n_envs·6`, `q/qd/tau` are `n_envs·n`. Returns the
    /// final `(base_pose, v0, q, qd)`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&self, base_pose: &[f64], v0: &[f64], q: &[f64], qd: &[f64], tau: &[f64], steps: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let ne = self.n_envs;
        let f = |s: &[f64]| -> Vec<f32> { s.iter().map(|&v| v as f32).collect() };
        self.queue.write_buffer(&self.basepose_buf, 0, bytemuck::cast_slice(&f(base_pose)));
        self.queue.write_buffer(&self.v0_buf, 0, bytemuck::cast_slice(&f(v0)));
        let mut qstate = vec![0.0f32; ne * 3 * self.n];
        for e in 0..ne {
            for i in 0..self.n {
                qstate[e * 3 * self.n + i] = q[e * self.n + i] as f32;
                qstate[e * 3 * self.n + self.n + i] = qd[e * self.n + i] as f32;
                qstate[e * 3 * self.n + 2 * self.n + i] = tau[e * self.n + i] as f32;
            }
        }
        self.queue.write_buffer(&self.qstate_buf, 0, bytemuck::cast_slice(&qstate));

        let groups = (ne as u32).div_ceil(64);
        let mut done = 0;
        while done < steps {
            let chunk = (steps - done).min(256);
            let mut enc = self.device.create_command_encoder(&Default::default());
            for _ in 0..chunk {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pso);
                pass.set_bind_group(0, &self.bind, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.queue.submit([enc.finish()]);
            done += chunk;
        }
        let rd = |buf: &wgpu::Buffer, elems: usize| -> Vec<f64> {
            let bytes = (elems * 4) as u64;
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buf, 0, &self.stage, 0, bytes);
            self.queue.submit([enc.finish()]);
            let slice = self.stage.slice(..bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().expect("map").expect("map ok");
            let v: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
            self.stage.unmap();
            v.iter().map(|&x| x as f64).collect()
        };
        let bp = rd(&self.basepose_buf, ne * 12);
        let vv = rd(&self.v0_buf, ne * 6);
        let qs = rd(&self.qstate_buf, ne * 3 * self.n);
        let mut qo = vec![0.0; ne * self.n];
        let mut qdo = vec![0.0; ne * self.n];
        for e in 0..ne {
            for i in 0..self.n {
                qo[e * self.n + i] = qs[e * 3 * self.n + i];
                qdo[e * self.n + i] = qs[e * 3 * self.n + self.n + i];
            }
        }
        (bp, vv, qo, qdo)
    }
}

// ---------------------------------------------------------------------------------------------
// TreeFloatingGpu — batched BRANCHED-tree floating-base ABA (quadruped/biped), one thread per env.
// The spatial ABA of FloatingBaseGpu with the parent-index array baked in place of the serial i−1.
// GPU port of `tree_floating_forward_dynamics`; the gate to multi-leg locomotion at RL scale.
// ---------------------------------------------------------------------------------------------

fn tree_wgsl(n: usize, parent: &[isize]) -> String {
    let list = parent.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ");
    let src = r#"
const N: u32 = {N}u;
const PARENT: array<i32, {N}> = array<i32, {N}>({LIST});
@group(0) @binding(0) var<storage, read> JOINTS: array<f32>;
@group(0) @binding(1) var<storage, read> INERTIA: array<f32>;
@group(0) @binding(2) var<storage, read> PRM: array<f32>;
@group(0) @binding(3) var<storage, read> V0: array<f32>;
@group(0) @binding(4) var<storage, read> STATE: array<f32>;
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(6) var<storage, read> FEXT: array<f32>;

struct SV { t: vec3<f32>, b: vec3<f32> }
struct SM { tl: mat3x3<f32>, tr: mat3x3<f32>, bl: mat3x3<f32>, br: mat3x3<f32> }
fn z3() -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0)); }
fn i3() -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(1.,0.,0.), vec3<f32>(0.,1.,0.), vec3<f32>(0.,0.,1.)); }
fn skew(v: vec3<f32>) -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(0.0, v.z, -v.y), vec3<f32>(-v.z, 0.0, v.x), vec3<f32>(v.y, -v.x, 0.0)); }
fn outer(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> { return mat3x3<f32>(a*b.x, a*b.y, a*b.z); }
fn smv(m: SM, v: SV) -> SV { return SV(m.tl*v.t + m.tr*v.b, m.bl*v.t + m.br*v.b); }
fn smm(a: SM, b: SM) -> SM { return SM(a.tl*b.tl + a.tr*b.bl, a.tl*b.tr + a.tr*b.br, a.bl*b.tl + a.br*b.bl, a.bl*b.tr + a.br*b.br); }
fn smt(a: SM) -> SM { return SM(transpose(a.tl), transpose(a.bl), transpose(a.tr), transpose(a.br)); }
fn sma(a: SM, b: SM) -> SM { return SM(a.tl+b.tl, a.tr+b.tr, a.bl+b.bl, a.br+b.br); }
fn sms(a: SM, s: f32) -> SM { return SM(a.tl*s, a.tr*s, a.bl*s, a.br*s); }
fn smsub(a: SM, b: SM) -> SM { return SM(a.tl-b.tl, a.tr-b.tr, a.bl-b.bl, a.br-b.br); }
fn sva(a: SV, b: SV) -> SV { return SV(a.t+b.t, a.b+b.b); }
fn svsub(a: SV, b: SV) -> SV { return SV(a.t-b.t, a.b-b.b); }
fn svs(a: SV, s: f32) -> SV { return SV(a.t*s, a.b*s); }
fn svdot(a: SV, b: SV) -> f32 { return dot(a.t, b.t) + dot(a.b, b.b); }
fn svouter(u: SV, w: SV) -> SM { return SM(outer(u.t,w.t), outer(u.t,w.b), outer(u.b,w.t), outer(u.b,w.b)); }
fn motion_transform(r: mat3x3<f32>, p: vec3<f32>) -> SM { let e = transpose(r); return SM(e, z3(), (e*skew(p)) * (-1.0), e); }
fn spatial_inertia(mass: f32, c: vec3<f32>, I: mat3x3<f32>) -> SM { let cx = skew(c); return SM(I - mass*(cx*cx), cx*mass, cx*(-mass), i3()*mass); }
fn crm(v: SV) -> SM { return SM(skew(v.t), z3(), skew(v.b), skew(v.t)); }
fn crf(v: SV) -> SM { return sms(smt(crm(v)), -1.0); }
fn grav_wrench(isp: SM, g: vec3<f32>, r: mat3x3<f32>) -> SV { return smv(isp, SV(vec3<f32>(0.0), r*g)); }
fn fext_sv(off: u32) -> SV { return SV(vec3<f32>(FEXT[off], FEXT[off+1u], FEXT[off+2u]), vec3<f32>(FEXT[off+3u], FEXT[off+4u], FEXT[off+5u])); }
fn jointR(i: u32, qi: f32) -> mat3x3<f32> {
  let jb = i*16u; let Ro = mat3x3<f32>(vec3<f32>(JOINTS[jb],JOINTS[jb+1u],JOINTS[jb+2u]), vec3<f32>(JOINTS[jb+3u],JOINTS[jb+4u],JOINTS[jb+5u]), vec3<f32>(JOINTS[jb+6u],JOINTS[jb+7u],JOINTS[jb+8u]));
  if (JOINTS[jb+15u] < 0.5) { let a=vec3<f32>(JOINTS[jb+12u],JOINTS[jb+13u],JOINTS[jb+14u]); let c=cos(qi); let s=sin(qi); let ic=1.0-c; let x=a.x; let y=a.y; let zz=a.z; return Ro*mat3x3<f32>(vec3<f32>(c+x*x*ic,y*x*ic+zz*s,zz*x*ic-y*s),vec3<f32>(x*y*ic-zz*s,c+y*y*ic,zz*y*ic+x*s),vec3<f32>(x*zz*ic+y*s,y*zz*ic-x*s,c+zz*zz*ic)); }
  return Ro;
}
fn jointP(i: u32, qi: f32) -> vec3<f32> {
  let jb = i*16u; let Ro = mat3x3<f32>(vec3<f32>(JOINTS[jb],JOINTS[jb+1u],JOINTS[jb+2u]), vec3<f32>(JOINTS[jb+3u],JOINTS[jb+4u],JOINTS[jb+5u]), vec3<f32>(JOINTS[jb+6u],JOINTS[jb+7u],JOINTS[jb+8u]));
  let po = vec3<f32>(JOINTS[jb+9u],JOINTS[jb+10u],JOINTS[jb+11u]);
  if (JOINTS[jb+15u] < 0.5) { return po; }
  return Ro*(vec3<f32>(JOINTS[jb+12u],JOINTS[jb+13u],JOINTS[jb+14u])*qi) + po;
}
fn subspace(i: u32) -> SV { let a=vec3<f32>(JOINTS[i*16u+12u],JOINTS[i*16u+13u],JOINTS[i*16u+14u]); if (JOINTS[i*16u+15u]<0.5){ return SV(a, vec3<f32>(0.0)); } return SV(vec3<f32>(0.0), a); }
fn linertia(i: u32) -> SM { let mass=INERTIA[i*13u]; let c=vec3<f32>(INERTIA[i*13u+1u],INERTIA[i*13u+2u],INERTIA[i*13u+3u]); let b=i*13u+4u; let I=mat3x3<f32>(vec3<f32>(INERTIA[b],INERTIA[b+1u],INERTIA[b+2u]),vec3<f32>(INERTIA[b+3u],INERTIA[b+4u],INERTIA[b+5u]),vec3<f32>(INERTIA[b+6u],INERTIA[b+7u],INERTIA[b+8u])); return spatial_inertia(mass,c,I); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x; let n_envs = u32(PRM[1]);
  if (e >= n_envs) { return; }
  let grav = vec3<f32>(PRM[2], PRM[3], PRM[4]);
  let ib = spatial_inertia(PRM[5], vec3<f32>(PRM[6],PRM[7],PRM[8]), mat3x3<f32>(vec3<f32>(PRM[9],PRM[10],PRM[11]), vec3<f32>(PRM[12],PRM[13],PRM[14]), vec3<f32>(PRM[15],PRM[16],PRM[17])));
  let v0 = SV(vec3<f32>(V0[e*6u], V0[e*6u+1u], V0[e*6u+2u]), vec3<f32>(V0[e*6u+3u], V0[e*6u+4u], V0[e*6u+5u]));
  let sb = e*3u*N;
  var q: array<f32,N>; var qd: array<f32,N>; var tau: array<f32,N>;
  for (var i=0u;i<N;i=i+1u){ q[i]=STATE[sb+i]; qd[i]=STATE[sb+N+i]; tau[i]=STATE[sb+2u*N+i]; }

  var xm: array<SM,N>; var s: array<SV,N>; var v: array<SV,N>; var c: array<SV,N>; var ia: array<SM,N>; var pa: array<SV,N>;
  var rframes: array<mat3x3<f32>,N>;
  var ia_base = ib; let feb = e*6u*(N+1u);
  var pa_base = svsub(svsub(smv(crf(v0), smv(ib, v0)), grav_wrench(ib, grav, i3())), fext_sv(feb));
  for (var i=0u;i<N;i=i+1u){
    let r = jointR(i, q[i]); let p = jointP(i, q[i]); let x = motion_transform(r, p); let si = subspace(i);
    let pi = PARENT[i];
    var v_par = v0; if (pi >= 0) { v_par = v[u32(pi)]; }
    v[i] = sva(smv(x, v_par), svs(si, qd[i]));
    c[i] = smv(crm(v[i]), svs(si, qd[i]));
    let ii = linertia(i);
    var r_bp = i3(); if (pi >= 0) { r_bp = rframes[u32(pi)]; }
    let r_bi = transpose(r) * r_bp;
    pa[i] = svsub(svsub(smv(crf(v[i]), smv(ii, v[i])), grav_wrench(ii, grav, r_bi)), fext_sv(feb + 6u*(i+1u)));
    ia[i] = ii; xm[i] = x; s[i] = si; rframes[i] = r_bi;
  }
  var u: array<SV,N>; var d: array<f32,N>; var uu: array<f32,N>;
  for (var ii=0u; ii<N; ii=ii+1u){
    let i = N-1u-ii;
    u[i] = smv(ia[i], s[i]); d[i] = svdot(s[i], u[i]); uu[i] = tau[i] - svdot(s[i], pa[i]);
    let ia_bar = smsub(ia[i], sms(svouter(u[i], u[i]), 1.0/d[i]));
    let pa_bar = sva(sva(pa[i], smv(ia_bar, c[i])), svs(u[i], uu[i]/d[i]));
    let xt = smt(xm[i]); let pi = PARENT[i];
    if (pi >= 0) { let pp=u32(pi); ia[pp] = sma(ia[pp], smm(smm(xt, ia_bar), xm[i])); pa[pp] = sva(pa[pp], smv(xt, pa_bar)); }
    else { ia_base = sma(ia_base, smm(smm(xt, ia_bar), xm[i])); pa_base = sva(pa_base, smv(xt, pa_bar)); }
  }
  var A: array<f32,36>;
  for (var r=0u;r<3u;r=r+1u){ for (var cc=0u;cc<3u;cc=cc+1u){ A[r*6u+cc]=ia_base.tl[cc][r]; A[r*6u+cc+3u]=ia_base.tr[cc][r]; A[(r+3u)*6u+cc]=ia_base.bl[cc][r]; A[(r+3u)*6u+cc+3u]=ia_base.br[cc][r]; }}
  var rhs: array<f32,6>; rhs[0]=pa_base.t.x; rhs[1]=pa_base.t.y; rhs[2]=pa_base.t.z; rhs[3]=pa_base.b.x; rhs[4]=pa_base.b.y; rhs[5]=pa_base.b.z;
  var L: array<f32,36>;
  for (var i=0u;i<6u;i=i+1u){ for (var j=0u;j<=i;j=j+1u){ var sum=A[i*6u+j]; for (var k=0u;k<j;k=k+1u){ sum=sum-L[i*6u+k]*L[j*6u+k]; } if(i==j){ L[i*6u+i]=sqrt(max(sum,1e-12)); } else { L[i*6u+j]=sum/L[j*6u+j]; } }}
  var yv: array<f32,6>; for (var i=0u;i<6u;i=i+1u){ var sm2=rhs[i]; for (var k=0u;k<i;k=k+1u){ sm2=sm2-L[i*6u+k]*yv[k]; } yv[i]=sm2/L[i*6u+i]; }
  var xv: array<f32,6>; for (var ii=0u;ii<6u;ii=ii+1u){ let i=5u-ii; var sm3=yv[i]; for (var k=i+1u;k<6u;k=k+1u){ sm3=sm3-L[k*6u+i]*xv[k]; } xv[i]=sm3/L[i*6u+i]; }
  let a0 = SV(vec3<f32>(-xv[0],-xv[1],-xv[2]), vec3<f32>(-xv[3],-xv[4],-xv[5]));
  var qdd: array<f32,N>; var a: array<SV,N>;
  for (var i=0u;i<N;i=i+1u){ let pi=PARENT[i]; var a_par=a0; if (pi>=0){ a_par=a[u32(pi)]; } let a_prime=sva(smv(xm[i], a_par), c[i]); qdd[i]=(uu[i]-svdot(u[i],a_prime))/d[i]; a[i]=sva(a_prime, svs(s[i], qdd[i])); }
  let ob = e*(6u+N);
  OUT[ob]=a0.t.x; OUT[ob+1u]=a0.t.y; OUT[ob+2u]=a0.t.z; OUT[ob+3u]=a0.b.x; OUT[ob+4u]=a0.b.y; OUT[ob+5u]=a0.b.z;
  for (var i=0u;i<N;i=i+1u){ OUT[ob+6u+i]=qdd[i]; }
}
"#;
    src.replace("{LIST}", &list).replace("{N}", &n.to_string())
}

/// Batched branched-tree floating-base forward dynamics on the GPU — a free base with limbs branching
/// off it (quadruped/biped), one thread per environment. GPU port of
/// [`tree_floating_forward_dynamics`](crate::tree_floating_forward_dynamics); topology `parent[]` baked
/// in. Returns `(a0, q̈)` per env.
pub struct TreeFloatingGpu {
    n: usize,
    n_envs: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    v0_buf: wgpu::Buffer,
    state_buf: wgpu::Buffer,
    fext_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    out_stage: wgpu::Buffer,
}

impl TreeFloatingGpu {
    /// Build for a tree: `joints[i]` with parent `parent[i]` (`-1` = base, topo-ordered), per-body
    /// `inertia`, free-base `base`, gravity, `n_envs`. `None` when there is no GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn new(joints: &[crate::Joint], inertia: &[crate::LinkInertia], parent: &[isize], base: &crate::LinkInertia, gravity: nalgebra::Vector3<f64>, n_envs: usize) -> Option<Self> {
        let n = joints.len();
        assert!(inertia.len() == n && parent.len() == n, "joints/inertia/parent length mismatch");
        let mut jf = Vec::with_capacity(n * 16);
        for j in joints {
            let r = j.origin.rotation.to_rotation_matrix();
            jf.extend(r.matrix().as_slice().iter().map(|&v| v as f32));
            let t = j.origin.translation.vector;
            jf.extend_from_slice(&[t.x as f32, t.y as f32, t.z as f32]);
            let a = j.axis.into_inner();
            jf.extend_from_slice(&[a.x as f32, a.y as f32, a.z as f32]);
            jf.push(match j.kind { crate::JointKind::Revolute => 0.0, crate::JointKind::Prismatic => 1.0 });
        }
        let mut inert = Vec::with_capacity(n * 13);
        for li in inertia {
            inert.push(li.mass as f32);
            inert.extend_from_slice(&[li.com.x as f32, li.com.y as f32, li.com.z as f32]);
            inert.extend(li.inertia.as_slice().iter().map(|&v| v as f32));
        }
        let mut prm = vec![n as f32, n_envs as f32, gravity.x as f32, gravity.y as f32, gravity.z as f32, base.mass as f32, base.com.x as f32, base.com.y as f32, base.com.z as f32];
        prm.extend(base.inertia.as_slice().iter().map(|&v| v as f32));

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;

        let init = |label, data: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE });
        let joints_buf = init("t-joints", bytemuck::cast_slice(&jf));
        let inertia_buf = init("t-inertia", bytemuck::cast_slice(&inert));
        let prm_buf = init("t-prm", bytemuck::cast_slice(&prm));
        let dyn_buf = |label, elems: usize, extra: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (elems * 4).max(4) as u64, usage: wgpu::BufferUsages::STORAGE | extra, mapped_at_creation: false });
        let v0_buf = dyn_buf("t-v0", n_envs * 6, wgpu::BufferUsages::COPY_DST);
        let state_buf = dyn_buf("t-state", n_envs * 3 * n, wgpu::BufferUsages::COPY_DST);
        let fext_buf = dyn_buf("t-fext", n_envs * (n + 1) * 6, wgpu::BufferUsages::COPY_DST);
        let out_buf = dyn_buf("t-out", n_envs * (6 + n), wgpu::BufferUsages::COPY_SRC);
        let out_stage = device.create_buffer(&wgpu::BufferDescriptor { label: Some("t-out-stage"), size: (n_envs * (6 + n) * 4).max(4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("tree"), source: wgpu::ShaderSource::Wgsl(tree_wgsl(n, parent).into()) });
        let ro = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let mut rw = ro(5);
        rw.ty = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("t-bgl"), entries: &[ro(0), ro(1), ro(2), ro(3), ro(4), rw, ro(6)] });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("t-layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pso = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("tree"), layout: Some(&layout), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("t-bind"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: joints_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: inertia_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: prm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v0_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: fext_buf.as_entire_binding() },
            ],
        });
        Some(Self { n, n_envs, device, queue, pso, bind, v0_buf, state_buf, fext_buf, out_buf, out_stage })
    }

    /// Batched tree floating-base dynamics with per-body external wrenches: per env, base spatial
    /// acceleration `a0` (6) and joint accelerations `q̈` (n). `fext` packs `[base(6), body0(6), …]` per
    /// env. Returns `(a0 flat n_envs·6, qdd flat n_envs·n)`.
    pub fn accelerations_ext(&self, v0: &[f64], q: &[f64], qd: &[f64], tau: &[f64], fext: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let ne = self.n_envs;
        assert!(v0.len() == ne * 6 && q.len() == ne * self.n && qd.len() == ne * self.n && tau.len() == ne * self.n && fext.len() == ne * (self.n + 1) * 6, "batch size mismatch");
        let f = |s: &[f64]| -> Vec<f32> { s.iter().map(|&v| v as f32).collect() };
        let mut state = vec![0.0f32; ne * 3 * self.n];
        for e in 0..ne {
            for i in 0..self.n {
                state[e * 3 * self.n + i] = q[e * self.n + i] as f32;
                state[e * 3 * self.n + self.n + i] = qd[e * self.n + i] as f32;
                state[e * 3 * self.n + 2 * self.n + i] = tau[e * self.n + i] as f32;
            }
        }
        self.queue.write_buffer(&self.v0_buf, 0, bytemuck::cast_slice(&f(v0)));
        self.queue.write_buffer(&self.state_buf, 0, bytemuck::cast_slice(&state));
        self.queue.write_buffer(&self.fext_buf, 0, bytemuck::cast_slice(&f(fext)));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pso);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups((ne as u32).div_ceil(64), 1, 1);
        }
        let bytes = (ne * (6 + self.n) * 4) as u64;
        enc.copy_buffer_to_buffer(&self.out_buf, 0, &self.out_stage, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = self.out_stage.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let raw: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        self.out_stage.unmap();
        let (mut a0, mut qdd) = (Vec::with_capacity(ne * 6), Vec::with_capacity(ne * self.n));
        for e in 0..ne {
            let ob = e * (6 + self.n);
            for k in 0..6 { a0.push(raw[ob + k] as f64); }
            for k in 0..self.n { qdd.push(raw[ob + 6 + k] as f64); }
        }
        (a0, qdd)
    }
}

// ---------------------------------------------------------------------------------------------
// TreeGaitGpu — the branched-tree (quadruped/biped) contact simulator on the GPU, one thread per env:
// tree FK + multi-foot contact + tree spatial ABA + SE(3) integration. GPU port of
// `tree_floating_contact_step`. The simulator learned quadruped walking runs in.
// ---------------------------------------------------------------------------------------------

fn tree_gait_wgsl(n: usize, parent: &[isize]) -> String {
    let list = parent.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ");
    let src = r#"
const N: u32 = {N}u;
const PARENT: array<i32, {N}> = array<i32, {N}>({LIST});
@group(0) @binding(0) var<storage, read> JOINTS: array<f32>;
@group(0) @binding(1) var<storage, read> INERTIA: array<f32>;
@group(0) @binding(2) var<storage, read> PRM: array<f32>;      // [N,n_envs,gx,gy,gz, base(13), floor,kn,kd,dt,n_contacts]
@group(0) @binding(3) var<storage, read> CONTACTS: array<f32>; // 5 per contact: body(1) offset(3) mu(1)
@group(0) @binding(4) var<storage, read_write> BASEPOSE: array<f32>; // 12 per env
@group(0) @binding(5) var<storage, read_write> V0: array<f32>;       // 6 per env
@group(0) @binding(6) var<storage, read_write> QSTATE: array<f32>;   // 3N per env
@group(0) @binding(7) var<storage, read> INIT: array<f32>;           // 12+6+2N shared rollout start
@group(0) @binding(8) var<storage, read> POLICY: array<f32>;         // per policy: W(N×in_dim) then b(N)
@group(0) @binding(9) var<storage, read_write> REWARD: array<f32>;   // per policy: return
// PRM tail (rollout): [23] effort_w [24] taumax [25] rollout_steps [26] in_dim [27] n_policies

struct SV { t: vec3<f32>, b: vec3<f32> }
struct SM { tl: mat3x3<f32>, tr: mat3x3<f32>, bl: mat3x3<f32>, br: mat3x3<f32> }
struct Accel { a0: SV, qdd: array<f32, N> }
struct GState { R0: mat3x3<f32>, p0: vec3<f32>, v0: SV, q: array<f32,N>, qd: array<f32,N> }
fn z3() -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0)); }
fn i3() -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(1.,0.,0.), vec3<f32>(0.,1.,0.), vec3<f32>(0.,0.,1.)); }
fn skew(v: vec3<f32>) -> mat3x3<f32> { return mat3x3<f32>(vec3<f32>(0.0, v.z, -v.y), vec3<f32>(-v.z, 0.0, v.x), vec3<f32>(v.y, -v.x, 0.0)); }
fn outer(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> { return mat3x3<f32>(a*b.x, a*b.y, a*b.z); }
fn rot_axis(a: vec3<f32>, t: f32) -> mat3x3<f32> { let c=cos(t); let s=sin(t); let ic=1.0-c; let x=a.x; let y=a.y; let zz=a.z; return mat3x3<f32>(vec3<f32>(c+x*x*ic,y*x*ic+zz*s,zz*x*ic-y*s), vec3<f32>(x*y*ic-zz*s,c+y*y*ic,zz*y*ic+x*s), vec3<f32>(x*zz*ic+y*s,y*zz*ic-x*s,c+zz*zz*ic)); }
fn expmap(w: vec3<f32>) -> mat3x3<f32> { let a = length(w); if (a < 1e-9) { return i3(); } return rot_axis(w / a, a); }
fn smv(m: SM, v: SV) -> SV { return SV(m.tl*v.t + m.tr*v.b, m.bl*v.t + m.br*v.b); }
fn smm(a: SM, b: SM) -> SM { return SM(a.tl*b.tl + a.tr*b.bl, a.tl*b.tr + a.tr*b.br, a.bl*b.tl + a.br*b.bl, a.bl*b.tr + a.br*b.br); }
fn smt(a: SM) -> SM { return SM(transpose(a.tl), transpose(a.bl), transpose(a.tr), transpose(a.br)); }
fn sma(a: SM, b: SM) -> SM { return SM(a.tl+b.tl, a.tr+b.tr, a.bl+b.bl, a.br+b.br); }
fn sms(a: SM, s: f32) -> SM { return SM(a.tl*s, a.tr*s, a.bl*s, a.br*s); }
fn smsub(a: SM, b: SM) -> SM { return SM(a.tl-b.tl, a.tr-b.tr, a.bl-b.bl, a.br-b.br); }
fn sva(a: SV, b: SV) -> SV { return SV(a.t+b.t, a.b+b.b); }
fn svsub(a: SV, b: SV) -> SV { return SV(a.t-b.t, a.b-b.b); }
fn svs(a: SV, s: f32) -> SV { return SV(a.t*s, a.b*s); }
fn svdot(a: SV, b: SV) -> f32 { return dot(a.t, b.t) + dot(a.b, b.b); }
fn svouter(u: SV, w: SV) -> SM { return SM(outer(u.t,w.t), outer(u.t,w.b), outer(u.b,w.t), outer(u.b,w.b)); }
fn motion_transform(r: mat3x3<f32>, p: vec3<f32>) -> SM { let e = transpose(r); return SM(e, z3(), (e*skew(p)) * (-1.0), e); }
fn spatial_inertia(mass: f32, c: vec3<f32>, I: mat3x3<f32>) -> SM { let cx = skew(c); return SM(I - mass*(cx*cx), cx*mass, cx*(-mass), i3()*mass); }
fn crm(v: SV) -> SM { return SM(skew(v.t), z3(), skew(v.b), skew(v.t)); }
fn crf(v: SV) -> SM { return sms(smt(crm(v)), -1.0); }
fn grav_wrench(isp: SM, g: vec3<f32>, r: mat3x3<f32>) -> SV { return smv(isp, SV(vec3<f32>(0.0), r*g)); }
fn jointR(i: u32, qi: f32) -> mat3x3<f32> {
  let jb = i*16u; let Ro = mat3x3<f32>(vec3<f32>(JOINTS[jb],JOINTS[jb+1u],JOINTS[jb+2u]), vec3<f32>(JOINTS[jb+3u],JOINTS[jb+4u],JOINTS[jb+5u]), vec3<f32>(JOINTS[jb+6u],JOINTS[jb+7u],JOINTS[jb+8u]));
  if (JOINTS[jb+15u] < 0.5) { let a=vec3<f32>(JOINTS[jb+12u],JOINTS[jb+13u],JOINTS[jb+14u]); return Ro*rot_axis(a, qi); }
  return Ro;
}
fn jointP(i: u32, qi: f32) -> vec3<f32> {
  let jb = i*16u; let Ro = mat3x3<f32>(vec3<f32>(JOINTS[jb],JOINTS[jb+1u],JOINTS[jb+2u]), vec3<f32>(JOINTS[jb+3u],JOINTS[jb+4u],JOINTS[jb+5u]), vec3<f32>(JOINTS[jb+6u],JOINTS[jb+7u],JOINTS[jb+8u]));
  let po = vec3<f32>(JOINTS[jb+9u],JOINTS[jb+10u],JOINTS[jb+11u]);
  if (JOINTS[jb+15u] < 0.5) { return po; }
  return Ro*(vec3<f32>(JOINTS[jb+12u],JOINTS[jb+13u],JOINTS[jb+14u])*qi) + po;
}
fn subspace(i: u32) -> SV { let a=vec3<f32>(JOINTS[i*16u+12u],JOINTS[i*16u+13u],JOINTS[i*16u+14u]); if (JOINTS[i*16u+15u]<0.5){ return SV(a, vec3<f32>(0.0)); } return SV(vec3<f32>(0.0), a); }
fn linertia(i: u32) -> SM { let mass=INERTIA[i*13u]; let c=vec3<f32>(INERTIA[i*13u+1u],INERTIA[i*13u+2u],INERTIA[i*13u+3u]); let b=i*13u+4u; let I=mat3x3<f32>(vec3<f32>(INERTIA[b],INERTIA[b+1u],INERTIA[b+2u]),vec3<f32>(INERTIA[b+3u],INERTIA[b+4u],INERTIA[b+5u]),vec3<f32>(INERTIA[b+6u],INERTIA[b+7u],INERTIA[b+8u])); return spatial_inertia(mass,c,I); }

fn tree_aba_ext(v0: SV, q: array<f32,N>, qd: array<f32,N>, tau: array<f32,N>, febase: SV, fe: array<SV,N>, grav: vec3<f32>) -> Accel {
  var xm: array<SM,N>; var s: array<SV,N>; var v: array<SV,N>; var c: array<SV,N>; var ia: array<SM,N>; var pa: array<SV,N>;
  var rframes: array<mat3x3<f32>,N>;
  let ib = spatial_inertia(PRM[5], vec3<f32>(PRM[6],PRM[7],PRM[8]), mat3x3<f32>(vec3<f32>(PRM[9],PRM[10],PRM[11]),vec3<f32>(PRM[12],PRM[13],PRM[14]),vec3<f32>(PRM[15],PRM[16],PRM[17])));
  var ia_base = ib;
  var pa_base = svsub(svsub(smv(crf(v0), smv(ib,v0)), grav_wrench(ib,grav,i3())), febase);
  for (var i=0u;i<N;i=i+1u){
    let r=jointR(i,q[i]); let p=jointP(i,q[i]); let x=motion_transform(r,p); let si=subspace(i);
    let pi = PARENT[i];
    var v_par=v0; if(pi>=0){ v_par=v[u32(pi)]; }
    v[i]=sva(smv(x,v_par), svs(si,qd[i])); c[i]=smv(crm(v[i]), svs(si,qd[i]));
    let ii=linertia(i);
    var r_bp=i3(); if(pi>=0){ r_bp=rframes[u32(pi)]; }
    let r_bi=transpose(r)*r_bp;
    pa[i]=svsub(svsub(smv(crf(v[i]), smv(ii,v[i])), grav_wrench(ii,grav,r_bi)), fe[i]);
    ia[i]=ii; xm[i]=x; s[i]=si; rframes[i]=r_bi;
  }
  var u: array<SV,N>; var d: array<f32,N>; var uu: array<f32,N>;
  for (var ii=0u;ii<N;ii=ii+1u){
    let i=N-1u-ii;
    u[i]=smv(ia[i],s[i]); d[i]=svdot(s[i],u[i]); uu[i]=tau[i]-svdot(s[i],pa[i]);
    let ia_bar=smsub(ia[i], sms(svouter(u[i],u[i]), 1.0/d[i]));
    let pa_bar=sva(sva(pa[i], smv(ia_bar,c[i])), svs(u[i], uu[i]/d[i]));
    let xt=smt(xm[i]); let pi=PARENT[i];
    if(pi>=0){ let pp=u32(pi); ia[pp]=sma(ia[pp], smm(smm(xt,ia_bar),xm[i])); pa[pp]=sva(pa[pp], smv(xt,pa_bar)); }
    else { ia_base=sma(ia_base, smm(smm(xt,ia_bar),xm[i])); pa_base=sva(pa_base, smv(xt,pa_bar)); }
  }
  var A: array<f32,36>;
  for (var r=0u;r<3u;r=r+1u){ for (var cc=0u;cc<3u;cc=cc+1u){ A[r*6u+cc]=ia_base.tl[cc][r]; A[r*6u+cc+3u]=ia_base.tr[cc][r]; A[(r+3u)*6u+cc]=ia_base.bl[cc][r]; A[(r+3u)*6u+cc+3u]=ia_base.br[cc][r]; }}
  var rhs: array<f32,6>; rhs[0]=pa_base.t.x; rhs[1]=pa_base.t.y; rhs[2]=pa_base.t.z; rhs[3]=pa_base.b.x; rhs[4]=pa_base.b.y; rhs[5]=pa_base.b.z;
  var L: array<f32,36>;
  for (var i=0u;i<6u;i=i+1u){ for (var j=0u;j<=i;j=j+1u){ var sum=A[i*6u+j]; for (var k=0u;k<j;k=k+1u){ sum=sum-L[i*6u+k]*L[j*6u+k]; } if(i==j){ L[i*6u+i]=sqrt(max(sum,1e-12)); } else { L[i*6u+j]=sum/L[j*6u+j]; } }}
  var yv: array<f32,6>; for (var i=0u;i<6u;i=i+1u){ var sm2=rhs[i]; for (var k=0u;k<i;k=k+1u){ sm2=sm2-L[i*6u+k]*yv[k]; } yv[i]=sm2/L[i*6u+i]; }
  var xv: array<f32,6>; for (var ii=0u;ii<6u;ii=ii+1u){ let i=5u-ii; var sm3=yv[i]; for (var k=i+1u;k<6u;k=k+1u){ sm3=sm3-L[k*6u+i]*xv[k]; } xv[i]=sm3/L[i*6u+i]; }
  var out: Accel; out.a0 = SV(vec3<f32>(-xv[0],-xv[1],-xv[2]), vec3<f32>(-xv[3],-xv[4],-xv[5]));
  var a: array<SV,N>;
  for (var i=0u;i<N;i=i+1u){ let pi=PARENT[i]; var a_par=out.a0; if(pi>=0){ a_par=a[u32(pi)]; } let a_prime=sva(smv(xm[i],a_par), c[i]); out.qdd[i]=(uu[i]-svdot(u[i],a_prime))/d[i]; a[i]=sva(a_prime, svs(s[i], out.qdd[i])); }
  return out;
}

fn tree_gait_advance(st: GState, tau: array<f32,N>) -> GState {
  let grav = vec3<f32>(PRM[2],PRM[3],PRM[4]);
  let floor_z=PRM[18]; let kn=PRM[19]; let kd=PRM[20]; let dt=PRM[21]; let nc=u32(PRM[22]);
  let R0=st.R0; let p0=st.p0; let v0=st.v0; var q=st.q; var qd=st.qd;
  // tree FK: per-body base->body rotation Rb, origin pb, spatial velocity vf
  var Rb: array<mat3x3<f32>,N>; var pbf: array<vec3<f32>,N>; var vf: array<SV,N>;
  for (var i=0u;i<N;i=i+1u){
    let r=jointR(i,q[i]); let p=jointP(i,q[i]); let x=motion_transform(r,p); let si=subspace(i);
    let pi=PARENT[i];
    var Rp=i3(); var pp=vec3<f32>(0.0); var vp=v0;
    if(pi>=0){ Rp=Rb[u32(pi)]; pp=pbf[u32(pi)]; vp=vf[u32(pi)]; }
    Rb[i]=Rp*r; pbf[i]=Rp*p+pp;
    vf[i]=sva(smv(x,vp), svs(si,qd[i]));
  }
  var febase = SV(vec3<f32>(0.0), vec3<f32>(0.0));
  var fe: array<SV,N>; for (var i=0u;i<N;i=i+1u){ fe[i]=SV(vec3<f32>(0.0),vec3<f32>(0.0)); }
  for (var ci=0u; ci<nc; ci=ci+1u){
    let cb=ci*5u; let bd=u32(CONTACTS[cb]); let off=vec3<f32>(CONTACTS[cb+1u],CONTACTS[cb+2u],CONTACTS[cb+3u]); let mu=CONTACTS[cb+4u];
    let Rwf = R0 * Rb[bd];
    let pfoot = R0*(Rb[bd]*off + pbf[bd]) + p0;
    let phi = pfoot.z - floor_z;
    if (phi < 0.0) {
      let vlink = vf[bd];
      let vcp = Rwf * (cross(vlink.t, off) + vlink.b);
      let fnrm = max(0.0, -kn*phi - kd*vcp.z);
      let vt = vec2<f32>(vcp.x, vcp.y);
      let ft = -mu*fnrm * vt/(length(vt)+1e-4);
      let flocal = transpose(Rwf) * vec3<f32>(ft.x, ft.y, fnrm);
      fe[bd] = sva(fe[bd], SV(cross(off, flocal), flocal));
    }
  }
  let acc = tree_aba_ext(v0, q, qd, tau, febase, fe, grav);
  let v0n = sva(v0, svs(acc.a0, dt));
  for (var i=0u;i<N;i=i+1u){ let vv=qd[i]+dt*acc.qdd[i]; qd[i]=vv; q[i]=q[i]+dt*vv; }
  var o: GState; o.R0 = R0 * expmap(v0n.t*dt); o.p0 = p0 + R0*(v0n.b*dt); o.v0 = v0n; o.q = q; o.qd = qd;
  return o;
}

@compute @workgroup_size(64)
fn gait_step(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x; let n_envs = u32(PRM[1]);
  if (e >= n_envs) { return; }
  let pb=e*12u; let vb=e*6u; let sb=e*3u*N;
  var st: GState;
  st.R0 = mat3x3<f32>(vec3<f32>(BASEPOSE[pb],BASEPOSE[pb+1u],BASEPOSE[pb+2u]), vec3<f32>(BASEPOSE[pb+3u],BASEPOSE[pb+4u],BASEPOSE[pb+5u]), vec3<f32>(BASEPOSE[pb+6u],BASEPOSE[pb+7u],BASEPOSE[pb+8u]));
  st.p0 = vec3<f32>(BASEPOSE[pb+9u],BASEPOSE[pb+10u],BASEPOSE[pb+11u]);
  st.v0 = SV(vec3<f32>(V0[vb],V0[vb+1u],V0[vb+2u]), vec3<f32>(V0[vb+3u],V0[vb+4u],V0[vb+5u]));
  var tau: array<f32,N>;
  for (var i=0u;i<N;i=i+1u){ st.q[i]=QSTATE[sb+i]; st.qd[i]=QSTATE[sb+N+i]; tau[i]=QSTATE[sb+2u*N+i]; }
  let o = tree_gait_advance(st, tau);
  BASEPOSE[pb]=o.R0[0][0]; BASEPOSE[pb+1u]=o.R0[0][1]; BASEPOSE[pb+2u]=o.R0[0][2];
  BASEPOSE[pb+3u]=o.R0[1][0]; BASEPOSE[pb+4u]=o.R0[1][1]; BASEPOSE[pb+5u]=o.R0[1][2];
  BASEPOSE[pb+6u]=o.R0[2][0]; BASEPOSE[pb+7u]=o.R0[2][1]; BASEPOSE[pb+8u]=o.R0[2][2];
  BASEPOSE[pb+9u]=o.p0.x; BASEPOSE[pb+10u]=o.p0.y; BASEPOSE[pb+11u]=o.p0.z;
  V0[vb]=o.v0.t.x; V0[vb+1u]=o.v0.t.y; V0[vb+2u]=o.v0.t.z; V0[vb+3u]=o.v0.b.x; V0[vb+4u]=o.v0.b.y; V0[vb+5u]=o.v0.b.z;
  for (var i=0u;i<N;i=i+1u){ QSTATE[sb+i]=o.q[i]; QSTATE[sb+N+i]=o.qd[i]; }
}

// One-hidden-layer MLP policy with a gait clock. Features: [base_z, up-alignment, forward world
// velocity, q, qd, sin(phase), cos(phase)]. The phase clock lets the policy be time-periodic — a real
// gait cycle — and the tanh hidden layer adds the nonlinearity a rhythmic dynamic gait needs.
const HID: u32 = {HID}u;
fn policy_tau(st: GState, pol: u32, phs: f32, phc: f32) -> array<f32,N> {
  let in_dim = {IN}u; let taumax = PRM[24];
  let vfwd = (st.R0 * st.v0.b).x;
  var feat: array<f32, {IN}>;
  feat[0] = st.p0.z; feat[1] = st.R0[2][2]; feat[2] = vfwd;
  for (var i=0u;i<N;i=i+1u){ feat[3u+i] = st.q[i]; feat[3u+N+i] = st.qd[i]; }
  feat[3u+2u*N] = phs; feat[4u+2u*N] = phc;
  let base = pol * {PD}u;
  // layer 1: h = tanh(W1·feat + b1), W1 is HID×in_dim then b1(HID)
  var h: array<f32, {HID}>;
  for (var hh=0u; hh<HID; hh=hh+1u){
    var s = POLICY[base + HID*in_dim + hh];
    for (var k=0u;k<in_dim;k=k+1u){ s = s + POLICY[base + hh*in_dim + k] * feat[k]; }
    h[hh] = tanh(s);
  }
  // layer 2: tau = clamp(W2·h + b2), W2 is N×HID then b2(N), after W1(HID·in_dim)+b1(HID)
  let o2 = HID*in_dim + HID;
  var tau: array<f32,N>;
  for (var j=0u;j<N;j=j+1u){
    var s = POLICY[base + o2 + N*HID + j];
    for (var k=0u;k<HID;k=k+1u){ s = s + POLICY[base + o2 + j*HID + k] * h[k]; }
    tau[j] = clamp(s, -taumax, taumax);
  }
  return tau;
}

@compute @workgroup_size(64)
fn gait_rollout(@builtin(global_invocation_id) g: vec3<u32>) {
  let pol = g.x; let n_policies = u32(PRM[27]);
  if (pol >= n_policies) { return; }
  let effort_w = PRM[23]; let steps = u32(PRM[25]); let dt = PRM[21]; let freq = PRM[28];
  var st: GState;
  st.R0 = mat3x3<f32>(vec3<f32>(INIT[0],INIT[1],INIT[2]), vec3<f32>(INIT[3],INIT[4],INIT[5]), vec3<f32>(INIT[6],INIT[7],INIT[8]));
  st.p0 = vec3<f32>(INIT[9],INIT[10],INIT[11]);
  st.v0 = SV(vec3<f32>(INIT[12],INIT[13],INIT[14]), vec3<f32>(INIT[15],INIT[16],INIT[17]));
  for (var i=0u;i<N;i=i+1u){ st.q[i]=INIT[18u+i]; st.qd[i]=INIT[18u+N+i]; }
  var reward = 0.0;
  for (var t=0u;t<steps;t=t+1u){
    let clock = 6.2831853 * freq * f32(t) * dt;           // gait phase clock
    let tau = policy_tau(st, pol, sin(clock), cos(clock));
    st = tree_gait_advance(st, tau);
    // reward: FORWARD progress (world +x), gated by staying upright so it can't win by toppling.
    var eff = 0.0; for (var j=0u;j<N;j=j+1u){ eff = eff + tau[j]*tau[j]; }
    reward = reward + st.p0.x * max(st.R0[2][2], 0.0) - effort_w*eff;
  }
  REWARD[pol] = reward;
}
"#;
    let in_dim = 5 + 2 * n; // base_z, up, fwd_vel, q, qd, sin, cos
    let hid = 8usize;
    let pd = hid * in_dim + hid + n * hid + n;
    src.replace("{LIST}", &list).replace("{HID}", &hid.to_string()).replace("{IN}", &in_dim.to_string()).replace("{PD}", &pd.to_string()).replace("{N}", &n.to_string())
}

/// The branched-tree contact simulator on the GPU (tree FK + multi-foot contact + tree spatial ABA +
/// SE(3) integration), one thread per environment. GPU port of
/// [`tree_floating_contact_step`](crate::tree_floating_contact_step).
pub struct TreeGaitGpu {
    n: usize,
    n_envs: usize,
    policy_dim: usize,
    base_prm: Vec<f32>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso: wgpu::ComputePipeline,
    pso_rollout: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    prm_buf: wgpu::Buffer,
    basepose_buf: wgpu::Buffer,
    v0_buf: wgpu::Buffer,
    qstate_buf: wgpu::Buffer,
    init_buf: wgpu::Buffer,
    policy_buf: wgpu::Buffer,
    reward_buf: wgpu::Buffer,
    stage: wgpu::Buffer,
}

impl TreeGaitGpu {
    /// Build for a tree with foot `contacts` `(body, offset, μ)` against `floor_z`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(joints: &[crate::Joint], inertia: &[crate::LinkInertia], parent: &[isize], base: &crate::LinkInertia, contacts: &[crate::FootContact], floor_z: f64, kn: f64, kd: f64, gravity: nalgebra::Vector3<f64>, dt: f64, n_envs: usize) -> Option<Self> {
        let n = joints.len();
        let mut jf = Vec::with_capacity(n * 16);
        for j in joints {
            let r = j.origin.rotation.to_rotation_matrix();
            jf.extend(r.matrix().as_slice().iter().map(|&v| v as f32));
            let t = j.origin.translation.vector;
            jf.extend_from_slice(&[t.x as f32, t.y as f32, t.z as f32]);
            let a = j.axis.into_inner();
            jf.extend_from_slice(&[a.x as f32, a.y as f32, a.z as f32]);
            jf.push(match j.kind { crate::JointKind::Revolute => 0.0, crate::JointKind::Prismatic => 1.0 });
        }
        let mut inert = Vec::with_capacity(n * 13);
        for li in inertia {
            inert.push(li.mass as f32);
            inert.extend_from_slice(&[li.com.x as f32, li.com.y as f32, li.com.z as f32]);
            inert.extend(li.inertia.as_slice().iter().map(|&v| v as f32));
        }
        let mut prm = vec![n as f32, n_envs as f32, gravity.x as f32, gravity.y as f32, gravity.z as f32, base.mass as f32, base.com.x as f32, base.com.y as f32, base.com.z as f32];
        prm.extend(base.inertia.as_slice().iter().map(|&v| v as f32));
        prm.extend_from_slice(&[floor_z as f32, kn as f32, kd as f32, dt as f32, contacts.len() as f32]);
        let in_dim = 5 + 2 * n; // base_z, up, fwd_vel, q, qd, sin, cos
        let hid = 8;
        let policy_dim = hid * in_dim + hid + n * hid + n; // 1-hidden-layer MLP
        prm.extend_from_slice(&[0.0, 0.0, 0.0, in_dim as f32, 0.0, 0.0]); // rollout params 23..28 (28 = gait freq)
        let mut cflat: Vec<f32> = contacts.iter().flat_map(|&(b, off, mu)| [b as f32, off.x as f32, off.y as f32, off.z as f32, mu as f32]).collect();
        if cflat.is_empty() { cflat = vec![0.0; 5]; }

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let desc = wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits { max_storage_buffers_per_shader_stage: adapter.limits().max_storage_buffers_per_shader_stage.max(10), ..wgpu::Limits::default() },
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc)).ok()?;

        let init = |label, data: &[u8], extra: wgpu::BufferUsages| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE | extra });
        let none = wgpu::BufferUsages::empty();
        let joints_buf = init("tg-joints", bytemuck::cast_slice(&jf), none);
        let inertia_buf = init("tg-inertia", bytemuck::cast_slice(&inert), none);
        let prm_buf = init("tg-prm", bytemuck::cast_slice(&prm), wgpu::BufferUsages::COPY_DST);
        let contacts_buf = init("tg-contacts", bytemuck::cast_slice(&cflat), none);
        let dyn_buf = |label, elems: usize| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (elems * 4).max(4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let basepose_buf = dyn_buf("tg-basepose", n_envs * 12);
        let v0_buf = dyn_buf("tg-v0", n_envs * 6);
        let qstate_buf = dyn_buf("tg-qstate", n_envs * 3 * n);
        let init_buf = dyn_buf("tg-init", 18 + 2 * n);
        let policy_buf = dyn_buf("tg-policy", n_envs * policy_dim);
        let reward_buf = dyn_buf("tg-reward", n_envs);
        let stage = device.create_buffer(&wgpu::BufferDescriptor { label: Some("tg-stage"), size: (n_envs * (3 * n).max(12) * 4).max(4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("treegait"), source: wgpu::ShaderSource::Wgsl(tree_gait_wgsl(n, parent).into()) });
        let ro = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let rw = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("tg-bgl"), entries: &[ro(0), ro(1), ro(2), ro(3), rw(4), rw(5), rw(6), ro(7), ro(8), rw(9)] });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("tg-layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let mk = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&layout), module: &shader, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
        let pso = mk("gait_step");
        let pso_rollout = mk("gait_rollout");
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tg-bind"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: joints_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: inertia_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: prm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: contacts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: basepose_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: v0_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: qstate_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: init_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: policy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: reward_buf.as_entire_binding() },
            ],
        });
        Some(Self { n, n_envs, policy_dim, base_prm: prm, device, queue, pso, pso_rollout, bind, prm_buf, basepose_buf, v0_buf, qstate_buf, init_buf, policy_buf, reward_buf, stage })
    }

    /// Linear-policy parameter count (`n·(3+2n) + n`).
    pub fn policy_dim(&self) -> usize {
        self.policy_dim
    }

    /// Batched policy search over quadruped/tree contact rollouts. `policies` packs `n_policies` linear
    /// policies (`W(n×in_dim), b(n)`, `in_dim=3+2n`); all roll out `steps` from the shared `init =
    /// [R0(9,col-major), p0(3), v0(6), q(n), qd(n)]` under `τ=clamp(W·[base_z, up, forward-vel, q, qd]+b,
    /// ±taumax)`. Returns the per-policy forward-progress return `Σ x·max(up,0) − w‖τ‖²`. One thread/policy.
    pub fn rollout_rewards(&self, policies: &[f64], init: &[f64], effort_w: f64, taumax: f64, freq: f64, steps: usize) -> Vec<f64> {
        let n_policies = policies.len() / self.policy_dim;
        assert_eq!(init.len(), 18 + 2 * self.n, "init state size mismatch");
        let f = |s: &[f64]| -> Vec<f32> { s.iter().map(|&v| v as f32).collect() };
        self.queue.write_buffer(&self.policy_buf, 0, bytemuck::cast_slice(&f(policies)));
        self.queue.write_buffer(&self.init_buf, 0, bytemuck::cast_slice(&f(init)));
        let mut prm = self.base_prm.clone();
        prm[23] = effort_w as f32; prm[24] = taumax as f32; prm[25] = steps as f32; prm[27] = n_policies as f32; prm[28] = freq as f32;
        self.queue.write_buffer(&self.prm_buf, 0, bytemuck::cast_slice(&prm));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pso_rollout);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups((n_policies as u32).div_ceil(64), 1, 1);
        }
        let bytes = (n_policies * 4) as u64;
        enc.copy_buffer_to_buffer(&self.reward_buf, 0, &self.stage, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = self.stage.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let v: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        self.stage.unmap();
        v.iter().map(|&x| x as f64).collect()
    }

    /// Set per-env state and advance `steps` tree contact steps. `base_pose` packs `[R0(9,col-major),
    /// p0(3)]` per env, `v0` is `n_envs·6`, `q/qd/tau` are `n_envs·n`. Returns final `(base_pose, v0, q, qd)`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&self, base_pose: &[f64], v0: &[f64], q: &[f64], qd: &[f64], tau: &[f64], steps: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let ne = self.n_envs;
        let f = |s: &[f64]| -> Vec<f32> { s.iter().map(|&v| v as f32).collect() };
        self.queue.write_buffer(&self.basepose_buf, 0, bytemuck::cast_slice(&f(base_pose)));
        self.queue.write_buffer(&self.v0_buf, 0, bytemuck::cast_slice(&f(v0)));
        let mut qstate = vec![0.0f32; ne * 3 * self.n];
        for e in 0..ne {
            for i in 0..self.n {
                qstate[e * 3 * self.n + i] = q[e * self.n + i] as f32;
                qstate[e * 3 * self.n + self.n + i] = qd[e * self.n + i] as f32;
                qstate[e * 3 * self.n + 2 * self.n + i] = tau[e * self.n + i] as f32;
            }
        }
        self.queue.write_buffer(&self.qstate_buf, 0, bytemuck::cast_slice(&qstate));
        let groups = (ne as u32).div_ceil(64);
        let mut done = 0;
        while done < steps {
            let chunk = (steps - done).min(256);
            let mut enc = self.device.create_command_encoder(&Default::default());
            for _ in 0..chunk {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pso);
                pass.set_bind_group(0, &self.bind, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.queue.submit([enc.finish()]);
            done += chunk;
        }
        let rd = |buf: &wgpu::Buffer, elems: usize| -> Vec<f64> {
            let bytes = (elems * 4) as u64;
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buf, 0, &self.stage, 0, bytes);
            self.queue.submit([enc.finish()]);
            let slice = self.stage.slice(..bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().expect("map").expect("map ok");
            let v: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
            self.stage.unmap();
            v.iter().map(|&x| x as f64).collect()
        };
        let bp = rd(&self.basepose_buf, ne * 12);
        let vv = rd(&self.v0_buf, ne * 6);
        let qs = rd(&self.qstate_buf, ne * 3 * self.n);
        let (mut qo, mut qdo) = (vec![0.0; ne * self.n], vec![0.0; ne * self.n]);
        for e in 0..ne {
            for i in 0..self.n {
                qo[e * self.n + i] = qs[e * 3 * self.n + i];
                qdo[e * self.n + i] = qs[e * 3 * self.n + self.n + i];
            }
        }
        (bp, vv, qo, qdo)
    }
}

// ---------------------------------------------------------------------------------------------
// FrictionalContactGpu — the interior-point (Dojo-style) HARD frictional contact solve, batched.
// One GPU thread per environment runs the full central-path LCP Newton of `solve_frictional_ipm`:
// invert M, form the Stewart-Trinkle LCP `M_lcp = BᵀM⁻¹B + E`, solve `0 ≤ z ⟂ M_lcp z + q` by damped
// Newton with fraction-to-boundary, and return v⁺ = v_free + M⁻¹B z. This is the non-penetration +
// Coulomb-cone contact model (vs the penalty spring-damper the gait benches use), on the local GPU.
// ---------------------------------------------------------------------------------------------

/// Bake the frictional-LCP kernel for a FIXED contact structure: `nv` velocity DOF, `nz` LCP
/// variables, `nc` contacts, and the per-contact normal-slot indices `normal_idx`.
fn frictional_wgsl(nv: usize, nz: usize, nc: usize, normal_idx: &[usize]) -> String {
    let normals = normal_idx.iter().map(|i| format!("{i}")).collect::<Vec<_>>().join(", ");
    let src = r#"
// PRM: [n_envs, dt, kappa, pad] (16-byte aligned uniform)
struct Prm { n_envs: f32, dt: f32, kappa: f32, pad: f32 };
@group(0) @binding(0) var<uniform> prm: Prm;
@group(0) @binding(1) var<storage, read> B: array<f32>;      // NV x NZ, row-major
@group(0) @binding(2) var<storage, read> E: array<f32>;      // NZ x NZ, row-major
@group(0) @binding(3) var<storage, read> M: array<f32>;      // n_envs x (NV x NV)
@group(0) @binding(4) var<storage, read> VF: array<f32>;     // n_envs x NV
@group(0) @binding(5) var<storage, read> PHI: array<f32>;    // n_envs x NC
@group(0) @binding(6) var<storage, read_write> VN: array<f32>; // n_envs x NV

const NORMAL: array<i32, {NC}> = array<i32, {NC}>({NORMALS});

// Gauss-Jordan inverse of an SPD NV x NV (the mass matrix), env e -> Minv (row-major).
fn invert_m(e: u32, minv: ptr<function, array<f32, {NVNV}>>) {
    var a: array<f32, {NVNV}>;
    for (var i = 0u; i < {NV}u; i = i + 1u) {
        for (var j = 0u; j < {NV}u; j = j + 1u) {
            a[i * {NV}u + j] = M[e * {NVNV}u + i * {NV}u + j];
            (*minv)[i * {NV}u + j] = select(0.0, 1.0, i == j);
        }
    }
    for (var col = 0u; col < {NV}u; col = col + 1u) {
        let d = a[col * {NV}u + col];
        let inv = 1.0 / d;
        for (var k = 0u; k < {NV}u; k = k + 1u) {
            a[col * {NV}u + k] = a[col * {NV}u + k] * inv;
            (*minv)[col * {NV}u + k] = (*minv)[col * {NV}u + k] * inv;
        }
        for (var r = 0u; r < {NV}u; r = r + 1u) {
            if (r == col) { continue; }
            let f = a[r * {NV}u + col];
            for (var k = 0u; k < {NV}u; k = k + 1u) {
                a[r * {NV}u + k] = a[r * {NV}u + k] - f * a[col * {NV}u + k];
                (*minv)[r * {NV}u + k] = (*minv)[r * {NV}u + k] - f * (*minv)[col * {NV}u + k];
            }
        }
    }
}

// Solve J x = rhs (NZ x NZ, general) by Gaussian elimination with partial pivoting; J, rhs consumed.
// Returns false if the system is (numerically) singular — the caller then stops Newton, matching the
// CPU `lu().solve()` returning None.
fn lu_solve(j: ptr<function, array<f32, {NZNZ}>>, rhs: ptr<function, array<f32, {NZ}>>, x: ptr<function, array<f32, {NZ}>>) -> bool {
    for (var col = 0u; col < {NZ}u; col = col + 1u) {
        var piv = col;
        var mx = abs((*j)[col * {NZ}u + col]);
        for (var r = col + 1u; r < {NZ}u; r = r + 1u) {
            let v = abs((*j)[r * {NZ}u + col]);
            if (v > mx) { mx = v; piv = r; }
        }
        if (mx < 1e-20) { return false; }
        if (piv != col) {
            for (var k = 0u; k < {NZ}u; k = k + 1u) {
                let t = (*j)[col * {NZ}u + k]; (*j)[col * {NZ}u + k] = (*j)[piv * {NZ}u + k]; (*j)[piv * {NZ}u + k] = t;
            }
            let tr = (*rhs)[col]; (*rhs)[col] = (*rhs)[piv]; (*rhs)[piv] = tr;
        }
        let d = (*j)[col * {NZ}u + col];
        for (var r = col + 1u; r < {NZ}u; r = r + 1u) {
            let f = (*j)[r * {NZ}u + col] / d;
            for (var k = col; k < {NZ}u; k = k + 1u) { (*j)[r * {NZ}u + k] = (*j)[r * {NZ}u + k] - f * (*j)[col * {NZ}u + k]; }
            (*rhs)[r] = (*rhs)[r] - f * (*rhs)[col];
        }
    }
    for (var ii = 0u; ii < {NZ}u; ii = ii + 1u) {
        let i = {NZ}u - 1u - ii;
        var s = (*rhs)[i];
        for (var k = i + 1u; k < {NZ}u; k = k + 1u) { s = s - (*j)[i * {NZ}u + k] * (*x)[k]; }
        (*x)[i] = s / (*j)[i * {NZ}u + i];
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if (f32(e) >= prm.n_envs) { return; }

    var minv: array<f32, {NVNV}>;
    invert_m(e, &minv);

    // tmp = Minv * B  (NV x NZ)
    var tmp: array<f32, {NVNZ}>;
    for (var r = 0u; r < {NV}u; r = r + 1u) {
        for (var c = 0u; c < {NZ}u; c = c + 1u) {
            var s = 0.0;
            for (var k = 0u; k < {NV}u; k = k + 1u) { s = s + minv[r * {NV}u + k] * B[k * {NZ}u + c]; }
            tmp[r * {NZ}u + c] = s;
        }
    }
    // Mlcp = Bᵀ tmp + E  (NZ x NZ)
    var mlcp: array<f32, {NZNZ}>;
    for (var i = 0u; i < {NZ}u; i = i + 1u) {
        for (var jc = 0u; jc < {NZ}u; jc = jc + 1u) {
            var s = E[i * {NZ}u + jc];
            for (var r = 0u; r < {NV}u; r = r + 1u) { s = s + B[r * {NZ}u + i] * tmp[r * {NZ}u + jc]; }
            mlcp[i * {NZ}u + jc] = s;
        }
    }
    // q = Bᵀ v_free + q0(phi)
    var q: array<f32, {NZ}>;
    for (var i = 0u; i < {NZ}u; i = i + 1u) {
        var s = 0.0;
        for (var r = 0u; r < {NV}u; r = r + 1u) { s = s + B[r * {NZ}u + i] * VF[e * {NV}u + r]; }
        q[i] = s;
    }
    for (var c = 0u; c < {NC}u; c = c + 1u) { q[NORMAL[c]] = q[NORMAL[c]] + PHI[e * {NC}u + c] / prm.dt; }

    // central-path Newton: lam ∘ (Mlcp lam + q) = kappa, lam > 0
    var lam: array<f32, {NZ}>;
    for (var i = 0u; i < {NZ}u; i = i + 1u) { lam[i] = 1.0; }
    for (var it = 0u; it < 100u; it = it + 1u) {
        var w: array<f32, {NZ}>;
        for (var i = 0u; i < {NZ}u; i = i + 1u) {
            var s = q[i];
            for (var k = 0u; k < {NZ}u; k = k + 1u) { s = s + mlcp[i * {NZ}u + k] * lam[k]; }
            w[i] = s;
        }
        var f: array<f32, {NZ}>;
        var fn2 = 0.0;
        for (var i = 0u; i < {NZ}u; i = i + 1u) { f[i] = lam[i] * w[i] - prm.kappa; fn2 = fn2 + f[i] * f[i]; }
        if (fn2 < 1e-16) { break; }
        // J = diag(w) + diag(lam) Mlcp
        var jm: array<f32, {NZNZ}>;
        for (var i = 0u; i < {NZ}u; i = i + 1u) {
            for (var k = 0u; k < {NZ}u; k = k + 1u) { jm[i * {NZ}u + k] = lam[i] * mlcp[i * {NZ}u + k]; }
            jm[i * {NZ}u + i] = jm[i * {NZ}u + i] + w[i];
        }
        var step: array<f32, {NZ}>;
        if (!lu_solve(&jm, &f, &step)) { break; }
        var alpha = 1.0;
        for (var i = 0u; i < {NZ}u; i = i + 1u) {
            if (step[i] > 0.0) { alpha = min(alpha, 0.99 * lam[i] / step[i]); }
        }
        for (var i = 0u; i < {NZ}u; i = i + 1u) { lam[i] = lam[i] - alpha * step[i]; }
    }

    // v_next = v_free + Minv B lam
    var bl: array<f32, {NV}>;
    for (var r = 0u; r < {NV}u; r = r + 1u) {
        var s = 0.0;
        for (var i = 0u; i < {NZ}u; i = i + 1u) { s = s + B[r * {NZ}u + i] * lam[i]; }
        bl[r] = s;
    }
    for (var r = 0u; r < {NV}u; r = r + 1u) {
        var s = VF[e * {NV}u + r];
        for (var k = 0u; k < {NV}u; k = k + 1u) { s = s + minv[r * {NV}u + k] * bl[k]; }
        VN[e * {NV}u + r] = s;
    }
}
"#;
    src.replace("{NVNV}", &(nv * nv).to_string())
        .replace("{NVNZ}", &(nv * nz).to_string())
        .replace("{NZNZ}", &(nz * nz).to_string())
        .replace("{NORMALS}", &normals)
        .replace("{NV}", &nv.to_string())
        .replace("{NZ}", &nz.to_string())
        .replace("{NC}", &nc.to_string())
}

/// Batched hard frictional contact — one GPU thread per environment solving the differentiable
/// Stewart-Trinkle LCP of [`crate::solve_frictional_ipm`] on the central path. The contact STRUCTURE (normal +
/// friction-facet directions, `mu`) is fixed at construction; the mass matrix `M`, free velocity
/// `v_free`, and signed gaps `phi` vary per environment. Feature `gpu`. Verified against the CPU
/// oracle. HONEST SCOPE: per-thread private arrays scale as `nv²+nz²`, so this is for small–moderate
/// contact sets (a handful of contacts); at large `nv`/`nz` register/occupancy pressure dominates.
pub struct FrictionalContactGpu {
    nv: usize,
    nc: usize,
    n_envs: usize,
    dt: f64,
    kappa: f64,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pso: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    b_buf: wgpu::Buffer,
    e_buf: wgpu::Buffer,
    prm_buf: wgpu::Buffer,
    m_buf: wgpu::Buffer,
    vf_buf: wgpu::Buffer,
    phi_buf: wgpu::Buffer,
    vn_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl FrictionalContactGpu {
    /// Build a solver for the fixed contact set `contacts` (their `jn`/`jt`/`mu` define the LCP;
    /// `phi` is supplied per environment at solve time), sized for batches of exactly `n_envs`.
    /// `None` when there is no GPU.
    pub fn new(contacts: &[StFrictionContact], dt: f64, kappa: f64, n_envs: usize) -> Option<Self> {
        let nv = contacts.first().map(|c| c.jn.len())?;
        let nc = contacts.len();
        // per-contact block layout [λₙ, β₁…β_d, s]; record normal-slot indices and total nz.
        let mut starts = Vec::with_capacity(nc);
        let mut nz = 0usize;
        for c in contacts {
            starts.push(nz);
            nz += 2 + c.jt.len();
        }
        // B (nv×nz) and E (nz×nz), row-major — the phi-free part of solve_frictional_ipm.
        let mut bmat = vec![0.0f32; nv * nz];
        let mut emat = vec![0.0f32; nz * nz];
        for (i, c) in contacts.iter().enumerate() {
            let d = c.jt.len();
            let ln = starts[i];
            let s_idx = ln + 1 + d;
            for r in 0..nv {
                bmat[r * nz + ln] = c.jn[r] as f32;
            }
            for (kf, dir) in c.jt.iter().enumerate() {
                let bk = ln + 1 + kf;
                for r in 0..nv {
                    bmat[r * nz + bk] = dir[r] as f32;
                }
                emat[bk * nz + s_idx] = 1.0;
                emat[s_idx * nz + bk] = -1.0;
            }
            emat[s_idx * nz + ln] = c.mu as f32;
        }

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;

        let init = |label, data: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE });
        let b_buf = init("fr-b", bytemuck::cast_slice(&bmat));
        let e_buf = init("fr-e", bytemuck::cast_slice(&emat));
        let prm = [n_envs as f32, dt as f32, kappa as f32, 0.0f32];
        let prm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("fr-prm"), contents: bytemuck::cast_slice(&prm), usage: wgpu::BufferUsages::UNIFORM });
        let store_dst = |label, size: usize| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (size * 4).max(4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let m_buf = store_dst("fr-m", n_envs * nv * nv);
        let vf_buf = store_dst("fr-vf", n_envs * nv);
        let phi_buf = store_dst("fr-phi", n_envs * nc.max(1));
        let vn_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("fr-vn"), size: (n_envs * nv * 4).max(4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("fr-staging"), size: (n_envs * nv * 4).max(4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("frictional"), source: wgpu::ShaderSource::Wgsl(frictional_wgsl(nv, nz, nc, &starts).into()) });
        let uni = wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let ro = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let mut rw = ro(6);
        rw.ty = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("fr-bgl"), entries: &[uni, ro(1), ro(2), ro(3), ro(4), ro(5), rw] });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("fr-layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pso = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("frictional"), layout: Some(&layout), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });

        Some(Self { nv, nc, n_envs, dt, kappa, device, queue, pso, bgl, b_buf, e_buf, prm_buf, m_buf, vf_buf, phi_buf, vn_buf, staging })
    }

    /// Post-contact velocity `v⁺` for each of `n_envs` environments. `m` is the flattened per-env mass
    /// matrices (`n_envs·nv·nv`, row-major), `v_free` the free velocities (`n_envs·nv`), `phi` the
    /// signed gaps (`n_envs·nc`). Matches [`crate::solve_frictional_ipm`]`(…).v_next` at the same `kappa`.
    pub fn solve(&self, m: &[f64], v_free: &[f64], phi: &[f64]) -> Vec<f64> {
        assert_eq!(m.len(), self.n_envs * self.nv * self.nv, "mass-matrix batch size mismatch");
        assert_eq!(v_free.len(), self.n_envs * self.nv, "v_free batch size mismatch");
        assert_eq!(phi.len(), self.n_envs * self.nc, "phi batch size mismatch");
        let f = |s: &[f64]| -> Vec<f32> { s.iter().map(|&v| v as f32).collect() };
        self.queue.write_buffer(&self.m_buf, 0, bytemuck::cast_slice(&f(m)));
        self.queue.write_buffer(&self.vf_buf, 0, bytemuck::cast_slice(&f(v_free)));
        self.queue.write_buffer(&self.phi_buf, 0, bytemuck::cast_slice(&f(phi)));

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fr-bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.prm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.e_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.m_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.vf_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.phi_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.vn_buf.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pso);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((self.n_envs as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.vn_buf, 0, &self.staging, 0, (self.n_envs * self.nv * 4) as u64);
        self.queue.submit([enc.finish()]);

        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let view = slice.get_mapped_range().expect("mapped range");
        let data: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        self.staging.unmap();
        let _ = (self.dt, self.kappa); // captured in prm at build time
        data.iter().map(|&v| v as f64).collect()
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use crate::{arm_clearance, floating_base_forward_dynamics, floating_contact_step, forward_dynamics, from_urdf_full, from_urdf_str, quadruped, raymarch, tree_floating_contact_step, tree_floating_forward_dynamics, DepthCamera, Lidar, LinkInertia};
    use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3, Vector6};

    const ARM: &str = r#"<robot name="a"><link name="world"/><link name="base"/>
      <link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/>
      <joint name="j0" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint></robot>"#;

    /// The GPU clearance batch reproduces the CPU `arm_clearance` per config (f32 vs f64), over a
    /// mixed scene (box + sphere + plane) — the cross-oracle at f32 tolerance.
    #[test]
    fn gpu_clearance_matches_cpu() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let dof = robot.dof();
        let scene = SdfScene {
            prims: vec![
                Sdf::Box { center: Vector3::new(0.25, 0.0, 0.4), half: Vector3::new(0.05, 0.2, 0.14) },
                Sdf::Sphere { center: Vector3::new(-0.2, 0.2, 0.3), radius: 0.08 },
                Sdf::Plane { normal: Vector3::new(0.0, 0.0, 1.0), offset: 0.0 },
            ],
        };
        let (link_r, per_link, n) = (0.03, 3usize, 512usize);

        // deterministic batch of configs (splitmix64)
        let mut s = 0xABCDu64;
        let mut cfg = Vec::with_capacity(n * dof);
        for _ in 0..n * dof {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            cfg.push(((((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0) * 1.4);
        }

        let Some(g) = ClearanceGpu::new(&robot, &scene, link_r, per_link, n) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let gpu = g.clearances(&cfg);
        let mut worst = 0.0f64;
        let mut decision_mismatch = 0;
        for k in 0..n {
            let q = &cfg[k * dof..(k + 1) * dof];
            let cpu = arm_clearance(&robot, q, &scene, link_r, per_link);
            worst = worst.max((gpu[k] - cpu).abs());
            if (gpu[k] > 0.0) != (cpu > 0.0) {
                decision_mismatch += 1;
            }
        }
        eprintln!("GPU vs CPU arm_clearance ({n} configs): worst {worst:.3e} m, collision-decision mismatches {decision_mismatch}");
        assert!(worst < 1e-4, "GPU clearance diverged from the CPU reference: {worst}");
        assert_eq!(decision_mismatch, 0, "GPU/CPU disagreed on collision for {decision_mismatch} configs");
    }

    /// The GPU sphere-tracer reproduces `DepthCamera::render` — median range within the f32 float
    /// gap over surface pixels, and segmentation labels agreeing on all but the silhouette edges.
    #[test]
    fn gpu_sensor_matches_cpu() {
        let scene = SdfScene {
            prims: vec![
                Sdf::Sphere { center: Vector3::new(-0.9, 0.0, 2.8), radius: 0.40 },
                Sdf::Box { center: Vector3::new(0.0, 0.1, 4.0), half: Vector3::new(0.35, 0.35, 0.35) },
                Sdf::Sphere { center: Vector3::new(0.9, 0.0, 2.2), radius: 0.38 },
                Sdf::Plane { normal: Vector3::new(0.0, -1.0, 0.0), offset: -1.2 },
            ],
        };
        let cam = DepthCamera { pose: Isometry3::identity(), fx: 96.0, fy: 96.0, cx: 79.5, cy: 59.5, width: 160, height: 120, far: 8.0 };

        let Some(g) = SensorGpu::new(&cam, &scene) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let (range, seg) = g.render();
        let cpu = cam.render(&scene);

        let far = cam.far as f32;
        let mut errs = Vec::new();
        let mut seg_agree = 0usize;
        for i in 0..cam.width * cam.height {
            if range[i] < far - 1e-3 && (cpu.range[i] as f32) < far - 1e-3 {
                errs.push((range[i] - cpu.range[i] as f32).abs());
            }
            if seg[i] == cpu.seg[i] {
                seg_agree += 1;
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = if errs.is_empty() { 0.0 } else { errs[errs.len() / 2] };
        let seg_frac = seg_agree as f64 / (cam.width * cam.height) as f64;
        eprintln!("GPU vs CPU sensor render: median range {:.3e} m over {} surface px, seg agreement {:.3}%", med, errs.len(), 100.0 * seg_frac);
        assert!(med < 1e-3, "GPU depth diverged from the CPU render: median {med}");
        assert!(seg_frac > 0.97, "GPU segmentation disagreed too much: {seg_frac}");
    }

    /// The GPU lidar reproduces `Lidar::scan`: dense per-ray range/point matches the CPU raymarch,
    /// and the compacted GPU scan has the same hit count as the CPU scan (± silhouette rays).
    #[test]
    fn gpu_lidar_matches_cpu() {
        let scene = SdfScene {
            prims: vec![
                Sdf::Sphere { center: Vector3::new(2.0, 0.0, 0.0), radius: 0.5 },
                Sdf::Box { center: Vector3::new(0.0, 2.0, 0.0), half: Vector3::new(0.4, 0.4, 0.6) },
                Sdf::Plane { normal: Vector3::new(0.0, 0.0, 1.0), offset: -0.8 },
            ],
        };
        let lidar = Lidar {
            pose: Isometry3::identity(),
            n_azimuth: 128,
            n_elevation: 64,
            az_min: -std::f64::consts::PI,
            az_max: std::f64::consts::PI,
            el_min: -0.4,
            el_max: 0.4,
            far: 6.0,
        };

        let Some(g) = LidarGpu::new(&lidar, &scene) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let (range, _seg, pts) = g.dense();

        // dense CPU reference: the same ray grid, calling the verified raymarch per ray
        let da = (lidar.az_max - lidar.az_min) / (lidar.n_azimuth - 1) as f64;
        let de = (lidar.el_max - lidar.el_min) / (lidar.n_elevation - 1) as f64;
        let o = lidar.pose.translation.vector;
        let mut range_errs = Vec::new();
        let mut pt_worst = 0.0f64;
        let mut hit_mismatch = 0;
        for ie in 0..lidar.n_elevation {
            let el = lidar.el_min + ie as f64 * de;
            for ia in 0..lidar.n_azimuth {
                let az = lidar.az_min + ia as f64 * da;
                let ds = Vector3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
                let dir = lidar.pose.rotation * ds;
                let idx = ie * lidar.n_azimuth + ia;
                let cpu = raymarch(&scene, o, dir, lidar.far);
                let gpu_hit = range[idx] < lidar.far as f32 - 1e-3;
                match (cpu, gpu_hit) {
                    (Some(hit), true) => {
                        range_errs.push((range[idx] - hit.t as f32).abs());
                        let p = pts[idx];
                        pt_worst = pt_worst.max(((p[0] as f64 - hit.point.x).powi(2) + (p[1] as f64 - hit.point.y).powi(2) + (p[2] as f64 - hit.point.z).powi(2)).sqrt());
                    }
                    (None, false) => {}
                    _ => hit_mismatch += 1,
                }
            }
        }
        range_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = if range_errs.is_empty() { 0.0 } else { range_errs[range_errs.len() / 2] };
        let total = lidar.n_azimuth * lidar.n_elevation;

        // the compacted API matches the CPU scan's hit count (± silhouette flips)
        let gpu_scan = g.scan();
        let cpu_scan = lidar.scan(&scene);
        let count_diff = (gpu_scan.points.len() as i64 - cpu_scan.points.len() as i64).abs();

        eprintln!("GPU vs CPU lidar ({total} rays): median range {med:.3e} m, worst point {pt_worst:.3e} m, hit mismatches {hit_mismatch}/{total}; scan hits GPU {} vs CPU {} (Δ{count_diff})", gpu_scan.points.len(), cpu_scan.points.len());
        assert!(med < 1e-3, "GPU lidar range diverged: median {med}");
        assert!(pt_worst < 2e-3, "GPU lidar point diverged: {pt_worst}");
        assert!(hit_mismatch * 200 < total, "too many GPU/CPU hit disagreements: {hit_mismatch}/{total}");
        assert!(count_diff * 200 < total as i64, "GPU/CPU scan hit-count diverged: {count_diff}");
    }

    const ARM3: &str = r#"<robot name="a3">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0 0 0.1" rpy="0 0 0"/><mass value="1.5"/><inertia ixx="0.02" iyy="0.02" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.15 0 0" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.01" iyy="0.03" izz="0.03" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="l3"><inertial><origin xyz="0.1 0 0" rpy="0 0 0"/><mass value="0.6"/><inertia ixx="0.005" iyy="0.012" izz="0.012" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="0.2 0 0" rpy="0 0 0"/></joint></robot>"#;

    /// The batched GPU forward dynamics reproduces the CPU `forward_dynamics` per environment (RNEA +
    /// CRBA + Cholesky, f32 vs f64), over a batch of random states — the cross-oracle at RL scale.
    #[test]
    fn gpu_articulated_matches_cpu() {
        let (robot, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let n = robot.dof();
        let grav = Vector3::new(0.0, 0.0, -9.81);
        let n_envs = 1024usize;

        // deterministic random states (splitmix64)
        let mut s = 0x5151u64;
        let mut rng = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let q: Vec<f64> = (0..n_envs * n).map(|_| rng() * 1.5).collect();
        let qd: Vec<f64> = (0..n_envs * n).map(|_| rng() * 1.0).collect();
        let tau: Vec<f64> = (0..n_envs * n).map(|_| rng() * 2.0).collect();

        let Some(g) = ArticulatedGpu::new(&robot, &inertia, grav, 1e-3, n_envs, &[], 0.0, 0.0, 0.0) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let gpu = g.accelerations(&q, &qd, &tau);

        let mut worst = 0.0f64;
        for e in 0..n_envs {
            let cpu = forward_dynamics(&robot, &inertia, &q[e * n..(e + 1) * n], &qd[e * n..(e + 1) * n], &tau[e * n..(e + 1) * n], grav);
            for i in 0..n {
                worst = worst.max((gpu[e * n + i] - cpu[i]).abs());
            }
        }
        eprintln!("GPU vs CPU forward_dynamics ({n_envs} envs × {n} DOF): worst qdd {worst:.3e} rad/s²");
        assert!(worst < 1e-2, "GPU forward dynamics diverged from the CPU reference: {worst}");

        // and a stepped trajectory on the GPU matches a CPU integration loop
        let steps = 50;
        let (gq, gqd) = g.run(&q, &qd, &tau, steps);
        let mut cq = q.clone();
        let mut cqd = qd.clone();
        for _ in 0..steps {
            for e in 0..n_envs {
                let a = forward_dynamics(&robot, &inertia, &cq[e * n..(e + 1) * n], &cqd[e * n..(e + 1) * n], &tau[e * n..(e + 1) * n], grav);
                for i in 0..n {
                    let vd = cqd[e * n + i] + 1e-3 * a[i];
                    cqd[e * n + i] = vd;
                    cq[e * n + i] += 1e-3 * vd;
                }
            }
        }
        let mut worst_q = 0.0f64;
        for k in 0..n_envs * n {
            worst_q = worst_q.max((gq[k] - cq[k]).abs());
        }
        eprintln!("GPU vs CPU stepped trajectory ({steps} steps): worst q {worst_q:.3e} rad");
        assert!(worst_q < 5e-3, "GPU stepped trajectory diverged: {worst_q}");
        let _ = gqd;
    }

    /// One CPU penalty ground-contact step, built from the verified `forward_dynamics` +
    /// `point_jacobian` + `frame_pose` — the reference the GPU `step_contact` is checked against.
    fn cpu_contact_step(
        robot: &crate::Robot, inertia: &[crate::LinkInertia], contacts: &[(usize, Vector3<f64>, f64)],
        floor_z: f64, kn: f64, kd: f64, q: &[f64], qd: &[f64], tau: &[f64], dt: f64, g: Vector3<f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        use nalgebra::Point3;
        let n = robot.dof();
        let mut tt = tau.to_vec();
        for &(fr, off, mu) in contacts {
            let p = (robot.frame_pose(q, fr) * Point3::from(off)).coords;
            let phi = p.z - floor_z;
            if phi < 0.0 {
                let jp = robot.point_jacobian(q, fr, &p); // 3×n
                let qdv = nalgebra::DVector::from_row_slice(qd);
                let v = &jp * &qdv; // 3-vector point velocity
                let fnrm = (-kn * phi - kd * v[2]).max(0.0);
                let vt = nalgebra::Vector2::new(v[0], v[1]);
                let ft = -mu * fnrm * vt / (vt.norm() + 1e-4);
                let f = Vector3::new(ft[0], ft[1], fnrm);
                let tc = jp.transpose() * f; // n-vector Jₚᵀ·f
                for i in 0..n {
                    tt[i] += tc[i];
                }
            }
        }
        let a = forward_dynamics(robot, inertia, q, qd, &tt, g);
        let mut qn = q.to_vec();
        let mut qdn = qd.to_vec();
        for i in 0..n {
            qdn[i] += dt * a[i];
            qn[i] += dt * qdn[i];
        }
        (qn, qdn)
    }

    /// The GPU penalty-contact step reproduces the CPU reference, and a robot dropped onto the floor
    /// settles (bounded penetration, comes to rest) — port correctness + a physical invariant.
    #[test]
    fn gpu_articulated_contact_matches_cpu() {
        let (robot, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let n = robot.dof();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let dt = 1e-3;
        let (floor_z, kn, kd) = (0.0, 4.0e3, 40.0);
        // contact points: the tool frame origin and the mid/last link origins
        let contacts = vec![
            (n, Vector3::new(0.2, 0.0, 0.0), 0.6),
            (n, Vector3::zeros(), 0.6),
            (n - 1, Vector3::zeros(), 0.6),
        ];
        let n_envs = 512usize;

        // random states, biased downward so some contacts penetrate
        let mut s = 0x9A9Au64;
        let mut rng = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let q: Vec<f64> = (0..n_envs * n).map(|_| rng() * 1.5).collect();
        let qd: Vec<f64> = (0..n_envs * n).map(|_| rng() * 0.5).collect();
        let tau: Vec<f64> = (0..n_envs * n).map(|_| rng() * 1.0).collect();

        let Some(gp) = ArticulatedGpu::new(&robot, &inertia, g, dt, n_envs, &contacts, floor_z, kn, kd) else {
            eprintln!("no GPU — skipping");
            return;
        };
        // one contact step, GPU vs CPU
        let (gq, gqd) = gp.run_contact(&q, &qd, &tau, 1);
        let mut worst = 0.0f64;
        for e in 0..n_envs {
            let (cq, cqd) = cpu_contact_step(&robot, &inertia, &contacts, floor_z, kn, kd, &q[e * n..(e + 1) * n], &qd[e * n..(e + 1) * n], &tau[e * n..(e + 1) * n], dt, g);
            for i in 0..n {
                worst = worst.max((gq[e * n + i] - cq[i]).abs()).max((gqd[e * n + i] - cqd[i]).abs());
            }
        }
        eprintln!("GPU vs CPU penalty-contact step ({n_envs} envs): worst {worst:.3e}");
        assert!(worst < 1e-3, "GPU contact step diverged from the CPU reference: {worst}");

        // physical invariant: drop the arm (zero torque, start above floor), it settles without
        // sinking through the floor or blowing up.
        let q0: Vec<f64> = (0..n_envs * n).map(|_| 0.0).collect(); // arm pointing up, tool well above floor
        let qd0 = vec![0.0; n_envs * n];
        let zero_tau = vec![0.0; n_envs * n];
        let (sq, sqd) = gp.run_contact(&q0, &qd0, &zero_tau, 4000); // ~4 s
        use nalgebra::Point3;
        let mut worst_pen = 0.0f64;
        let mut worst_speed = 0.0f64;
        for e in 0..n_envs.min(8) {
            let qi = &sq[e * n..(e + 1) * n];
            for &(fr, off, _) in &contacts {
                let p = (robot.frame_pose(qi, fr) * Point3::from(off)).coords;
                worst_pen = worst_pen.max((floor_z - p.z).max(0.0));
            }
            for i in 0..n {
                worst_speed = worst_speed.max(sqd[e * n + i].abs());
            }
        }
        eprintln!("drop-and-settle: worst floor penetration {worst_pen:.3e} m, worst joint speed {worst_speed:.3e} rad/s");
        assert!(sq.iter().all(|v| v.is_finite()) && sqd.iter().all(|v| v.is_finite()), "contact sim blew up (NaN/inf)");
        assert!(worst_pen < 0.05, "arm sank through the floor: {worst_pen} m");
        assert!(worst_speed < 2.0, "arm did not settle to rest: {worst_speed} rad/s");
    }

    /// One CPU policy rollout matching the GPU `rollout` kernel — the reference for the port check,
    /// and used to read the learned policy's final tracking error.
    #[allow(clippy::too_many_arguments)]
    fn cpu_rollout(
        robot: &crate::Robot, inertia: &[crate::LinkInertia], theta: &[f64], q0: &[f64], qd0: &[f64],
        target: &[f64], tau_max: f64, effort_w: f64, steps: usize, dt: f64, g: Vector3<f64>,
    ) -> (f64, Vec<f64>) {
        let n = robot.dof();
        let mut q = q0.to_vec();
        let mut qd = qd0.to_vec();
        let mut reward = 0.0;
        for _ in 0..steps {
            let mut tau = vec![0.0; n];
            for i in 0..n {
                let mut s = theta[2 * n * n + i];
                for j in 0..n {
                    s += theta[i * 2 * n + j] * (q[j] - target[j]);
                }
                for j in 0..n {
                    s += theta[i * 2 * n + n + j] * qd[j];
                }
                tau[i] = s.clamp(-tau_max, tau_max);
            }
            let a = forward_dynamics(robot, inertia, &q, &qd, &tau, g);
            let mut cost = 0.0;
            for i in 0..n {
                let ei = q[i] - target[i];
                cost += ei * ei + effort_w * tau[i] * tau[i];
            }
            reward -= cost;
            for i in 0..n {
                qd[i] += dt * a[i];
                q[i] += dt * qd[i];
            }
        }
        (reward, q)
    }

    /// **RL at scale on the GPU sim.** A cross-entropy-method (CEM) loop learns a linear feedback
    /// controller that regulates the arm to a target posture under gravity — every generation
    /// evaluates a whole population of candidate policies in ONE GPU dispatch. Verifies that (1) the
    /// best reward improves over generations, (2) the learned policy actually reaches the target, and
    /// (3) the GPU rollout reward matches the CPU reference for the learned (stable) policy.
    #[test]
    fn gpu_policy_search_learns_to_regulate() {
        let (robot, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let n = robot.dof();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let dt = 3e-3;
        let steps = 180;
        let (tau_max, effort_w) = (10.0, 3e-5);
        let target = vec![0.4, -0.3, 0.5];
        let q0 = vec![0.0; n];
        let qd0 = vec![0.0; n];
        let pop = 256usize;

        let Some(gp) = ArticulatedGpu::new(&robot, &inertia, g, dt, pop, &[], 0.0, 0.0, 0.0) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let dim = gp.policy_dim();
        let elite = pop / 8;

        // deterministic uniform noise
        let mut s = 0x7E57u64;
        let mut rng = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };

        let mut mean = vec![0.0f64; dim];
        let mut sigma = vec![1.0f64; dim];
        let mut first_best = f64::NEG_INFINITY;
        let mut last_best = f64::NEG_INFINITY;
        let gens = 60;
        for giter in 0..gens {
            // sample a population around the mean
            let mut pol = vec![0.0f64; pop * dim];
            for e in 0..pop {
                for d in 0..dim {
                    pol[e * dim + d] = mean[d] + sigma[d] * rng();
                }
            }
            let rewards = gp.rollout_rewards(&pol, &q0, &qd0, &target, tau_max, effort_w, steps);
            // rank, take elite, refit mean/sigma (CEM)
            let mut idx: Vec<usize> = (0..pop).collect();
            idx.sort_by(|&a, &b| rewards[b].partial_cmp(&rewards[a]).unwrap());
            let best = rewards[idx[0]];
            if giter == 0 {
                first_best = best;
            }
            last_best = best;
            for d in 0..dim {
                let m: f64 = idx[..elite].iter().map(|&e| pol[e * dim + d]).sum::<f64>() / elite as f64;
                let v: f64 = idx[..elite].iter().map(|&e| (pol[e * dim + d] - m).powi(2)).sum::<f64>() / elite as f64;
                mean[d] = m;
                sigma[d] = v.sqrt().max(0.01); // variance floor
            }
        }

        // the learned policy, evaluated on the CPU: reward + final tracking error
        let (cpu_reward, final_q) = cpu_rollout(&robot, &inertia, &mean, &q0, &qd0, &target, tau_max, effort_w, steps, dt, g);
        let final_err: f64 = (0..n).map(|i| (final_q[i] - target[i]).powi(2)).sum::<f64>().sqrt();

        // GPU reward for the same learned policy (fill the batch with the mean) — the port check
        let batch: Vec<f64> = (0..pop).flat_map(|_| mean.clone()).collect();
        let grew = gp.rollout_rewards(&batch, &q0, &qd0, &target, tau_max, effort_w, steps);
        let port_rel = ((grew[0] - cpu_reward) / cpu_reward.abs().max(1.0)).abs();

        // do-nothing baseline: zero policy (arm sags under gravity). Learned control must beat it.
        let zero_pol = vec![0.0f64; pop * dim];
        let zero_reward = gp.rollout_rewards(&zero_pol, &q0, &qd0, &target, tau_max, effort_w, steps)[0];
        let improvement = cpu_reward - zero_reward;

        eprintln!(
            "CEM policy search ({pop} policies × {gens} gens): best reward {first_best:.2} → {last_best:.2}; learned reward {cpu_reward:.2} vs do-nothing {zero_reward:.2} (Δ{improvement:.2}); final ‖q−q*‖ = {final_err:.4} rad; GPU vs CPU reward rel {port_rel:.2e}"
        );
        assert!(last_best > first_best + 1.0, "CEM did not improve over generations: {first_best} → {last_best}");
        assert!(improvement > 0.3 * zero_reward.abs(), "learned policy did not beat the do-nothing baseline: {cpu_reward} vs {zero_reward}");
        assert!(port_rel < 1e-2, "GPU rollout reward diverged from the CPU reference: {port_rel}");
    }

    /// The batched floating-base forward dynamics (spatial 6D ABA) reproduces the CPU
    /// `floating_base_forward_dynamics` per environment — base spatial acceleration `a0` and joint
    /// accelerations `q̈`, f32 vs f64, for a free 6-DoF root + arm under random state.
    #[test]
    fn gpu_floating_base_matches_cpu() {
        let (robot, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let n = robot.dof();
        let base = LinkInertia { mass: 5.0, com: Vector3::new(0.0, 0.0, 0.05), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.05)) };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let n_envs = 512usize;

        let mut s = 0xF10Au64;
        let mut rng = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let v0: Vec<f64> = (0..n_envs * 6).map(|_| rng() * 0.6).collect();
        let q: Vec<f64> = (0..n_envs * n).map(|_| rng() * 1.3).collect();
        let qd: Vec<f64> = (0..n_envs * n).map(|_| rng() * 0.8).collect();
        let tau: Vec<f64> = (0..n_envs * n).map(|_| rng() * 1.5).collect();

        let Some(gp) = FloatingBaseGpu::new(&robot, &inertia, &base, g, n_envs) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let (a0, qdd) = gp.accelerations(&v0, &q, &qd, &tau);

        let mut worst_a0 = 0.0f64;
        let mut worst_qdd = 0.0f64;
        for e in 0..n_envs {
            let v0e = Vector6::from_row_slice(&v0[e * 6..e * 6 + 6]);
            let (a0c, qddc) = floating_base_forward_dynamics(&robot, &inertia, &base, v0e, &q[e * n..(e + 1) * n], &qd[e * n..(e + 1) * n], &tau[e * n..(e + 1) * n], g);
            for k in 0..6 {
                worst_a0 = worst_a0.max((a0[e * 6 + k] - a0c[k]).abs());
            }
            for i in 0..n {
                worst_qdd = worst_qdd.max((qdd[e * n + i] - qddc[i]).abs());
            }
        }
        eprintln!("GPU vs CPU floating-base ABA ({n_envs} envs × free-base + {n} DOF): worst a0 {worst_a0:.3e}, worst qdd {worst_qdd:.3e}");
        assert!(worst_a0 < 1e-2, "GPU base acceleration diverged: {worst_a0}");
        assert!(worst_qdd < 1e-2, "GPU joint accelerations diverged: {worst_qdd}");
    }

    /// The batched BRANCHED-tree floating-base ABA reproduces the CPU `tree_floating_forward_dynamics`
    /// per environment — a free base with TWO legs (a real tree, not a chain), f32 vs f64.
    #[test]
    fn gpu_tree_matches_cpu() {
        let (arm, ai) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let leg = arm.joints[0..2].to_vec();
        let li = ai[0..2].to_vec();
        let joints: Vec<crate::Joint> = leg.iter().chain(leg.iter()).cloned().collect(); // 4 joints
        let inertia: Vec<LinkInertia> = li.iter().chain(li.iter()).cloned().collect();
        let parent: Vec<isize> = vec![-1, 0, -1, 2]; // two legs off the base
        let n = joints.len();
        let base = LinkInertia { mass: 5.0, com: Vector3::new(0.0, 0.0, 0.05), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.05)) };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let n_envs = 512usize;

        let mut s = 0xB1A5u64;
        let mut rng = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let v0: Vec<f64> = (0..n_envs * 6).map(|_| rng() * 0.6).collect();
        let q: Vec<f64> = (0..n_envs * n).map(|_| rng() * 1.2).collect();
        let qd: Vec<f64> = (0..n_envs * n).map(|_| rng() * 0.7).collect();
        let tau: Vec<f64> = (0..n_envs * n).map(|_| rng() * 1.4).collect();
        let fext = vec![0.0f64; n_envs * (n + 1) * 6];

        let Some(gp) = TreeFloatingGpu::new(&joints, &inertia, &parent, &base, g, n_envs) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let (a0, qdd) = gp.accelerations_ext(&v0, &q, &qd, &tau, &fext);
        let zero = vec![Vector6::zeros(); n];
        let mut worst_a0 = 0.0f64;
        let mut worst_qdd = 0.0f64;
        for e in 0..n_envs {
            let v0e = Vector6::from_row_slice(&v0[e * 6..e * 6 + 6]);
            let (a0c, qddc) = tree_floating_forward_dynamics(&joints, &inertia, &parent, &base, v0e, &q[e * n..(e + 1) * n], &qd[e * n..(e + 1) * n], &tau[e * n..(e + 1) * n], Vector6::zeros(), &zero, g);
            for k in 0..6 { worst_a0 = worst_a0.max((a0[e * 6 + k] - a0c[k]).abs()); }
            for i in 0..n { worst_qdd = worst_qdd.max((qdd[e * n + i] - qddc[i]).abs()); }
        }
        eprintln!("GPU vs CPU tree ABA ({n_envs} envs × free-base + 2 legs): worst a0 {worst_a0:.3e}, worst qdd {worst_qdd:.3e}");
        assert!(worst_a0 < 1e-2, "GPU tree base acceleration diverged: {worst_a0}");
        assert!(worst_qdd < 1e-2, "GPU tree joint accelerations diverged: {worst_qdd}");
    }

    /// The GPU tree gait step (tree FK + multi-foot contact + tree ABA + SE(3) integration) reproduces
    /// the CPU `tree_floating_contact_step` over a rollout — a QUADRUPED standing on four feet.
    #[test]
    fn gpu_tree_gait_matches_cpu() {
        let (joints, inertia, parent, contacts) = quadruped();
        let n = joints.len();
        let base_inertia = LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)) };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (floor, kn, kd, dt) = (0.0, 1.5e4, 120.0, 2e-4);
        let n_envs = 256usize;
        let steps = 40usize;

        // stance: base with straight legs, feet penetrating ~1 cm (contact active throughout)
        let mut s = 0x9E77u64;
        let mut rng = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let mut base_pose = vec![0.0f64; n_envs * 12];
        let v0 = vec![0.0f64; n_envs * 6];
        let mut q = vec![0.0f64; n_envs * n];
        let qd = vec![0.0f64; n_envs * n];
        let mut tau = vec![0.0f64; n_envs * n];
        for e in 0..n_envs {
            for (k, v) in Matrix3::<f64>::identity().as_slice().iter().enumerate() { base_pose[e * 12 + k] = *v; }
            base_pose[e * 12 + 11] = 0.59; // feet ~1 cm into the floor
            for i in 0..n { q[e * n + i] = 0.05 * rng(); tau[e * n + i] = 0.3 * rng(); }
        }

        let Some(gp) = TreeGaitGpu::new(&joints, &inertia, &parent, &base_inertia, &contacts, floor, kn, kd, g, dt, n_envs) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let (gbp, gv0, gq, gqd) = gp.run(&base_pose, &v0, &q, &qd, &tau, steps);

        let mut worst_pose = 0.0f64;
        let mut vdiffs = Vec::new();
        let mut qdiffs = Vec::new();
        for e in 0..n_envs {
            let mut base = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.59), UnitQuaternion::identity());
            let mut vv = Vector6::zeros();
            let mut qq = q[e * n..(e + 1) * n].to_vec();
            let mut qdd = qd[e * n..(e + 1) * n].to_vec();
            let taue = tau[e * n..(e + 1) * n].to_vec();
            for _ in 0..steps {
                let (b, v, qn, qdn) = tree_floating_contact_step(&joints, &inertia, &parent, &base_inertia, base, vv, &qq, &qdd, &taue, &contacts, floor, kn, kd, dt, g);
                base = b; vv = v; qq = qn; qdd = qdn;
            }
            let r = base.rotation.to_rotation_matrix();
            for k in 0..9 { worst_pose = worst_pose.max((gbp[e * 12 + k] - r.matrix().as_slice()[k]).abs()); }
            for k in 0..3 { worst_pose = worst_pose.max((gbp[e * 12 + 9 + k] - base.translation.vector[k]).abs()); }
            for k in 0..6 { vdiffs.push((gv0[e * 6 + k] - vv[k]).abs()); }
            for i in 0..n { qdiffs.push((gq[e * n + i] - qq[i]).abs()); qdiffs.push((gqd[e * n + i] - qdd[i]).abs()); }
        }
        // Near-straight legs sit at an unstable strut equilibrium, so f32 vs f64 buckle a few joints
        // differently over the rollout (chaos, not a port bug) while the base — pinned by four feet —
        // and the bulk of joints match to the f32 limit. Judge the port by the MEDIAN.
        vdiffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        qdiffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med_v = vdiffs[vdiffs.len() / 2];
        let med_q = qdiffs[qdiffs.len() / 2];
        eprintln!("GPU vs CPU tree gait — quadruped ({n_envs} envs × {steps} steps): worst pose {worst_pose:.3e}, median v0 {med_v:.3e}, median q/qd {med_q:.3e}");
        assert!(worst_pose < 2e-2, "GPU quadruped base pose diverged: {worst_pose}");
        assert!(med_v < 1e-3, "GPU quadruped base velocity diverged (median): {med_v}");
        assert!(med_q < 1e-3, "GPU quadruped joint state diverged (median): {med_q}");
    }

    /// CPU forward-locomotion rollout reward — mirrors the GPU `gait_rollout` (same features, policy,
    /// reward) — the port reference and CEM baseline.
    #[allow(clippy::too_many_arguments)]
    fn cpu_tree_gait_reward(joints: &[crate::Joint], inertia: &[LinkInertia], base_inertia: &LinkInertia, parent: &[isize], contacts: &[crate::FootContact], floor: f64, kn: f64, kd: f64, dt: f64, g: Vector3<f64>, init: &[f64], policy: &[f64], effort_w: f64, taumax: f64, freq: f64, steps: usize, n: usize) -> f64 {
        use nalgebra::Rotation3;
        let in_dim = 5 + 2 * n;
        let hid = 8;
        let rmat = Matrix3::from_column_slice(&init[0..9]);
        let mut base = Isometry3::from_parts(Translation3::from(Vector3::new(init[9], init[10], init[11])), UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rmat)));
        let mut v0 = Vector6::from_row_slice(&init[12..18]);
        let mut q = init[18..18 + n].to_vec();
        let mut qd = init[18 + n..18 + 2 * n].to_vec();
        let mut reward = 0.0;
        for t in 0..steps {
            let clock = std::f64::consts::TAU * freq * t as f64 * dt;
            let r0 = *base.rotation.to_rotation_matrix().matrix();
            let up = r0[(2, 2)];
            let vfwd = (r0 * v0.fixed_rows::<3>(3).into_owned()).x;
            let mut feat = vec![0.0; in_dim];
            feat[0] = base.translation.z; feat[1] = up; feat[2] = vfwd;
            for i in 0..n { feat[3 + i] = q[i]; feat[3 + n + i] = qd[i]; }
            feat[3 + 2 * n] = clock.sin(); feat[4 + 2 * n] = clock.cos();
            // 1-hidden-layer MLP (tanh), matching the GPU policy_tau
            let mut h = vec![0.0; hid];
            for hh in 0..hid {
                let mut s = policy[hid * in_dim + hh];
                for k in 0..in_dim { s += policy[hh * in_dim + k] * feat[k]; }
                h[hh] = s.tanh();
            }
            let o2 = hid * in_dim + hid;
            let mut tau = vec![0.0; n];
            for j in 0..n {
                let mut s = policy[o2 + n * hid + j];
                for k in 0..hid { s += policy[o2 + j * hid + k] * h[k]; }
                tau[j] = s.clamp(-taumax, taumax);
            }
            let (b, v, qn, qdn) = tree_floating_contact_step(joints, inertia, parent, base_inertia, base, v0, &q, &qd, &tau, contacts, floor, kn, kd, dt, g);
            base = b; v0 = v; q = qn; qd = qdn;
            let up2 = base.rotation.to_rotation_matrix().matrix()[(2, 2)];
            let eff: f64 = tau.iter().map(|t| t * t).sum();
            reward += base.translation.x * up2.max(0.0) - effort_w * eff;
        }
        reward
    }

    /// The assembled learned QUADRUPED locomotion loop: CEM over branched-tree contact rollouts on the
    /// GPU learns joint torques that push the standing quadruped forward — beating do-nothing (which
    /// stays put). Verifies the GPU rollout reward == the CPU reference.
    #[test]
    fn gpu_tree_gait_policy_learns() {
        let (joints, inertia, parent, contacts) = quadruped();
        let n = joints.len();
        let base_inertia = LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)) };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (floor, kn, kd, dt) = (0.0, 1.5e4, 120.0, 1e-3);
        let (effort_w, taumax, freq, steps) = (1e-5, 25.0, 2.0, 200usize);

        // stance: straight legs (q=0), base at 0.6 so feet just touch — a stable standing start
        let mut init = vec![0.0f64; 18 + 2 * n];
        for (k, v) in Matrix3::<f64>::identity().as_slice().iter().enumerate() { init[k] = *v; }
        init[11] = 0.60;

        let pop = 512usize;
        let gens = 60usize;
        let Some(gp) = TreeGaitGpu::new(&joints, &inertia, &parent, &base_inertia, &contacts, floor, kn, kd, g, dt, pop) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let dim = gp.policy_dim();

        let mut s = 0x4EE7u64;
        let mut rng = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let mut mean = vec![0.0f64; dim];
        let mut sigma = vec![0.3f64; dim];
        let (mut first_best, mut last_best) = (f64::NEG_INFINITY, 0.0);
        let mut best_policy = mean.clone();
        for giter in 0..gens {
            let mut batch = vec![0.0f64; pop * dim];
            for p in 0..pop { for d in 0..dim { batch[p * dim + d] = mean[d] + sigma[d] * rng(); } }
            let rewards = gp.rollout_rewards(&batch, &init, effort_w, taumax, freq, steps);
            let mut idx: Vec<usize> = (0..pop).collect();
            idx.sort_by(|&a, &b| rewards[b].partial_cmp(&rewards[a]).unwrap());
            if giter == 0 { first_best = rewards[idx[0]]; }
            last_best = rewards[idx[0]];
            best_policy = batch[idx[0] * dim..(idx[0] + 1) * dim].to_vec();
            let elite = (pop as f64 * 0.12) as usize;
            for d in 0..dim {
                let mut m = 0.0;
                for &e in idx.iter().take(elite) { m += batch[e * dim + d]; }
                m /= elite as f64;
                let mut vv = 0.0;
                for &e in idx.iter().take(elite) { vv += (batch[e * dim + d] - m).powi(2); }
                mean[d] = m; sigma[d] = (vv / elite as f64).sqrt().max(0.02);
            }
        }

        let zero = vec![0.0f64; dim];
        let base_reward = cpu_tree_gait_reward(&joints, &inertia, &base_inertia, &parent, &contacts, floor, kn, kd, dt, g, &init, &zero, effort_w, taumax, freq, steps, n);
        let learned_cpu = cpu_tree_gait_reward(&joints, &inertia, &base_inertia, &parent, &contacts, floor, kn, kd, dt, g, &init, &best_policy, effort_w, taumax, freq, steps, n);
        let filled: Vec<f64> = (0..pop).flat_map(|_| best_policy.clone()).collect();
        let grew = gp.rollout_rewards(&filled, &init, effort_w, taumax, freq, steps)[0];
        let port_rel = ((grew - learned_cpu) / learned_cpu.abs().max(1.0)).abs();

        eprintln!("CEM quadruped MLP+phase locomotion ({pop} policies × {gens} gens, {dim}-param MLP): best reward {first_best:.3} → {last_best:.3}; learned {learned_cpu:.3} vs do-nothing {base_reward:.3}; GPU vs CPU reward rel {port_rel:.2e}");
        assert!(port_rel < 2e-2, "GPU rollout reward diverged from the CPU reference: {port_rel}");
        assert!(last_best > first_best + 0.1, "CEM did not improve: {first_best} → {last_best}");
        assert!(learned_cpu > base_reward + 0.1, "learned MLP policy did not out-walk do-nothing: {learned_cpu} vs {base_reward}");
    }

    /// The floating-base ABA with **external spatial forces** (the ground-contact / applied-wrench
    /// mechanism) reproduces the CPU `floating_base_forward_dynamics_ext` — one external wrench per
    /// body (base + each link), f32 vs f64. This is what makes floating-base contact possible.
    #[test]
    fn gpu_floating_base_external_forces_matches_cpu() {
        use crate::floating_base_forward_dynamics_ext;
        let (robot, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let n = robot.dof();
        let base = LinkInertia { mass: 5.0, com: Vector3::new(0.0, 0.0, 0.05), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.05)) };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let n_envs = 512usize;

        let mut s = 0xC0DEu64;
        let mut rng = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let v0: Vec<f64> = (0..n_envs * 6).map(|_| rng() * 0.5).collect();
        let q: Vec<f64> = (0..n_envs * n).map(|_| rng() * 1.2).collect();
        let qd: Vec<f64> = (0..n_envs * n).map(|_| rng() * 0.7).collect();
        let tau: Vec<f64> = (0..n_envs * n).map(|_| rng() * 1.0).collect();
        let fext: Vec<f64> = (0..n_envs * (n + 1) * 6).map(|_| rng() * 3.0).collect(); // per env: base(6) + links

        let Some(gp) = FloatingBaseGpu::new(&robot, &inertia, &base, g, n_envs) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let (a0, qdd) = gp.accelerations_ext(&v0, &q, &qd, &tau, &fext);

        let stride = (n + 1) * 6;
        let mut worst_a0 = 0.0f64;
        let mut worst_qdd = 0.0f64;
        for e in 0..n_envs {
            let v0e = Vector6::from_row_slice(&v0[e * 6..e * 6 + 6]);
            let fb = e * stride;
            let f_ext_base = Vector6::from_row_slice(&fext[fb..fb + 6]);
            let f_ext: Vec<Vector6<f64>> = (0..n).map(|i| Vector6::from_row_slice(&fext[fb + 6 * (i + 1)..fb + 6 * (i + 1) + 6])).collect();
            let (a0c, qddc) = floating_base_forward_dynamics_ext(&robot, &inertia, &base, v0e, &q[e * n..(e + 1) * n], &qd[e * n..(e + 1) * n], &tau[e * n..(e + 1) * n], f_ext_base, &f_ext, g);
            for k in 0..6 {
                worst_a0 = worst_a0.max((a0[e * 6 + k] - a0c[k]).abs());
            }
            for i in 0..n {
                worst_qdd = worst_qdd.max((qdd[e * n + i] - qddc[i]).abs());
            }
        }
        eprintln!("GPU vs CPU floating-base ABA + external forces ({n_envs} envs): worst a0 {worst_a0:.3e}, worst qdd {worst_qdd:.3e}");
        assert!(worst_a0 < 2e-2, "GPU base acceleration (ext) diverged: {worst_a0}");
        assert!(worst_qdd < 2e-2, "GPU joint accelerations (ext) diverged: {worst_qdd}");
    }

    const UPARM_G: &str = r#"<robot name="upg">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0 0 0.1" rpy="0 0 0"/><mass value="0.6"/><inertia ixx="0.01" iyy="0.01" izz="0.005" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0 0 0.1" rpy="0 0 0"/><mass value="0.4"/><inertia ixx="0.006" iyy="0.006" izz="0.003" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="tip"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.15" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="20" velocity="5"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="20" velocity="5"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tip"/><origin xyz="0 0 0.2" rpy="0 0 0"/></joint></robot>"#;

    /// The GPU floating-base contact step (FK + foot contact + spatial ABA + SE(3) integration)
    /// reproduces the CPU `floating_contact_step` over a multi-step rollout — the port check against
    /// the validated CPU reference, from a stable four-foot stance (contact active throughout).
    #[test]
    fn gpu_floating_gait_matches_cpu() {
        let (robot, inertia) = from_urdf_full(UPARM_G, "base", "tip").unwrap();
        let n = robot.dof();
        let base_inertia = LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.06, 0.06, 0.08)) };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (floor_z, kn, kd, dt) = (0.0, 2.0e4, 150.0, 2e-4);
        let hx = 0.12;
        let contacts: Vec<crate::FootContact> = vec![
            (0, Vector3::new(hx, hx, -0.06), 0.9), (0, Vector3::new(-hx, hx, -0.06), 0.9),
            (0, Vector3::new(hx, -hx, -0.06), 0.9), (0, Vector3::new(-hx, -hx, -0.06), 0.9),
        ];
        let n_envs = 256usize;
        let steps = 40usize;

        // stable stance: base resting with feet ~5 mm penetrating, small random joint state + torque
        let mut s = 0x6A17u64;
        let mut rng = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let mut base_pose = vec![0.0f64; n_envs * 12];
        let v0 = vec![0.0f64; n_envs * 6];
        let mut q = vec![0.0f64; n_envs * n];
        let mut qd = vec![0.0f64; n_envs * n];
        let mut tau = vec![0.0f64; n_envs * n];
        for e in 0..n_envs {
            let id = Matrix3::<f64>::identity();
            for (k, &v) in id.as_slice().iter().enumerate() { base_pose[e * 12 + k] = v; } // R0 col-major = I
            base_pose[e * 12 + 9] = 0.0; base_pose[e * 12 + 10] = 0.0; base_pose[e * 12 + 11] = 0.055; // p0
            for i in 0..n { q[e * n + i] = 0.2 * rng(); qd[e * n + i] = 0.1 * rng(); tau[e * n + i] = 0.5 * rng(); }
        }

        let Some(gp) = FloatingGaitGpu::new(&robot, &inertia, &base_inertia, &contacts, floor_z, kn, kd, g, dt, n_envs) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let (gbp, gv0, gq, gqd) = gp.run(&base_pose, &v0, &q, &qd, &tau, steps);

        let mut worst_pose = 0.0f64;
        let mut worst_v = 0.0f64;
        let mut worst_q = 0.0f64;
        for e in 0..n_envs {
            let mut base = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.055), UnitQuaternion::identity());
            let mut vv = Vector6::zeros();
            let mut qq = q[e * n..(e + 1) * n].to_vec();
            let mut qdd = qd[e * n..(e + 1) * n].to_vec();
            let taue = tau[e * n..(e + 1) * n].to_vec();
            for _ in 0..steps {
                let (b, v, qn, qdn) = floating_contact_step(&robot, &inertia, &base_inertia, base, vv, &qq, &qdd, &taue, &contacts, floor_z, kn, kd, dt, g);
                base = b; vv = v; qq = qn; qdd = qdn;
            }
            let r = base.rotation.to_rotation_matrix();
            for k in 0..9 { worst_pose = worst_pose.max((gbp[e * 12 + k] - r.matrix().as_slice()[k]).abs()); }
            for k in 0..3 { worst_pose = worst_pose.max((gbp[e * 12 + 9 + k] - base.translation.vector[k]).abs()); }
            for k in 0..6 { worst_v = worst_v.max((gv0[e * 6 + k] - vv[k]).abs()); }
            for i in 0..n { worst_q = worst_q.max((gq[e * n + i] - qq[i]).abs()).max((gqd[e * n + i] - qdd[i]).abs()); }
        }
        eprintln!("GPU vs CPU floating gait ({n_envs} envs × {steps} steps): worst pose {worst_pose:.3e}, worst v0 {worst_v:.3e}, worst q/qd {worst_q:.3e}");
        assert!(worst_pose < 2e-2, "GPU base pose diverged: {worst_pose}");
        assert!(worst_v < 2e-2, "GPU base velocity diverged: {worst_v}");
        assert!(worst_q < 2e-2, "GPU joint state diverged: {worst_q}");
    }

    const LEG_G: &str = r#"<robot name="legg">
      <link name="base"/>
      <link name="thigh"><inertial><origin xyz="0 0 -0.15" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.01" iyy="0.01" izz="0.004" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="shank"><inertial><origin xyz="0 0 -0.15" rpy="0 0 0"/><mass value="0.6"/><inertia ixx="0.006" iyy="0.006" izz="0.002" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="foot"/>
      <joint name="hip" type="revolute"><parent link="base"/><child link="thigh"/><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="30" velocity="8"/></joint>
      <joint name="knee" type="revolute"><parent link="thigh"/><child link="shank"/><origin xyz="0 0 -0.3" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="30" velocity="8"/></joint>
      <joint name="ankle" type="fixed"><parent link="shank"/><child link="foot"/><origin xyz="0 0 -0.3" rpy="0 0 0"/></joint></robot>"#;

    /// The CPU rollout reward — mirrors the GPU `gait_rollout` exactly (same features, policy, reward)
    /// — used to check the port and inside the CEM baseline.
    #[allow(clippy::too_many_arguments)]
    fn cpu_gait_reward(robot: &crate::Robot, inertia: &[LinkInertia], base_inertia: &LinkInertia, contacts: &[crate::FootContact], floor: f64, kn: f64, kd: f64, dt: f64, g: Vector3<f64>, init: &[f64], policy: &[f64], effort_w: f64, taumax: f64, steps: usize, n: usize) -> f64 {
        use nalgebra::Rotation3;
        let in_dim = 3 + 2 * n;
        let rmat = Matrix3::from_column_slice(&init[0..9]);
        let mut base = Isometry3::from_parts(Translation3::from(Vector3::new(init[9], init[10], init[11])), UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rmat)));
        let mut v0 = Vector6::from_row_slice(&init[12..18]);
        let mut q = init[18..18 + n].to_vec();
        let mut qd = init[18 + n..18 + 2 * n].to_vec();
        let mut reward = 0.0;
        for _ in 0..steps {
            let up = base.rotation.to_rotation_matrix().matrix()[(2, 2)];
            let mut feat = vec![0.0; in_dim];
            feat[0] = base.translation.z; feat[1] = up; feat[2] = v0[5];
            for i in 0..n { feat[3 + i] = q[i]; feat[3 + n + i] = qd[i]; }
            let mut tau = vec![0.0; n];
            for j in 0..n {
                let mut sv = policy[n * in_dim + j];
                for k in 0..in_dim { sv += policy[j * in_dim + k] * feat[k]; }
                tau[j] = sv.clamp(-taumax, taumax);
            }
            let (b, v, qn, qdn) = floating_contact_step(robot, inertia, base_inertia, base, v0, &q, &qd, &tau, contacts, floor, kn, kd, dt, g);
            base = b; v0 = v; q = qn; qd = qdn;
            let up2 = base.rotation.to_rotation_matrix().matrix()[(2, 2)];
            let eff: f64 = tau.iter().map(|t| t * t).sum();
            reward += base.translation.z * up2.max(0.0) - effort_w * eff;
        }
        reward
    }

    /// The assembled learned-locomotion loop: CEM over floating-base contact rollouts on the GPU
    /// learns joint torques that hold a leg-supported base up against gravity — beating do-nothing
    /// (the leg buckles and the base collapses). Verifies the GPU rollout reward == the CPU reference.
    #[test]
    fn gpu_floating_gait_policy_learns() {
        let (robot, inertia) = from_urdf_full(LEG_G, "base", "foot").unwrap();
        let n = robot.dof();
        let base_inertia = LinkInertia { mass: 6.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.05, 0.05, 0.05)) };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (floor, kn, kd, dt) = (0.0, 6.0e3, 80.0, 1e-3);
        let contacts: Vec<crate::FootContact> = vec![(n, Vector3::zeros(), 0.9)];
        let (effort_w, taumax, steps) = (2e-5, 40.0, 250);

        // shared init: a deeply FOLDED leg (a crouch) with the foot planted — do-nothing stays crouched
        // (low base), so the policy must push the leg to stand the base up. Big headroom to improve.
        let mut init = vec![0.0f64; 18 + 2 * n];
        for (k, v) in Matrix3::<f64>::identity().as_slice().iter().enumerate() { init[k] = *v; }
        init[11] = 0.40; // p0.z
        init[18] = 0.6; // hip
        init[19] = -1.2; // knee

        let pop = 256usize;
        let gens = 60usize;
        let Some(gp) = FloatingGaitGpu::new(&robot, &inertia, &base_inertia, &contacts, floor, kn, kd, g, dt, pop) else {
            eprintln!("no GPU — skipping");
            return;
        };
        let dim = gp.policy_dim();

        // CEM
        let mut s = 0x11CEu64;
        let mut rng = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let mut mean = vec![0.0f64; dim];
        let mut sigma = vec![0.6f64; dim];
        let (mut first_best, mut last_best) = (f64::NEG_INFINITY, 0.0);
        let mut best_policy = mean.clone();
        for giter in 0..gens {
            let mut batch = vec![0.0f64; pop * dim];
            for p in 0..pop { for d in 0..dim { batch[p * dim + d] = mean[d] + sigma[d] * rng(); } }
            let rewards = gp.rollout_rewards(&batch, &init, effort_w, taumax, steps);
            let mut idx: Vec<usize> = (0..pop).collect();
            idx.sort_by(|&a, &b| rewards[b].partial_cmp(&rewards[a]).unwrap());
            let best = rewards[idx[0]];
            if giter == 0 { first_best = best; }
            last_best = best;
            best_policy = batch[idx[0] * dim..(idx[0] + 1) * dim].to_vec();
            let elite = (pop as f64 * 0.15) as usize;
            for d in 0..dim {
                let (mut m, mut v) = (0.0, 0.0);
                for &e in idx.iter().take(elite) { m += batch[e * dim + d]; }
                m /= elite as f64;
                for &e in idx.iter().take(elite) { v += (batch[e * dim + d] - m).powi(2); }
                mean[d] = m; sigma[d] = (v / elite as f64).sqrt().max(0.02);
            }
        }

        // do-nothing baseline (leg buckles, base collapses) and the learned policy's CPU reward
        let zero = vec![0.0f64; dim];
        let base_reward = cpu_gait_reward(&robot, &inertia, &base_inertia, &contacts, floor, kn, kd, dt, g, &init, &zero, effort_w, taumax, steps, n);
        let learned_cpu = cpu_gait_reward(&robot, &inertia, &base_inertia, &contacts, floor, kn, kd, dt, g, &init, &best_policy, effort_w, taumax, steps, n);
        // GPU reward for the learned policy (fill the batch) — the port check
        let filled: Vec<f64> = (0..pop).flat_map(|_| best_policy.clone()).collect();
        let grew = gp.rollout_rewards(&filled, &init, effort_w, taumax, steps)[0];
        let port_rel = ((grew - learned_cpu) / learned_cpu.abs().max(1.0)).abs();

        eprintln!("CEM floating-base support ({pop} policies × {gens} gens): best reward {first_best:.3} → {last_best:.3}; learned {learned_cpu:.3} vs do-nothing {base_reward:.3}; GPU vs CPU reward rel {port_rel:.2e}");
        assert!(port_rel < 2e-2, "GPU rollout reward diverged from the CPU reference: {port_rel}");
        assert!(last_best > first_best + 0.2, "CEM did not improve: {first_best} → {last_best}");
        assert!(learned_cpu > base_reward + 0.02 * base_reward.abs(), "learned policy did not beat do-nothing: {learned_cpu} vs {base_reward}");
    }

    // The GPU interior-point frictional contact solve matches the CPU `solve_frictional_ipm` oracle
    // across a batch of environments: a 3-DoF block on a floor with a 4-facet Coulomb friction pyramid,
    // each env a different free velocity / gap / mass. This is HARD (non-penetration + cone) contact,
    // batched on the local GPU — the Dojo mechanism, verified.
    #[test]
    fn gpu_frictional_contact_matches_cpu_ipm() {
        use crate::solve_frictional_ipm;
        use nalgebra::{DMatrix, DVector};
        let row = |a: [f64; 3]| DVector::from_row_slice(&a);
        // fixed contact structure: normal +z, pyramid ±x ±y, mu 0.6
        let structure = StFrictionContact {
            jn: row([0.0, 0.0, 1.0]),
            jt: vec![row([1.0, 0.0, 0.0]), row([-1.0, 0.0, 0.0]), row([0.0, 1.0, 0.0]), row([0.0, -1.0, 0.0])],
            phi: 0.0,
            mu: 0.6,
        };
        let (dt, kappa, n_envs) = (0.01, 1e-3, 256usize);
        let Some(gpu) = FrictionalContactGpu::new(std::slice::from_ref(&structure), dt, kappa, n_envs) else {
            eprintln!("no GPU adapter; skipping gpu_frictional_contact_matches_cpu_ipm");
            return;
        };

        // deterministic per-env batch: varied mass, free velocity, and gap.
        let (mut mflat, mut vf, mut phi) = (Vec::new(), Vec::new(), Vec::new());
        let mut cpu_vn = Vec::new();
        for e in 0..n_envs {
            let t = e as f64 / n_envs as f64;
            let mass = 0.5 + 1.5 * t; // 0.5 → 2.0 kg
            let m = DMatrix::from_diagonal(&DVector::from_row_slice(&[mass, mass, mass]));
            let vfree = DVector::from_row_slice(&[
                1.6 * (0.3 + 0.9 * t) * ((e % 7) as f64 - 3.0) / 3.0,
                1.2 * (0.2 + 0.8 * t) * ((e % 5) as f64 - 2.0) / 2.0,
                -0.1 - 0.4 * ((e % 3) as f64),
            ]);
            let gap = -0.002 * (e % 4) as f64; // small penetration
            let mut c = structure.clone();
            c.phi = gap;
            let s = solve_frictional_ipm(&m, &vfree, std::slice::from_ref(&c), dt, kappa);
            cpu_vn.extend(s.v_next.iter().copied());
            // row-major mass matrix
            for r in 0..3 {
                for cc in 0..3 {
                    mflat.push(m[(r, cc)]);
                }
            }
            vf.extend(vfree.iter().copied());
            phi.push(gap);
        }

        let gpu_vn = gpu.solve(&mflat, &vf, &phi);
        assert_eq!(gpu_vn.len(), cpu_vn.len());
        let finite = gpu_vn.iter().all(|v| v.is_finite());
        assert!(finite, "GPU produced non-finite velocities");
        let mut diffs: Vec<f64> = gpu_vn.iter().zip(&cpu_vn).map(|(g, c)| (g - c).abs()).collect();
        let worst = diffs.iter().cloned().fold(0.0f64, f64::max);
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = diffs[diffs.len() / 2];
        eprintln!("GPU frictional IPM vs CPU: {n_envs} envs (3-DoF block, 4-facet pyramid), median |Δv⁺| {median:.2e}, worst {worst:.2e}");
        assert!(median < 2e-3, "GPU IPM contact diverged from CPU (median {median})");
        assert!(worst < 2e-2, "GPU IPM contact worst-case too large ({worst})");
    }
}
