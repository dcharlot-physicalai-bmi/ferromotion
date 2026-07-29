//! **The wgpu GPU path for batch collision-checking** — the parallel hot loop of sampling-based
//! planning. An RRT tree is sequential, but its inner question — does a candidate joint configuration
//! drive the arm through an obstacle? — is asked over thousands of candidates and is embarrassingly
//! parallel. [`ClearanceGpu`] evaluates [`arm_clearance`] for a whole batch of configurations in one
//! WGSL dispatch: one GPU thread per config runs forward kinematics, places the arm's swept collision
//! spheres, and min-reduces their signed distance to the [`SdfScene`] — exactly the CPU reference,
//! which is the oracle it is verified against. wgpu-portable (Metal/Vulkan/DX12 + WebGPU). Feature `gpu`.

use crate::{JointKind, Robot, Sdf, SdfScene};
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
/// [`arm_clearance`], over a whole batch of configurations at once.
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
    /// configurations. `link_r` and `per_link` mirror [`arm_clearance`]. `None` when there is no GPU.
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

        let prims: Vec<f32> = scene.prims.iter().flat_map(|s| prim_floats(s)).collect();
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
    /// `[q0…q_{dof-1}, …]`, length `n_configs·dof`). Negative = in collision. Matches [`arm_clearance`].
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
/// [`DepthCamera::render`], one GPU thread per pixel, over the full [`SdfScene`].
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
        let mut rw2 = rw.clone();
        rw2.binding = 3;
        let mut rw_range = rw.clone();
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
/// [`Lidar::scan`], one GPU thread per ray, over the full [`SdfScene`].
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

    /// The compacted hit point cloud — only rays that hit, in the same order as [`Lidar::scan`].
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

#[cfg(test)]
mod verification {
    use super::*;
    use crate::{arm_clearance, from_urdf_str, raymarch, DepthCamera, Lidar};
    use nalgebra::{Isometry3, Vector3};

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
}
