//! **MuJoCo's sensors** (`<sensor>`), read from an MJCF file and evaluated at a state — `mj_sensorPos`,
//! `mj_sensorVel` and `mj_sensorAcc` from MuJoCo 3.13.0's `engine_sensor.c`, for the sensor types a robot
//! model declares: joint, tendon and actuator positions and velocities, site and body frames (position,
//! orientation, axes, linear and angular velocity and acceleration), the IMU (gyro, accelerometer,
//! velocimeter, magnetometer), a site's force and torque, touch, and actuator forces.
//!
//! The acceleration stage needs `mj_rnePostConstraint`'s quantities, and they are formed here per BODY as
//! MuJoCo has them, not per link of this port's tree (which merges a jointless body into the one it rides):
//! each body's Newton–Euler wrench `m·(a_c − g)`, `I·α + ω×I·ω`, summed over the sensor body's subtree,
//! less the contact and connect forces acting on that subtree, is `cfrc_int` — the force the parent exerts.
//!
//! Each sensor's `sensordata` is what MuJoCo writes, `cutoff` applied as MuJoCo applies it (both sides for a
//! real-valued sensor, never for an axis or a quaternion). A sensor this port does not evaluate is kept, in
//! file order, as [`SensorKind::Unsupported`] with the reason — so the list lines up with MuJoCo's
//! `sensor_*` arrays index for index.

use crate::mjcf::El;
use crate::{Iso, MjcfTree};
use nalgebra::{UnitQuaternion, Vector3};

/// `mjtSensor`, for the types evaluated here.
#[derive(Clone, Debug, PartialEq)]
pub enum SensorKind {
    JointPos,
    JointVel,
    TendonPos,
    TendonVel,
    ActuatorPos,
    ActuatorVel,
    BallQuat,
    BallAngVel,
    FramePos,
    FrameXAxis,
    FrameYAxis,
    FrameZAxis,
    FrameQuat,
    FrameLinVel,
    FrameAngVel,
    Velocimeter,
    Gyro,
    Magnetometer,
    /// `jointactuatorfrc`: the actuator force on the joint's dof, `qfrc_actuator[dofadr]`
    JointActFrc,
    /// `actuatorfrc`: `actuator_force`
    ActuatorFrc,
    Clock,
    /// the site's linear acceleration less gravity (PROPER acceleration), in the site frame
    Accelerometer,
    /// `cfrc_int` of the site's body — the wrench its parent exerts on its subtree — at the site, in its frame
    Force,
    Torque,
    /// the normal force of every contact on the site's body whose normal ray passes through the site's zone
    Touch,
    /// the frame's linear acceleration less gravity, in the world
    FrameLinAcc,
    FrameAngAcc,
    /// A sensor this port does not evaluate, and why.
    Unsupported(String),
}

/// What a sensor reads.
#[derive(Clone, Debug, PartialEq)]
pub enum SensorObject {
    /// index into [`MjcfTree::joints`]
    Joint(usize),
    /// index into [`MjcfTree::tendons`]
    Tendon(usize),
    /// index into [`MjcfTree::actuators`]
    Actuator(usize),
    Site(String),
    /// `objtype="body"`: the body's INERTIAL frame (`xipos`, `ximat`)
    Body(String),
    /// `objtype="xbody"`: the body's own frame (`xpos`, `xmat`)
    XBody(String),
    /// index into [`MjcfTree::geoms`]
    Geom(usize),
    None,
}

/// One `<sensor>` child, resolved.
#[derive(Clone, Debug)]
pub struct MjcfSensor {
    pub name: String,
    pub kind: SensorKind,
    pub object: SensorObject,
    /// `reftype`/`refname`: the frame a frame sensor is expressed in; `None` is the world
    pub reference: Option<SensorObject>,
    /// `cutoff`, zero when absent (no clipping)
    pub cutoff: f64,
}

impl MjcfSensor {
    /// `sensor_dim`.
    pub fn dim(&self) -> usize {
        use SensorKind::*;
        match self.kind {
            JointPos | JointVel | TendonPos | TendonVel | ActuatorPos | ActuatorVel | JointActFrc | ActuatorFrc | Clock | Touch => 1,
            BallQuat | FrameQuat => 4,
            Unsupported(_) => 0,
            _ => 3,
        }
    }

    /// Whether `cutoff` clips this sensor (`mjDATATYPE_REAL`); axes and quaternions are never clipped.
    fn clipped(&self) -> bool {
        use SensorKind::*;
        !matches!(self.kind, FrameXAxis | FrameYAxis | FrameZAxis | FrameQuat | BallQuat | Unsupported(_))
    }
}

/// Every `<sensor>` section's children, in file order.
pub(crate) fn parse_sensors(root: &El, t: &MjcfTree) -> Result<Vec<MjcfSensor>, String> {
    let mut out = Vec::new();
    for section in root.children_named("sensor") {
        for el in &section.children {
            let name = el.attr("name").unwrap_or("").to_string();
            let cutoff = match el.attr("cutoff") {
                Some(v) => v.trim().parse::<f64>().map_err(|e| format!("sensor '{name}' cutoff: {e}"))?,
                None => 0.0,
            };
            let joint = |k: &str| el.attr(k).and_then(|n| t.joints.iter().position(|j| j.name == n)).map(SensorObject::Joint);
            let site = || el.attr("site").map(|s| SensorObject::Site(s.to_string()));
            let frame = |kind: Option<&str>, n: Option<&str>| -> Option<SensorObject> {
                let n = n?;
                match kind? {
                    "site" => Some(SensorObject::Site(n.to_string())),
                    "body" => Some(SensorObject::Body(n.to_string())),
                    "xbody" => Some(SensorObject::XBody(n.to_string())),
                    "geom" => t.geoms.iter().position(|g| g.name == n).map(SensorObject::Geom),
                    _ => None,
                }
            };
            use SensorKind::*;
            let (kind, object) = match el.name.as_str() {
                "jointpos" => (JointPos, joint("joint")),
                "jointvel" => (JointVel, joint("joint")),
                "ballquat" => (BallQuat, joint("joint")),
                "ballangvel" => (BallAngVel, joint("joint")),
                "jointactuatorfrc" => (JointActFrc, joint("joint")),
                "tendonpos" => (TendonPos, el.attr("tendon").and_then(|n| t.tendons.iter().position(|x| x.name == n)).map(SensorObject::Tendon)),
                "tendonvel" => (TendonVel, el.attr("tendon").and_then(|n| t.tendons.iter().position(|x| x.name == n)).map(SensorObject::Tendon)),
                "actuatorpos" | "actuatorvel" | "actuatorfrc" => {
                    let k = match el.name.as_str() {
                        "actuatorpos" => ActuatorPos,
                        "actuatorvel" => ActuatorVel,
                        _ => ActuatorFrc,
                    };
                    (k, el.attr("actuator").and_then(|n| t.actuators.iter().position(|a| a.name == n)).map(SensorObject::Actuator))
                }
                "gyro" => (Gyro, site()),
                "accelerometer" => (Accelerometer, site()),
                "force" => (Force, site()),
                "torque" => (Torque, site()),
                "touch" => (Touch, site()),
                "velocimeter" => (Velocimeter, site()),
                "magnetometer" => (Magnetometer, site()),
                "framepos" | "framexaxis" | "frameyaxis" | "framezaxis" | "framequat" | "framelinvel" | "frameangvel" | "framelinacc" | "frameangacc" => {
                    let k = match el.name.as_str() {
                        "framepos" => FramePos,
                        "framexaxis" => FrameXAxis,
                        "frameyaxis" => FrameYAxis,
                        "framezaxis" => FrameZAxis,
                        "framequat" => FrameQuat,
                        "framelinvel" => FrameLinVel,
                        "frameangvel" => FrameAngVel,
                        "framelinacc" => FrameLinAcc,
                        _ => FrameAngAcc,
                    };
                    (k, frame(el.attr("objtype"), el.attr("objname")))
                }
                "clock" => (Clock, Some(SensorObject::None)),
                other => (Unsupported(format!("<{other}>")), Some(SensorObject::None)),
            };
            let (kind, object) = match object {
                Some(o) => (kind, o),
                None if matches!(kind, Unsupported(_)) => (kind, SensorObject::None),
                None => (Unsupported(format!("<{}>: its object is not one this port resolves", el.name)), SensorObject::None),
            };
            let reference = match (el.attr("reftype"), el.attr("refname")) {
                (Some(_), Some(_)) => match frame(el.attr("reftype"), el.attr("refname")) {
                    Some(r) => Some(r),
                    None => {
                        out.push(MjcfSensor { name, kind: Unsupported(format!("<{}>: its reference frame is not one this port resolves", el.name)), object, reference: None, cutoff });
                        continue;
                    }
                },
                _ => None,
            };
            out.push(MjcfSensor { name, kind, object, reference, cutoff });
        }
    }
    Ok(out)
}

impl MjcfTree {
    /// **`sensordata`**, one entry per sensor in [`MjcfTree::sensors`] (`None` where the sensor is not
    /// evaluated here), at a state in MuJoCo's layout: `qpos`, `qvel`, the control and activations the
    /// actuator sensors read, and `time`. An acceleration-stage sensor runs the forward pass
    /// ([`MjcfTree::forward_mujoco_at`]); pass one you already have to [`MjcfTree::sensordata_with`].
    pub fn sensordata(&self, qpos: &[f64], qvel: &[f64], ctrl: &[f64], act: &[f64], time: f64) -> Result<Vec<Option<Vec<f64>>>, String> {
        use SensorKind::*;
        let acc_stage = self.sensors.iter().any(|s| matches!(s.kind, Accelerometer | Force | Torque | Touch | FrameLinAcc | FrameAngAcc));
        let forward = if acc_stage { Some(self.forward_mujoco_at(&self.q_from_qpos(qpos, &self.qposadr())?, qvel, ctrl, act, time)?) } else { None };
        self.sensordata_with(qpos, qvel, ctrl, act, time, forward.as_ref())
    }

    /// [`MjcfTree::sensordata`] with the forward pass at this state already run; acceleration-stage sensors
    /// read `None` without one.
    pub fn sensordata_with(&self, qpos: &[f64], qvel: &[f64], ctrl: &[f64], act: &[f64], time: f64, forward: Option<&crate::MjcfForward>) -> Result<Vec<Option<Vec<f64>>>, String> {
        self.sensordata_applied(qpos, qvel, ctrl, act, time, forward, &crate::MjcfApplied::default())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sensordata_applied(&self, qpos: &[f64], qvel: &[f64], ctrl: &[f64], act: &[f64], time: f64, forward: Option<&crate::MjcfForward>, applied: &crate::MjcfApplied) -> Result<Vec<Option<Vec<f64>>>, String> {
        let q = self.q_from_qpos(qpos, &self.qposadr())?;
        let tinv = self.free_basis(&q).try_inverse().ok_or("gimbal lock: the Euler base's basis map is singular")?;
        let qd: Vec<f64> = (&tinv * nalgebra::DVector::from_row_slice(qvel)).iter().copied().collect();
        let frames = crate::tree_frames(&self.tree, &q);
        let qposadr = self.qposadr();
        let need_act = self.sensors.iter().any(|s| matches!(s.kind, SensorKind::ActuatorPos | SensorKind::ActuatorVel | SensorKind::ActuatorFrc));
        let act_state = need_act.then(|| self.actuator_state(&q, &qd));
        let need_ten = self.sensors.iter().any(|s| matches!(s.kind, SensorKind::TendonPos | SensorKind::TendonVel));
        let ten = need_ten.then(|| (self.ten_length(&q), self.ten_moment(&q)));
        let qfrc_act = self.sensors.iter().any(|s| s.kind == SensorKind::JointActFrc).then(|| self.qfrc_actuator_at(&q, &qd, ctrl, act, time));
        let act_force = self.sensors.iter().any(|s| s.kind == SensorKind::ActuatorFrc).then(|| self.actuator_force_at(&q, &qd, ctrl, act, time));
        // an object's world pose and the tree joint it rides (None: welded to the world)
        let pose = |o: &SensorObject| -> Option<(Iso, Option<usize>)> {
            match o {
                SensorObject::Site(n) => match self.site_frames.get(n) {
                    Some((j, off)) => Some((frames[*j] * off, Some(*j))),
                    None => self.world_fixed.get(&format!("site:{n}")).map(|p| (*p, None)),
                },
                SensorObject::XBody(n) | SensorObject::Body(n) => {
                    let (x, ride) = if n == "world" {
                        (Iso::identity(), None)
                    } else {
                        match self.body_frames.get(n) {
                            Some((j, pre)) => (frames[*j] * pre, Some(*j)),
                            None => (*self.world_fixed.get(&format!("body:{n}"))?, None),
                        }
                    };
                    if matches!(o, SensorObject::Body(_)) {
                        let (ipos, irot) = self.inertial_rt(n);
                        let local = Iso::from_parts(ipos.into(), UnitQuaternion::from_matrix(&irot));
                        Some((x * local, ride))
                    } else {
                        Some((x, ride))
                    }
                }
                SensorObject::Geom(i) => {
                    let g = &self.geoms[*i];
                    Some(match g.joint {
                        Some(j) => (frames[j] * g.pose, Some(j)),
                        None => (g.pose, None),
                    })
                }
                _ => None,
            }
        };
        // `mj_objectVelocity`: (angular, linear) of the frame's ORIGIN, in the world
        let velocity = |p: &Iso, ride: Option<usize>| -> (Vector3<f64>, Vector3<f64>) {
            let Some(r) = ride else { return (Vector3::zeros(), Vector3::zeros()) };
            let qdv = nalgebra::DVector::from_row_slice(&qd);
            let jl = crate::tree_jacobian::tree_point_jacobian(&self.tree.joints, &self.tree.parent, &frames, Some(r), p.translation.vector);
            let ja = crate::tree_jacobian::tree_angular_jacobian(&self.tree.joints, &self.tree.parent, &frames, Some(r));
            let (w, v) = (&ja * &qdv, &jl * &qdv);
            (Vector3::new(w[0], w[1], w[2]), Vector3::new(v[0], v[1], v[2]))
        };
        // `q̈` in this port's basis: `a_mj = T·a + Ṫ·v`
        let qdd: Option<nalgebra::DVector<f64>> = forward.map(|f| {
            let v = nalgebra::DVector::from_row_slice(&qd);
            &tinv * (nalgebra::DVector::from_row_slice(&f.qacc) - self.free_basis_dot(&q, &qd) * v)
        });
        // (angular, linear less gravity) of a frame's origin, in the world: `cacc` carried to the point, with
        // the world's own `cacc` at `−g` so a body at rest reads the support holding it up
        let acceleration = |p: &Iso, ride: Option<usize>| -> Option<(Vector3<f64>, Vector3<f64>)> {
            let qdd = qdd.as_ref()?;
            let Some(r) = ride else { return Some((Vector3::zeros(), -self.gravity)) };
            let x = p.translation.vector;
            let jl = crate::tree_jacobian::tree_point_jacobian(&self.tree.joints, &self.tree.parent, &frames, Some(r), x);
            let ja = crate::tree_jacobian::tree_angular_jacobian(&self.tree.joints, &self.tree.parent, &frames, Some(r));
            let (ab, lb) = crate::tree_jacobian::tree_spatial_bias_acceleration(&self.tree.joints, &self.tree.parent, &frames, &qd, Some(r), x);
            let (al, a) = (&ja * qdd, &jl * qdd);
            Some((Vector3::new(al[0], al[1], al[2]) + ab, Vector3::new(a[0], a[1], a[2]) + lb - self.gravity))
        };
        // ⛔ `mj_objectAcceleration` itself: a DOF-LESS body (static, or mocap) returns ZERO — not the `−g` its
        // `cacc` holds. An accelerometer bolted to the world reads 0, not 9.81 (`google_barkour_v0`).
        let object_acceleration = |p: &Iso, ride: Option<usize>| -> Option<(Vector3<f64>, Vector3<f64>)> {
            match ride {
                Some(_) => acceleration(p, ride),
                None => qdd.as_ref().map(|_| (Vector3::zeros(), Vector3::zeros())),
            }
        };
        let wrench = self.sensors.iter().any(|s| matches!(s.kind, SensorKind::Force | SensorKind::Torque)).then(|| forward.map(|f| self.body_wrenches(&frames, &qd, &velocity, &acceleration, f, applied))).flatten();
        let contact_forces = forward.map(|f| self.contact_forces(f));
        let mut out = Vec::with_capacity(self.sensors.len());
        for s in &self.sensors {
            use SensorKind::*;
            let value: Option<Vec<f64>> = match (&s.kind, &s.object) {
                (Accelerometer, o) => pose(o).and_then(|(p, ride)| object_acceleration(&p, ride).map(|(_, a)| (p.rotation.inverse() * a).as_slice().to_vec())),
                (FrameLinAcc | FrameAngAcc, o) => pose(o).and_then(|(p, ride)| object_acceleration(&p, ride).map(|(al, a)| if s.kind == FrameLinAcc { a } else { al }.as_slice().to_vec())),
                (Force | Torque, SensorObject::Site(n)) => pose(&s.object).and_then(|(p, _)| {
                    let body = self.site_body.get(n)?;
                    let (f, tau) = wrench.as_ref()?.subtree(body, p.translation.vector, &self.body_parent);
                    Some((p.rotation.inverse() * if s.kind == Force { f } else { tau }).as_slice().to_vec())
                }),
                (Touch, SensorObject::Site(n)) => pose(&s.object).and_then(|(p, _)| {
                    let body = self.site_body.get(n)?;
                    let (shape, size) = self.site_shapes.get(n).copied()?;
                    let mut sum = 0.0;
                    for c in contact_forces.as_ref()? {
                        let on = [&self.geoms[c.geom[0]].body, &self.geoms[c.geom[1]].body];
                        if on[0] != body && on[1] != body {
                            continue;
                        }
                        if c.local[0] <= 0.0 {
                            continue;
                        }
                        let mut ray = c.frame[0] * c.local[0];
                        ray /= ray.norm();
                        if on[1] == body {
                            ray = -ray;
                        }
                        if ray_geom(&p, &size, shape, &c.pos, &ray) >= 0.0 {
                            sum += c.local[0];
                        }
                    }
                    Some(vec![sum])
                }),
                (JointPos, SensorObject::Joint(j)) => Some(vec![qpos[qposadr[*j]]]),
                (JointVel, SensorObject::Joint(j)) => Some(vec![qvel[self.joints[*j].first]]),
                (BallQuat, SensorObject::Joint(j)) => {
                    let a = qposadr[*j];
                    let n = (qpos[a] * qpos[a] + qpos[a + 1] * qpos[a + 1] + qpos[a + 2] * qpos[a + 2] + qpos[a + 3] * qpos[a + 3]).sqrt();
                    Some((0..4).map(|k| qpos[a + k] / n).collect())
                }
                (BallAngVel, SensorObject::Joint(j)) => Some(qvel[self.joints[*j].first..self.joints[*j].first + 3].to_vec()),
                (JointActFrc, SensorObject::Joint(j)) => qfrc_act.as_ref().map(|f| vec![f[self.joints[*j].first]]),
                (TendonPos, SensorObject::Tendon(i)) => ten.as_ref().map(|(l, _)| vec![l[*i]]),
                (TendonVel, SensorObject::Tendon(i)) => ten.as_ref().map(|(_, m)| vec![m[*i].iter().map(|(d, c)| c * qd[*d]).sum()]),
                (ActuatorPos, SensorObject::Actuator(i)) => act_state.as_ref().map(|st| vec![st[*i].length]),
                (ActuatorVel, SensorObject::Actuator(i)) => act_state.as_ref().map(|st| vec![st[*i].velocity]),
                (ActuatorFrc, SensorObject::Actuator(i)) => act_force.as_ref().map(|f| vec![f[*i]]),
                (Clock, _) => Some(vec![time]),
                (Magnetometer, o) => pose(o).map(|(p, _)| (p.rotation.inverse() * self.magnetic).as_slice().to_vec()),
                (Gyro | Velocimeter, o) => pose(o).map(|(p, ride)| {
                    let (w, v) = velocity(&p, ride);
                    let local = p.rotation.inverse() * if s.kind == Gyro { w } else { v };
                    local.as_slice().to_vec()
                }),
                (FramePos | FrameXAxis | FrameYAxis | FrameZAxis | FrameQuat | FrameLinVel | FrameAngVel, o) => {
                    let Some((p, ride)) = pose(o) else {
                        out.push(None);
                        continue;
                    };
                    let reference = match &s.reference {
                        Some(r) => match pose(r) {
                            Some(rp) => Some(rp),
                            None => {
                                out.push(None);
                                continue;
                            }
                        },
                        None => None,
                    };
                    let v: Vector3<f64> = match s.kind {
                        FramePos => match &reference {
                            Some((rp, _)) => rp.rotation.inverse() * (p.translation.vector - rp.translation.vector),
                            None => p.translation.vector,
                        },
                        FrameXAxis | FrameYAxis | FrameZAxis => {
                            let axis = p.rotation * Vector3::ith(if s.kind == FrameXAxis { 0 } else if s.kind == FrameYAxis { 1 } else { 2 }, 1.0);
                            match &reference {
                                Some((rp, _)) => rp.rotation.inverse() * axis,
                                None => axis,
                            }
                        }
                        FrameQuat => {
                            let quat = match &reference {
                                Some((rp, _)) => rp.rotation.inverse() * p.rotation,
                                None => p.rotation,
                            };
                            out.push(Some(vec![quat.w, quat.i, quat.j, quat.k]));
                            continue;
                        }
                        _ => {
                            let (w, lin) = velocity(&p, ride);
                            let (w, lin) = match &reference {
                                // relative to a moving frame, expressed in it: `mj_computeSensorVel`
                                Some((rp, rride)) => {
                                    let (wr, vr) = velocity(rp, *rride);
                                    let rel_w = w - wr;
                                    let rel_v = lin - vr + (p.translation.vector - rp.translation.vector).cross(&wr);
                                    (rp.rotation.inverse() * rel_w, rp.rotation.inverse() * rel_v)
                                }
                                None => (w, lin),
                            };
                            if s.kind == FrameLinVel {
                                lin
                            } else {
                                w
                            }
                        }
                    };
                    Some(v.as_slice().to_vec())
                }
                _ => None,
            };
            // `apply_cutoff`: a real-valued sensor is clipped on both sides
            out.push(value.map(|mut v| {
                if s.cutoff > 0.0 && s.clipped() {
                    for x in v.iter_mut() {
                        *x = x.clamp(-s.cutoff, s.cutoff);
                    }
                }
                v
            }));
        }
        Ok(out)
    }
}

/// One contact's force as `mj_contactForce` reports it: in the contact frame, `(normal, t1, t2, τn, τ1, τ2)`.
#[derive(Clone, Debug)]
pub struct ContactForce {
    pub geom: [usize; 2],
    pub pos: Vector3<f64>,
    /// `[normal, t1, t2]`, the normal from geom 1 toward geom 2
    pub frame: [Vector3<f64>; 3],
    pub local: [f64; 6],
}

/// `mj_rnePostConstraint`'s ingredients, per BODY: each body's Newton–Euler wrench and the external forces
/// (contacts, connects) acting on it.
#[derive(Clone, Debug)]
struct BodyWrenches {
    /// per body: (centre of mass, `m·(a_c − g)`, `I·α + ω×I·ω` about the centre of mass)
    bodies: Vec<(String, Vector3<f64>, Vector3<f64>, Vector3<f64>)>,
    /// external forces: (body, point of application, force, torque)
    external: Vec<(String, Vector3<f64>, Vector3<f64>, Vector3<f64>)>,
}

impl BodyWrenches {
    /// `cfrc_int` of `root`, about `point`: the wrench its parent exerts on its subtree — what the subtree
    /// needs for its motion, less what the world applies to it from outside.
    fn subtree(&self, root: &str, point: Vector3<f64>, parents: &std::collections::BTreeMap<String, String>) -> (Vector3<f64>, Vector3<f64>) {
        let inside = |b: &str| {
            let mut x = b;
            for _ in 0..4096 {
                if x == root {
                    return true;
                }
                match parents.get(x) {
                    Some(p) => x = p.as_str(),
                    None => return false,
                }
            }
            false
        };
        let (mut f, mut tau) = (Vector3::zeros(), Vector3::zeros());
        for (b, c, fb, tb) in &self.bodies {
            if inside(b) {
                f += fb;
                tau += tb + (c - point).cross(fb);
            }
        }
        for (b, x, fe, te) in &self.external {
            if inside(b) {
                f -= fe;
                tau -= te + (x - point).cross(fe);
            }
        }
        (f, tau)
    }
}

impl MjcfTree {
    /// **Every active contact's force**, as `mj_contactForce` decodes it from the solve: a pyramid's edge
    /// forces summed into the normal and differenced into the tangents (`mju_decodePyramid`), an elliptic
    /// cone's rows as they are.
    pub fn contact_forces(&self, forward: &crate::MjcfForward) -> Vec<ContactForce> {
        let active: Vec<&crate::MjcfContact> = forward.collision.contacts.iter().filter(|c| !c.record.exclude).collect();
        let rows = |dim: usize| if dim == 1 { 1 } else if self.cone == crate::mujoco_contact::Cone::Pyramidal { 2 * (dim - 1) } else { dim };
        let total: usize = active.iter().map(|c| rows(c.record.dim)).sum();
        let force = &forward.solve.force;
        let mut row = force.len().saturating_sub(total);
        let mut out = Vec::with_capacity(active.len());
        for c in active {
            let (dim, n) = (c.record.dim, rows(c.record.dim));
            let f = &force[row..row + n];
            let mut local = [0.0; 6];
            if dim == 1 {
                local[0] = f[0];
            } else if self.cone == crate::mujoco_contact::Cone::Pyramidal {
                local[0] = f.iter().sum();
                for i in 0..dim - 1 {
                    local[i + 1] = (f[2 * i] - f[2 * i + 1]) * c.record.friction[i];
                }
            } else {
                local[..dim].copy_from_slice(f);
            }
            out.push(ContactForce { geom: c.geom, pos: c.record.pos, frame: c.record.frame, local });
            row += n;
        }
        out
    }

    /// Each body's Newton–Euler wrench and the external forces on it, at the state `forward` solved.
    fn body_wrenches(
        &self,
        frames: &[Iso],
        _qd: &[f64],
        velocity: &dyn Fn(&Iso, Option<usize>) -> (Vector3<f64>, Vector3<f64>),
        acceleration: &dyn Fn(&Iso, Option<usize>) -> Option<(Vector3<f64>, Vector3<f64>)>,
        forward: &crate::MjcfForward,
        applied: &crate::MjcfApplied,
    ) -> BodyWrenches {
        let mut bodies = Vec::new();
        for name in self.body_parent.keys() {
            let (x, ride) = match self.body_frames.get(name) {
                Some((j, pre)) => (frames[*j] * pre, Some(*j)),
                None => match self.world_fixed.get(&format!("body:{name}")) {
                    Some(p) => (*p, None),
                    None => continue,
                },
            };
            let mass = self.body_mass.get(name).copied().unwrap_or(0.0);
            let (ipos, irot) = self.inertial_rt(name);
            let com = x * nalgebra::Point3::from(ipos);
            let at = Iso::from_parts(com.coords.into(), x.rotation);
            let (w, _) = velocity(&at, ride);
            let Some((al, a)) = acceleration(&at, ride) else { continue };
            let r = x.rotation.to_rotation_matrix().into_inner() * irot;
            let i = self.body_iinertia.get(name).copied().unwrap_or([0.0; 3]);
            let iw = r * nalgebra::Matrix3::from_diagonal(&Vector3::new(i[0], i[1], i[2])) * r.transpose();
            bodies.push((name.clone(), com.coords, a * mass, iw * al + w.cross(&(iw * w))));
        }
        let mut external = Vec::new();
        // `xfrc_applied`: a world force and torque at the body's centre of mass
        for (b, w) in &applied.xfrc {
            if let Some((_, com, _, _)) = bodies.iter().find(|(n, ..)| n == b) {
                external.push((b.clone(), *com, Vector3::new(w[0], w[1], w[2]), Vector3::new(w[3], w[4], w[5])));
            }
        }
        for c in self.contact_forces(forward) {
            let f = c.frame[0] * c.local[0] + c.frame[1] * c.local[1] + c.frame[2] * c.local[2];
            let tau = c.frame[0] * c.local[3] + c.frame[1] * c.local[4] + c.frame[2] * c.local[5];
            // geom 2's body is pushed by `f`, geom 1's by `−f`
            external.push((self.geoms[c.geom[0]].body.clone(), c.pos, -f, -tau));
            external.push((self.geoms[c.geom[1]].body.clone(), c.pos, f, tau));
        }
        // a connect's three rows are a world force on body 1 at its anchor, and the opposite on body 2
        let mut row = 0usize;
        for e in &self.equalities {
            match &e.kind {
                crate::EqualityKind::Connect { side1, side2, bodies: b, .. } => {
                    let f = Vector3::new(forward.solve.force[row], forward.solve.force[row + 1], forward.solve.force[row + 2]);
                    let at = |(ride, local): &(Option<usize>, Vector3<f64>)| match ride {
                        Some(r) => (frames[*r] * nalgebra::Point3::from(*local)).coords,
                        None => *local,
                    };
                    external.push((b[0].clone(), at(side1), f, Vector3::zeros()));
                    external.push((b[1].clone(), at(side2), -f, Vector3::zeros()));
                    row += 3;
                }
                // a weld's six: the force at each anchor, and its last three rows taken as a world torque,
                // as `mj_rnePostConstraint` takes them
                crate::EqualityKind::Weld { side1, side2, bodies: b, .. } => {
                    let fr = &forward.solve.force;
                    let f = Vector3::new(fr[row], fr[row + 1], fr[row + 2]);
                    let tau = Vector3::new(fr[row + 3], fr[row + 4], fr[row + 5]);
                    let at = |(ride, local): &(Option<usize>, Vector3<f64>)| match ride {
                        Some(r) => (frames[*r] * nalgebra::Point3::from(*local)).coords,
                        None => *local,
                    };
                    external.push((b[0].clone(), at(side1), f, tau));
                    external.push((b[1].clone(), at(side2), -f, -tau));
                    row += 6;
                }
                _ => row += 1,
            }
        }
        BodyWrenches { bodies, external }
    }
}

/// `mju_rayGeom` for a site's shape: the distance along `vec` from `pnt` to the shape's surface, or −1.
/// `pose` is the shape's frame; `vec` need not be unit.
pub fn ray_geom(pose: &Iso, size: &[f64; 3], shape: crate::mujoco_collision::GeomType, pnt: &Vector3<f64>, vec: &Vector3<f64>) -> f64 {
    use crate::mujoco_collision::GeomType::*;
    const MINVAL: f64 = 1e-15;
    let pos = pose.translation.vector;
    // `ray_quad`: the smallest non-negative root of a·x² + 2b·x + c = 0, and both roots
    let quad = |a: f64, b: f64, c: f64| -> (f64, [f64; 2]) {
        let det = b * b - a * c;
        if det < 0.0 || a < MINVAL {
            return (-1.0, [-1.0, -1.0]);
        }
        let det = det.sqrt();
        let x = [(-b - det) / a, (-b + det) / a];
        (if x[0] >= 0.0 { x[0] } else if x[1] >= 0.0 { x[1] } else { -1.0 }, x)
    };
    let sphere = |r2: f64| {
        let dif = pnt - pos;
        quad(vec.dot(vec), vec.dot(&dif), dif.dot(&dif) - r2).0
    };
    // `ray_map`: the ray in the shape's frame
    let lpnt = pose.rotation.inverse() * (pnt - pos);
    let lvec = pose.rotation.inverse() * vec;
    match shape {
        Sphere => sphere(size[0] * size[0]),
        Capsule => {
            let ssz = size[0] + size[1];
            if sphere(ssz * ssz) < 0.0 {
                return -1.0;
            }
            let mut x = -1.0f64;
            let (sol, _) = quad(lvec[0] * lvec[0] + lvec[1] * lvec[1], lvec[0] * lpnt[0] + lvec[1] * lpnt[1], lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - size[0] * size[0]);
            if sol >= 0.0 && (lpnt[2] + sol * lvec[2]).abs() <= size[1] && (x < 0.0 || sol < x) {
                x = sol;
            }
            for side in [1.0f64, -1.0] {
                let ldif = Vector3::new(lpnt[0], lpnt[1], lpnt[2] - side * size[1]);
                let (_, xx) = quad(lvec.dot(&lvec), lvec.dot(&ldif), ldif.dot(&ldif) - size[0] * size[0]);
                for r in xx {
                    let z = lpnt[2] + r * lvec[2];
                    let on_cap = if side > 0.0 { z >= size[1] } else { z <= -size[1] };
                    if r >= 0.0 && on_cap && (x < 0.0 || r < x) {
                        x = r;
                    }
                }
            }
            x
        }
        Ellipsoid => {
            let s = [1.0 / (size[0] * size[0]), 1.0 / (size[1] * size[1]), 1.0 / (size[2] * size[2])];
            let a = s[0] * lvec[0] * lvec[0] + s[1] * lvec[1] * lvec[1] + s[2] * lvec[2] * lvec[2];
            let b = s[0] * lvec[0] * lpnt[0] + s[1] * lvec[1] * lpnt[1] + s[2] * lvec[2] * lpnt[2];
            let c = s[0] * lpnt[0] * lpnt[0] + s[1] * lpnt[1] * lpnt[1] + s[2] * lpnt[2] * lpnt[2] - 1.0;
            quad(a, b, c).0
        }
        Cylinder => {
            if sphere(size[0] * size[0] + size[1] * size[1]) < 0.0 {
                return -1.0;
            }
            let mut x = -1.0f64;
            if lvec[2].abs() > MINVAL {
                for side in [-1.0f64, 1.0] {
                    let sol = (side * size[1] - lpnt[2]) / lvec[2];
                    if sol >= 0.0 {
                        let (p0, p1) = (lpnt[0] + sol * lvec[0], lpnt[1] + sol * lvec[1]);
                        if p0 * p0 + p1 * p1 <= size[0] * size[0] && (x < 0.0 || sol < x) {
                            x = sol;
                        }
                    }
                }
            }
            let (sol, _) = quad(lvec[0] * lvec[0] + lvec[1] * lvec[1], lvec[0] * lpnt[0] + lvec[1] * lpnt[1], lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - size[0] * size[0]);
            if sol >= 0.0 && (lpnt[2] + sol * lvec[2]).abs() <= size[1] && (x < 0.0 || sol < x) {
                x = sol;
            }
            x
        }
        Box => {
            if sphere(size[0] * size[0] + size[1] * size[1] + size[2] * size[2]) < 0.0 {
                return -1.0;
            }
            let iface = [[1, 2], [0, 2], [0, 1]];
            let mut x = -1.0f64;
            for i in 0..3 {
                if lvec[i].abs() > MINVAL {
                    for side in [-1.0f64, 1.0] {
                        let sol = (side * size[i] - lpnt[i]) / lvec[i];
                        if sol >= 0.0 {
                            let (a, b) = (iface[i][0], iface[i][1]);
                            let (p0, p1) = (lpnt[a] + sol * lvec[a], lpnt[b] + sol * lvec[b]);
                            if p0.abs() <= size[a] && p1.abs() <= size[b] && (x < 0.0 || sol < x) {
                                x = sol;
                            }
                        }
                    }
                }
            }
            x
        }
        _ => -1.0,
    }
}

#[cfg(test)]
mod tests {
    use crate::tree_from_mjcf_str;

    /// **Every sensor type carried here, against MuJoCo 3.13.0.** A box resting on the floor (two contacts,
    /// elliptic cone) with an arm and a hand on it, all moving and driven: the IMU on a rotated site, frame
    /// sensors on sites, bodies, an xbody and a geom, one relative to another site, the arm's force/torque
    /// site, a touch pad under the box, an accelerometer bolted to the world (MuJoCo reads ZERO there, not
    /// 9.81), and the joint, tendon and actuator sensors.
    #[test]
    fn sensordata_matches_mujoco_for_every_carried_type() {
        let t = tree_from_mjcf_str(MODEL).unwrap();
        let qpos = [0.01, -0.02, 0.0995, 0.9992993349829556, 0.020005992692351465, -0.030008989038527195, 0.010002996346175732, 0.4, -0.3];
        let qvel = [0.1, -0.05, 0.02, 0.3, -0.2, 0.5, 0.7, -1.1];
        let want: [(&str, &[f64]); 25] = [
            ("acc", &[23.181544604206575, 13.88477684505398, 1.3919929181570692]),
            ("gyro", &[0.34520653548373165, -0.32648087258791186, 0.39273742843476783]),
            ("vel", &[0.05022373246784056, -0.0797544687162279, 0.012503530396151553]),
            ("mag", &[0.03505542137735844, -0.40156418309040165, 0.31227763975506484]),
            ("quat", &[0.9420442500061783, 0.051857074353291066, -0.222136166031594, 0.2460060946614899]),
            ("flv", &[0.027851966348661723, 0.20383174965567055, -0.09627424039973914]),
            ("fav", &[-0.7776439457620931, 0.4484215377265438, 0.9025784944369857]),
            ("fpos", &[0.2980836808118855, -0.014243701962763741, 0.10850707934250553]),
            ("fz", &[-0.05957568583581564, -0.04058431000169101, 0.9973984416665582]),
            ("frel", &[0.2599154515683291, -0.20458812710633756, -0.2815815424793571]),
            ("fla", &[1.4265649280147203, -19.33684365431787, -16.54622312723881]),
            ("faa", &[17635.920175201736, 677.3142882168684, -6331.34793531815]),
            ("force", &[0.29535628591532165, -6.825224090153219, -5.455843861196219]),
            ("torque", &[2.754792744224062, -0.7453605614470984, -1.7182597475950478]),
            ("touch", &[49.54987483405773]),
            // a zone ABOVE the contacts: the floor's normal points up into it, but the box is the contact's
            // second body, so MuJoCo turns the ray round and it misses
            ("touch_above", &[0.0]),
            // a free base transmits nothing to the world: its subtree's motion is ALL contact and gravity,
            // so these are rounding — and only if the contact forces are booked to the right bodies
            ("base_force", &[1.3481260466667459e-14, 9.825534589075099e-15, -1.357144767350102e-14]),
            ("base_torque", &[-2.9026565847538335e-15, 8.678329066444899e-15, 1.2685329281310302e-14]),
            ("static_acc", &[0.0, 0.0, 0.0]),
            ("jp", &[0.4]),
            ("jv", &[-1.1]),
            ("jaf", &[1.6]),
            ("af", &[2.5]),
            ("tp", &[0.55]),
            ("ap", &[-0.3]),
        ];
        let got = t.sensordata(&qpos, &qvel, &[0.8, 0.2], &[], 0.0).unwrap();
        assert_eq!(t.sensors.len(), want.len());
        for ((s, v), (name, w)) in t.sensors.iter().zip(&got).zip(want) {
            assert_eq!(s.name, name);
            let v = v.as_ref().unwrap_or_else(|| panic!("sensor '{name}' was not evaluated: {:?}", s.kind));
            assert_eq!(v.len(), w.len(), "{name}");
            for (k, (a, b)) in v.iter().zip(w).enumerate() {
                assert!((a - b).abs() <= 1e-8 * b.abs().max(1.0), "{name}[{k}]: {a} vs MuJoCo {b}");
            }
        }
    }

    const MODEL: &str = r#"<mujoco><option cone="elliptic" magnetic="0.1 -0.4 0.3"/>
<worldbody>
  <geom name="floor" type="plane" size="2 2 0.1"/>
  <site name="world_imu" pos="0.1 0 0.3"/>
  <body name="base" pos="0 0 0.0999">
    <freejoint/>
    <geom name="base_box" type="box" size="0.15 0.1 0.1" mass="3"/>
    <site name="imu" pos="0.05 0.02 0.08" euler="10 -20 30"/>
    <site name="pad" type="box" size="0.16 0.11 0.02" pos="0 0 -0.1"/>
    <site name="base_ft" pos="-0.05 0.03 0.02" euler="0 30 0"/>
    <site name="pad_above" type="box" size="0.16 0.11 0.02" pos="0 0 -0.07"/>
    <body name="arm" pos="0.15 0 0.05">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.6" contype="0" conaffinity="0"/>
      <site name="ft" pos="0.02 0 0" euler="0 0 45"/>
      <body name="hand" pos="0.3 0 0">
        <joint name="wrist" type="hinge" axis="1 0 0"/>
        <geom type="box" size="0.03 0.04 0.02" mass="0.2" contype="0" conaffinity="0"/>
        <site name="tip" pos="0.03 0 0"/>
      </body>
    </body>
  </body>
</worldbody>
<tendon><fixed name="coupling"><joint joint="shoulder" coef="1"/><joint joint="wrist" coef="-0.5"/></fixed></tendon>
<actuator><motor name="m1" joint="shoulder" gear="2"/><position name="p1" joint="wrist" kp="5"/></actuator>
<sensor>
  <accelerometer name="acc" site="imu"/>
  <gyro name="gyro" site="imu"/>
  <velocimeter name="vel" site="imu"/>
  <magnetometer name="mag" site="imu"/>
  <framequat name="quat" objtype="site" objname="imu"/>
  <framelinvel name="flv" objtype="site" objname="tip"/>
  <frameangvel name="fav" objtype="xbody" objname="hand"/>
  <framepos name="fpos" objtype="body" objname="arm"/>
  <framezaxis name="fz" objtype="geom" objname="base_box"/>
  <framepos name="frel" objtype="site" objname="tip" reftype="site" refname="imu"/>
  <framelinacc name="fla" objtype="site" objname="tip"/>
  <frameangacc name="faa" objtype="xbody" objname="hand"/>
  <force name="force" site="ft"/>
  <torque name="torque" site="ft"/>
  <touch name="touch" site="pad"/>
  <touch name="touch_above" site="pad_above"/>
  <force name="base_force" site="base_ft"/>
  <torque name="base_torque" site="base_ft"/>
  <accelerometer name="static_acc" site="world_imu"/>
  <jointpos name="jp" joint="shoulder"/>
  <jointvel name="jv" joint="wrist"/>
  <jointactuatorfrc name="jaf" joint="shoulder"/>
  <actuatorfrc name="af" actuator="p1"/>
  <tendonpos name="tp" tendon="coupling"/>
  <actuatorpos name="ap" actuator="p1"/>
</sensor></mujoco>"#;
}
