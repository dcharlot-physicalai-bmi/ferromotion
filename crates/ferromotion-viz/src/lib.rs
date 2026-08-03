//! **Rerun logging for ferromotion** — robots, trajectories, and calibration curves in the
//! [rerun.io](https://rerun.io) viewer, the Rust-native observability layer the 2026 ecosystem
//! standardized on. This is deliberately a *companion* crate: the model-based core stays
//! dependency-light and wasm-clean, while native tooling gets streaming 3-D visualization in a
//! few lines:
//!
//! ```no_run
//! # let (robot, q, waypoint_traj) = ferromotion_viz::doc_fixture();
//! use ferromotion_viz::*;
//! let rec = rerun::RecordingStreamBuilder::new("my_robot").spawn()?;
//! log_robot(&rec, "robot", &robot, &q)?;                      // the chain, as points + strip
//! log_trajectory(&rec, "traj", &robot, &waypoint_traj, 300)?; // joints + EE path over time
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Everything logs through the standard entity-path/timeline model, so recordings compose with
//! whatever else the process logs and save to `.rrd` for offline inspection (`.save(path)`).

use ferromotion_core::Robot;
use rerun::{LineStrips3D, Points3D, RecordingStream, Scalars};

/// Anything sampleable as a multi-DoF motion: `[position, velocity, acceleration]` per DoF at `t`.
/// Implemented for both ruckig trajectory kinds; implement it to log your own.
pub trait SampledMotion {
    fn duration(&self) -> f64;
    fn sample(&self, t: f64) -> Vec<[f64; 3]>;
}

impl SampledMotion for ferromotion_ruckig::Trajectory {
    fn duration(&self) -> f64 {
        self.duration
    }
    fn sample(&self, t: f64) -> Vec<[f64; 3]> {
        self.at(t)
    }
}

impl SampledMotion for ferromotion_ruckig::WaypointTrajectory {
    fn duration(&self) -> f64 {
        self.duration
    }
    fn sample(&self, t: f64) -> Vec<[f64; 3]> {
        self.at(t)
    }
}

fn chain_points(robot: &Robot, q: &[f64]) -> Vec<[f32; 3]> {
    let mut pts = Vec::with_capacity(robot.dof() + 2);
    for k in 0..=robot.dof() {
        let p = robot.frame_pose(q, k).translation.vector;
        pts.push([p[0] as f32, p[1] as f32, p[2] as f32]);
    }
    let ee = robot.fk(q).translation.vector;
    pts.push([ee[0] as f32, ee[1] as f32, ee[2] as f32]);
    pts
}

/// Log the robot's kinematic chain at configuration `q`: joint origins as points and the chain as
/// a line strip, under `entity_path/{joints,links}`.
pub fn log_robot(rec: &RecordingStream, entity_path: &str, robot: &Robot, q: &[f64]) -> rerun::RecordingStreamResult<()> {
    let pts = chain_points(robot, q);
    rec.log(format!("{entity_path}/joints"), &Points3D::new(pts.iter().copied()).with_radii([0.012]))?;
    rec.log(format!("{entity_path}/links"), &LineStrips3D::new([pts]))?;
    Ok(())
}

/// Log a motion over its whole duration (`samples` steps) on the `t` timeline: per-DoF position /
/// velocity scalars under `entity_path/q{i}` and `entity_path/qd{i}`, the robot chain animated
/// under `entity_path/robot`, and the end-effector path as a growing strip under
/// `entity_path/ee_path`.
pub fn log_trajectory(
    rec: &RecordingStream,
    entity_path: &str,
    robot: &Robot,
    motion: &impl SampledMotion,
    samples: usize,
) -> rerun::RecordingStreamResult<()> {
    let mut ee_path: Vec<[f32; 3]> = Vec::with_capacity(samples + 1);
    for k in 0..=samples {
        let t = motion.duration() * k as f64 / samples as f64;
        rec.set_duration_secs("t", t);
        let s = motion.sample(t);
        let q: Vec<f64> = s.iter().map(|d| d[0]).collect();
        for (i, d) in s.iter().enumerate() {
            rec.log(format!("{entity_path}/q{i}"), &Scalars::new([d[0]]))?;
            rec.log(format!("{entity_path}/qd{i}"), &Scalars::new([d[1]]))?;
        }
        log_robot(rec, &format!("{entity_path}/robot"), robot, &q)?;
        let ee = robot.fk(&q).translation.vector;
        ee_path.push([ee[0] as f32, ee[1] as f32, ee[2] as f32]);
        rec.log(format!("{entity_path}/ee_path"), &LineStrips3D::new([ee_path.clone()]))?;
    }
    Ok(())
}

/// Log a named scalar curve (calibration RMS, loss, a parameter estimate…) over the `t` timeline.
pub fn log_curve(rec: &RecordingStream, entity_path: &str, values: &[f64]) -> rerun::RecordingStreamResult<()> {
    for (i, &v) in values.iter().enumerate() {
        rec.set_duration_secs("t", i as f64);
        rec.log(entity_path.to_owned(), &Scalars::new([v]))?;
    }
    Ok(())
}

/// Doc-example fixture (a small arm, a pose, a waypoint trajectory). Hidden from docs.
#[doc(hidden)]
pub fn doc_fixture() -> (Robot, Vec<f64>, ferromotion_ruckig::WaypointTrajectory) {
    use ferromotion_core::{Iso, Joint};
    use nalgebra::{Translation3, UnitQuaternion, Vector3};
    let mk = |z: f64| Iso::from_parts(Translation3::new(0.0, 0.0, z), UnitQuaternion::identity());
    let robot = Robot {
        joints: vec![Joint::revolute(mk(0.1), Vector3::z()), Joint::revolute(mk(0.3), Vector3::y())],
        ee_offset: mk(0.25),
    };
    let lims = vec![ferromotion_ruckig::Limits::new(1.5, 4.0, 15.0); 2];
    let wt = ferromotion_ruckig::plan_waypoints(&[vec![0.0, 0.0], vec![0.7, 0.4], vec![1.2, -0.2]], &lims);
    (robot, vec![0.3, -0.4], wt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole logging path works end-to-end: a robot + waypoint trajectory recorded to a real
    /// `.rrd` file, non-empty on disk. (Visual inspection is the `view_arm` example's job.)
    #[test]
    fn logs_a_robot_and_trajectory_to_rrd() {
        let dir = std::env::temp_dir().join("ferromotion_viz_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arm.rrd");
        let _ = std::fs::remove_file(&path);

        let (robot, q, wt) = doc_fixture();
        {
            let rec = rerun::RecordingStreamBuilder::new("ferromotion_viz_test")
                .save(&path)
                .expect("open rrd for writing");
            log_robot(&rec, "robot", &robot, &q).unwrap();
            log_trajectory(&rec, "traj", &robot, &wt, 60).unwrap();
            log_curve(&rec, "calib/rms", &[1.0, 0.3, 0.05, 0.01]).unwrap();
            let _ = rec.flush_blocking(); // best-effort flush; a closed viewer is not an error here
        }
        let meta = std::fs::metadata(&path).expect("rrd written");
        assert!(meta.len() > 1000, "rrd should be non-trivial: {} bytes", meta.len());
    }
}
