//! Spawn the rerun viewer and watch a 2-DOF arm execute a non-stop waypoint trajectory —
//! `cargo run -p ferromotion-viz --example view_arm` (needs the rerun viewer installed).
use ferromotion_viz::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (robot, q, wt) = doc_fixture();
    let rec = rerun::RecordingStreamBuilder::new("ferromotion_arm").spawn()?;
    log_robot(&rec, "pose", &robot, &q)?;
    log_trajectory(&rec, "traj", &robot, &wt, 400)?;
    Ok(())
}
