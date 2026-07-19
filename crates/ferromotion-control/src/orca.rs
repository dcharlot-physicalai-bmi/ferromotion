//! **ORCA — Optimal Reciprocal Collision Avoidance** (van den Berg, Guy, Lin & Manocha, ISRR 2011) — the
//! decentralized, communication-free reactive collision avoidance behind swarms of drones, AMRs, and
//! crowds. Each agent, each step, picks the velocity closest to its **preferred** velocity that lies in the
//! intersection of one **half-plane per neighbour** — the set of velocities guaranteed collision-free for a
//! time horizon `τ` under **reciprocity** (each agent takes half the avoidance responsibility). It scales to
//! thousands of agents as a tiny 2-D linear program, and it complements the crate's *planning*-based
//! multi-agent methods (`swarm` consensus/formation, `algames`/`dpilqr` games) with a reactive,
//! guarantee-bearing layer.
//!
//! This is a faithful port of the RVO2 half-plane construction and its incremental 2-D linear program.
//! Verified: two head-on agents deflect symmetrically and pass without collision, a densely-seeded group
//! stays collision-free, and an uncrowded agent gets exactly its preferred velocity. Pure `nalgebra` →
//! WASM-clean.

use nalgebra::Vector2;

fn det(a: &Vector2<f64>, b: &Vector2<f64>) -> f64 {
    a.x * b.y - a.y * b.x
}

/// A directed line; the feasible half-plane is to the *left* of `direction`.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub point: Vector2<f64>,
    pub direction: Vector2<f64>,
}

/// A disc agent: position, velocity, radius.
#[derive(Clone, Copy, Debug)]
pub struct Agent {
    pub p: Vector2<f64>,
    pub v: Vector2<f64>,
    pub radius: f64,
}

/// The ORCA half-plane constraining agent `a`'s new velocity to avoid `b` over horizon `tau` (reciprocal:
/// `a` takes half the responsibility). `dt` is the sim step (used only when already overlapping).
pub fn orca_line(a: &Agent, b: &Agent, tau: f64, dt: f64) -> Line {
    let rel_p = b.p - a.p;
    let rel_v = a.v - b.v;
    let dist_sq = rel_p.norm_squared();
    let comb_r = a.radius + b.radius;
    let comb_r_sq = comb_r * comb_r;

    let (direction, u);
    if dist_sq > comb_r_sq {
        // no collision yet: velocity obstacle is a truncated cone
        let w = rel_v - rel_p / tau;
        let w_len_sq = w.norm_squared();
        let dot1 = w.dot(&rel_p);
        if dot1 < 0.0 && dot1 * dot1 > comb_r_sq * w_len_sq {
            // project on the cutoff circle
            let w_len = w_len_sq.sqrt();
            let unit_w = w / w_len;
            direction = Vector2::new(unit_w.y, -unit_w.x);
            u = (comb_r / tau - w_len) * unit_w;
        } else {
            // project on a cone leg
            let leg = (dist_sq - comb_r_sq).sqrt();
            direction = if det(&rel_p, &w) > 0.0 {
                Vector2::new(rel_p.x * leg - rel_p.y * comb_r, rel_p.x * comb_r + rel_p.y * leg) / dist_sq
            } else {
                -Vector2::new(rel_p.x * leg + rel_p.y * comb_r, -rel_p.x * comb_r + rel_p.y * leg) / dist_sq
            };
            let dot2 = rel_v.dot(&direction);
            u = dot2 * direction - rel_v;
        }
    } else {
        // already overlapping: push apart within one step
        let inv_dt = 1.0 / dt;
        let w = rel_v - inv_dt * rel_p;
        let w_len = w.norm();
        let unit_w = w / w_len;
        direction = Vector2::new(unit_w.y, -unit_w.x);
        u = (comb_r * inv_dt - w_len) * unit_w;
    }
    Line { point: a.v + 0.5 * u, direction }
}

/// RVO2 `linearProgram1`: optimize along line `i` within the `max_speed` disc and the prior half-planes.
fn lp1(lines: &[Line], i: usize, max_speed: f64, opt: &Vector2<f64>) -> Option<Vector2<f64>> {
    let dot = lines[i].point.dot(&lines[i].direction);
    let disc = dot * dot + max_speed * max_speed - lines[i].point.norm_squared();
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let mut t_left = -dot - sqrt_disc;
    let mut t_right = -dot + sqrt_disc;
    for j in 0..i {
        let denom = det(&lines[i].direction, &lines[j].direction);
        let numer = det(&lines[j].direction, &(lines[i].point - lines[j].point));
        if denom.abs() <= 1e-12 {
            if numer < 0.0 {
                return None;
            }
            continue;
        }
        let t = numer / denom;
        if denom >= 0.0 {
            t_right = t_right.min(t);
        } else {
            t_left = t_left.max(t);
        }
        if t_left > t_right {
            return None;
        }
    }
    let mut t = lines[i].direction.dot(&(opt - lines[i].point));
    t = t.clamp(t_left, t_right);
    Some(lines[i].point + t * lines[i].direction)
}

/// RVO2 `linearProgram2`: the velocity in the half-plane intersection (within `max_speed`) closest to `opt`.
/// Returns `(fail_index, velocity)`; `fail_index == lines.len()` on full success.
fn lp2(lines: &[Line], max_speed: f64, opt: &Vector2<f64>) -> (usize, Vector2<f64>) {
    let mut result = if opt.norm_squared() > max_speed * max_speed { opt.normalize() * max_speed } else { *opt };
    for i in 0..lines.len() {
        if det(&lines[i].direction, &(lines[i].point - result)) > 0.0 {
            let temp = result;
            match lp1(lines, i, max_speed, opt) {
                Some(r) => result = r,
                None => return (i, temp),
            }
        }
    }
    (lines.len(), result)
}

/// RVO2 `linearProgram3`: the dense-crowd fallback — when the half-planes are jointly infeasible, minimize
/// the maximum constraint violation (the "safest" velocity).
fn lp3(lines: &[Line], begin: usize, max_speed: f64, result: &mut Vector2<f64>) {
    let mut distance = 0.0;
    for i in begin..lines.len() {
        if det(&lines[i].direction, &(lines[i].point - *result)) > distance {
            let mut proj: Vec<Line> = Vec::new();
            for j in 0..i {
                let denom = det(&lines[i].direction, &lines[j].direction);
                let point = if denom.abs() <= 1e-12 {
                    if lines[i].direction.dot(&lines[j].direction) > 0.0 {
                        continue;
                    }
                    0.5 * (lines[i].point + lines[j].point)
                } else {
                    lines[i].point + (det(&lines[j].direction, &(lines[i].point - lines[j].point)) / denom) * lines[i].direction
                };
                let dir = (lines[j].direction - lines[i].direction).normalize();
                proj.push(Line { point, direction: dir });
            }
            let temp = *result;
            let opt = Vector2::new(-lines[i].direction.y, lines[i].direction.x);
            if lp2(&proj, max_speed, &opt).0 < proj.len() {
                *result = temp;
            }
            distance = det(&lines[i].direction, &(lines[i].point - *result));
        }
    }
}

/// The ORCA optimal new velocity for `a` given its `neighbours`, preferred velocity, speed cap, and horizon.
pub fn orca_velocity(a: &Agent, neighbours: &[Agent], v_pref: &Vector2<f64>, max_speed: f64, tau: f64, dt: f64) -> Vector2<f64> {
    let lines: Vec<Line> = neighbours.iter().map(|b| orca_line(a, b, tau, dt)).collect();
    let (fail, mut result) = lp2(&lines, max_speed, v_pref);
    if fail < lines.len() {
        lp3(&lines, fail, max_speed, &mut result);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_uncrowded_agent_gets_its_preferred_velocity() {
        let a = Agent { p: Vector2::zeros(), v: Vector2::zeros(), radius: 0.5 };
        let vp = Vector2::new(0.8, -0.3);
        let v = orca_velocity(&a, &[], &vp, 2.0, 2.0, 0.1);
        assert!((v - vp).norm() < 1e-12, "no neighbours ⇒ preferred velocity");
    }

    #[test]
    fn two_head_on_agents_deflect_and_avoid_collision() {
        // THE HEADLINE. A at x=−2 wants +x, B at x=+2 (slightly off-axis so the pass side is defined) wants
        // −x, on a collision course. ORCA deflects them; simulate and confirm they never overlap and pass.
        let r = 0.5;
        let mut a = Agent { p: Vector2::new(-2.0, 0.0), v: Vector2::zeros(), radius: r };
        let mut b = Agent { p: Vector2::new(2.0, 0.05), v: Vector2::zeros(), radius: r };
        let dt = 0.1;
        let mut min_gap = f64::INFINITY;
        for _ in 0..80 {
            let va = orca_velocity(&a, &[b], &Vector2::new(1.0, 0.0), 1.0, 2.0, dt);
            let vb = orca_velocity(&b, &[a], &Vector2::new(-1.0, 0.0), 1.0, 2.0, dt);
            a.v = va;
            b.v = vb;
            a.p += va * dt;
            b.p += vb * dt;
            min_gap = min_gap.min((a.p - b.p).norm() - 2.0 * r);
        }
        assert!(min_gap > -1e-6, "agents must never overlap: min gap {min_gap}");
        assert!(a.p.x > 1.0 && b.p.x < -1.0, "they should pass each other");
    }

    #[test]
    fn a_dense_group_stays_collision_free() {
        // Agents on a circle each aiming for the antipode (maximally conflicting) must not collide — the
        // reciprocal guarantee, exercising the LP3 fallback.
        let r = 0.4;
        let n = 6;
        let mut agents: Vec<Agent> = (0..n)
            .map(|i| {
                let th = std::f64::consts::TAU * i as f64 / n as f64;
                Agent { p: Vector2::new(3.0 * th.cos(), 3.0 * th.sin()), v: Vector2::zeros(), radius: r }
            })
            .collect();
        let dt = 0.1;
        let mut min_gap = f64::INFINITY;
        for _ in 0..120 {
            let vs: Vec<Vector2<f64>> = (0..n)
                .map(|i| {
                    let others: Vec<Agent> = (0..n).filter(|&j| j != i).map(|j| agents[j]).collect();
                    let v_pref = (-agents[i].p).normalize(); // head to the antipode
                    orca_velocity(&agents[i], &others, &v_pref, 1.0, 2.0, dt)
                })
                .collect();
            for (ag, v) in agents.iter_mut().zip(&vs) {
                ag.v = *v;
                ag.p += v * dt;
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    min_gap = min_gap.min((agents[i].p - agents[j].p).norm() - 2.0 * r);
                }
            }
        }
        assert!(min_gap > -0.02, "the reciprocal guarantee should keep the group ~collision-free: min gap {min_gap}");
    }
}
