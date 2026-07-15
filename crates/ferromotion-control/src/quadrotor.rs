//! **Quadrotor differential flatness + minimum-snap trajectories** (Mellinger & Kumar, ICRA 2011) —
//! the aerial embodiment.
//!
//! A quadrotor is **differentially flat** in the outputs `σ = (x, y, z, ψ)`: given `σ(t)` and its
//! derivatives, the *entire* state and input are recovered algebraically, with no integration.
//! Thrust must produce the commanded acceleration, `T·z_B = m(a + g·e_z)`, which fixes both the
//! thrust magnitude and the body-`z` axis; yaw then fixes the remaining freedom. Differentiating that
//! relation shows **jerk determines angular velocity** (`ż_B = (m/T)(j − (z_B·j)z_B) = ω × z_B`), and
//! one more derivative shows snap determines the torques — which is exactly why the canonical
//! quadrotor trajectory is **minimum-snap**: a piecewise polynomial minimizing `∫(d⁴x/dt⁴)²` through
//! waypoints, solved here as an equality-constrained QP via its KKT system. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

/// A point on a flat trajectory: position and its derivatives, plus yaw and yaw rate.
#[derive(Clone, Copy, Debug)]
pub struct FlatState {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub acc: Vector3<f64>,
    pub jerk: Vector3<f64>,
    pub yaw: f64,
    pub yaw_rate: f64,
}

/// The recovered quadrotor state/input: thrust, attitude, and body-frame angular velocity.
#[derive(Clone, Copy, Debug)]
pub struct QuadState {
    /// Total thrust (N).
    pub thrust: f64,
    /// Attitude `R` (world←body); columns are the body axes.
    pub rotation: Matrix3<f64>,
    /// Angular velocity in the **body** frame `(p, q, r)`.
    pub omega: Vector3<f64>,
}

/// **Flatness map**: recover thrust, attitude, and angular velocity from the flat outputs.
/// `g` is the gravity magnitude (e.g. 9.81); gravity acts along `−e_z`.
pub fn flat_to_state(f: &FlatState, mass: f64, g: f64) -> QuadState {
    // Thrust must supply the acceleration plus hold up weight: T·z_B = m(a + g·e_z).
    let fvec = f.acc + Vector3::new(0.0, 0.0, g);
    let thrust = mass * fvec.norm();
    let z_b = fvec.normalize();
    // Yaw picks the remaining rotational freedom about z_B.
    let x_c = Vector3::new(f.yaw.cos(), f.yaw.sin(), 0.0);
    let y_b = z_b.cross(&x_c).normalize();
    let x_b = y_b.cross(&z_b);
    let rotation = Matrix3::from_columns(&[x_b, y_b, z_b]);

    // Jerk fixes the angular velocity: ż_B = (m/T)(j − (z_B·j)z_B), and ż_B = ω × z_B.
    let h = (mass / thrust) * (f.jerk - z_b.dot(&f.jerk) * z_b);
    let p = -h.dot(&y_b);
    let q = h.dot(&x_b);
    let r = f.yaw_rate * Vector3::z().dot(&z_b); // yaw rate projected onto the body axis
    QuadState { thrust, rotation, omega: Vector3::new(p, q, r) }
}

/// `p^(k)(t)` as a row of coefficients: `p^(k)(t) = row · c` for a degree-7 polynomial.
fn deriv_row(k: usize, t: f64) -> [f64; 8] {
    let mut row = [0.0; 8];
    for (i, r) in row.iter_mut().enumerate() {
        if i >= k {
            // i!/(i−k)! · t^{i−k}
            let mut coef = 1.0;
            for m in 0..k {
                coef *= (i - m) as f64;
            }
            *r = coef * t.powi((i - k) as i32);
        }
    }
    row
}

/// Snap cost matrix for one segment of duration `t`: `∫₀ᵗ (p⁗)² dτ = cᵀ Q c`.
fn snap_q(t: f64) -> [[f64; 8]; 8] {
    let mut q = [[0.0; 8]; 8];
    for i in 4..8 {
        for j in 4..8 {
            let ci = (i * (i - 1) * (i - 2) * (i - 3)) as f64; // i!/(i−4)!
            let cj = (j * (j - 1) * (j - 2) * (j - 3)) as f64;
            let p = (i + j - 7) as f64;
            q[i][j] = ci * cj * t.powf(p) / p;
        }
    }
    q
}

/// A minimum-snap piecewise polynomial: degree-7 coefficients per segment, in **local** time.
#[derive(Clone, Debug)]
pub struct MinSnap {
    pub coeffs: Vec<[f64; 8]>,
    pub durations: Vec<f64>,
}

impl MinSnap {
    /// Evaluate the `k`-th derivative at global time `t`.
    pub fn eval(&self, t: f64, k: usize) -> f64 {
        let mut t0 = 0.0;
        for (i, &d) in self.durations.iter().enumerate() {
            if t < t0 + d || i == self.durations.len() - 1 {
                let row = deriv_row(k, (t - t0).clamp(0.0, d));
                return (0..8).map(|j| row[j] * self.coeffs[i][j]).sum();
            }
            t0 += d;
        }
        0.0
    }

    /// Total snap cost `Σ ∫ (p⁗)²`.
    pub fn snap_cost(&self) -> f64 {
        let mut c = 0.0;
        for (seg, &d) in self.durations.iter().enumerate() {
            let q = snap_q(d);
            for i in 0..8 {
                for j in 0..8 {
                    c += self.coeffs[seg][i] * q[i][j] * self.coeffs[seg][j];
                }
            }
        }
        c
    }
}

/// Build the equality constraints `A c = b`: waypoints, `C³` interior continuity, rest at both ends.
pub(crate) fn build_constraints(positions: &[f64], durations: &[f64]) -> (DMatrix<f64>, DVector<f64>) {
    let m = durations.len();
    let nv = 8 * m;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    let push = |row: Vec<f64>, v: f64, rows: &mut Vec<Vec<f64>>, rhs: &mut Vec<f64>| {
        rows.push(row);
        rhs.push(v);
    };
    for i in 0..m {
        // Segment endpoints hit the waypoints (this also gives position continuity).
        let mut r0 = vec![0.0; nv];
        let d0 = deriv_row(0, 0.0);
        r0[i * 8..i * 8 + 8].copy_from_slice(&d0);
        push(r0, positions[i], &mut rows, &mut rhs);

        let mut r1 = vec![0.0; nv];
        let d1 = deriv_row(0, durations[i]);
        r1[i * 8..i * 8 + 8].copy_from_slice(&d1);
        push(r1, positions[i + 1], &mut rows, &mut rhs);
    }
    // C³ continuity (derivatives 1..3) at interior knots.
    for i in 0..m.saturating_sub(1) {
        for k in 1..4 {
            let mut r = vec![0.0; nv];
            let end = deriv_row(k, durations[i]);
            let start = deriv_row(k, 0.0);
            for j in 0..8 {
                r[i * 8 + j] = end[j];
                r[(i + 1) * 8 + j] = -start[j];
            }
            push(r, 0.0, &mut rows, &mut rhs);
        }
    }
    // Rest at both ends.
    for k in 1..4 {
        let mut r = vec![0.0; nv];
        let s = deriv_row(k, 0.0);
        r[0..8].copy_from_slice(&s);
        push(r, 0.0, &mut rows, &mut rhs);

        let mut r = vec![0.0; nv];
        let e = deriv_row(k, durations[m - 1]);
        r[(m - 1) * 8..(m - 1) * 8 + 8].copy_from_slice(&e);
        push(r, 0.0, &mut rows, &mut rhs);
    }

    let nc = rows.len();
    let mut a = DMatrix::zeros(nc, nv);
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            a[(i, j)] = v;
        }
    }
    (a, DVector::from_vec(rhs))
}

/// Minimum-snap trajectory through `positions` at the given segment `durations`, starting and ending
/// at rest (zero velocity/acceleration/jerk), with `C³` continuity at the interior knots.
/// Solves `min cᵀQc s.t. Ac = b` via the KKT system.
pub fn min_snap(positions: &[f64], durations: &[f64]) -> MinSnap {
    let m = durations.len();
    let nv = 8 * m;
    let (a, b) = build_constraints(positions, durations);
    let nc = a.nrows();

    // ---- KKT: [[2Q, Aᵀ], [A, 0]] [c; λ] = [0; b] ----
    let mut qbig = DMatrix::zeros(nv, nv);
    for (seg, &d) in durations.iter().enumerate() {
        let q = snap_q(d);
        for i in 0..8 {
            for j in 0..8 {
                qbig[(seg * 8 + i, seg * 8 + j)] = 2.0 * q[i][j];
            }
        }
    }
    for i in 0..nv {
        qbig[(i, i)] += 1e-8; // regularize the snap-free (low-order) directions
    }
    let dim = nv + nc;
    let mut kkt = DMatrix::zeros(dim, dim);
    kkt.view_mut((0, 0), (nv, nv)).copy_from(&qbig);
    kkt.view_mut((0, nv), (nv, nc)).copy_from(&a.transpose());
    kkt.view_mut((nv, 0), (nc, nv)).copy_from(&a);
    let mut rv = DVector::zeros(dim);
    for i in 0..nc {
        rv[nv + i] = b[i];
    }
    let sol = kkt.lu().solve(&rv).expect("min-snap KKT solvable");

    let coeffs = (0..m).map(|s| std::array::from_fn(|j| sol[s * 8 + j])).collect();
    MinSnap { coeffs, durations: durations.to_vec() }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 9.81;
    const MASS: f64 = 1.2;

    fn flat(pos: Vector3<f64>, acc: Vector3<f64>, jerk: Vector3<f64>, yaw: f64, yaw_rate: f64) -> FlatState {
        FlatState { pos, vel: Vector3::zeros(), acc, jerk, yaw, yaw_rate }
    }

    #[test]
    fn hover_is_level_thrust_equals_weight() {
        let s = flat_to_state(&flat(Vector3::zeros(), Vector3::zeros(), Vector3::zeros(), 0.0, 0.0), MASS, G);
        assert!((s.thrust - MASS * G).abs() < 1e-12, "hover thrust {} ≠ mg", s.thrust);
        assert!((s.rotation - Matrix3::identity()).norm() < 1e-12, "hover attitude not level");
        assert!(s.omega.norm() < 1e-12, "hover should not rotate");
    }

    #[test]
    fn thrust_and_attitude_reproduce_the_commanded_acceleration_and_yaw() {
        for (acc, yaw) in [(Vector3::new(2.0, -1.0, 3.0), 0.4), (Vector3::new(-4.0, 0.5, -2.0), -1.1)] {
            let s = flat_to_state(&flat(Vector3::zeros(), acc, Vector3::zeros(), yaw, 0.0), MASS, G);
            // T·z_B/m − g·e_z must equal the commanded acceleration.
            let z_b = s.rotation.column(2).into_owned();
            let a_rec = s.thrust / MASS * z_b - Vector3::new(0.0, 0.0, G);
            assert!((a_rec - acc).norm() < 1e-9, "acceleration not reproduced: {a_rec:?} vs {acc:?}");
            // The yaw convention: x_B is x_C projected onto the plane ⊥ z_B, i.e. y_B ⊥ x_C.
            // (When tilted, x_B's *heading* is not ψ — only the level case gives that.)
            let x_c = Vector3::new(yaw.cos(), yaw.sin(), 0.0);
            let y_b = s.rotation.column(1).into_owned();
            assert!(y_b.dot(&x_c).abs() < 1e-9, "yaw convention violated: y_B not ⊥ x_C");
            // Attitude is a proper rotation.
            assert!((s.rotation.transpose() * s.rotation - Matrix3::identity()).norm() < 1e-12);
            assert!((s.rotation.determinant() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn jerk_determines_angular_velocity() {
        // Non-circular check of the ω-from-jerk map: differentiate the recovered body-z axis along a
        // smooth flat trajectory and confirm ż_B = ω × z_B.
        let traj = |t: f64| -> FlatState {
            // A smooth wiggling acceleration profile ⇒ well-defined jerk.
            let acc = Vector3::new(2.0 * (1.3 * t).sin(), 1.5 * (0.9 * t).cos(), 1.0 * (0.7 * t).sin());
            let jerk = Vector3::new(2.0 * 1.3 * (1.3 * t).cos(), -1.5 * 0.9 * (0.9 * t).sin(), 1.0 * 0.7 * (0.7 * t).cos());
            flat(Vector3::zeros(), acc, jerk, 0.3 * t, 0.3)
        };
        let eps = 1e-6;
        for &t in &[0.0, 0.7, 1.9, 3.3] {
            let s = flat_to_state(&traj(t), MASS, G);
            let z_b = s.rotation.column(2).into_owned();
            // Numerical ż_B from the recovered attitude.
            let zp = flat_to_state(&traj(t + eps), MASS, G).rotation.column(2).into_owned();
            let zm = flat_to_state(&traj(t - eps), MASS, G).rotation.column(2).into_owned();
            let zdot_num = (zp - zm) / (2.0 * eps);
            // ω × z_B with ω from the flatness map (body → world).
            let omega_world = s.rotation * s.omega;
            let zdot_flat = omega_world.cross(&z_b);
            assert!((zdot_num - zdot_flat).norm() < 1e-4, "ż_B mismatch at t={t}: {zdot_num:?} vs {zdot_flat:?}");
        }
    }

    #[test]
    fn min_snap_hits_waypoints_is_c3_continuous_and_starts_and_ends_at_rest() {
        let pos = [0.0, 1.0, 0.5, 2.0];
        let dur = [1.0, 1.2, 0.8];
        let ms = min_snap(&pos, &dur);

        // Waypoints.
        let mut t = 0.0;
        for (i, &d) in dur.iter().enumerate() {
            assert!((ms.eval(t, 0) - pos[i]).abs() < 1e-6, "waypoint {i} missed");
            t += d;
        }
        assert!((ms.eval(t, 0) - pos[3]).abs() < 1e-6, "final waypoint missed");

        // Rest at both ends (velocity, acceleration, jerk).
        for k in 1..4 {
            assert!(ms.eval(0.0, k).abs() < 1e-6, "start derivative {k} ≠ 0");
            assert!(ms.eval(t, k).abs() < 1e-6, "end derivative {k} ≠ 0");
        }

        // C³ continuity across the interior knots.
        let eps = 1e-7;
        let mut tk = 0.0;
        for &d in &dur[..dur.len() - 1] {
            tk += d;
            for k in 0..4 {
                let (l, r) = (ms.eval(tk - eps, k), ms.eval(tk + eps, k));
                assert!((l - r).abs() < 1e-3, "derivative {k} discontinuous at knot t={tk}: {l} vs {r}");
            }
        }
    }

    #[test]
    fn min_snap_is_optimal_among_feasible_trajectories() {
        // Any perturbation that stays feasible (lies in the constraint null space) must raise the
        // snap cost — the defining property of the optimum.
        let pos = [0.0, 1.0, 0.5, 2.0];
        let dur = [1.0, 1.2, 0.8];
        let ms = min_snap(&pos, &dur);
        let base = ms.snap_cost();
        assert!(base > 0.0);

        // Null space of A via the eigenvectors of AᵀA with ~zero eigenvalue (nalgebra's SVD is thin,
        // so its V has no null-space rows; AᵀA is nv×nv and gives the full basis).
        let (a, _) = build_constraints(&pos, &dur);
        let ata = a.transpose() * &a;
        let eig = ata.symmetric_eigen();
        let emax = eig.eigenvalues.max();
        let mut tested = 0;
        for i in 0..eig.eigenvalues.len() {
            if eig.eigenvalues[i] > 1e-9 * emax {
                continue; // not a null-space direction
            }
            let d: Vec<f64> = (0..a.ncols()).map(|j| eig.eigenvectors[(j, i)]).collect();
            // Confirm it really is feasible-preserving.
            let ad = &a * DVector::from_vec(d.clone());
            assert!(ad.norm() < 1e-6, "not a null-space direction");
            for &eps in &[1e-3, -1e-3] {
                let mut p = ms.clone();
                for (s, seg) in p.coeffs.iter_mut().enumerate() {
                    for j in 0..8 {
                        seg[j] += eps * d[s * 8 + j];
                    }
                }
                assert!(p.snap_cost() > base, "feasible perturbation lowered the snap cost: {} vs {base}", p.snap_cost());
            }
            tested += 1;
            if tested >= 4 {
                break;
            }
        }
        assert!(tested > 0, "expected a non-trivial null space to perturb within");
    }
}
