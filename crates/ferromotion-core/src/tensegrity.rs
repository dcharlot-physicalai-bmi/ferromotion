//! **Tensegrity form-finding** (Schek's force-density method, 1974) — the last member of the
//! cable-and-strut family: structures that hold their shape with *no external support*, from a balance
//! of compression struts floating in a net of tension cables. Their geometry is not designed directly;
//! it *emerges* from a self-equilibrium of internal forces.
//!
//! The tool is the **force density** `q_m = t_m / L_m` (tension over length) of each member. With the
//! signed incidence matrix `C`, the force-density matrix `D = Cᵀ diag(q) C` (a weighted graph Laplacian)
//! gives the nodal equilibrium: with no external load, `D X = 0` for each coordinate, so the shape is a
//! **null vector of D**. Cables carry `q > 0` (tension), struts `q < 0` (compression), and only a
//! special ratio of force densities makes `D` rank-deficient enough to yield a real 3-D form — which is
//! exactly what "self-stress" means. Form-finding fixes a few nodes and solves the rest. Pure `nalgebra`
//! → WASM-clean.

use nalgebra::{DMatrix, DVector, Vector3};

/// A member connecting two nodes, in tension (cable) or compression (strut), with a force density.
#[derive(Clone, Copy, Debug)]
pub struct Member {
    pub i: usize,
    pub j: usize,
    /// `true` = cable (should carry `q > 0`), `false` = strut (`q < 0`).
    pub cable: bool,
    /// Force density `q = tension / length`.
    pub q: f64,
}

/// A tensegrity: node positions plus the members that connect them.
#[derive(Clone, Debug)]
pub struct Tensegrity {
    pub nodes: Vec<Vector3<f64>>,
    pub members: Vec<Member>,
}

impl Tensegrity {
    pub fn n(&self) -> usize {
        self.nodes.len()
    }

    /// Net force at node `k` from the force-density model: member `m=(i,j)` pulls node `i` toward `j`
    /// with `q_m (x_j − x_i)`. Zero at every node ⇒ self-equilibrium (with no external load).
    pub fn nodal_residual(&self, k: usize) -> Vector3<f64> {
        let mut f = Vector3::zeros();
        for m in &self.members {
            if m.i == k {
                f += m.q * (self.nodes[m.j] - self.nodes[k]);
            } else if m.j == k {
                f += m.q * (self.nodes[m.i] - self.nodes[k]);
            }
        }
        f
    }

    /// Largest nodal force-imbalance over all nodes (0 ⇒ self-equilibrated).
    pub fn max_residual(&self) -> f64 {
        (0..self.n()).map(|k| self.nodal_residual(k).norm()).fold(0.0, f64::max)
    }

    pub fn is_self_equilibrated(&self, tol: f64) -> bool {
        self.max_residual() < tol
    }

    /// Cables in tension (`q > 0`) and struts in compression (`q < 0`) — the sign convention a valid
    /// tensegrity must respect.
    pub fn signs_valid(&self) -> bool {
        self.members.iter().all(|m| if m.cable { m.q > 0.0 } else { m.q < 0.0 })
    }

    /// The force-density matrix `D = Cᵀ diag(q) C` (a weighted Laplacian): `D[i][i] = Σ q` of members
    /// at `i`, `D[i][j] = −q_m` for a member joining `i,j`. Self-equilibrium ⇔ `D X = 0` per coordinate.
    pub fn force_density_matrix(&self) -> DMatrix<f64> {
        let n = self.n();
        let mut d = DMatrix::zeros(n, n);
        for m in &self.members {
            d[(m.i, m.i)] += m.q;
            d[(m.j, m.j)] += m.q;
            d[(m.i, m.j)] -= m.q;
            d[(m.j, m.i)] -= m.q;
        }
        d
    }

    /// Force-density form-finding: with `fixed` nodes pinned at their current positions and the rest
    /// free, solve `D_ff X_f = −D_fn X_n` for the free-node coordinates that put the structure in
    /// self-equilibrium under the assigned force densities. Updates `self.nodes` in place.
    pub fn form_find(&mut self, fixed: &[usize]) {
        let n = self.n();
        let is_fixed = |k: usize| fixed.contains(&k);
        let free: Vec<usize> = (0..n).filter(|&k| !is_fixed(k)).collect();
        if free.is_empty() {
            return;
        }
        let d = self.force_density_matrix();
        let nf = free.len();
        let mut dff = DMatrix::zeros(nf, nf);
        for (a, &i) in free.iter().enumerate() {
            for (b, &j) in free.iter().enumerate() {
                dff[(a, b)] = d[(i, j)];
            }
        }
        let Some(dff_inv) = dff.try_inverse() else { return };
        for axis in 0..3 {
            // rhs = −D_fn X_n (contribution of the fixed nodes)
            let mut rhs = DVector::zeros(nf);
            for (a, &i) in free.iter().enumerate() {
                let mut s = 0.0;
                for &j in fixed {
                    s += d[(i, j)] * self.nodes[j][axis];
                }
                rhs[a] = -s;
            }
            let xf = &dff_inv * rhs;
            for (a, &i) in free.iter().enumerate() {
                self.nodes[i][axis] = xf[a];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    /// The canonical planar "X" tensegrity: a unit square of tension cables (the 4 sides) with two
    /// compression struts on the diagonals. Self-equilibrium requires `q_strut = −q_cable`.
    fn square_x(qc: f64, qs: f64) -> Tensegrity {
        Tensegrity {
            nodes: vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(1.0, 1.0, 0.0), v(0.0, 1.0, 0.0)],
            members: vec![
                Member { i: 0, j: 1, cable: true, q: qc },
                Member { i: 1, j: 2, cable: true, q: qc },
                Member { i: 2, j: 3, cable: true, q: qc },
                Member { i: 3, j: 0, cable: true, q: qc },
                Member { i: 0, j: 2, cable: false, q: qs }, // diagonal strut
                Member { i: 1, j: 3, cable: false, q: qs }, // diagonal strut
            ],
        }
    }

    #[test]
    fn the_x_square_self_equilibrates_only_at_the_right_force_ratio() {
        // THE INVARIANT. With q_strut = −q_cable the square is in self-equilibrium (zero nodal force);
        // any other ratio leaves a residual — self-stress is a knife-edge property of the q's.
        let good = square_x(1.0, -1.0);
        assert!(good.max_residual() < 1e-12, "balanced X should self-equilibrate: {}", good.max_residual());
        assert!(good.signs_valid(), "cables in tension, struts in compression");
        let bad = square_x(1.0, -0.5);
        assert!(bad.max_residual() > 0.1, "the wrong ratio must NOT be in equilibrium: {}", bad.max_residual());
    }

    #[test]
    fn the_force_density_matrix_annihilates_the_equilibrium_geometry() {
        // D X = 0 per coordinate at self-equilibrium — the geometry lies in the null space of D.
        let t = square_x(1.0, -1.0);
        let d = t.force_density_matrix();
        for axis in 0..2 {
            let x = DVector::from_iterator(t.n(), t.nodes.iter().map(|p| p[axis]));
            assert!((&d * &x).norm() < 1e-12, "D·X should vanish on axis {axis}");
        }
        // D is symmetric (Laplacian structure)
        assert!((d.clone() - d.transpose()).norm() < 1e-12);
    }

    #[test]
    fn form_finding_places_a_free_node_at_self_equilibrium() {
        // Force-density form-finding solves D_ff X_f = −D_fn X_n for the free nodes. A tensegrity's D is
        // deliberately rank-deficient (that degeneracy is what admits a self-stressed shape), so the
        // pinned set must cover the deficiency for D_ff to be invertible — here, pin three corners and
        // recover the fourth. Perturb it, re-solve, and it returns to the square in self-equilibrium.
        let mut t = square_x(1.0, -1.0);
        t.nodes[3] = v(-0.2, 0.8, 0.3); // knock the free corner off
        t.form_find(&[0, 1, 2]);
        assert!((t.nodes[3] - v(0.0, 1.0, 0.0)).norm() < 1e-9, "node 3 → (0,1,0), got {:?}", t.nodes[3]);
        assert!(t.max_residual() < 1e-9, "recovered shape should be self-equilibrated: {}", t.max_residual());
    }

    /// A prestressed **cable net** (Schek's original force-density application): a 3×3 grid, boundary
    /// pinned, all members cables. Unlike a tensegrity its D_ff is a proper positive-definite Laplacian,
    /// so form-finding pulls the interior node to its taut minimal-energy position (the centroid here).
    #[test]
    fn form_finding_pulls_a_cable_net_taut() {
        // 5 nodes: 4 pinned around a unit diamond + 1 interior; 4 cables to the interior.
        let mut t = Tensegrity {
            nodes: vec![v(1.0, 0.0, 0.0), v(-1.0, 0.0, 0.0), v(0.0, 1.0, 0.0), v(0.0, -1.0, 0.0), v(0.4, -0.3, 0.6)],
            members: vec![
                Member { i: 4, j: 0, cable: true, q: 1.0 },
                Member { i: 4, j: 1, cable: true, q: 1.0 },
                Member { i: 4, j: 2, cable: true, q: 1.0 },
                Member { i: 4, j: 3, cable: true, q: 1.0 },
            ],
        };
        t.form_find(&[0, 1, 2, 3]); // pin the boundary
        // equal cable force densities to symmetric anchors ⇒ the interior node sits at their centroid (origin)
        assert!(t.nodes[4].norm() < 1e-9, "interior node should settle at the centroid, got {:?}", t.nodes[4]);
        assert!(t.nodal_residual(4).norm() < 1e-9, "interior node in force balance");
    }

    #[test]
    fn a_strut_carries_compression_a_cable_carries_tension() {
        // Sanity on the physics: in the balanced X, the diagonal struts push their endpoints apart
        // (they are longer than the sides) and the side cables pull theirs together.
        let t = square_x(1.0, -1.0);
        let diag_len = (t.nodes[2] - t.nodes[0]).norm();
        let side_len = (t.nodes[1] - t.nodes[0]).norm();
        assert!(diag_len > side_len, "struts (diagonals) span farther than cables (sides)");
        // member force = q·L; struts negative (compression), cables positive (tension)
        for m in &t.members {
            let force = m.q * (t.nodes[m.j] - t.nodes[m.i]).norm();
            if m.cable {
                assert!(force > 0.0, "cable should be in tension");
            } else {
                assert!(force < 0.0, "strut should be in compression");
            }
        }
    }
}
