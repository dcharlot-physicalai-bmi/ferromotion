//! Per-call-site contraction. The published default for every site is [`Contraction::Contracted`].
//!
//! A site may name another contraction only in `docs/acceptance/oracles/manifest.json` after a
//! matched oracle reproduces that choice. No such selection is in force. Arm64 golden flags are
//! unproven provenance.

/// How one call site rounds a product-sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Contraction {
    /// Published `mul_add` chain. This is the build default.
    Contracted,
    /// Each product rounds, then the C expression adds left to right.
    /// Observed to match exported `mju_mulQuat` in the pinned Linux x86_64 wheel
    /// on the vector in the unit test. Not selected for the default build.
    Unfused,
}

/// `mul_quat` build selection. Contracted until a manifest entry changes it.
pub const MUL_QUAT: Contraction = Contraction::Contracted;

pub fn mul_quat(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    mul_quat_with(MUL_QUAT, a, b)
}

pub fn mul_quat_with(policy: Contraction, a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    match policy {
        Contraction::Contracted => [
            (-a[3]).mul_add(b[3], (-a[2]).mul_add(b[2], a[0].mul_add(b[0], -(a[1] * b[1])))),
            (-a[3]).mul_add(b[2], a[2].mul_add(b[3], a[0].mul_add(b[1], a[1] * b[0]))),
            a[3].mul_add(b[1], a[2].mul_add(b[0], a[0].mul_add(b[2], -(a[1] * b[3])))),
            a[3].mul_add(b[0], (-a[2]).mul_add(b[1], a[0].mul_add(b[3], a[1] * b[2]))),
        ],
        Contraction::Unfused => {
            let (p00, p11, p22, p33) = (a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]);
            let (p01, p10, p23, p32) = (a[0] * b[1], a[1] * b[0], a[2] * b[3], a[3] * b[2]);
            let (p02, p13, p20, p31) = (a[0] * b[2], a[1] * b[3], a[2] * b[0], a[3] * b[1]);
            let (p03, p12, p21, p30) = (a[0] * b[3], a[1] * b[2], a[2] * b[1], a[3] * b[0]);
            [
                ((p00 - p11) - p22) - p33,
                ((p01 + p10) + p23) - p32,
                ((p02 - p13) + p20) + p31,
                ((p03 + p12) - p21) + p30,
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mul_quat_stays_contracted_and_unfused_matches_the_pinned_wheel_vector() {
        assert_eq!(MUL_QUAT, Contraction::Contracted);
        let manifest = include_str!("../../../docs/acceptance/oracles/manifest.json");
        assert!(manifest.contains("\"mul_quat\": \"contracted\""));
        let a = [0.9, 0.1, -0.2, 0.3];
        let b = [0.8, -0.1, 0.4, 0.2];
        let contracted = mul_quat(&a, &b);
        assert_eq!(contracted.map(f64::to_bits), mul_quat_with(Contraction::Contracted, &a, &b).map(f64::to_bits));
        let wheel = [
            0x3fe8000000000000,
            0xbfc5c28f5c28f5c2,
            0x3fc3333333333333,
            0x3fdc28f5c28f5c29,
        ];
        let unfused = mul_quat_with(Contraction::Unfused, &a, &b);
        assert_eq!(unfused.map(f64::to_bits), wheel);
        assert_ne!(contracted.map(f64::to_bits), wheel);
    }
}
