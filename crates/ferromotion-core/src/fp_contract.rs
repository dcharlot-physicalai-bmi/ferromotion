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
    let out = mul_quat_with(MUL_QUAT, a, b);
    let mut inputs = [0.0; 8];
    inputs[..4].copy_from_slice(a);
    inputs[4..].copy_from_slice(b);
    record_op("mul_quat", &inputs, &out);
    out
}

/// One executed call. Present only while [`capture_ops`] is running.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpSample {
    pub op: &'static str,
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
}

use std::cell::RefCell;

thread_local! {
    static OP_TRACE: RefCell<Option<Vec<OpSample>>> = const { RefCell::new(None) };
}

/// Record `op` when a capture is active. A capture that is off does not store the call.
pub fn record_op(op: &'static str, inputs: &[f64], outputs: &[f64]) {
    OP_TRACE.with(|slot| {
        let mut guard = slot.borrow_mut();
        if let Some(buf) = guard.as_mut() {
            buf.push(OpSample {
                op,
                inputs: inputs.iter().copied().map(f64::to_bits).collect(),
                outputs: outputs.iter().copied().map(f64::to_bits).collect(),
            });
        }
    });
}

/// Run `f` and return the instrumented calls it made, in order.
pub fn capture_ops<R>(f: impl FnOnce() -> R) -> (R, Vec<OpSample>) {
    OP_TRACE.with(|slot| *slot.borrow_mut() = Some(Vec::new()));
    let value = f();
    let samples = OP_TRACE.with(|slot| slot.borrow_mut().take().unwrap_or_default());
    (value, samples)
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
        let (again, samples) = capture_ops(|| mul_quat(&a, &b));
        assert_eq!(again.map(f64::to_bits), contracted.map(f64::to_bits));
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].op, "mul_quat");
        assert_eq!(samples[0].outputs, contracted.map(f64::to_bits).to_vec());
        assert_ne!(samples[0].outputs, wheel.to_vec());
    }
}
