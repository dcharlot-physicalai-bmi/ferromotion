//! ferromotion-models — **robot arms built from the tables their makers and their textbooks publish.**
//!
//! Every constructor here is a [`Robot`] assembled by [`Robot::from_dh`] from a Denavit–Hartenberg table,
//! and every table states where it came from: the manufacturer document, the paper, or the textbook
//! edition, with the convention that source uses ([`DhConvention::Standard`] or
//! [`DhConvention::Modified`]) and any unit conversion applied. DH parameters are public engineering
//! data. Files — URDFs, meshes — carry licenses, and none are vendored here.
//!
//! Each model is verified rather than transcribed: forward kinematics at a documented pose or a
//! hand-computed one, the same central-difference Jacobian and Hessian check every `Robot` in the
//! workspace is held to, and a stated mutation showing the convention argument is load-bearing — a table
//! read under the wrong convention builds a plausible-looking wrong arm, and that is the failure mode
//! these tests exist to catch.
//!
//! Where a source gives no joint limit, effort or velocity, the model says so by leaving it `None`
//! rather than inventing one. Pure `nalgebra` → WASM-clean.

pub use ferromotion_core::{DhConvention, DhRow, Robot};
