//! # ferromotion-learn — differentiable, physics-informed learning for physical AI
//!
//! Where [`ferromotion_core`] gives you the *physics* — rigid-body dynamics, variational integrators,
//! contact, estimation — this crate gives you the *learning*: the differentiable machinery for models that
//! blend physics priors with data. It is the on-device, pure-Rust, WASM-clean counterpart to the
//! PyTorch-based physics-informed-ML stacks: no BLAS, no Python, no GPU required.
//!
//! The organizing idea (from the physics-informed ML literature) is that a physics prior can enter a learned
//! model at one of three places:
//! - **guided** — in the *data / features* (engineered inputs, geometric projections);
//! - **informed** — in the *loss* (penalize violation of a differential equation — a PINN);
//! - **encoded** — in the *architecture* (Lagrangian/Hamiltonian nets, Neural ODEs, structure-preserving
//!   integrators).
//!
//! Everything rests on one keystone: **automatic differentiation**. [`autodiff`] is reverse-mode (for
//! gradients of a scalar loss w.r.t. many parameters — backpropagation); [`dual`] is forward-mode (for exact
//! higher-order derivatives w.r.t. a model's *inputs*, which PINNs and Lagrangian nets need). Both are
//! verified against finite differences.

mod autodiff;
mod capstone;
mod delan;
mod diff_control;
mod dual;
pub mod calib;
pub mod nls;
mod hnn;
mod msnn;
mod neural_ode;
mod nn;
mod pinn;
mod sindy;

pub use autodiff::{Grad, Tape, Var};
pub use capstone::ModelBasedControl;
pub use delan::Delan;
pub use diff_control::PidController;
pub use dual::{Dual, HyperDual};
pub use hnn::Hnn;
pub use msnn::Msnn;
pub use neural_ode::NeuralOde;
pub use nn::Mlp;
pub use pinn::Pinn;
pub use sindy::{monomial_exponents, Sindy};
