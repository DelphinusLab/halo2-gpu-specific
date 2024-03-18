//! # halo2_proofs
#![feature(local_key_cell_methods)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Build without warnings on stable 1.51 and later.
#![allow(unknown_lints)]
// Disable old lint warnings until our MSRV is at least 1.51.
#![allow(renamed_and_removed_lints)]
// Use the old lint name to build without warnings until our MSRV is at least 1.51.
#![allow(clippy::unknown_clippy_lints)]
// The actual lints we want to disable.
#![allow(
    clippy::op_ref,
    clippy::assign_op_pattern,
    clippy::too_many_arguments,
    clippy::suspicious_arithmetic_impl,
    clippy::many_single_char_names,
    clippy::same_item_push,
    clippy::upper_case_acronyms
)]
#![deny(broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
// Remove this once we update pasta_curves
#![allow(unused_imports)]

pub(crate) mod parallel;

pub mod arithmetic;
pub mod circuit;
pub use pairing;
mod multicore;
pub mod plonk;
pub mod poly;
pub mod transcript;

pub mod dev;
pub mod helpers;

#[macro_use]
extern crate lazy_static;
