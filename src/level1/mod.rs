//! This is a list of function.
//! s-function:
//! - [x] SROTG - setup Givens rotation
//!
//! - [x] SROTMG - setup modified Givens rotation
//!
//! - [x] SROT - apply Givens rotation
//!
//! - [x] SROTM - apply modified Givens rotation
//!
//! - [x] SSWAP - swap x and y
//!
//! - [x] SSCAL - x = a*x
//!
//! - [x] SCOPY - copy x into y
//!
//! - [x] SAXPY - y = a*x + y
//!
//! - [x] SDOT - dot product
//!
//! - [ ] SDSDOT - dot product with extended precision accumulation
//!
//! - [ ] SNRM2 - Euclidean norm
//!
//! - [x] SASUM - sum of absolute values
//!
//! - [ ] ISAMAX - index of max abs value
//!
//!

mod naive;
pub use naive::*;
