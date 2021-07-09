//! This is a list of function.
//! s-function:
//! - [x] SROTG - setup Givens rotation
//!
//! - [x] SROTMG - setup modified Givens rotation
//!
//! - [ ] SROT - apply Givens rotation
//!
//! - [ ] SROTM - apply modified Givens rotation
//!
//! - [x] SSWAP - swap x and y
//!
//! - [x] SSCAL - x = a*x
//!
//! - [ ] SCOPY - copy x into y
//!
//! - [ ] SAXPY - y = a*x + y
//!
//! - [ ] SDOT - dot product
//!
//! - [ ] SDSDOT - dot product with extended precision accumulation
//!
//! - [ ] SNRM2 - Euclidean norm
//!
//! - [ ] SCNRM2 - Euclidean norm
//!
//! - [ ] SASUM - sum of absolute values
//!
//! - [ ] ISAMAX - index of max abs value
//!
//!

mod naive;
pub use naive::*;
