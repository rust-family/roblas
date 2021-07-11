//! The level 1 functions of blas.
//!
//! This is a list of functions:
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
//! - [x] SDSDOT - dot product with extended precision accumulation
//!
//! - [x] SNRM2 - Euclidean norm
//!
//! - [x] SASUM - sum of absolute values
//!
//! - [x] ISAMAX - index of max abs value
//!
//! - [x] ISAMIN - index of min abs value
//!
//! This is a list of d-functions:
//! - [ ] DROTG - setup Givens rotation
//!
//! - [ ] DROTMG - setup modified Givens rotation
//!
//! - [ ] DROT - apply Givens rotation
//!
//! - [ ] DROTM - apply modified Givens rotation
//!
//! - [ ] DSWAP - swap x and y
//!
//! - [ ] DSCAL - x = a*x
//!
//! - [ ] DCOPY - copy x into y
//!
//! - [ ] DAXPY - y = a*x + y
//!
//! - [ ] DDOT - dot product
//!
//! - [ ] DSDOT - dot product with extended precision accumulation
//!
//! - [ ] DNRM2 - Euclidean norm
//!
//! - [ ] DZNRM2 - Euclidean norm
//!
//! - [ ] DASUM - sum of absolute values
//!
//! - [ ] IDAMAX - index of max abs value
//!
//! - [ ] IDAMIN - index of min abs value

mod naive;
pub use naive::*;
