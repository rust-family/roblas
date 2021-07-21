//! The level 1 functions of cblas.
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
//! - [x] DROTG - setup Givens rotation
//!
//! - [x] DROTMG - setup modified Givens rotation
//!
//! - [x] DROT - apply Givens rotation
//!
//! - [x] DROTM - apply modified Givens rotation
//!
//! - [x] DSWAP - swap x and y
//!
//! - [x] DSCAL - x = a*x
//!
//! - [x] DCOPY - copy x into y
//!
//! - [x] DAXPY - y = a*x + y
//!
//! - [x] DDOT - dot product
//!
//! - [x] DSDOT - dot product with extended precision accumulation
//!
//! - [x] DNRM2 - Euclidean norm
//!
//! - [x] DZNRM2 - Euclidean norm
//!
//! - [x] DASUM - sum of absolute values
//!
//! - [x] IDAMAX - index of max abs value
//!
//! - [x] IDAMIN - index of min abs value
//!
//! This is a list of c-functions:
//! - [x] CROTG - setup Givens rotation
//!
//! - [x] CSROT - apply Givens rotation
//!
//! - [x] CSWAP - swap x and y
//!
//! - [x] CSCAL - x = a*x
//!
//! - [x] CSSCAL - x = a*x
//!
//! - [x] CCOPY - copy x into y
//!
//! - [x] CAXPY - y = a*x + y
//!
//! - [x] CDOTU - dot product
//!
//! - [x] CDOTC - dot product, conjugating the first vector
//!
//! - [x] SCASUM - sum of absolute values
//!
//! - [ ] ICAMAX - index of max abs value
//!
//! - [ ] ICAMIN - index of min abs value

mod naive;
pub use naive::*;
