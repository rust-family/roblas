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
//! - [ ] SSUM - sum of values(**not included in blas**), should not be implemented now
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
//! - [ ] DSUM - sum of values(**not included in blas**), should not be implemented now
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
//! - [ ] SCNRM2 - Euclidean norm
//!
//! - [ ] SCSUM - sum of values(**not included in blas**), should not be implemented now
//!
//! - [x] SCASUM - sum of absolute values
//!
//! - [x] ICAMAX - index of max abs value
//!
//! - [x] ICAMIN - index of min abs value
//!
//! This is a list of z-functions:
//! - [x] ZROTG - setup Givens rotation
//!
//! - [x] ZSROT - apply Givens rotation
//!
//! - [x] ZSWAP - swap x and y
//!
//! - [x] ZSCAL - x = a*x
//!
//! - [x] ZSSCAL - x = a*x
//!
//! - [x] ZCOPY - copy x into y
//!
//! - [x] ZAXPY - y = a*x + y
//!
//! - [x] ZDOTU - dot product
//!
//! - [x] ZDOTC - dot product, conjugating the first vector
//!
//! - [ ] DZNRM2 - Euclidean norm
//!
//! - [ ] DZSUM - sum of values(**not included in blas**), should not be implemented now
//!
//! - [x] DZASUM - sum of absolute values
//!
//! - [x] IZAMAX - index of max abs value
//!
//! - [x] IZAMIN - index of min abs value

mod naive;
pub use naive::*;
