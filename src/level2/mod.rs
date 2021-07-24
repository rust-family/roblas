//! The level 2 functions of cblas
//!
//! This is a list of functions:
//! s-functions:
//! - [ ] SGEMV - matrix vector multiply
//!
//! - [ ] SGBMV - banded matrix vector multiply
//!
//! - [ ] SSYMV - symmetric matrix vector multiply
//!
//! - [ ] SSBMV - symmetric banded matrix vector multiply
//!
//! - [ ] SSPMV - symmetric packed matrix vector multiply
//!
//! - [ ] STRMV - triangular matrix vector multiply
//!
//! - [ ] STBMV - triangular banded matrix vector multiply
//!
//! - [ ] STPMV - triangular packed matrix vector multiply
//!
//! - [ ] STRSV - solving triangular matrix problems
//!
//! - [ ] STBSV - solving triangular banded matrix problems
//!
//! - [ ] STPSV - solving triangular packed matrix problems
//!
//! - [ ] SGER - performs the rank 1 operation A := alpha*x*y' + A
//!
//! - [ ] SSYR - performs the symmetric rank 1 operation A := alpha*x*x' + A
//!
//! - [ ] SSPR - symmetric packed rank 1 operation A := alpha*x*x' + A
//!
//! - [ ] SSYR2 - performs the symmetric rank 2 operation, A := alpha*x*y' + alpha*y*x' + A
//!
//! - [ ] SSPR2 - performs the symmetric packed rank 2 operation, A := alpha*x*y' + alpha*y*x' + A
//!
mod naive;
pub use naive::*;
