use super::core;
use crate::common::{BlasInt, CBlasOrder, CBlasTranspose};

/// SGEMV perform one of the matrix-vector operations $y=\alpha * op(\boldsymbol{A}) * \vec{x} + \beta * \vec{y}$.
///
/// # Description
/// SGEMV performs one of the matrix-vector operations
/// $$\vec{y}=\alpha * \boldsymbol{A} * \vec{x} + \beta * \vec{y}$$
/// or
/// $$\vec{y}=\alpha * \boldsymbol{A}^T * \vec{x} + \beta * \vec{y}$$
///
/// # Arguments
/// `order`(in) - indicate if matrix `a` is row-major or col-major.
///
/// `trans_a`(in) - whether matrix `a` would be transposed, that is, the $op(\boldsymbol{A})$
///
/// `m`(in) - the number of rows.
///
/// `n`(in) - the number of cols.
///
/// `alpha`(in) - constant scaling factor $\alpha$.
///
/// `lda`(in) - if `a` is col-major, `lda` should be the number of rows of the outer matrix;
/// else, `lda` should be the number of cols of the outer matrix.
///
/// `x`(in) - vector x.
///
/// `inc_x`(in) - increment of vector x.
///
/// `beta`(in) - constant scaling factor $\beta$.
///
/// `y`(in, out) - vector y.
///
/// `inc_y`(in) - increment of vector y.
///
#[no_mangle]
#[inline(always)]
pub unsafe extern "C" fn cblas_sgemv(
    order: CBlasOrder,
    trans_a: CBlasTranspose,
    m: BlasInt,
    n: BlasInt,
    alpha: f32,
    a: *const f32,
    lda: BlasInt,
    x: *const f32,
    inc_x: BlasInt,
    beta: f32,
    y: *mut f32,
    inc_y: BlasInt,
) {
    let ta: char;
    if order == CBlasOrder::ColMajor {
        match trans_a {
            CBlasTranspose::NoTrans => {
                ta = 'N';
            }
            CBlasTranspose::Trans => {
                ta = 'T';
            }
            CBlasTranspose::ConjTrans => {
                ta = 'N';
            }
            _ => {
                xerbla!(
                    false,
                    2,
                    "cblas_sgemv",
                    "Illegal TransA setting, {:?}\n",
                    trans_a
                );
            }
        }
        core::sd_gemv(ta, m, n, alpha, a, lda, x, inc_x, beta, y, inc_y);
    } else if order == CBlasOrder::RowMajor {
        match trans_a {
            CBlasTranspose::NoTrans => {
                ta = 'T';
            }
            CBlasTranspose::Trans => {
                ta = 'N';
            }
            CBlasTranspose::ConjTrans => {
                ta = 'N';
            }
            _ => {
                xerbla!(
                    false,
                    2,
                    "cblas_sgemv",
                    "Illegal TransA setting, {:?}\n",
                    trans_a
                );
            }
        }
        core::sd_gemv(ta, n, m, alpha, a, lda, x, inc_x, beta, y, inc_y);
    } else {
        xerbla!(
            false,
            2,
            "cblas_sgemv",
            "Illegal layout setting, {:?}\n",
            order
        );
    }
}
