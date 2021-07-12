use crate::common::BlasInt;
use num_complex::{Complex32, Complex};
use super::common;

/// CSWAP interchanges two complex vectors.
///
/// # Description
/// CSWAP  swaps two complex vectors, it interchanges n values of vector x and
/// vector y.  incx and incy specify the increment between two consecutive
/// elements of respectively vector x and y.
///
/// This routine performs the following vector operation:
///
/// $$x \leftrightarrow y$$
///
/// where x and y are complex vectors.
///
/// # Arguments
/// * `n`(in) - Number of vector elements to be swapped. If n <= 0, this routine returns without computation.
///
/// * `x`(in, out) - Array of dimension $(n-1) * |inc_x| + 1$.
///
/// * `inc_x`(in) - Increment between elements of x. If $incx = 0$, the results will be unpredictable.
///
/// * `y`(in, out) - Array of dimension $(n-1) * |inc_y| + 1$.
///
/// * `inc_y`(in) - Increment between elements of y. If $incy = 0$, the results will be unpredictable.
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_cswap(n: BlasInt, x: *mut Complex32, inc_x: BlasInt, y: *mut Complex32, inc_y: BlasInt) {
    common::a_swap(n, x, inc_x, y, inc_y);
}


/// CSCAL scales a vector by a constant.
///
/// # Description
/// CSCAL scales a complex vector with a complex scalar.  CSCAL scales the vector
/// x of length n and increment inc_x by the constant $\alpha$.
///
/// Ths routine performs the following vector operation:
///
/// $$\vec{x} \gets \alpha \vec{x}$$
///
/// # Arguments
///
/// * `n`(in) - number of elements in input vector(s)
///
/// * `p_alpha`(in) - On entry, specifies pointer to the scalar alpha
///
/// * `x`(in, out) - array, dimension ( 1 + ( N - 1 )*abs( `inc_x` ) )
///
/// * `inc_x`(in) - Storage spacing between elements of `x`
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_cscal(n: BlasInt, p_alpha: *const Complex32, x: *mut Complex32, inc_x: BlasInt) {
    common::cz_scal(n, p_alpha, x, inc_x);
}
