use crate::common::{BlasInt, Complex32};
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


/// CSCAL scales a complex vector by a complex constant.
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

/// CSSCAL scales a complex vector by a real constant.
///
/// # Description
/// CSSCAL scales a complex vector with a real scalar. CSSCAL scales the vector
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
/// * `alpha`(in) - On entry, SA specifies the scalar alpha
///
/// * `x`(in, out) - array, dimension ( 1 + ( N - 1 )*abs( `inc_x` ) )
///
/// * `inc_x`(in) - Storage spacing between elements of `x`
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_csscal(n: BlasInt, alpha: f32, x: *mut Complex32, inc_x: BlasInt) {
    common::cz_sscal(n, alpha, x, inc_x);
}

/// CCOPY copies a vector, x, to a vector, y.
///
/// # Description
/// The copy routines copy one vector to another:
/// $$ \vec{y} \gets \vec{x} $$
///
/// where $\vec{x}$ and $\vec{y}$ are vectors of n elements.
///
/// # Arguments
/// * `n`(in) - Number of vector elements to be copied.
///
/// * `x`(in) - Vector from which to copy.
///
/// * `inc_x`(in) - Increment between elements of x. If inc_x = 0, the results will be unpredictable.
///
/// * `y`(out) - array of dimension (n-1) * |inc_y| + 1, result vector.
///
/// * `inc_y`(in) - Increment between elements of y.  If inc_y = 0, the results will be unpredictable.
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_ccopy(n: BlasInt, x: *const Complex32, inc_x: BlasInt, y: *mut Complex32, inc_y: BlasInt) {
    common::a_copy(n, x, inc_x, y, inc_y);
}
