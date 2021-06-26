use crate::common::BlasInt;

/// SSCAL scales a vector by a constant. Uses unrolled loops for increment equal to 1.
///
/// # Arguments
///
/// * `n` - number of elements in input vector(s)
/// * `alpha` - On entry, SA specifies the scalar alpha
/// * `x` - array, dimension ( 1 + ( N - 1 )*abs( `inc_x` ) )
/// * `inc_x` - storage spacing between elements of `x`
///
/// # Formula
///
/// $\vec{x} \to \alpha \vec{x}$
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_sscal(n: BlasInt, alpha: f32, mut x: *mut f32, inc_x: BlasInt) {
    // TODO: this function IS NOT well-implemented, there is still some room to improve
    for _ in 0..n {
        *x = alpha * (*x);
        x = x.add(inc_x as usize);
    }
}