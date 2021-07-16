use crate::common::{BlasInt, Complex32};
use super::common;

/// CROTG construct givens plane rotation.
///
/// # Description
///
/// CROTG computes the elements of a rotation matrix such that:
///
/// $$
///     \left [  \begin{matrix}
///         c & s \\\\
///         -\bar{s} & c
///     \end{matrix} \right ] *
///     \left [ \begin{matrix}
///         a \\\\
///         b
///     \end{matrix} \right ] =
///     \left [ \begin{matrix}
///         r \\\\
///         0
///     \end{matrix} \right ]
/// $$
///
/// where $r=\frac{a}{\sqrt{\bar{a} * a}} * \sqrt{\bar{a} * a + \bar{b} * b}$ and the notation $\bar{z}$
/// represents the complex conjugate of z.
///
/// The Givens plane rotation can be used to introduce zero elements into
/// a matrix selectively.
///
/// # Arguments
///
/// `a`(in, out) - First  vector  component.  On input, the first component of the vector to be rotated.
/// On output, a is overwritten by the unique complex number r, whose size in the complex plane is the
/// Euclidean norm of the complex vector (a, b), and whose direction in the complex plane is the same
/// as that of the original complex element a.
///
/// if $|a| \ne 0$, then $r= \frac{a}{|a|} * \sqrt{\bar{a} * a + \bar{b} * b}$.
///
/// if $|a| = 0$, then $r = b$.
///
/// `b`(in) - Second vector component. The second component of the vector to be rotated.
///
/// `c`(out) - Cosine of the angle of rotation:
///
/// if $|a| \ne 0$, then $c= \frac{|a|}{\sqrt{\bar{a} * a + \bar{b} * b}}$.
///
/// if $|a| = 0$, then $c = 0$.
///
/// `s`(out) - Sine of the angle of rotation:
///
/// if $|a| \ne 0$, then $c= \frac{a}{|a|} * \frac{\bar{b}}{\sqrt{\bar{a} * a + \bar{b} * b}}$.
///
/// if $|a| = 0$, then $s=(1.0,0.0)$.
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_crotg(a: *mut Complex32, b: *mut Complex32, c: *mut f32, s: *mut Complex32) {
    common::cz_rotg(a, b, c, s);
}

/// CSROT performs rotation of points in the plane.
///
/// # Description
/// CSROT applies a plane rotation, where the cos and sin (C and S) is real and the vectors X and Y are complex.
///
/// # Arguments
/// * `n`(in) - The number of elements in the vectors X and Y.
///
/// * `x`(in, out) - Array  of dimension (n-1) * |inc_x| + 1.
/// On input, the vector X.
/// On output, X is overwritten with C\*X + S\*Y
///
/// * `inc_x`(in) - Increment between elements of x. If inc_x = 0, the results will be unpredictable.
///
/// * `y`(in, out) - Array of dimension (n-1) * |inc_y| + 1.
/// On input, the vector Y.
/// On output, Y is overwritten with -S\*X + C\*Y
///
/// * `inc_y`(in) - Increment between elements of y. If inc_y = 0, the results will be unpredictable.
///
/// * `c`(in) - Cosine of the angle of rotation.
///
/// * `s`(in) - Sine of the angle of rotation.
/// C and S define a rotation:
/// $$
///     \left [ \begin{matrix}
///         c & s \\\\
///         -s & c
///     \end{matrix} \right ]
/// $$
/// where C\*C + S\*S = 1.0.
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_csrot(n: BlasInt, x: *mut Complex32, inc_x: BlasInt, y: *mut Complex32, inc_y: BlasInt, c: f32, s: f32) {
    common::cz_srot(n, x, inc_x, y, inc_y, c, s);
}

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

///CAXPY constant times a vector plus a vector.
/// 
/// # Description
/// 
//CAXPY  adds  a  scalar  multiple of a complex vector to another complex
///vector.
///
///CAXPY computes a constant alpha times a vector x plus a vector y.   The
///result overwrites the initial values of vector y.
/// 
///This routine performs the following vector operation:
/// 
/// $$ y \gets alpha*x + y $$
/// 
/// incx and incy specify the increment between two consecutive
///elements of respectively vector x and y.

#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_caxpy(n: BlasInt, ca: *mut Complex32, cx: *mut Complex32, inc_x: BlasInt, cy: *mut Complex32, inc_y: BlasInt){
    common::cz_axpy(n, ca, cx, inc_x, cy, inc_y);
}
