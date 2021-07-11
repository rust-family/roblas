use crate::common::BlasInt;

/// SROTG construct givens plane rotation.
///
/// # Description
///
/// SROTG computes the elements of a rotation matrix such that:
///
/// $$
///     \left [  \begin{matrix}
///         c & s \\\\
///         -s & c
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
/// where $r=\pm \sqrt{a^2 + b^2}$ and $c^2 + s^2 = 1$
///
/// The Givens plane rotation can be used to introduce zero elements into
/// a matrix selectively.
///
/// # Arguments
///
/// `a`(in, out) - First  vector  component.  On input, the first component of the vector to be rotated.
/// On output, a is overwritten by r, the first  component of the vector in the rotated coordinate system where:
/// $$r = sgn(a) * \sqrt{a^2 + b^2},~\text{if |a| > |b|}$$
/// $$r = sgn(b) * \sqrt{a^2 + b^2},~\text{if |a| <= |b|}$$
///
/// `b`(in, out) - On input, the second component of the vector to be rotated.  On output, b contains z, where:
/// $$z=s,~\text{if |a| > |b|}$$
/// $$z=\frac{1}{c},~\text{if |a| <= |b| and c != 0 and r != 0}$$
/// $$z=1,~\text{if |a| <= |b| and c = 0 and r != 0}$$
/// $$z=0,~\text{if r = 0}$$
///
/// `c`(out) - Cosine of the angle of rotation:
/// $$c=\frac{a}{r},~\text{if r != 0}$$
/// $$c=1,~\text{if r = 0}$$
///
/// `s`(out) - Sine of the angle of rotation:
/// $$s=\frac{b}{r},~\text{if r != 0}$$
/// $$s=0,~\text{if r = 0}$$
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_srotg(a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32) {
    let old_a = *a;
    let old_b = *b;
    let mut roe = *b;
    if old_a.abs() > old_b.abs() {
        roe = old_a;
    }
    let scale = old_a.abs() + old_b.abs();
    if scale == 0.0 {
        *c = 1.0;
        *s = 0.0;
        let r = 0.0;
        let z = 0.0;
        *a = r;
        *b = z;
    } else {
        let mut r = scale * ((old_a / scale).powi(2) + (old_b / scale).powi(2)).sqrt();
        r *= roe.signum();
        *c = old_a / r;
        *s = old_b / r;
        let mut z = 1.0;
        if old_a.abs() < old_b.abs() {
            z = *s;
        }
        if old_b.abs() >= old_a.abs() && *c != 0.0 {
            z = 1.0 / *c;
        }
        *a = r;
        *b = z;
    }
}

/// SROTMG computes the elements of a modified Givens plane rotation matrix.
///
/// # Description
/// Given Cartesian coordinates ($b_1$, $b_2$) of an input vector, the rotmg routines compute
/// the components of a modified Givens transformation matrix $\boldsymbol{H}$ that zeros the y-component of the resulting vector:
///
/// $$
///     \left [ \begin{matrix}
///         b_1 \\\\
///         0
///     \end{matrix} \right ] \gets
///     \boldsymbol{H}
///     \left [ \begin{matrix}
///         b_1 \sqrt{d_1} \\\\
///         b_2 \sqrt{d_2}
///     \end{matrix} \right ]
/// $$
///
/// # Arguments
/// * `d1`(in, out) - On input, the scaling factor for the x-coordinate of the input vector.
/// On output, the first diagonal element of the updated matrix.
///
/// * `d2`(in, out) - On input, The scaling factor for the y-coordinate of the input vector.
/// On output, the second diagonal element of the updated matrix.
///
/// * `b1`(in, out) - On input, the x-coordinate of the input vector.
/// On output, the x-coordinate of the rotated vector before scaling by the updated matrix.
///
/// * `b2`(in) - The y-coordinate of the input vector.
///
/// * `params`(out) - Array of size 5. `*params.add(0)` contains a switch, `flag`. The other array elements
/// `params[1-4]` contain the components of the array `H`: $h_{11},h_{21},h_{12},h_{22}$, respectively.
/// Depending on the values of `flag`, the components of `H` are set as follows:
///     * `flag = -1.0`:
///     $$\boldsymbol{H}=\left [ \begin{matrix} h_{11} & h_{12} \\\\ h_{21} & h_{22} \end{matrix} \right ]$$
///     * `flag = 0.0`:
///     $$\boldsymbol{H}=\left [ \begin{matrix} 1.0 & h_{12} \\\\ h_{21} & 1.0 \end{matrix} \right ]$$
///     * `flag = 1.0`:
///     $$\boldsymbol{H}=\left [ \begin{matrix} h_{11} & 1.0 \\\\ -1.0 & h_{22} \end{matrix} \right ]$$
///     * `flag = -2.0`:
///     $$\boldsymbol{H}=\left [ \begin{matrix} 1.0 & 0.0 \\\\ 0.0 & 1.0 \end{matrix} \right ]$$
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_srotmg(d1: *mut f32, d2: *mut f32, b1: *mut f32, b2: f32, params: *mut f32) {
    // constant
    let zero = 0_f32;
    let one = 1_f32;
    let two = 2_f32;
    let gam = 4096_f32;
    let gamsq = 1.67772e7_f32;
    let rgamsq = 5.96046e-8_f32;

    // process variable
    let mut flag = 0_f32;
    let mut h11 = 0_f32;
    let mut h12 = 0_f32;
    let mut h21 = 0_f32;
    let mut h22 = 0_f32;

    if *d1 < zero {
        // zero H, D and b1
        flag = -one;
        h11 = zero;
        h12 = zero;
        h21 = zero;
        h22 = zero;

        *d1 = zero;
        *d2 = zero;
        *b1 = zero;
    } else {
        // d1 non-negative
        let p2 = *d2 * b2;
        if p2 == zero {
            flag = -two;
            *params = flag;
            return;
        }
        // regular case
        let p1 = *d1 * *b1;
        let q2 = p2 * b2;
        let q1 = p1 * *b1;

        if q1.abs() > q2.abs() {
            h21 = -b2 / *b1;
            h12 = p2 / p1;

            let mut u = one - h12 * h21;

            if u > zero {
                flag = zero;
                *d1 /= u;
                *d2 /= u;
                *b1 *= u;
            } else {
                if q2 < zero {
                    // zero H, D and b1
                    flag = -one;
                    h11 = zero;
                    h12 = zero;
                    h21 = zero;
                    h22 = zero;

                    *d1 = zero;
                    *d2 = zero;
                    *b1 = zero;
                } else {
                    flag = one;
                    h11 = p1 / p2;
                    h22 = *b1 / b2;
                    u = one + h11 * h22;
                    let tmp = *d2 / u;
                    *d2 = *d1 / u;
                    *d1 = tmp;
                    *b1 = b2 * u;
                }
            }
        }

        // scale check
        if *d1 != zero {
            while *d1 <= rgamsq || *d1 >= gamsq {
                if flag == zero {
                    h11 = one;
                    h22 = one;
                    flag = -one;
                } else {
                    h21 = -one;
                    h12 = one;
                    flag = -one;
                }
                if *d1 <= rgamsq {
                    *d1 *= gam.powi(2);
                    *b1 /= gam;
                    h11 /= gam;
                    h12 /= gam;
                } else {
                    *d1 /= gam.powi(2);
                    *b1 *= gam;
                    h11 *= gam;
                    h12 *= gam;
                }
            }
        }

        if *d2 != zero {
            while (*d2).abs() <= rgamsq || (*d2).abs() >= gamsq {
                if flag == zero {
                    h11 = one;
                    h22 = one;
                    flag = -one;
                } else {
                    h21 = -one;
                    h12 = one;
                    flag = -one;
                }
                if (*d2).abs() <= rgamsq {
                    *d2 *= gam.powi(2);
                    h21 /= gam;
                    h22 /= gam;
                } else {
                    *d2 /= gam.powi(2);
                    h21 *= gam;
                    h22 *= gam;
                }
            }
        }
    }

    if flag < zero {
        *params.add(1) = h11;
        *params.add(2) = h21;
        *params.add(3) = h12;
        *params.add(4) = h22;
    } else if flag == zero {
        *params.add(2) = h21;
        *params.add(3) = h12;
    } else {
        *params.add(1) = h11;
        *params.add(4) = h22;
    }

    *params = flag;
}

/// SROT performs rotation of points in the plane.
///
/// # Description
/// SROT   applies  a  plane  rotation matrix to a real sequence of ordered pairs:
/// $$ (x_i, y_i),i=1,\cdots,n $$
///
/// # Arguments
/// * `n`(in) - Number of ordered pairs (planar points in SROT) to be  rotated.
/// If n <= 0, this routine returns without computation.
///
/// * `sx`(in, out) - Array  of dimension (n-1) * |inc_x| + 1.
/// On input, array x contains the x-coordinate of each planar point to be rotated.
/// On output, array x contains the x-coordinate of each rotated planar point.
///
/// * `sy`(in, out) - Array of dimension (n-1) * |inc_y| + 1.
/// On input, array y contains the y-coordinate of each planar point to be rotated.
/// On output, array y contains the y-coordinate of each rotated planar point.
///
/// * `inc_x`(in) - Increment between elements of x. If inc_x = 0, the results will be unpredictable.
///
/// * `inc_y`(in) - Increment between elements of y. If inc_y = 0, the results will be unpredictable.
///
/// * `c`(in) - Cosine of the angle of rotation.
///
/// * `s`(in) - Sine of the angle of rotation.
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_srot(n: BlasInt, sx: *mut f32, inc_x: BlasInt, sy: *mut f32, inc_y: BlasInt, c: f32, s: f32) {
    let inc_x_us = inc_x as usize;
    let inc_y_us = inc_y as usize;
    if n <= 0 { return; }
    if inc_x == 1 && inc_y == 1 {
        for i in 0..n as usize {
            let stemp = c * *sx.add(i) + s * *sy.add(i);
            *sy.add(i) = c * *sy.add(i) - s * *sx.add(i);
            *sx.add(i) = stemp;
        }
    } else {
        let mut ix = 1_usize;
        let mut iy = 1_usize;
        if inc_x < 0 {
            ix = ((1 - n) * inc_x + 1) as usize;
        }
        if inc_y < 0 {
            iy = ((1 - n) * inc_y + 1) as usize;
        }
        for _ in 1..n {
            let stemp = c * *sx.add(ix) + s * *sy.add(iy);
            *sy.add(iy) = c * *sy.add(iy) - s * *sx.add(ix);
            *sx.add(ix) = stemp;
            ix += inc_x_us;
            iy += inc_y_us;
        }
    }
}


/// SROTM - BLAS level one. Applies a modified Givens rotation.
///
/// # Description
///  SROTM applies the modified Givens plane rotation constructed by SROTMG.
///
/// # Arguments
/// * `n`(in) - Number of ordered pairs (planar points in SROT) to be  rotated.
/// If n <= 0, this routine returns without computation.
///
/// * `sx`(in, out) - Array  of dimension (n-1) * |inc_x| + 1.
/// On input, array x contains the x-coordinate of each planar point to be rotated.
/// On output, array x contains the x-coordinate of each rotated planar point.
///
/// * `sy`(in, out) - Array of dimension (n-1) * |inc_y| + 1.
/// On input, array y contains the y-coordinate of each planar point to be rotated.
/// On output, array y contains the y-coordinate of each rotated planar point.
///
/// * `inc_x`(in) - Increment between elements of x. If inc_x = 0, the results will be unpredictable.
///
/// * `inc_y`(in) - Increment between elements of y. If inc_y = 0, the results will be unpredictable.
///
/// * `param`(in) - REAL array of dimension 5. Contains rotation matrix information.
/// The key parameter, param(0), may have one of four values: 1.0,  0.0,  -1.0,  or -2.0
/// Depending on the values of `flag`, the components of `H` are set as follows:
///     * `flag = -1.0`:
///     $$\boldsymbol{H}=\left [ \begin{matrix} h_{11} & h_{12} \\\\ h_{21} & h_{22} \end{matrix} \right ]$$
///     * `flag = 0.0`:
///     $$\boldsymbol{H}=\left [ \begin{matrix} 1.0 & h_{12} \\\\ h_{21} & 1.0 \end{matrix} \right ]$$
///     * `flag = 1.0`:
///     $$\boldsymbol{H}=\left [ \begin{matrix} h_{11} & 1.0 \\\\ -1.0 & h_{22} \end{matrix} \right ]$$
///     * `flag = -2.0`:
///     $$\boldsymbol{H}=\left [ \begin{matrix} 1.0 & 0.0 \\\\ 0.0 & 1.0 \end{matrix} \right ]$$
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_srotm(n: BlasInt, sx: *mut f32, inc_x: BlasInt, sy: *mut f32, inc_y: BlasInt, param: *const f32) {
    // Const var
    let zero = 0_f32;
    let two = 2_f32;
    // Subroutine
    let flag = *param.add(0);
    if n <= 0 || flag + two == zero { return; }
    if inc_x == inc_y && inc_x > 0 {
        let n_steps = (n * inc_x) as usize;
        if flag < zero {
            let sh11 = *param.add(1);
            let sh12 = *param.add(3);
            let sh21 = *param.add(2);
            let sh22 = *param.add(4);
            for i in (0..n_steps).step_by(inc_x as usize) {
                let w = *sx.add(i);
                let z = *sy.add(i);
                *sx.add(i) = w * sh11 + z * sh12;
                *sy.add(i) = w * sh21 + z * sh22;
            }
        }else if flag == zero {
            let sh12 = *param.add(3);
            let sh21 = *param.add(2);
            for i in (0..n_steps).step_by(inc_y as usize) {
                let w = *sx.add(i);
                let z = *sy.add(i);
                *sx.add(i) = w + z * sh12;
                *sy.add(i) = w * sh21 + z;
            }
        }else {
            let sh11 = *param.add(1);
            let sh22 = *param.add(4);
            for i in (0..n_steps).step_by(inc_x as usize) {
                let w = *sx.add(i);
                let z = *sy.add(i);
                *sx.add(i) = w * sh11 + z;
                *sy.add(i) = -w + sh22 * z;
            }
        }
    } else {
        let mut kx = 1_usize;
        let mut ky = 1_usize;
        if inc_x < 0 {
            kx = (1 + (1 - n) * inc_x) as usize;
        }
        if inc_y < 0 {
            ky = (1 + (1 - n) * inc_y) as usize;
        }
        if flag < zero {
            let sh11 = *param.add(1);
            let sh12 = *param.add(3);
            let sh21 = *param.add(2);
            let sh22 = *param.add(4);
            for _ in 0..n {
                let w = *sx.add(kx);
                let z = *sy.add(ky);
                *sx.add(kx) = w * sh11 + z * sh12;
                *sy.add(ky) = w * sh21 + z * sh22;
                kx += inc_x as usize;
                ky += inc_y as usize;
            }
        }else if flag == zero {
            let sh12 = *param.add(3);
            let sh21 = *param.add(2);
            for _ in 0..n {
                let w = *sx.add(kx);
                let z = *sy.add(ky);
                *sx.add(kx) = w + z * sh12;
                *sy.add(ky) = w * sh21 + z;
                kx += inc_x as usize;
                ky += inc_y as usize;
            }
        }else {
            let sh11 = *param.add(1);
            let sh22 = *param.add(4);
            for _ in 0..n {
                let w = *sx.add(kx);
                let z = *sy.add(ky);
                *sx.add(kx) = w * sh11 + z;
                *sy.add(ky) = -w + sh22 * z;
                kx += inc_x as usize;
                ky += inc_y as usize;
            }
        }
    }

}

/// SSWAP interchanges two vectors.
///
/// # Description
/// SSWAP  swaps two real vectors, it interchanges n values of vector x and
/// vector y.  incx and incy specify the increment between two consecutive
/// elements of respectively vector x and y.
///
/// This routine performs the following vector operation:
///
/// $$x \leftrightarrow y$$
///
/// where x and y are real vectors.
///
/// # Arguments
/// * `n`(in) - Number of vector elements to be swapped. If n <= 0, this routine returns without computation.
///
/// * `x`(in, out) - Array of dimension $(n-1) * |incx| + 1$.
///
/// * `inc_x`(in) - Increment between elements of x. If $incx = 0$, the results will be unpredictable.
///
/// * `y`(in, out) - Array of dimension $(n-1) * |incy| + 1$.
///
/// * `inc_y`(in) - Increment between elements of y. If $incy = 0$, the results will be unpredictable.
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_sswap(n: BlasInt, x: *mut f32, inc_x: BlasInt, y: *mut f32, inc_y: BlasInt) {
    if n < 0 {
        return;
    }
    if inc_x == 1 && inc_y == 1 {
        // code for both increments equal to 1
        let m = (n % 3) as usize;
        if m != 0 {
            for i in 0_usize..m {
                let tmp = *x.add(i);
                *x.add(i) = *y.add(i);
                *y.add(i) = tmp;
            }
        }
        for i in (m..(n as usize)).step_by(3) {
            let tmp = *x.add(i);
            *x.add(i) = *y.add(i);
            *y.add(i) = tmp;
            let tmp = *x.add(i + 1);
            *x.add(i + 1) = *y.add(i + 1);
            *y.add(i + 1) = tmp;
            let tmp = *x.add(i + 2);
            *x.add(i + 2) = *y.add(i + 2);
            *y.add(i + 2) = tmp;
        }
    } else {
        // code for unequal increments or equal increments not equal to 1
        let mut ix = 0_usize;
        let mut iy = 0_usize;
        if inc_x < 0 {
            ix = (-inc_x * (n - 1)) as usize;
        }
        if inc_y < 0 {
            iy = (-inc_y * (n - 1)) as usize;
        }
        for _ in 0..n {
            let tmp = *x.add(ix);
            *x.add(ix) = *y.add(iy);
            *y.add(iy) = tmp;
            ix = ((ix as BlasInt) + inc_x) as usize;
            iy = ((iy as BlasInt) + inc_y) as usize;
        }
    }
}

/// SSCAL scales a vector by a constant.
///
/// # Description
/// SSCAL scales a real vector with a real scalar.  SSCAL scales the vector
/// x of length n and increment incx by the constant $\alpha$.
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
pub unsafe extern fn cblas_sscal(n: BlasInt, alpha: f32, x: *mut f32, inc_x: BlasInt) {
    if n < 0 || inc_x < 0 {
        return;
    }
    if inc_x == 1 {
        // optimized code for increment equal to 1
        let m = (n % 5) as usize;
        if m != 0 {
            for i in 0_usize..m {
                let pos = x.add(i);
                *pos = alpha * (*pos);
            }
            if n < 5 {
                return;
            }
        }
        for i in (m..(n as usize)).step_by(5) {
            let pos = x.add(i);
            *pos = alpha * (*pos);
            *pos.add(1) = alpha * (*pos.add(1));
            *pos.add(2) = alpha * (*pos.add(2));
            *pos.add(3) = alpha * (*pos.add(3));
            *pos.add(4) = alpha * (*pos.add(4));
        }
    } else {
        // normal code for increment not equal to 1
        let n_inc_x = (n * inc_x) as usize;
        for i in (0..n_inc_x).step_by(inc_x as usize) {
            let pos = x.add(i);
            *pos = alpha * (*pos);
        }
    }
}

/// SCOPY copies a vector, x, to a vector, y.
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
/// * `inc_x`(in) - Increment between elements of x. If incx = 0, the results will be unpredictable.
///
/// * `y`(out) - array of dimension (n-1) * |incy| + 1, result vector.
///
/// * `inc_y`(in) - Increment between elements of y.  If incy = 0, the results will be unpredictable.
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_scopy(n: BlasInt, x: *const f32, inc_x: BlasInt, y: *mut f32, inc_y: BlasInt) {
    if n <= 0 {
        return;
    }
    if inc_x == 1 && inc_y == 1 {
        // code for increment equal to 1
        let m = (n % 7) as usize;
        if m != 0 {
            for i in 0_usize..m {
                *y.add(i) = *x.add(i);
            }
            if n < 7 {
                return;
            }
        }
        for i in (m..(n as usize)).step_by(7) {
            *y.add(i) = *x.add(i);
            *y.add(i + 1) = *x.add(i + 1);
            *y.add(i + 2) = *x.add(i + 2);
            *y.add(i + 3) = *x.add(i + 3);
            *y.add(i + 4) = *x.add(i + 4);
            *y.add(i + 5) = *x.add(i + 5);
            *y.add(i + 6) = *x.add(i + 6);
        }
    } else {
        // code for increment not equal to 1
        let mut ix = 0_usize;
        let mut iy = 0_usize;
        if inc_x < 0 {
            ix = (-inc_x * (n - 1)) as usize;
        }
        if inc_y < 0 {
            iy = (-inc_y * (n - 1)) as usize;
        }
        for _ in 0_usize..(n as usize) {
            *y.add(iy) = *x.add(ix);
            ix += inc_x as usize;
            iy += inc_y as usize;
        }
    }
}


/// SAXPY adds a scalar multiple of a real vector to another real vector.
///
/// # Description
/// SAXPY computes a constant alpha times a vector x plus a vector y.  The
/// result overwrites the initial values of vector y.
///
///        This routine performs the following vector operation:
///
///            $$y\leftarrow\alpha x + y$$
///
///        inc_x and inc_y specify the increment between two consecutive
///        elements of respectively vector x and y.
///
/// # Arguments
/// * `n`(in) - Number of elements in each vector.
///
/// * `sa`(in) - On entry, SA specifies the scalar alpha.
/// If alpha = 0 this routine returns without any computation.
///
/// * `sx`(in) - Array  of dimension $(n-1) * |inc_x| + 1$.  Contains the vector to be scaled before summation.
///
/// * `inc_x`(in) - Increment between elements of x. If inc_x = 0, the results will be unpredictable.
///
/// * `sy`(in, out) - array of dimension $(n-1) * |inc_y| + 1$, result vector.
/// Before calling the routine, y contains the vector to be summed.
/// After the routine ends, y contains the result of the summation.
///
/// * `inc_y`(in) - Increment between elements of y.  If inc_y = 0, the results will be unpredictable.
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_saxpy(n: BlasInt, sa: f32, sx: *const f32, inc_x: BlasInt, sy: *mut f32, inc_y: BlasInt) {
    let zero = 0_f32;
    if n <= 0 { return; }
    if sa == zero { return; }
    if inc_x == 1 && inc_y == 1 {
        let m = n % 4;
        if m != 0 {
            for i in 0..m as usize {
                *sy.add(i) += sa * *sx.add(i);
            }
        }
        if n < 4 { return; }
        let mp1 = m as usize;
        for i in (mp1..(n as usize)).step_by(4) {
            *sy.add(i) += sa * *sx.add(i);
            *sy.add(i + 1) += sa * *sx.add(i + 1);
            *sy.add(i + 2) += sa * *sx.add(i + 2);
            *sy.add(i + 3) += sa * *sx.add(i + 3);
        }
    } else {
        let mut ix = 0_usize;
        let mut iy = 0_usize;
        if inc_x < 0 {
            ix = (-inc_x * (n - 1)) as usize;
        }
        if inc_y < 0 {
            iy = (-inc_y * (n - 1)) as usize;
        }
        for _ in 0..n {
            *sy.add(iy) += sa * *sx.add(ix);
            ix += inc_x as usize;
            iy += inc_y as usize;
        }
    }
}


/// SDOT computes a dot product of two real vectors (l real inner product).
///
/// # Description
/// This routine performs the following vector operation:
///
///     $$\mathrm{Result}=x^{\mathrm{T}}y=\sum_{i=0}^{n-1}x(i)\cdot y(i)$$
///
/// # Arguments
/// * `n`(in) - Number of elements in each vector.
///
/// * `sx`(in) - Array  of dimension $(n-1) * |inc_x| + 1$.  Array sx contains the first vector operand.
///
/// * `inc_x`(in) - Increment between elements of x. If inc_x = 0, the results will be unpredictable.
///
/// * `sy`(in) - array of dimension $(n-1) * |inc_y| + 1$.  Array sy contains the second vector operand.
///
/// * `inc_y`(in) - Increment between elements of y.  If inc_y = 0, the results will be unpredictable.
///
/// # Return values
///
/// The result of dot product operation
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_sdot(n: BlasInt, sx: *const f32, inc_x: BlasInt, sy: *const f32, inc_y: BlasInt) -> f32 {
    let zero = 0_f32;
    let mut stemp = 0_f32;
    if n <= 0 { return zero; }
    if inc_x == 1 && inc_y == 1 {
        let m = n % 5;
        for i in 0..m as usize {
            stemp += *sx.add(i) * *sy.add(i);
        }
        if n < 5 {
            return stemp;
        }
        let mp1 = m as usize;
        for i in (mp1..(n as usize)).step_by(5) {
            stemp += *sx.add(i) * *sy.add(i) + *sx.add(i+1) * *sy.add(i+1) + *sx.add(i+2) * *sy.add(i+2) + *sx.add(i+3) * *sy.add(i+3) + *sx.add(i+4) * *sy.add(i+4);
        }
    } else {
        let mut ix = 0_usize;
        let mut iy = 0_usize;
        if inc_x < 0 {
            ix = (-inc_x * (n - 1)) as usize;
        }
        if inc_y < 0 {
            iy = (-inc_y * (n - 1)) as usize;
        }
        for _ in 0..n {
            stemp += *sx.add(ix) * *sy.add(iy);
            ix += inc_x as usize;
            iy += inc_y as usize;
        }
    }
    stemp
}

/// SASUM sums the absolute values of the elements of a real vector.
///
/// # Description
/// This routine performs the following vector operation:
/// $$ result \gets \sum_{i=1}^{n} |x_i| $$
///
/// # Arguments
/// * `n`(in) - Number of vector elements to be summed.
///
/// * `x`(in) - Array of dimension (n-1) * abs(incx) + 1. Vector that contains elements to be summed.
///
/// * `inc_x`(in) - Increment between elements of x. If incx = 0, the results will be unpredictable.
///
/// # Return values
/// Sum of the absolute values of the elements of the vector x. If $n <= 0$, SASUM is set to 0.
///
#[no_mangle]
#[inline(always)]
pub unsafe extern fn cblas_sasum(n: BlasInt, x: *mut f32, inc_x: BlasInt) -> f32 {
    let mut sasum = 0_f32;
    let mut tmp = 0_f32;
    if n <= 0 || inc_x <= 0 {
        return sasum;
    }
    if inc_x == 1 {
        // code for increment equal to 1
        let m = (n % 6) as usize;
        if m != 0 {
            for i in 0_usize..m {
                tmp += (*x.add(i)).abs();
            }
            if n < 6 {
                sasum = tmp;
                return sasum;
            }
        }
        for i in (m..(n as usize)).step_by(6) {
            tmp += (*x.add(i)).abs() + (*x.add(i + 1)).abs() +
                (*x.add(i + 2)).abs() + (*x.add(i + 3)).abs() +
                (*x.add(i + 4)).abs() + (*x.add(i + 5)).abs()
        }
    } else {
        // code for increment not equal to 1
        let n_inc_x = (n * inc_x) as usize;
        for i in (0_usize..n_inc_x).step_by(inc_x as usize) {
            tmp += (*x.add(i)).abs()
        }
    }
    sasum = tmp;
    sasum
}


