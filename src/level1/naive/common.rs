use crate::common::{BlasInt, BlasIndex};
use std::ops::{Mul, AddAssign, Add, Div, MulAssign, DivAssign};
use num_traits::{Signed, Float, FromPrimitive};
use num_traits::real::Real;

#[inline(always)]
pub unsafe fn sd_rotg<T>(a: *mut T, b: *mut T, c: *mut T, s: *mut T)
    where T: Float + From<i8> + MulAssign
{
    let one = From::from(1);
    let zero = From::from(0);
    let old_a = *a;
    let old_b = *b;
    let mut roe = *b;
    if old_a.abs() > old_b.abs() {
        roe = old_a;
    }
    let scale = old_a.abs() + old_b.abs();
    if scale == zero {
        *c = one;
        *s = zero;
        let r = zero;
        let z = zero;
        *a = r;
        *b = z;
    } else {
        let mut r = scale * ((old_a / scale).powi(2) + (old_b / scale).powi(2)).sqrt();
        r *= roe.signum();
        *c = old_a / r;
        *s = old_b / r;
        let mut z = one;
        if old_a.abs() < old_b.abs() {
            z = *s;
        }
        if old_b.abs() >= old_a.abs() && *c != zero {
            z = one / *c;
        }
        *a = r;
        *b = z;
    }
}

#[inline(always)]
pub unsafe fn sd_rotmg<T>(d1: *mut T, d2: *mut T, b1: *mut T, b2: T, params: *mut T)
    where T: Float + FromPrimitive + From<u16> + DivAssign + MulAssign
{
    // constant
    let zero: T = From::from(0);
    let one: T = From::from(1);
    let two: T = From::from(2);
    let gam: T = From::from(4096);
    let gamsq: T = FromPrimitive::from_f64(1.67772e7_f64).unwrap();
    let rgamsq: T = FromPrimitive::from_f64(5.96046e-8_f64).unwrap();


    // process variable
    let mut flag = zero;
    let mut h11 = zero;
    let mut h12 = zero;
    let mut h21 = zero;
    let mut h22 = zero;

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

#[inline(always)]
pub unsafe fn sd_rot<T>(n: BlasInt, x: *mut T, inc_x: BlasInt, y: *mut T, inc_y: BlasInt, c: T, s: T)
    where T: Float
{
    let inc_x_us = inc_x as usize;
    let inc_y_us = inc_y as usize;
    if n <= 0 { return; }
    if inc_x == 1 && inc_y == 1 {
        for i in 0..n as usize {
            let stemp = c * *x.add(i) + s * *y.add(i);
            *y.add(i) = c * *y.add(i) - s * *x.add(i);
            *x.add(i) = stemp;
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
            let stemp = c * *x.add(ix) + s * *y.add(iy);
            *y.add(iy) = c * *y.add(iy) - s * *x.add(ix);
            *x.add(ix) = stemp;
            ix += inc_x_us;
            iy += inc_y_us;
        }
    }
}

#[inline(always)]
pub unsafe fn sd_rotm<T>(n: BlasInt, x: *mut T, inc_x: BlasInt, y: *mut T, inc_y: BlasInt, param: *const T)
    where T: Float + From<i8>
{
    // Const var
    let zero = From::from(0);
    let two = From::from(2);
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
                let w = *x.add(i);
                let z = *y.add(i);
                *x.add(i) = w * sh11 + z * sh12;
                *y.add(i) = w * sh21 + z * sh22;
            }
        }else if flag == zero {
            let sh12 = *param.add(3);
            let sh21 = *param.add(2);
            for i in (0..n_steps).step_by(inc_y as usize) {
                let w = *x.add(i);
                let z = *y.add(i);
                *x.add(i) = w + z * sh12;
                *y.add(i) = w * sh21 + z;
            }
        }else {
            let sh11 = *param.add(1);
            let sh22 = *param.add(4);
            for i in (0..n_steps).step_by(inc_x as usize) {
                let w = *x.add(i);
                let z = *y.add(i);
                *x.add(i) = w * sh11 + z;
                *y.add(i) = -w + sh22 * z;
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
                let w = *x.add(kx);
                let z = *y.add(ky);
                *x.add(kx) = w * sh11 + z * sh12;
                *y.add(ky) = w * sh21 + z * sh22;
                kx += inc_x as usize;
                ky += inc_y as usize;
            }
        }else if flag == zero {
            let sh12 = *param.add(3);
            let sh21 = *param.add(2);
            for _ in 0..n {
                let w = *x.add(kx);
                let z = *y.add(ky);
                *x.add(kx) = w + z * sh12;
                *y.add(ky) = w * sh21 + z;
                kx += inc_x as usize;
                ky += inc_y as usize;
            }
        }else {
            let sh11 = *param.add(1);
            let sh22 = *param.add(4);
            for _ in 0..n {
                let w = *x.add(kx);
                let z = *y.add(ky);
                *x.add(kx) = w * sh11 + z;
                *y.add(ky) = -w + sh22 * z;
                kx += inc_x as usize;
                ky += inc_y as usize;
            }
        }
    }
}

/// Apply to all type of swap function.
#[inline(always)]
pub unsafe fn a_swap<T>(n: BlasInt, x: *mut T, inc_x: BlasInt, y: *mut T, inc_y: BlasInt)
    where T: Copy
{
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

#[inline(always)]
pub unsafe fn sd_scal<T>(n: BlasInt, alpha: T, x: *mut T, inc_x: BlasInt)
    where T: Copy + Mul<Output = T>
{
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

#[inline(always)]
pub unsafe fn cz_scal<T>(n: BlasInt, p_alpha: *const T, x: *mut T, inc_x: BlasInt)
    where T: Copy + Mul<Output = T>
{
    if n < 0 || inc_x < 0 {
        return;
    }
    let alpha = *p_alpha;
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

#[inline(always)]
pub unsafe fn a_copy<T>(n: BlasInt, x: *const T, inc_x: BlasInt, y: *mut T, inc_y: BlasInt)
    where T: Copy
{
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

#[inline(always)]
pub unsafe fn sd_axpy<T>(n: BlasInt, a: T, x: *const T, inc_x: BlasInt, y: *mut T, inc_y: BlasInt)
    where T: Copy + From<i8> + PartialEq + Mul<Output = T> + AddAssign
{
    let zero = T::from(0);
    if n <= 0 { return; }
    if a == zero { return; }
    if inc_x == 1 && inc_y == 1 {
        let m = n % 4;
        if m != 0 {
            for i in 0..m as usize {
                *y.add(i) += a * *x.add(i);
            }
        }
        if n < 4 { return; }
        let mp1 = m as usize;
        for i in (mp1..(n as usize)).step_by(4) {
            *y.add(i) += a * *x.add(i);
            *y.add(i + 1) += a * *x.add(i + 1);
            *y.add(i + 2) += a * *x.add(i + 2);
            *y.add(i + 3) += a * *x.add(i + 3);
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
            *y.add(iy) += a * *x.add(ix);
            ix += inc_x as usize;
            iy += inc_y as usize;
        }
    }
}

#[inline(always)]
pub unsafe fn sd_sdot<T>(n: BlasInt, x: *const T, inc_x: BlasInt, y: *const T, inc_y: BlasInt) -> T
    where T: Copy + From<i8> + PartialEq + Mul<Output = T> + Add<Output = T> + AddAssign
{
    let zero = T::from(0);
    let mut stemp = T::from(0);
    if n <= 0 { return zero; }
    if inc_x == 1 && inc_y == 1 {
        let m = n % 5;
        for i in 0..m as usize {
            stemp += *x.add(i) * *y.add(i);
        }
        if n < 5 {
            return stemp;
        }
        let mp1 = m as usize;
        for i in (mp1..(n as usize)).step_by(5) {
            stemp += *x.add(i) * *y.add(i) + *x.add(i+1) * *y.add(i+1) + *x.add(i+2) * *y.add(i+2) + *x.add(i+3) * *y.add(i+3) + *x.add(i+4) * *y.add(i+4);
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
            stemp += *x.add(ix) * *y.add(iy);
            ix += inc_x as usize;
            iy += inc_y as usize;
        }
    }
    stemp
}

#[inline(always)]
pub unsafe fn sd_asum<T>(n: BlasInt, x: *const T, inc_x: BlasInt) -> T
    where T: Copy + From<i8> + PartialEq + Mul<Output = T> + Add<Output = T> + AddAssign + Signed
{
    let mut sasum = From::from(0);
    let mut tmp = From::from(0);
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

#[inline(always)]
pub unsafe fn sd_nrm2<T>(n: BlasInt, x: *const T, inc_x: BlasInt) -> T
    where T: From<i8> + PartialEq + Mul<Output = T> + Add<Output = T> + AddAssign + Float
{
    let one: T = From::from(1);
    let zero: T = From::from(0);
    if n < 1 || inc_x < 1 {
        zero
    } else if n == 1 {
        (*x).abs()
    } else {
        let mut scale = zero;
        let mut ssq = one;
        for ix in (0_usize..(n * inc_x) as usize).step_by(inc_x as usize) {
            if *x.add(ix) != zero {
                let abs_xi = (*x.add(ix)).abs();
                if scale < abs_xi {
                    ssq = one + ssq * (scale / abs_xi).powi(2);
                    scale = abs_xi;
                } else {
                    ssq += (abs_xi / scale).powi(2);
                }
            }
        }
        scale * ssq.sqrt()
    }
}

#[inline(always)]
pub unsafe fn sd_iamax<T>(n: BlasInt, x: *const T, inc_x: BlasInt) -> BlasIndex
    where T: Copy + PartialOrd + Signed
{
    if n < 1 || inc_x <= 0 {
        0
    } else if n == 1 {
        0
    } else if inc_x == 1 {
        // code for increment equal to 1
        let mut iamax = 0;
        let mut smax = (*x).abs();
        for i in 1_usize..n as usize {
            let tmp = (*x.add(i)).abs();
            if tmp > smax {
                iamax = i;
                smax = tmp;
            }
        }
        iamax
    } else {
        // code for increment not equal to 1
        let mut iamax = 0;
        let mut smax = (*x).abs();
        for i in ((inc_x as usize)..((n * inc_x) as usize)).step_by(inc_x as usize) {
            let tmp = (*x.add(i)).abs();
            if tmp > smax {
                iamax = i;
                smax = tmp;
            }
        }
        iamax
    }
}

#[inline(always)]
pub unsafe fn sd_iamin<T>(n: BlasInt, x: *const T, inc_x: BlasInt) -> BlasIndex
    where T: Copy + PartialOrd + Signed
{
    if n < 1 || inc_x <= 0 {
        0
    } else if n == 1 {
        0
    } else if inc_x == 1 {
        // code for increment equal to 1
        let mut iamin = 0;
        let mut smin = (*x).abs();
        for i in 1_usize..n as usize {
            let tmp = (*x.add(i)).abs();
            if tmp < smin {
                iamin = i;
                smin = tmp;
            }
        }
        iamin
    } else {
        // code for increment not equal to 1
        let mut iamin = 0;
        let mut smin = (*x).abs();
        for i in ((inc_x as usize)..((n * inc_x) as usize)).step_by(inc_x as usize) {
            let tmp = (*x.add(i)).abs();
            if tmp < smin {
                iamin = i;
                smin = tmp;
            }
        }
        iamin
    }
}