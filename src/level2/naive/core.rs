use crate::common::BlasInt;
use crate::utils::{col_major_index, letter_same};
use num_traits::Float;
use std::cmp::max;
use std::ops::AddAssign;

#[inline(always)]
pub unsafe fn sd_gemv<T>(
    trans: char,
    m: BlasInt,
    n: BlasInt,
    alpha: T,
    a: *const T,
    lda: BlasInt,
    x: *const T,
    inc_x: BlasInt,
    beta: T,
    y: *mut T,
    inc_y: BlasInt,
) where
    T: Float + From<i8> + AddAssign,
{
    let zero: T = From::from(0);
    let one: T = From::from(1);
    // first, check `trans`
    let mut info = 0;
    if !letter_same(trans, 'N') && !letter_same(trans, 'T') && !letter_same(trans, 'C') {
        info = 1;
    } else if m < 0 {
        info = 2;
    } else if n < 0 {
        info = 3;
    } else if lda < max(1, m) {
        info = 6;
    } else if inc_x == 0 {
        info = 8;
    } else if inc_y == 0 {
        info = 11;
    }
    if info != 0 {
        xerbla!(false, info, "SGEMV");
    }

    // quick return if possible
    if m == 0 || n == 0 || (alpha == zero && beta == one) {
        return;
    }

    let len_x;
    let len_y;
    // set `len_x` and `len_y`, the lengths of the vectors x and y,
    // and set up the start points in X and Y.
    if letter_same(trans, 'N') {
        len_x = n as usize;
        len_y = m as usize;
    } else {
        len_x = m as usize;
        len_y = n as usize;
    }
    let kx;
    if inc_x > 0 {
        kx = 0_usize;
    } else {
        kx = (len_x - 1) * (-inc_x) as usize;
    }
    let ky;
    if inc_y > 0 {
        ky = 0_usize;
    } else {
        ky = (len_y - 1) * (-inc_y) as usize;
    }

    // Start the operations.
    // In this version the elements of A are accessed sequentially with one pass through A.
    //
    // First form y := beta * y
    if beta != one {
        if inc_y == 1 {
            if beta == zero {
                for i in 0..len_y {
                    *y.add(i) = zero;
                }
            } else {
                for i in 0..len_y {
                    *y.add(i) = beta * (*y.add(i));
                }
            }
        } else {
            if beta == zero {
                for i in (0..(len_y * inc_y as usize)).step_by(inc_y as usize) {
                    *y.add(i) = zero;
                }
            } else {
                for i in (0..(len_y * inc_y as usize)).step_by(inc_y as usize) {
                    *y.add(i) = beta * (*y.add(i));
                }
            }
        }
    }
    if alpha == zero {
        return;
    }
    if letter_same(trans, 'N') {
        // Form y := alpha * A * x + y
        let mut jx = kx;
        if inc_y == 1 {
            for j in 0..n as usize {
                let temp = alpha * *x.add(jx);
                for i in 0..m as usize {
                    *y.add(i) += temp * *a.add(col_major_index(i, j, lda));
                }
                jx += inc_x as usize;
            }
        } else {
            for j in 0..n as usize {
                let temp = alpha * *x.add(jx);
                let mut iy = ky;
                for i in 0..m as usize {
                    *y.add(iy) += temp * *a.add(col_major_index(i, j, lda));
                    iy += inc_y as usize;
                }
                jx += inc_x as usize;
            }
        }
    } else {
        // Form y := alpha * A^T * x + y
        let mut jy = ky;
        if inc_x == 1 {
            for j in 0..n as usize {
                let mut temp = zero;
                for i in 0..m as usize {
                    temp += *a.add(col_major_index(i, j, lda)) * *x.add(i);
                }
                *y.add(jy) += alpha * temp;
                jy += inc_y as usize;
            }
        } else {
            for j in 0..n as usize {
                let mut temp = zero;
                let mut ix = kx;
                for i in 0..m as usize {
                    temp += *a.add(col_major_index(i, j, lda)) * *x.add(ix);
                    ix += inc_x as usize;
                }
                *y.add(jy) += alpha * temp;
                jy += inc_y as usize;
            }
        }
    }
}
