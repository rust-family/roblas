use crate::common::BlasInt;
use crate::utils::letter_same;
use num_traits::Float;
use std::cmp::max;

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
    T: Float + From<i8>,
{
    let zero = From::from(0);
    let one = From::from(1);
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
    if beta != one {}
    todo!()
}
