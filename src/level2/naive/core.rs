use num_traits::Float;
use crate::common::BlasInt;

#[allow(unused_variables)]
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
    inc_y: BlasInt
)
    where T: Float
{
    todo!()
}
