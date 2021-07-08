#[cfg(test)]
mod s_test {
    use roblas::level1::*;
    use roblas::common::BlasInt;

    #[test]
    fn srotg1() {
        let mut a = 0_f32;
        let mut b = 1_f32;
        let mut c = 0_f32;
        let mut s = 0_f32;
        unsafe {
            cblas_srotg(&mut a, &mut b, &mut c, &mut s);
        }
        assert_eq!((a, b, c, s), (1_f32, 1_f32, 0_f32, 1_f32));
    }

    #[test]
    fn sswap1() {
        let mut x = vec![1_f32, 2_f32];
        let mut y = vec![3_f32, 4_f32];
        unsafe {
            cblas_sswap(2, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1);
        }
        assert_eq!(x, vec![3_f32, 4_f32]);
        assert_eq!(y, vec![1_f32, 2_f32]);
    }

    #[test]
    fn sswap2() {
        let mut x = vec![1_f32, 2_f32, 3_f32];
        let mut y = vec![4_f32, 5_f32, 6_f32];
        unsafe {
            cblas_sswap(2, x.as_mut_ptr(), -2, y.as_mut_ptr(), -2);
        }
        assert_eq!(x, vec![4_f32, 2_f32, 6_f32]);
        assert_eq!(y, vec![1_f32, 5_f32, 3_f32]);
    }

    #[test]
    fn sscal1() {
        let mut v = vec![1f32, 2f32, 3f32];
        unsafe {
            cblas_sscal(v.len() as BlasInt, 2f32, v.as_mut_ptr(), 1);
        }
        assert_eq!(v, vec![2f32, 4f32, 6f32]);
    }
}
