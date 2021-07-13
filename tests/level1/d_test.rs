#[cfg(test)]
mod d_test {
    use roblas::level1::*;
    use roblas::common::BlasInt;

    #[test]
    fn drotg1() {
        let mut a = 0_f64;
        let mut b = 1_f64;
        let mut c = 0_f64;
        let mut s = 0_f64;
        unsafe {
            cblas_drotg(&mut a, &mut b, &mut c, &mut s);
        }
        assert_eq!((a, b, c, s), (1_f64, 1_f64, 0_f64, 1_f64));
    }

    #[test]
    fn dswap1() {
        let mut x = vec![1_f64, 2_f64];
        let mut y = vec![3_f64, 4_f64];
        unsafe {
            cblas_dswap(2, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1);
        }
        assert_eq!(x, vec![3_f64, 4_f64]);
        assert_eq!(y, vec![1_f64, 2_f64]);
    }

    #[test]
    fn dswap2() {
        let mut x = vec![1_f64, 2_f64, 3_f64];
        let mut y = vec![4_f64, 5_f64, 6_f64];
        unsafe {
            cblas_dswap(2, x.as_mut_ptr(), -2, y.as_mut_ptr(), -2);
        }
        assert_eq!(x, vec![4_f64, 2_f64, 6_f64]);
        assert_eq!(y, vec![1_f64, 5_f64, 3_f64]);
    }

    #[test]
    fn dscal1() {
        let mut v = vec![1_f64, 2_f64, 3_f64];
        unsafe {
            cblas_dscal(v.len() as BlasInt, 2f64, v.as_mut_ptr(), 1);
        }
        assert_eq!(v, vec![2f64, 4f64, 6f64]);
    }

    #[test]
    fn dcopy1() {
        let v1 = vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64, 7_f64];
        let mut v2 = vec![0_f64; 7];
        unsafe {
            cblas_dcopy(7, v1.as_ptr(), 1, v2.as_mut_ptr(), 1);
        }
        assert_eq!(v1, v2);
    }

    #[test]
    fn dnrm1() {
        let v1 = vec![-3_f64, 4_f64];
        let result;
        unsafe {
            result = cblas_dnrm2(2, v1.as_ptr(), 1);
        }
        assert_eq!(result, 5_f64);
    }

    #[test]
    fn dnrm2() {
        let v1 = vec![-5_f64, 5_f64, 12_f64];
        let result;
        unsafe {
            result = cblas_dnrm2(2, v1.as_ptr(), 2);
        }
        assert_eq!(result, 13_f64);
    }

    #[test]
    fn dasum1() {
        let mut v = vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64];
        let result;
        unsafe {
            result = cblas_dasum(6, v.as_mut_ptr(), 1);
        }
        assert_eq!(result, 21_f64);
    }

    #[test]
    fn daxpy() {
        let v1 = vec![1_f64, 2_f64, 3_f64];
        let mut v2 = vec![2_f64, 3_f64, 4_f64];
        unsafe {
            cblas_daxpy(3, 0.5_f64, v1.as_ptr(), 1, v2.as_mut_ptr(), 1);
        }
        assert_eq!(v2, vec![2.5_f64, 4_f64, 5.5_f64]);
        let v3 = vec![1_f64, 2_f64, 3_f64, 1_f64, 2_f64, 3_f64, 1_f64, 2_f64, 3_f64];
        let mut v4 = vec![2_f64, 3_f64, 4_f64, 2_f64, 3_f64, 4_f64, 2_f64, 3_f64, 4_f64];
        unsafe {
            cblas_daxpy(9, 0.5_f64, v3.as_ptr(), 1, v4.as_mut_ptr(), 1);
        }
        assert_eq!(v4, vec![2.5_f64, 4_f64, 5.5_f64, 2.5_f64, 4_f64, 5.5_f64, 2.5_f64, 4_f64, 5.5_f64]);
    }

    #[test]
    fn ddot1() {
        let v1 = vec![1_f64, 2_f64, 3_f64, 4_f64];
        let mut result;
        unsafe {
            result = cblas_ddot(4, v1.as_ptr(), 1, v1.as_ptr(), 1);
        }
        assert_eq!(result, 30_f64);
        // Implementation to this function unroll the for-loop with step of 5, so add this testcase here
        let v2 = vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64,
                                1_f64, 2_f64, 3_f64, 4_f64, 5_f64];
        unsafe {
            result = cblas_ddot(10, v2.as_ptr(), 1, v2.as_ptr(), 1);
        }
        assert_eq!(result, 110_f64);
        let v3 = vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64,
                                1_f64, 2_f64, 3_f64, 4_f64, 5_f64,
                                1_f64, 2_f64, 3_f64,];
        unsafe {
            result = cblas_ddot(13, v3.as_ptr(), 1, v3.as_ptr(), 1);
        }
        assert_eq!(result, 124_f64);
    }

    #[test]
    fn dsdot1() {
        let v1 = vec![1_f32, 2_f32, 3_f32, 4_f32];
        let result;
        unsafe {
            result = cblas_dsdot(4, v1.as_ptr(), 1, v1.as_ptr(), 1);
        }
        assert_eq!(result, 30_f64);
    }

    #[test]
    fn isamax1() {
        let v1 = vec![2_f64, 1_f64, 5_f64, 5_f64];
        let result;
        unsafe {
            result = cblas_idamax(4, v1.as_ptr(), 1);
        }
        assert_eq!(result, 2);
    }

    #[test]
    fn isamin1() {
        let v1 = vec![2_f64, 1_f64, 5_f64, 5_f64];
        let result;
        unsafe {
            result = cblas_idamin(4, v1.as_ptr(), 1);
        }
        assert_eq!(result, 1);
    }
}
