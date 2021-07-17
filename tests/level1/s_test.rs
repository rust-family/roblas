#[cfg(test)]
mod s_test {
    use roblas::common::BlasInt;
    use roblas::level1::*;

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
        let mut v = vec![1_f32, 2_f32, 3_f32];
        unsafe {
            cblas_sscal(v.len() as BlasInt, 2f32, v.as_mut_ptr(), 1);
        }
        assert_eq!(v, vec![2f32, 4f32, 6f32]);
    }

    #[test]
    fn scopy1() {
        let v1 = vec![1_f32, 2_f32, 3_f32, 4_f32, 5_f32, 6_f32, 7_f32];
        let mut v2 = vec![0_f32; 7];
        unsafe {
            cblas_scopy(7, v1.as_ptr(), 1, v2.as_mut_ptr(), 1);
        }
        assert_eq!(v1, v2);
    }

    #[test]
    fn snrm1() {
        let v1 = vec![-3_f32, 4_f32];
        let result;
        unsafe {
            result = cblas_snrm2(2, v1.as_ptr(), 1);
        }
        assert_eq!(result, 5_f32);
    }

    #[test]
    fn snrm2() {
        let v1 = vec![-5_f32, 5_f32, 12_f32];
        let result;
        unsafe {
            result = cblas_snrm2(2, v1.as_ptr(), 2);
        }
        assert_eq!(result, 13_f32);
    }

    #[test]
    fn sasum1() {
        let mut v = vec![1_f32, 2_f32, 3_f32, 4_f32, 5_f32, 6_f32];
        let result;
        unsafe {
            result = cblas_sasum(6, v.as_mut_ptr(), 1);
        }
        assert_eq!(result, 21_f32);
    }

    #[test]
    fn saxpy() {
        let v1 = vec![1_f32, 2_f32, 3_f32];
        let mut v2 = vec![2_f32, 3_f32, 4_f32];
        unsafe {
            cblas_saxpy(3, 0.5_f32, v1.as_ptr(), 1, v2.as_mut_ptr(), 1);
        }
        assert_eq!(v2, vec![2.5_f32, 4_f32, 5.5_f32]);
        let v3 = vec![
            1_f32, 2_f32, 3_f32, 1_f32, 2_f32, 3_f32, 1_f32, 2_f32, 3_f32,
        ];
        let mut v4 = vec![
            2_f32, 3_f32, 4_f32, 2_f32, 3_f32, 4_f32, 2_f32, 3_f32, 4_f32,
        ];
        unsafe {
            cblas_saxpy(9, 0.5_f32, v3.as_ptr(), 1, v4.as_mut_ptr(), 1);
        }
        assert_eq!(
            v4,
            vec![2.5_f32, 4_f32, 5.5_f32, 2.5_f32, 4_f32, 5.5_f32, 2.5_f32, 4_f32, 5.5_f32]
        );
    }

    #[test]
    fn sdot1() {
        let v1 = vec![1_f32, 2_f32, 3_f32, 4_f32];
        let mut result;
        unsafe {
            result = cblas_sdot(4, v1.as_ptr(), 1, v1.as_ptr(), 1);
        }
        assert_eq!(result, 30_f32);
        // Implementation to this function unroll the for-loop with step of 5, so add this testcase here
        let v2 = vec![
            1_f32, 2_f32, 3_f32, 4_f32, 5_f32, 1_f32, 2_f32, 3_f32, 4_f32, 5_f32,
        ];
        unsafe {
            result = cblas_sdot(10, v2.as_ptr(), 1, v2.as_ptr(), 1);
        }
        assert_eq!(result, 110_f32);
        let v3 = vec![
            1_f32, 2_f32, 3_f32, 4_f32, 5_f32, 1_f32, 2_f32, 3_f32, 4_f32, 5_f32, 1_f32, 2_f32,
            3_f32,
        ];
        unsafe {
            result = cblas_sdot(13, v3.as_ptr(), 1, v3.as_ptr(), 1);
        }
        assert_eq!(result, 124_f32);
    }

    #[test]
    fn sdsdot1() {
        let v1 = vec![1_f32, 2_f32, 3_f32, 4_f32];
        let result;
        unsafe {
            result = cblas_sdsdot(4, -100_f32, v1.as_ptr(), 1, v1.as_ptr(), 1);
        }
        assert_eq!(result, -70_f32);
    }

    #[test]
    fn isamax1() {
        let v1 = vec![2_f32, 1_f32, 5_f32, 5_f32];
        let result;
        unsafe {
            result = cblas_isamax(4, v1.as_ptr(), 1);
        }
        assert_eq!(result, 2);
    }

    #[test]
    fn isamin1() {
        let v1 = vec![2_f32, 1_f32, 5_f32, 5_f32];
        let result;
        unsafe {
            result = cblas_isamin(4, v1.as_ptr(), 1);
        }
        assert_eq!(result, 1);
    }
}
