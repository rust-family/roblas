#[cfg(test)]
mod c_test {
    use roblas::common::{BlasInt, Complex32};
    use roblas::level1::*;

    #[test]
    fn cswap1() {
        let mut v1 = vec![Complex32::from(1_f32), Complex32::from(2_f32)];
        let mut v2 = vec![Complex32::from(3_f32), Complex32::from(4_f32)];
        unsafe {
            cblas_cswap(2, v1.as_mut_ptr(), 1, v2.as_mut_ptr(), 1);
        }
        assert_eq!(v1, vec![Complex32::from(3_f32), Complex32::from(4_f32)]);
        assert_eq!(v2, vec![Complex32::from(1_f32), Complex32::from(2_f32)]);
    }

    #[test]
    fn cscal1() {
        let mut v1 = vec![
            Complex32::from(1_f32),
            Complex32::from(2_f32),
            Complex32::from(3_f32),
        ];
        let a = Complex32::from(2_f32);
        unsafe {
            cblas_cscal(v1.len() as BlasInt, &a, v1.as_mut_ptr(), 1);
        }
        assert_eq!(
            v1,
            vec![
                Complex32::from(2_f32),
                Complex32::from(4_f32),
                Complex32::from(6_f32)
            ]
        );
    }

    #[test]
    fn csscal1() {
        let mut v1 = vec![
            Complex32::from(1_f32),
            Complex32::from(2_f32),
            Complex32::from(3_f32),
        ];
        unsafe {
            cblas_csscal(v1.len() as BlasInt, 2_f32, v1.as_mut_ptr(), 1);
        }
        assert_eq!(
            v1,
            vec![
                Complex32::from(2_f32),
                Complex32::from(4_f32),
                Complex32::from(6_f32)
            ]
        );
    }

    #[test]
    fn ccopy1() {
        let mut v1 = Vec::with_capacity(7);
        for i in 1..=7 {
            v1.push(Complex32::new(i as f32, (i + 1) as f32));
        }
        let mut v2 = vec![Complex32::default(); 7];
        unsafe {
            cblas_ccopy(7, v1.as_ptr(), 1, v2.as_mut_ptr(), 1);
        }
        assert_eq!(v1, v2);
    }

    #[test]
    fn cdotu() {
        let v1 = vec![
            Complex32::new(1_f32,1_f32),
            Complex32::new(1_f32,-1_f32),
            Complex32::new(-1_f32,1_f32),
            Complex32::new(-1_f32,-1_f32),
        ];
        let v2 = vec![
            Complex32::new(3_f32,-4_f32),
            Complex32::new(6_f32,-2_f32),
            Complex32::new(1_f32,2_f32),
            Complex32::new(4_f32,3_f32),
        ];
        let  result;
        unsafe {
            result = cblas_cdotu(4, v1.as_ptr(), 1, v2.as_ptr(), 1);
        }
        let  expect = Complex32::new(0_f32,-16_f32);
        assert_eq!(result, expect);
        // Implementation to this function unroll the for-loop with step of 5, so add this testcase here
    }

    #[test]
    fn cdotc() {
        let v1 = vec![
            Complex32::new(1_f32,1_f32),
            Complex32::new(1_f32,-1_f32),
        ];
        let v2 = vec![
            Complex32::new(3_f32,-4_f32),
            Complex32::new(6_f32,-2_f32),
        ];
        let  result;
        unsafe {
            result = cblas_cdotc(2, v1.as_ptr(), 1, v2.as_ptr(), 1);
        }
        let  expect = Complex32::new(7_f32,-3_f32);
        assert_eq!(result, expect);
        // Implementation to this function unroll the for-loop with step of 5, so add this testcase here
    }
}
