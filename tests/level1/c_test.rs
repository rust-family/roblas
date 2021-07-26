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
            Complex32::new(1_f32, 1_f32),
            Complex32::new(1_f32, -1_f32),
            Complex32::new(0_f32, 0_f32),
        ];
        let v2 = vec![
            Complex32::new(3_f32, -4_f32),
            Complex32::new(6_f32, -2_f32),
            Complex32::new(0_f32, 0_f32),
        ];
        let result1;
        unsafe {
            result1 = cblas_cdotu(3, v1.as_ptr(), 1, v2.as_ptr(), 1);
        }
        let expect1 = Complex32::new(11_f32, -9_f32);

        let result2;
        unsafe {
            result2 = cblas_cdotu(3, v1.as_ptr(), 2, v2.as_ptr(), 2);
        }
        let expect2 = Complex32::new(7_f32, -1_f32);

        let result3;
        unsafe {
            result3 = cblas_cdotu(3, v1.as_ptr(), -1, v2.as_ptr(), -1);
        }
        let expect3 = Complex32::new(11_f32, -9_f32);
        assert_eq!(result1, expect1);
        assert_eq!(result2, expect2);
        assert_eq!(result3, expect3);
    }

    #[test]
    fn cdotc() {
        let v1 = vec![Complex32::new(1_f32, 1_f32), Complex32::new(1_f32, -1_f32)];
        let v2 = vec![Complex32::new(3_f32, -4_f32), Complex32::new(6_f32, -2_f32)];
        let result1;
        unsafe {
            result1 = cblas_cdotc(2, v1.as_ptr(), 1, v2.as_ptr(), 1);
        }
        let expect1 = Complex32::new(7_f32, -3_f32);
        assert_eq!(result1, expect1);
    }
}
