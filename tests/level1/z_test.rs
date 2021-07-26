#[cfg(test)]
mod z_test {
    use roblas::common::{BlasInt, Complex64};
    use roblas::level1::*;

    #[test]
    fn zswap1() {
        let mut v1 = vec![Complex64::from(1_f64), Complex64::from(2_f64)];
        let mut v2 = vec![Complex64::from(3_f64), Complex64::from(4_f64)];
        unsafe {
            cblas_zswap(2, v1.as_mut_ptr(), 1, v2.as_mut_ptr(), 1);
        }
        assert_eq!(v1, vec![Complex64::from(3_f64), Complex64::from(4_f64)]);
        assert_eq!(v2, vec![Complex64::from(1_f64), Complex64::from(2_f64)]);
    }

    #[test]
    fn zscal1() {
        let mut v1 = vec![
            Complex64::from(1_f64),
            Complex64::from(2_f64),
            Complex64::from(3_f64),
        ];
        let a = Complex64::from(2_f64);
        unsafe {
            cblas_zscal(v1.len() as BlasInt, &a, v1.as_mut_ptr(), 1);
        }
        assert_eq!(
            v1,
            vec![
                Complex64::from(2_f64),
                Complex64::from(4_f64),
                Complex64::from(6_f64)
            ]
        );
    }

    #[test]
    fn zsscal1() {
        let mut v1 = vec![
            Complex64::from(1_f64),
            Complex64::from(2_f64),
            Complex64::from(3_f64),
        ];
        unsafe {
            cblas_zsscal(v1.len() as BlasInt, 2_f64, v1.as_mut_ptr(), 1);
        }
        assert_eq!(
            v1,
            vec![
                Complex64::from(2_f64),
                Complex64::from(4_f64),
                Complex64::from(6_f64)
            ]
        );
    }

    #[test]
    fn zcopy1() {
        let mut v1 = Vec::with_capacity(7);
        for i in 1..=7 {
            v1.push(Complex64::new(i as f64, (i + 1) as f64));
        }
        let mut v2 = vec![Complex64::default(); 7];
        unsafe {
            cblas_zcopy(7, v1.as_ptr(), 1, v2.as_mut_ptr(), 1);
        }
        assert_eq!(v1, v2);
    }

    #[test]
    fn zdotu() {
        let v1 = vec![Complex64::new(1_f64, 1_f64), Complex64::new(1_f64, -1_f64)];
        let v2 = vec![Complex64::new(3_f64, -4_f64), Complex64::new(6_f64, -2_f64)];
        let result;
        unsafe {
            result = cblas_zdotu(2, v1.as_ptr(), 1, v2.as_ptr(), 1);
        }
        let expect = Complex64::new(11_f64, -9_f64);
        assert_eq!(result, expect);
    }

    #[test]
    fn zdotc() {
        let v1 = vec![Complex64::new(1_f64, 1_f64), Complex64::new(1_f64, -1_f64)];
        let v2 = vec![Complex64::new(3_f64, -4_f64), Complex64::new(6_f64, -2_f64)];
        let result;
        unsafe {
            result = cblas_zdotc(2, v1.as_ptr(), 1, v2.as_ptr(), 1);
        }
        let expect = Complex64::new(7_f64, -3_f64);
        assert_eq!(result, expect);
    }

    #[test]
    fn izamax() {
        let v1 = vec![
            Complex64::new(1_f64, 1_f64),
            Complex64::new(1_f64, -2_f64),
            Complex64::new(1_f64, 10_f64),
            Complex64::new(1_f64, 15_f64),
            Complex64::new(1_f64, 11_f64),
        ];
        let result1;
        unsafe {
            result1 = cblas_izamax(5, v1.as_ptr(), 2);
        }
        let expect1 = 4 as usize;
        let result2;
        unsafe {
            result2 = cblas_izamax(5, v1.as_ptr(), 1);
        }
        let expect2 = 3 as usize;

        let result3;
        unsafe {
            result3 = cblas_izamax(5, v1.as_ptr(), -1);
        }
        let expect3 = 0 as usize;
        assert_eq!(result1, expect1);
        assert_eq!(result2, expect2);
        assert_eq!(result3, expect3);
    }
}
