#[cfg(test)]
mod c_test {
    use roblas::level1::*;
    use roblas::common::{BlasInt, Complex32};

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
        let mut v1 = vec![Complex32::from(1_f32), Complex32::from(2_f32), Complex32::from(3_f32)];
        let a = Complex32::from(2_f32);
        unsafe {
            cblas_cscal(v1.len() as BlasInt, &a, v1.as_mut_ptr(), 1);
        }
        assert_eq!(v1, vec![Complex32::from(2_f32), Complex32::from(4_f32), Complex32::from(6_f32)]);
    }

    #[test]
    fn csscal1() {
        let mut v1 = vec![Complex32::from(1_f32), Complex32::from(2_f32), Complex32::from(3_f32)];
        unsafe {
            cblas_csscal(v1.len() as BlasInt, 2_f32, v1.as_mut_ptr(), 1);
        }
        assert_eq!(v1, vec![Complex32::from(2_f32), Complex32::from(4_f32), Complex32::from(6_f32)]);
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
}
