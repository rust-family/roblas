#[cfg(test)]
mod s_test {
    use roblas::level1::cblas_sscal;
    use roblas::common::BlasInt;

    #[test]
    fn sscal1() {
        let mut v = vec![1f32, 2f32, 3f32];
        unsafe {
            cblas_sscal(v.len() as BlasInt, 2f32, v.as_mut_ptr(), 1);
        }
        assert_eq!(v, vec![2f32, 4f32, 6f32]);
    }
}
