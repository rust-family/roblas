#[cfg(test)]
mod s_test {
    use roblas::common::{CBlasOrder, CBlasTranspose};
    use roblas::level2::*;

    #[test]
    fn sgemv1() {
        // col major test
        //
        // 3 * [1 2 3] * [1] + 2 * [-3]
        //     [3 4 5]   [2]       [-2]
        //               [3]
        let a = vec![1_f32, 3_f32, 2_f32, 4_f32, 3_f32, 5_f32];
        let x = vec![1_f32, 2_f32, 3_f32];
        let mut y = vec![-3_f32, -2_f32];
        unsafe {
            cblas_sgemv(
                CBlasOrder::ColMajor,
                CBlasTranspose::NoTrans,
                2,3,
                3_f32,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                2_f32,
                y.as_mut_ptr(),
                1
            )
        }
        assert_eq!(y, vec![36_f32, 74_f32])
    }

    #[test]
    fn sgemv2() {
        // row major test
        //
        // 3 * [1 2 3] * [1] + 2 * [-3]
        //     [3 4 5]   [2]       [-2]
        //               [3]
        let a = vec![1_f32, 2_f32, 3_f32, 3_f32, 4_f32, 5_f32];
        let x = vec![1_f32, 2_f32, 3_f32];
        let mut y = vec![-3_f32, -2_f32];
        unsafe {
            cblas_sgemv(
                CBlasOrder::RowMajor,
                CBlasTranspose::NoTrans,
                2,3,
                3_f32,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                2_f32,
                y.as_mut_ptr(),
                1
            )
        }
        assert_eq!(y, vec![36_f32, 74_f32])
    }
}
