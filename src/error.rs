/// mimic `cblas_xerbla` function, with a extra `row_major` boolean parameter indicating the layout of matrix
macro_rules! xerbla {
    ($row_major:expr,$param_info:expr,$rout:expr,$form:expr) => {
        let param_info = crate::error::param_info_transform($row_major, $param_info, $rout);
        if param_info != 0 {
            eprintln!("Parameter {} to routine {} was incorrect", $param_info, $rout);
        }
        eprint!($form);
        std::process::exit(-1);
    };
    ($row_major:expr,$param_info:expr,$rout:expr,$form:expr,$($args:tt)*) => {
        let param_info = crate::error::param_info_transform($row_major, $param_info, $rout);
        if param_info != 0 {
            eprintln!("Parameter {} to routine {} was incorrect", $param_info, $rout);
        }
        eprint!($form, $($args)*);
        std::process::exit(-1);
    };
}

pub fn param_info_transform(row_major: bool, param_info: isize, rout: &str) -> isize {
    if !row_major {
        param_info
    } else {
        if rout.contains("gemm") {
            match param_info {
                5 => 4,
                4 => 5,
                11 => 9,
                9 => 11,
                _ => param_info
            }
        } else if rout.contains("symm") || rout.contains("hemm") {
            match param_info {
                5 => 4,
                4 => 5,
                _ => param_info
            }
        } else if rout.contains("trmm") || rout.contains("trsm") {
            match param_info {
                7 => 6,
                6 => 7,
                _ => param_info
            }
        } else if rout.contains("gemv") {
            match param_info {
                4 => 3,
                3 => 4,
                _ => param_info
            }
        } else if rout.contains("gbmv") {
            match param_info {
                4 => 3,
                3 => 4,
                6 => 5,
                5 => 6,
                _ => param_info
            }
        } else if rout.contains("ger") {
            match param_info {
                3 => 2,
                2 => 3,
                8 => 6,
                6 => 8,
                _ => param_info
            }
        } else if rout.contains("her2") || rout.contains("hpr2") || rout.contains("her2k") {
            match param_info {
                8 => 6,
                6 => 8,
                _ => param_info
            }
        } else {
            param_info
        }
    }
}
