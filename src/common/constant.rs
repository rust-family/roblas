/// Layout of matrix in memory.
#[repr(i32)]
#[derive(Debug, PartialEq)]
pub enum CBlasOrder {
    RowMajor = 101,
    ColMajor = 102,
}
pub type CBlasLayout = CBlasOrder;

/// Indicate whether a matrix is transposed.
#[repr(i32)]
#[derive(Debug, PartialEq)]
pub enum CBlasTranspose {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
    ConjNoTrans = 114,
}

#[repr(i32)]
#[derive(Debug, PartialEq)]
pub enum CBlasUpLo {
    Upper = 121,
    Lower = 122,
}

#[repr(i32)]
#[derive(Debug, PartialEq)]
pub enum CBlasDiag {
    NonUnit = 131,
    Unit = 132,
}

#[repr(i32)]
#[derive(Debug, PartialEq)]
pub enum CBlasSide {
    Left = 141,
    Right = 142,
}
