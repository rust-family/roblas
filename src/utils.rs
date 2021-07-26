use crate::common::BlasInt;

/// compare two letter, case-insensitive
#[inline(always)]
pub fn letter_same(l: char, r: char) -> bool {
    l.to_ascii_uppercase() == r.to_ascii_uppercase()
}

/// compute the offset(index) of an element of 2-dim column-major array.
///
/// # Arguments
/// `i` - row of the element.
///
/// `j` - column of the element.
///
/// `lda` - leading dimension of the 2-dim array.
/// If the array is in column-major, `lda` should be the same the number of rows of the 2-dim array.
#[inline(always)]
pub fn col_major_index(i: usize, j: usize, lda: BlasInt) -> usize {
    i + j * lda as usize
}
