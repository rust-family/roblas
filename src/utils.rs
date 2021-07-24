/// compare two letter, case-insensitive
#[inline(always)]
pub fn letter_same(l: char, r: char) -> bool {
    l.to_ascii_uppercase() == r.to_ascii_uppercase()
}
