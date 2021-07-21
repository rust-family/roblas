/*
   Copyright 2021 Leonardo da Link

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
//! A BLAS(Basic Linear Algebra Subprograms) library implemented in pure rust. For now this project
//! is under development. The first available version is expected to be released at the end of 2021.
// #![no_std]

pub mod common;
#[macro_use]
mod error;
pub mod level1;
pub mod level2;
