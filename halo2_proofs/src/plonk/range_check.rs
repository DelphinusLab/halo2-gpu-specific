use ff::Field;
use num::FromPrimitive;
use num_derive::FromPrimitive;

use super::Advice;
use super::Column;

#[derive(Clone, Copy, Debug, FromPrimitive)]
pub enum Range {
    U16,
}

#[derive(Clone, Debug)]
pub struct RangeCheckRel<F: Field> {
    pub origin: Column<Advice>,
    pub sort: Column<Advice>,
    pub min: (u32, F),
    pub max: (u32, F),
    pub step: (u32, F),
}

#[derive(Clone, Debug)]
pub struct Argument<F: Field>(pub Vec<RangeCheckRel<F>>);

impl<F: Field> Argument<F> {
    pub(crate) fn new() -> Self {
        Self(vec![])
    }
}
