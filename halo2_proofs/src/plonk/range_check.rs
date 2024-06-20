use ff::Field;
use num::FromPrimitive;
use num_derive::FromPrimitive;
use pairing::arithmetic::FieldExt;

use super::Advice;
use super::Column;

#[derive(Clone, Debug)]
pub struct RangeCheckRel<F: Field> {
    pub origin: Column<Advice>,
    pub sort: Column<Advice>,
    pub min: (u32, F),
    pub max: (u32, F),
    pub step: (u32, F),
}

impl<F: Field> RangeCheckRel<F> {
    pub fn new(
        origin: Column<Advice>,
        sort: Column<Advice>,
        min: (u32, F),
        max: (u32, F),
        step: (u32, F),
    ) -> Self {
        assert_ne!(step.0, 0);
        assert!(min.0 <= max.0);

        RangeCheckRel {
            origin,
            sort,
            min,
            max,
            step,
        }
    }
}

pub(crate) struct RangeCheckRelAssigner {
    current: u32,
    maximal: u32,
    step: u32,
}

impl Iterator for RangeCheckRelAssigner {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.current;

        if value < self.maximal {
            self.current = u32::min(value + self.step, self.maximal);

            Some(value)
        } else if self.current == self.maximal {
            self.current += self.step;

            Some(value)
        } else {
            None
        }
    }
}

impl<F: Field> From<&RangeCheckRel<F>> for RangeCheckRelAssigner {
    fn from(value: &RangeCheckRel<F>) -> Self {
        RangeCheckRelAssigner {
            current: value.min.0,
            maximal: value.max.0,
            step: value.step.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Argument<F: Field>(pub Vec<RangeCheckRel<F>>);

impl<F: Field> Argument<F> {
    pub(crate) fn new() -> Self {
        Self(vec![])
    }
}
