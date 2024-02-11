use std::{cmp, sync::{Arc, Mutex}};
use ff::Field;

use crate::{circuit::{layouter::{RegionLayouter, SharedRegion}, Cell}, plonk::{Advice, Any, Assigned, Column, Error, Fixed, Instance, Selector, TableColumn}};
use std::collections::HashSet;
use crate::circuit::{layouter::RegionColumn, RegionIndex};

/// The shape of a region. For a region at a certain index, we track
/// the set of columns it uses as well as the number of rows it uses.
#[derive(Clone, Debug)]
pub struct RegionSetup<F: Field> {
    pub(super) region_index: RegionIndex,
    pub(super) columns: HashSet<RegionColumn>,
    pub(super) row_count: usize,
    pub(super) constants: Vec<(Assigned<F>, Cell)>,
}

impl<F: Field> RegionSetup<F> {
    /// Create a new `RegionShape` for a region at `region_index`.
    pub fn new(region_index: RegionIndex) -> Self {
        RegionSetup {
            region_index,
            columns: HashSet::default(),
            row_count: 0,
            constants: vec![],
        }
    }

    /// Get the `region_index` of a `RegionShape`.
    pub fn region_index(&self) -> RegionIndex {
        self.region_index
    }

    /// Get a reference to the set of `columns` used in a `RegionShape`.
    pub fn columns(&self) -> &HashSet<RegionColumn> {
        &self.columns
    }

    /// Get the `row_count` of a `RegionShape`.
    pub fn row_count(&self) -> usize {
        self.row_count
    }
}


impl<F: Field> RegionLayouter<F> for SharedRegion<RegionSetup<F>> {
    fn enable_selector<'v>(
        &'v self,
        _: &'v (dyn Fn() -> String + 'v),
        selector: &Selector,
        offset: usize,
    ) -> Result<(), Error> {
        // Track the selector's fixed column as part of the region's shape.
        let mut region = self.0.lock().unwrap();
        region.columns.insert((*selector).into());
        region.row_count = cmp::max(region.row_count, offset + 1);
        Ok(())
    }

    fn assign_advice<'v>(
        &'v self,
        _: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        offset: usize,
        _to: &'v mut (dyn FnMut() -> Result<Assigned<F>, Error> + 'v),
    ) -> Result<Cell, Error> {
        let mut region = self.0.lock().unwrap();
        region.columns.insert(Column::<Any>::from(column).into());
        region.row_count = cmp::max(region.row_count, offset + 1);

        Ok(Cell {
            region_index: region.region_index,
            row_offset: offset,
            column: column.into(),
        })
    }

    fn assign_advice_from_constant<'v>(
        &'v self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        offset: usize,
        constant: Assigned<F>,
    ) -> Result<Cell, Error> {
        // The rest is identical to witnessing an advice cell.
        let advice = self.assign_advice(annotation, column, offset, &mut || Ok(constant))?;
        self.constrain_constant(advice, constant)?;
        Ok(advice)
    }

    fn assign_advice_from_instance<'v>(
        &self,
        _: &'v (dyn Fn() -> String + 'v),
        _: Column<Instance>,
        _: usize,
        advice: Column<Advice>,
        offset: usize,
    ) -> Result<(Cell, Option<F>), Error> {
        let mut region = self.0.lock().unwrap();
        region.columns.insert(Column::<Any>::from(advice).into());
        region.row_count = cmp::max(region.row_count, offset + 1);

        Ok((
            Cell {
                region_index: region.region_index,
                row_offset: offset,
                column: advice.into(),
            },
            None,
        ))
    }

    fn assign_fixed<'v>(
        &'v self,
        _: &'v (dyn Fn() -> String + 'v),
        column: Column<Fixed>,
        offset: usize,
        _to: &'v mut (dyn FnMut() -> Result<Assigned<F>, Error> + 'v),
    ) -> Result<Cell, Error> {
        let mut region = self.0.lock().unwrap();
        region.columns.insert(Column::<Any>::from(column).into());
        region.row_count = cmp::max(region.row_count, offset + 1);

        Ok(Cell {
            region_index: region.region_index,
            row_offset: offset,
            column: column.into(),
        })
    }

    fn constrain_constant(&self, cell: Cell, constant: Assigned<F>) -> Result<(), Error> {
        // Global constants don't affect the region shape.
        let mut region = self.0.lock().unwrap();
        region.constants.push((constant, cell));
        Ok(())
    }

    fn constrain_equal(&self, _left: Cell, _right: Cell) -> Result<(), Error> {
        // Equality constraints don't affect the region shape.
        Ok(())
    }
}
