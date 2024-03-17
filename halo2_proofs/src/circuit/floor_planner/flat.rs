use std::cmp;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;

use ark_std::end_timer;
use ark_std::start_timer;
use ff::Field;

use crate::parallel::Parallel;
use crate::{
    circuit::{
        layouter::{RegionColumn, RegionLayouter, TableLayouter},
        Cell, Layouter, Region, RegionIndex, RegionStart, Table,
    },
    plonk::{
        Advice, Any, Assigned, Assignment, Circuit, Column, Error, Fixed, FloorPlanner, Instance,
        Selector, TableColumn,
    },
};

mod region;

use super::single_pass::SimpleTableLayouter;

/// A simple [`FlatFloorPlanner`] that performs minimal optimizations.
#[derive(Debug)]
pub struct FlatFloorPlanner;

impl FloorPlanner for FlatFloorPlanner {
    fn synthesize<F: Field, CS: Assignment<F>, C: Circuit<F>>(
        cs: &CS,
        circuit: &C,
        config: C::Config,
        constants: Vec<Column<Fixed>>,
    ) -> Result<(), Error> {
        if !cs.is_in_prove_mode() {
            let layouter = FlatShapeLayouter::new(cs)?;
            circuit
                .without_witnesses()
                .synthesize(config.clone(), layouter.clone())?;

            let mut constants_to_assign = Arc::try_unwrap(layouter.dynamic)
                .unwrap()
                .into_inner()
                .unwrap()
                .constants_to_assign;

            constants_to_assign.sort_by(|(_, cell_a), (_, cell_b)| {
                if cell_a.column != cell_b.column {
                    cell_a.column.cmp(&cell_b.column)
                } else {
                    cell_a.row_offset.cmp(&cell_b.row_offset)
                }
            });

            // Assign constants. For the simple floor planner, we assign constants in order in
            // the first `constants` column.
            // we assume the first constants starts at zero if constants_to_assign is not empty
            if constants.is_empty() {
                if !constants_to_assign.is_empty() {
                    return Err(Error::NotEnoughColumnsForConstants);
                } else {
                    Ok::<(), Error>(())
                }
            } else {
                let constants_column = constants[0];
                let mut next_constant_row = 0;
                for (constant, advice) in constants_to_assign {
                    cs.assign_fixed(
                        || format!("Constant({:?})", constant.evaluate()),
                        constants_column.clone(),
                        next_constant_row,
                        || Ok(constant),
                    )?;
                    cs.copy(
                        constants_column.into(),
                        next_constant_row,
                        advice.column,
                        advice.row_offset,
                    )?;
                    next_constant_row += 1;
                }
                Ok(())
            }?;
        }

        let layouter = FlatChipLayouter::new(cs)?;
        circuit.synthesize(config, layouter)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FlatShapeDynamic<F: Field> {
    constants_to_assign: Vec<(Assigned<F>, Cell)>,
    /// number of regions
    nb_regions: usize,
    /// Stores the table fixed columns.
    table_columns: Vec<TableColumn>,
    _marker: PhantomData<F>,
}

/// A [`Layouter`] for a single-chip circuit.
#[derive(Clone)]
pub struct FlatShapeLayouter<'a, F: Field, CS: Assignment<F> + 'a> {
    cs: &'a CS,
    dynamic: Arc<Mutex<FlatShapeDynamic<F>>>,
}

unsafe impl<'a, F: Field, CS: Assignment<F> + 'a> Send for FlatShapeLayouter<'a, F, CS> {
    // No need to provide methods; it's a marker trait
}

impl<'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug for FlatShapeLayouter<'a, F, CS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dynamic = self.dynamic.lock().unwrap();
        f.debug_struct("FlatChipLayouter")
            .field("nb_regions", &dynamic.nb_regions)
            .finish()
    }
}

impl<'a, F: Field, CS: Assignment<F>> FlatShapeLayouter<'a, F, CS> {
    /// Creates a new single-chip layouter.
    pub fn new(cs: &'a CS) -> Result<Self, Error> {
        let ret = FlatShapeLayouter {
            cs,
            dynamic: Arc::new(Mutex::new(FlatShapeDynamic {
                nb_regions: 0,
                table_columns: vec![],
                constants_to_assign: vec![],
                _marker: PhantomData,
            })),
        };
        Ok(ret)
    }
}

impl<'a, F: Field, CS: Assignment<F> + 'a> Layouter<F> for FlatShapeLayouter<'a, F, CS> {
    type Root = Self;

    fn assign_region<A, AR, N, NR>(&self, name: N, assignment: A) -> Result<AR, Error>
    where
        A: Fn(&Region<F>) -> Result<AR, Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        let mut dynamic = self.dynamic.lock().unwrap();
        let region_index = dynamic.nb_regions;
        dynamic.nb_regions += 1;
        drop(dynamic);

        let name = name().into();

        self.cs.enter_region(|| name.clone());

        // Get shape of the region.
        let shape = region::RegionSetup::new(region_index.into());
        let shared_region = Parallel::new(shape);

        let region: &dyn RegionLayouter<F> = &shared_region;
        let result = assignment(&region.into())?;
        self.cs.exit_region();

        let mut shape = shared_region.into_inner();

        let mut dynamic = self.dynamic.lock().unwrap();
        dynamic.constants_to_assign.append(&mut shape.constants);

        Ok(result)
    }

    fn assign_table<A, N, NR>(&self, name: N, assignment: A) -> Result<(), Error>
    where
        A: Fn(Table<F>) -> Result<(), Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        // Maintenance hazard: there is near-duplicate code in `v1::AssignmentPass::assign_table`.
        // Assign table cells.
        let name = name().into();

        let mut dynamic = self.dynamic.lock().unwrap();
        self.cs.enter_region(|| name.clone());
        let table = SimpleTableLayouter::new(self.cs, &dynamic.table_columns);
        {
            let table: &dyn TableLayouter<F> = &table;
            assignment(table.into())
        }?;

        let default_and_assigned = table.default_and_assigned.lock().unwrap().clone();
        self.cs.exit_region();

        // Check that all table columns have the same length `first_unused`,
        // and all cells up to that length are assigned.
        let first_unused = {
            match default_and_assigned
                .values()
                .map(|(_, assigned)| {
                    if assigned.iter().all(|b| *b) {
                        Some(assigned.len())
                    } else {
                        None
                    }
                })
                .reduce(|acc, item| match (acc, item) {
                    (Some(a), Some(b)) if a == b => Some(a),
                    _ => None,
                }) {
                Some(Some(len)) => len,
                _ => return Err(Error::Synthesis), // TODO better error
            }
        };

        // Record these columns so that we can prevent them from being used again.
        for column in default_and_assigned.keys() {
            dynamic.table_columns.push(*column);
        }

        for (col, (default_val, _)) in default_and_assigned.iter() {
            // default_val must be Some because we must have assigned
            // at least one cell in each column, and in that case we checked
            // that all cells up to first_unused were assigned.
            self.cs
                .fill_from_row(col.inner(), first_unused, default_val.unwrap())?;
        }

        Ok(())
    }

    // Flat layouter does not arrange region that overlaps

    fn constrain_instance(
        &self,
        cell: Cell,
        instance: Column<Instance>,
        row: usize,
    ) -> Result<(), Error> {
        self.cs
            .copy(cell.column, cell.row_offset, instance.into(), row)
    }

    fn get_root(&self) -> &Self::Root {
        self
    }

    fn push_namespace<NR, N>(&self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.cs.push_namespace(name_fn)
    }

    fn pop_namespace(&self, gadget_name: Option<String>) {
        self.cs.pop_namespace(gadget_name)
    }
}

/// A [`Layouter`] for a single-chip circuit.
#[derive(Clone)]
pub struct FlatChipLayouter<'a, F: Field, CS: Assignment<F> + 'a> {
    cs: &'a CS,
    _mark: PhantomData<F>,
}

unsafe impl<'a, F: Field, CS: Assignment<F> + 'a> Send for FlatChipLayouter<'a, F, CS> {
    // No need to provide methods; it's a marker trait
}

impl<'a, F: Field, CS: Assignment<F>> FlatChipLayouter<'a, F, CS> {
    /// Creates a new single-chip layouter.
    pub fn new(cs: &'a CS) -> Result<Self, Error> {
        let ret = FlatChipLayouter {
            cs,
            _mark: PhantomData,
        };
        Ok(ret)
    }
}

impl<'a, F: Field, CS: Assignment<F> + 'a> Layouter<F> for FlatChipLayouter<'a, F, CS> {
    type Root = Self;

    fn assign_region<A, AR, N, NR>(&self, _name: N, assignment: A) -> Result<AR, Error>
    where
        A: Fn(&Region<'_, F>) -> Result<AR, Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        let region = FlatChipLayouterRegion::new(self.cs);
        let result = {
            let region: &dyn RegionLayouter<F> = &region;
            assignment(&region.into())
        }?;
        Ok(result)
    }

    fn assign_table<A, N, NR>(&self, _name: N, mut _assignment: A) -> Result<(), Error>
    where
        A: FnMut(Table<'_, F>) -> Result<(), Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        Ok(())
    }

    fn constrain_instance(
        &self,
        _cell: Cell,
        _instance: Column<Instance>,
        _row: usize,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn get_root(&self) -> &Self::Root {
        self
    }

    fn push_namespace<NR, N>(&self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.cs.push_namespace(name_fn)
    }

    fn pop_namespace(&self, gadget_name: Option<String>) {
        self.cs.pop_namespace(gadget_name)
    }
}

struct FlatChipLayouterRegion<'a, F: Field, CS: Assignment<F> + 'a> {
    cs: &'a CS,
    _mark: PhantomData<F>,
}

impl<'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug for FlatChipLayouterRegion<'a, F, CS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlatChipLayouterRegion").finish()
    }
}

impl<'a, F: Field, CS: Assignment<F> + 'a> FlatChipLayouterRegion<'a, F, CS> {
    fn new(cs: &'a CS) -> Self {
        FlatChipLayouterRegion {
            cs,
            _mark: PhantomData,
        }
    }
}

impl<'a, F: Field, CS: Assignment<F> + 'a> RegionLayouter<F> for FlatChipLayouterRegion<'a, F, CS> {
    fn enable_selector<'v>(
        &'v self,
        annotation: &'v (dyn Fn() -> String + 'v),
        selector: &Selector,
        offset: usize,
    ) -> Result<(), Error> {
        self.cs.enable_selector(annotation, selector, offset)
    }

    fn assign_advice<'v>(
        &'v self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        offset: usize,
        to: &'v mut (dyn FnMut() -> Result<Assigned<F>, Error> + 'v),
    ) -> Result<Cell, Error> {
        self.cs.assign_advice(annotation, column, offset, to)?;

        Ok(Cell {
            //region_index: self.region_index,
            region_index: RegionIndex(0), // no longer track the region index as in phase 2
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
        let advice = self.assign_advice(annotation, column, offset, &mut || Ok(constant))?;
        self.constrain_constant(advice, constant)?;

        Ok(advice)
    }

    fn assign_advice_from_instance<'v>(
        &self,
        annotation: &'v (dyn Fn() -> String + 'v),
        instance: Column<Instance>,
        row: usize,
        advice: Column<Advice>,
        offset: usize,
    ) -> Result<(Cell, Option<F>), Error> {
        let value = self.cs.query_instance(instance, row)?;

        let cell = self.assign_advice(annotation, advice, offset, &mut || {
            value.ok_or(Error::Synthesis).map(|v| v.into())
        })?;

        self.cs
            .copy(cell.column, cell.row_offset, instance.into(), row)?;

        Ok((cell, value))
    }

    fn assign_fixed<'v>(
        &'v self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Fixed>,
        offset: usize,
        to: &'v mut (dyn FnMut() -> Result<Assigned<F>, Error> + 'v),
    ) -> Result<Cell, Error> {
        self.cs.assign_fixed(annotation, column, offset, to)?;
        Ok(Cell {
            //region_index: self.region_index,
            region_index: RegionIndex(0), // no longer track the region index as in phase 2
            row_offset: offset,
            column: column.into(),
        })
    }

    fn constrain_constant(&self, _cell: Cell, _constant: Assigned<F>) -> Result<(), Error> {
        Ok(())
    }

    fn constrain_equal(&self, left: Cell, right: Cell) -> Result<(), Error> {
        self.cs
            .copy(left.column, left.row_offset, right.column, right.row_offset)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::FlatFloorPlanner;
    use crate::{
        dev::MockProver,
        plonk::{Advice, Circuit, Column, Error},
    };
    use pairing::bn256::Fr as Scalar;

    #[test]
    fn not_enough_columns_for_constants() {
        struct MyCircuit {}

        impl Circuit<Scalar> for MyCircuit {
            type Config = Column<Advice>;
            type FloorPlanner = FlatFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MyCircuit {}
            }

            fn configure(meta: &mut crate::plonk::ConstraintSystem<Scalar>) -> Self::Config {
                meta.advice_column()
            }

            fn synthesize(
                &self,
                config: Self::Config,
                layouter: impl crate::circuit::Layouter<Scalar>,
            ) -> Result<(), crate::plonk::Error> {
                layouter.assign_region(
                    || "assign constant",
                    |region| region.assign_advice_from_constant(|| "one", config, 0, Scalar::one()),
                )?;

                Ok(())
            }
        }

        let circuit = MyCircuit {};
        assert!(matches!(
            MockProver::run(3, &circuit, vec![]).unwrap_err(),
            Error::NotEnoughColumnsForConstants,
        ));
    }
}
