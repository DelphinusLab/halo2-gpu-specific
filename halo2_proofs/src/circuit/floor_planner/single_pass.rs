use std::cmp;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;

use ff::Field;

use crate::{
    circuit::{
        layouter::{RegionColumn, RegionLayouter, RegionShape, TableLayouter, SharedRegion},
        Cell, Layouter, Region, RegionIndex, RegionStart, Table,
    },
    plonk::{
        Advice, Any, Assigned, Assignment, Circuit, Column, Error, Fixed, FloorPlanner, Instance,
        Selector, TableColumn,
    },
};

/// A simple [`FloorPlanner`] that performs minimal optimizations.
///
/// This floor planner is suitable for debugging circuits. It aims to reflect the circuit
/// "business logic" in the circuit layout as closely as possible. It uses a single-pass
/// layouter that does not reorder regions for optimal packing.
#[derive(Debug)]
pub struct SimpleFloorPlanner;

impl FloorPlanner for SimpleFloorPlanner {
    fn synthesize<F: Field, CS: Assignment<F>, C: Circuit<F>>(
        cs: &CS,
        circuit: &C,
        config: C::Config,
        constants: Vec<Column<Fixed>>,
    ) -> Result<(), Error> {
        let layouter = SingleChipLayouter::new(cs, constants)?;
        circuit.synthesize(config, layouter)
    }
}

pub struct SingleChipDynamic<F: Field> {
    constants: Vec<Column<Fixed>>,
    /// Stores the starting row for each region.
    regions: Vec<RegionStart>,
    /// Stores the first empty row for each column.
    columns: HashMap<RegionColumn, usize>,
    /// Stores the table fixed columns.
    table_columns: Vec<TableColumn>,
    _marker: PhantomData<F>,
}

/// A [`Layouter`] for a single-chip circuit.
#[derive (Clone)]
pub struct SingleChipLayouter<'a, F: Field, CS: Assignment<F> + 'a> {
    cs: &'a CS,
    dynamic: Arc<Mutex<SingleChipDynamic<F>>>,
}

unsafe impl<'a, F: Field, CS: Assignment<F> + 'a>  Send for SingleChipLayouter<'a, F, CS> {
        // No need to provide methods; it's a marker trait
}

impl<'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug for SingleChipLayouter<'a, F, CS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dynamic = self.dynamic.lock().unwrap();
        f.debug_struct("SingleChipLayouter")
            .field("regions", &dynamic.regions)
            .field("columns", &dynamic.columns)
            .finish()
    }
}

impl<'a, F: Field, CS: Assignment<F>> SingleChipLayouter<'a, F, CS> {
    /// Creates a new single-chip layouter.
    pub fn new(cs: &'a CS, constants: Vec<Column<Fixed>>) -> Result<Self, Error> {
        let ret = SingleChipLayouter {
            cs,
            dynamic: Arc::new(Mutex::new(SingleChipDynamic {
                constants,
                regions: vec![],
                columns: HashMap::default(),
                table_columns: vec![],
                _marker: PhantomData,
            })),
        };
        Ok(ret)
    }
}

impl<'a, F: Field, CS: Assignment<F> + 'a> Layouter<F> for SingleChipLayouter<'a, F, CS> {
    type Root = Self;

    fn assign_region<A, AR, N, NR>(&self, name: N, assignment: A) -> Result<AR, Error>
    where
        A: Fn(Region<'_, F>) -> Result<AR, Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        let mut dynamic = self.dynamic.lock().unwrap();
        let region_index = dynamic.regions.len();

        // Get shape of the region.
        let shape = RegionShape::new(region_index.into());
        let shared_region = SharedRegion(Arc::new(Mutex::new(shape)));

        let region: &dyn RegionLayouter<F> = &shared_region;
        assignment(region.into())?;

        let shape = Arc::try_unwrap(shared_region.0).unwrap().into_inner().unwrap();

        // Lay out this region. We implement the simplest approach here: position the
        // region starting at the earliest row for which none of the columns are in use.
        let mut region_start = 0;
        for column in &shape.columns {
            region_start = cmp::max(region_start, dynamic.columns.get(column).cloned().unwrap_or(0));
        }
        dynamic.regions.push(region_start.into());

        // Update column usage information.
        for column in shape.columns {
            dynamic.columns.insert(column, region_start + shape.row_count);
        }

        drop(dynamic);

        // Assign region cells.
        self.cs.enter_region(name);

        let region = SingleChipLayouterRegion::new(self, region_index.into());
        let result = {
            let region: &dyn RegionLayouter<F> = &region;
            assignment(region.into())
        }?;

        let constants_to_assign = Arc::try_unwrap(region.constants).unwrap().into_inner().unwrap();
        self.cs.exit_region();


        let mut dynamic = self.dynamic.lock().unwrap();

        // Assign constants. For the simple floor planner, we assign constants in order in
        // the first `constants` column.
        if dynamic.constants.is_empty() {
            if !constants_to_assign.is_empty() {
                return Err(Error::NotEnoughColumnsForConstants);
            }
        } else {
            let constants_column = dynamic.constants[0];
            let regions = dynamic.regions.clone();

            let next_constant_row = dynamic
                .columns
                .entry(Column::<Any>::from(constants_column).into())
                .or_default();
            for (constant, advice) in constants_to_assign {
                self.cs.assign_fixed(
                    || format!("Constant({:?})", constant.evaluate()),
                    constants_column.clone(),
                    *next_constant_row,
                    || Ok(constant),
                )?;
                self.cs.copy(
                    constants_column.into(),
                    *next_constant_row,
                    advice.column,
                    *regions[*advice.region_index] + advice.row_offset,
                )?;
                *next_constant_row += 1;
            }
        }

        Ok(result)
    }

    fn assign_table<A, N, NR>(&self, name: N, mut assignment: A) -> Result<(), Error>
    where
        A: FnMut(Table<'_, F>) -> Result<(), Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        // Maintenance hazard: there is near-duplicate code in `v1::AssignmentPass::assign_table`.
        // Assign table cells.
        let mut dynamic = self.dynamic.lock().unwrap();
        self.cs.enter_region(name);
        let mut table = SimpleTableLayouter::new(self.cs, &dynamic.table_columns);
        {
            let table: &mut dyn TableLayouter<F> = &mut table;
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

    fn constrain_instance(
        &self,
        cell: Cell,
        instance: Column<Instance>,
        row: usize,
    ) -> Result<(), Error> {
        let dynamic = self.dynamic.lock().unwrap();
        self.cs.copy(
            cell.column,
            *dynamic.regions[*cell.region_index] + cell.row_offset,
            instance.into(),
            row,
        )
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

struct SingleChipLayouterRegion<'r, 'a, F: Field, CS: Assignment<F> + 'a> {
    layouter: &'r SingleChipLayouter<'a, F, CS>,
    region_index: RegionIndex,
    /// Stores the constants to be assigned, and the cells to which they are copied.
    constants: Arc<Mutex<Vec<(Assigned<F>, Cell)>>>,
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug
    for SingleChipLayouterRegion<'r, 'a, F, CS>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleChipLayouterRegion")
            .field("layouter", &self.layouter)
            .field("region_index", &self.region_index)
            .finish()
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> SingleChipLayouterRegion<'r, 'a, F, CS> {
    fn new(layouter: &'r SingleChipLayouter<'a, F, CS>, region_index: RegionIndex) -> Self {
        SingleChipLayouterRegion {
            layouter,
            region_index,
            constants: Arc::new(Mutex::new(vec![])),
        }
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> RegionLayouter<F>
    for SingleChipLayouterRegion<'r, 'a, F, CS>
{
    fn enable_selector<'v>(
        &'v self,
        annotation: &'v (dyn Fn() -> String + 'v),
        selector: &Selector,
        offset: usize,
    ) -> Result<(), Error> {
        self.layouter.cs.enable_selector(
            annotation,
            selector,
            *self.layouter.dynamic.lock().unwrap().regions[*self.region_index] + offset,
        )
    }

    fn assign_advice<'v>(
        &'v self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        offset: usize,
        to: &'v mut (dyn FnMut() -> Result<Assigned<F>, Error> + 'v),
    ) -> Result<Cell, Error> {
        let dynamic = self.layouter.dynamic.lock().unwrap();
        self.layouter.cs.assign_advice(
            annotation,
            column,
            *dynamic.regions[*self.region_index] + offset,
            to,
        )?;

        Ok(Cell {
            region_index: self.region_index,
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
        let value = self.layouter.cs.query_instance(instance, row)?;

        let cell = self.assign_advice(annotation, advice, offset, &mut || {
            value.ok_or(Error::Synthesis).map(|v| v.into())
        })?;

        let dynamic = self.layouter.dynamic.lock().unwrap();

        self.layouter.cs.copy(
            cell.column,
            *dynamic.regions[*cell.region_index] + cell.row_offset,
            instance.into(),
            row,
        )?;

        Ok((cell, value))
    }

    fn assign_fixed<'v>(
        &'v self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Fixed>,
        offset: usize,
        to: &'v mut (dyn FnMut() -> Result<Assigned<F>, Error> + 'v),
    ) -> Result<Cell, Error> {
        let dynamic = self.layouter.dynamic.lock().unwrap();
        self.layouter.cs.assign_fixed(
            annotation,
            column,
            *dynamic.regions[*self.region_index] + offset,
            to,
        )?;

        Ok(Cell {
            region_index: self.region_index,
            row_offset: offset,
            column: column.into(),
        })
    }

    fn constrain_constant(&self, cell: Cell, constant: Assigned<F>) -> Result<(), Error> {
        self.constants.lock().unwrap().push((constant, cell));
        Ok(())
    }

    fn constrain_equal(&self, left: Cell, right: Cell) -> Result<(), Error> {
        let dynamic = self.layouter.dynamic.lock().unwrap();
        self.layouter.cs.copy(
            left.column,
            *dynamic.regions[*left.region_index] + left.row_offset,
            right.column,
            *dynamic.regions[*right.region_index] + right.row_offset,
        )?;

        Ok(())
    }
}

/// The default value to fill a table column with.
///
/// - The outer `Option` tracks whether the value in row 0 of the table column has been
///   assigned yet. This will always be `Some` once a valid table has been completely
///   assigned.
/// - The inner `Option` tracks whether the underlying `Assignment` is evaluating
///   witnesses or not.
type DefaultTableValue<F> = Option<Option<Assigned<F>>>;

pub(crate) struct SimpleTableLayouter<'r, 'a, F: Field, CS: Assignment<F> + 'a> {
    cs: &'a CS,
    used_columns: &'r [TableColumn],
    // maps from a fixed column to a pair (default value, vector saying which rows are assigned)
    pub(crate) default_and_assigned: Arc<Mutex<HashMap<TableColumn, (DefaultTableValue<F>, Vec<bool>)>>>,
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug for SimpleTableLayouter<'r, 'a, F, CS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimpleTableLayouter")
            .field("used_columns", &self.used_columns)
            .field("default_and_assigned", &self.default_and_assigned)
            .finish()
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> SimpleTableLayouter<'r, 'a, F, CS> {
    pub(crate) fn new(cs: &'a CS, used_columns: &'r [TableColumn]) -> Self {
        SimpleTableLayouter {
            cs,
            used_columns,
            default_and_assigned: Arc::new(Mutex::new(HashMap::default())),
        }
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> TableLayouter<F>
    for SimpleTableLayouter<'r, 'a, F, CS>
{
    fn assign_cell<'v>(
        &'v self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: TableColumn,
        offset: usize,
        to: &'v mut (dyn FnMut() -> Result<Assigned<F>, Error> + 'v),
    ) -> Result<(), Error> {
        if self.used_columns.contains(&column) {
            return Err(Error::Synthesis); // TODO better error
        }

        let mut binding = self.default_and_assigned.lock().unwrap();

        let entry = binding.entry(column).or_default();

        let mut value = None;
        self.cs.assign_fixed(
            annotation,
            column.inner(),
            offset, // tables are always assigned starting at row 0
            || {
                let res = to();
                value = res.as_ref().ok().cloned();
                res
            },
        )?;

        match (entry.0.is_none(), offset) {
            // Use the value at offset 0 as the default value for this table column.
            (true, 0) => entry.0 = Some(value),
            // Since there is already an existing default value for this table column,
            // the caller should not be attempting to assign another value at offset 0.
            (false, 0) => return Err(Error::Synthesis), // TODO better error
            _ => (),
        }
        if entry.1.len() <= offset {
            entry.1.resize(offset + 1, false);
        }
        entry.1[offset] = true;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pairing::bn256::Fr as Scalar;

    use super::SimpleFloorPlanner;
    use crate::{
        dev::MockProver,
        plonk::{Advice, Circuit, Column, Error},
    };

    #[test]
    fn not_enough_columns_for_constants() {
        struct MyCircuit {}

        impl Circuit<Scalar> for MyCircuit {
            type Config = Column<Advice>;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MyCircuit {}
            }

            fn configure(meta: &mut crate::plonk::ConstraintSystem<Scalar>) -> Self::Config {
                meta.advice_column()
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl crate::circuit::Layouter<Scalar>,
            ) -> Result<(), crate::plonk::Error> {
                layouter.assign_region(
                    || "assign constant",
                    |mut region| {
                        region.assign_advice_from_constant(|| "one", config, 0, Scalar::one())
                    },
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
