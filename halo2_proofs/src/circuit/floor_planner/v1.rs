use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;

use ff::Field;

use crate::{
    circuit::{
        floor_planner::single_pass::SimpleTableLayouter,
        layouter::{RegionColumn, RegionLayouter, RegionShape, TableLayouter, SharedRegion},
        Cell, Layouter, Region, RegionIndex, RegionStart, Table,
    },
    plonk::{
        Advice, Any, Assigned, Assignment, Circuit, Column, Error, Fixed, FloorPlanner, Instance,
        Selector, TableColumn,
    },
};

mod strategy;

/// The version 1 [`FloorPlanner`] provided by `halo2`.
///
/// - No column optimizations are performed. Circuit configuration is left entirely to the
///   circuit designer.
/// - A dual-pass layouter is used to measures regions prior to assignment.
/// - Regions are measured as rectangles, bounded on the cells they assign.
/// - Regions are laid out using a greedy first-fit strategy, after sorting regions by
///   their "advice area" (number of advice columns * rows).
#[derive(Debug)]
pub struct V1;

struct V1PlanDynamic<F: Field> {
    regions: Vec<RegionStart>,
    /// Stores the constants to be assigned, and the cells to which they are copied.
    constants: Vec<(Assigned<F>, Cell)>,
    /// Stores the table fixed columns.
    table_columns: Vec<TableColumn>,
}

struct V1Plan<'a, F: Field, CS: Assignment<F> + 'a> {
    cs: &'a CS,
    dynamic: Arc<Mutex<V1PlanDynamic<F>>>,
}

impl<'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug for V1Plan<'a, F, CS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("floor_planner::V1Plan").finish()
    }
}

impl<'a, F: Field, CS: Assignment<F>> V1Plan<'a, F, CS> {
    /// Creates a new v1 layouter.
    pub fn new(cs: &'a CS) -> Result<Self, Error> {
        let ret = V1Plan {
            cs,
            dynamic: Arc::new(Mutex::new(V1PlanDynamic {
            regions: vec![],
            constants: vec![],
            table_columns: vec![],
            }))
        };
        Ok(ret)
    }
}

impl FloorPlanner for V1 {
    fn synthesize<F: Field, CS: Assignment<F>, C: Circuit<F>>(
        cs: &CS,
        circuit: &C,
        config: C::Config,
        constants: Vec<Column<Fixed>>,
    ) -> Result<(), Error> {
        let plan = V1Plan::new(cs)?;

        // First pass: measure the regions within the circuit.
        let mut measure = MeasurementPass::new();
        {
            let pass = &mut measure;
            circuit
                .without_witnesses()
                .synthesize(config.clone(), V1Pass::<_, CS>::measure(pass))?;
        }

        // Planning:
        // - Position the regions.
        let regions = measure.regions.lock().unwrap();
        let (regions, column_allocations) = strategy::slot_in_biggest_advice_first(regions.clone());

        let mut dynamic = plan.dynamic.lock().unwrap();

        dynamic.regions = regions;

        // - Determine how many rows our planned circuit will require.
        let first_unassigned_row = column_allocations
            .iter()
            .map(|(_, a)| a.unbounded_interval_start())
            .max()
            .unwrap_or(0);

        // - Position the constants within those rows.
        let fixed_allocations: Vec<_> = constants
            .into_iter()
            .map(|c| {
                (
                    c,
                    column_allocations
                        .get(&Column::<Any>::from(c).into())
                        .cloned()
                        .unwrap_or_default(),
                )
            })
            .collect();
        let constant_positions = || {
            fixed_allocations.iter().flat_map(|(c, a)| {
                let c = *c;
                a.free_intervals(0, Some(first_unassigned_row))
                    .flat_map(move |e| e.range().unwrap().map(move |i| (c, i)))
            })
        };

        // Second pass:
        // - Assign the regions.
        let mut assign = AssignmentPass::new(&plan);
        {
            let pass = &mut assign;
            circuit.synthesize(config, V1Pass::assign(pass))?;
        }

        // - Assign the constants.
        if constant_positions().count() < dynamic.constants.len() {
            return Err(Error::NotEnoughColumnsForConstants);
        }
        for ((fixed_column, fixed_row), (value, advice)) in
            constant_positions().zip(dynamic.constants.clone().into_iter())
        {
            plan.cs.assign_fixed(
                || format!("Constant({:?})", value.evaluate()),
                fixed_column,
                fixed_row,
                || Ok(value),
            )?;
            plan.cs.copy(
                fixed_column.into(),
                fixed_row,
                advice.column,
                *dynamic.regions[*advice.region_index] + advice.row_offset,
            )?;
        }

        Ok(())
    }
}

#[derive(Debug)]
enum Pass<'p, 'a, F: Field, CS: Assignment<F> + 'a> {
    Measurement(&'p MeasurementPass),
    Assignment(&'p AssignmentPass<'p, 'a, F, CS>),
}

/// A single pass of the [`V1`] layouter.
#[derive(Debug)]
pub struct V1Pass<'p, 'a, F: Field, CS: Assignment<F> + 'a>(Pass<'p, 'a, F, CS>);

impl<'p, 'a, F: Field, CS: Assignment<F> + 'a> V1Pass<'p, 'a, F, CS> {
    fn measure(pass: &'p mut MeasurementPass) -> Self {
        V1Pass(Pass::Measurement(pass))
    }

    fn assign(pass: &'p mut AssignmentPass<'p, 'a, F, CS>) -> Self {
        V1Pass(Pass::Assignment(pass))
    }
}

impl<'p, 'a, F: Field, CS: Assignment<F> + 'a> Layouter<F> for V1Pass<'p, 'a, F, CS> {
    type Root = Self;

    fn assign_region<A, AR, N, NR>(&self, name: N, assignment: A) -> Result<AR, Error>
    where
        A: Fn(Region<'_, F>) -> Result<AR, Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        match &self.0 {
            Pass::Measurement(pass) => pass.assign_region(assignment),
            Pass::Assignment(pass) => pass.assign_region(name, assignment),
        }
    }

    fn assign_table<A, N, NR>(&self, name: N, assignment: A) -> Result<(), Error>
    where
        A: FnMut(Table<'_, F>) -> Result<(), Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        match &self.0 {
            Pass::Measurement(_) => Ok(()),
            Pass::Assignment(pass) => pass.assign_table(name, assignment),
        }
    }

    fn constrain_instance(
        &self,
        cell: Cell,
        instance: Column<Instance>,
        row: usize,
    ) -> Result<(), Error> {
        match &self.0 {
            Pass::Measurement(_) => Ok(()),
            Pass::Assignment(pass) => pass.constrain_instance(cell, instance, row),
        }
    }

    fn get_root(&self) -> &Self::Root {
        self
    }

    fn push_namespace<NR, N>(&self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        if let Pass::Assignment(pass) = &self.0 {
            pass.plan.cs.push_namespace(name_fn);
        }
    }

    fn pop_namespace(&self, gadget_name: Option<String>) {
        if let Pass::Assignment(pass) = &self.0 {
            pass.plan.cs.pop_namespace(gadget_name);
        }
    }
}

/// Measures the circuit.
#[derive(Debug)]
pub struct MeasurementPass {
    regions: Arc<Mutex<Vec<SharedRegion<RegionShape>>>>,
}

impl MeasurementPass {
    fn new() -> Self {
        MeasurementPass { regions: Arc::new(Mutex::new(vec![])) }
    }

    fn assign_region<F: Field, A, AR>(&self, assignment: A) -> Result<AR, Error>
    where
        A: Fn(Region<'_, F>) -> Result<AR, Error>,
    {
        let region_index = self.regions.lock().unwrap().len();

        // Get shape of the region.
        let shape = SharedRegion(Arc::new(Mutex::new(RegionShape::new(region_index.into()))));
        let result = {
            let region: &dyn RegionLayouter<F> = &shape;
            assignment(region.into())
        }?;
        self.regions.lock().unwrap().push(shape);

        Ok(result)
    }
}

/// Assigns the circuit.
#[derive(Debug)]
pub struct AssignmentPass<'p, 'a, F: Field, CS: Assignment<F> + 'a> {
    plan: &'p V1Plan<'a, F, CS>,
    /// Counter tracking which region we need to assign next.
    region_index: Arc<Mutex<usize>>,
}

impl<'p, 'a, F: Field, CS: Assignment<F> + 'a> AssignmentPass<'p, 'a, F, CS> {
    fn new(plan: &'p V1Plan<'a, F, CS>) -> Self {
        AssignmentPass {
            plan,
            region_index: Arc::new(Mutex::new(0)),
        }
    }

    fn assign_region<A, AR, N, NR>(&self, name: N, assignment: A) -> Result<AR, Error>
    where
        A: Fn(Region<'_, F>) -> Result<AR, Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        // Get the next region we are assigning.
        let mut region_index = self.region_index.lock().unwrap();

        let index = *region_index;
        *region_index += 1;

        self.plan.cs.enter_region(name);
        let region = V1Region::new(self.plan, index.into());
        let result = {
            let region: &dyn RegionLayouter<F> = &region;
            assignment(region.into())
        }?;
        self.plan.cs.exit_region();

        Ok(result)
    }

    fn assign_table<A, AR, N, NR>(&self, name: N, mut assignment: A) -> Result<AR, Error>
    where
        A: FnMut(Table<'_, F>) -> Result<AR, Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        // Maintenance hazard: there is near-duplicate code in `SingleChipLayouter::assign_table`.

        // Assign table cells.
        self.plan.cs.enter_region(name);
        let mut dynamic = self.plan.dynamic.lock().unwrap();
        let mut table = SimpleTableLayouter::new(self.plan.cs, &dynamic.table_columns);
        let result = {
            let table: &mut dyn TableLayouter<F> = &mut table;
            assignment(table.into())
        }?;
        let default_and_assigned = table.default_and_assigned.lock().unwrap();
        self.plan.cs.exit_region();

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
            self.plan
                .cs
                .fill_from_row(col.inner(), first_unused, default_val.unwrap())?;
        }

        Ok(result)
    }

    fn constrain_instance(
        &self,
        cell: Cell,
        instance: Column<Instance>,
        row: usize,
    ) -> Result<(), Error> {
        let dynamic = self.plan.dynamic.lock().unwrap();
        self.plan.cs.copy(
            cell.column,
            *dynamic.regions[*cell.region_index] + cell.row_offset,
            instance.into(),
            row,
        )
    }
}

struct V1Region<'r, 'a, F: Field, CS: Assignment<F> + 'a> {
    plan: &'r V1Plan<'a, F, CS>,
    region_index: RegionIndex,
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug for V1Region<'r, 'a, F, CS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("V1Region")
            .field("plan", &self.plan)
            .field("region_index", &self.region_index)
            .finish()
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> V1Region<'r, 'a, F, CS> {
    fn new(plan: &'r V1Plan<'a, F, CS>, region_index: RegionIndex) -> Self {
        V1Region { plan, region_index }
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> RegionLayouter<F> for V1Region<'r, 'a, F, CS> {
    fn enable_selector<'v>(
        &'v self,
        annotation: &'v (dyn Fn() -> String + 'v),
        selector: &Selector,
        offset: usize,
    ) -> Result<(), Error> {
        let dynamic = self.plan.dynamic.lock().unwrap();
        self.plan.cs.enable_selector(
            annotation,
            selector,
            *dynamic.regions[*self.region_index] + offset,
        )
    }

    fn assign_advice<'v>(
        &'v self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        offset: usize,
        to: &'v mut (dyn FnMut() -> Result<Assigned<F>, Error> + 'v),
    ) -> Result<Cell, Error> {
        let dynamic = self.plan.dynamic.lock().unwrap();
        self.plan.cs.assign_advice(
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
        let value = self.plan.cs.query_instance(instance, row)?;
        let dynamic = self.plan.dynamic.lock().unwrap();

        let cell = self.assign_advice(annotation, advice, offset, &mut || {
            value.ok_or(Error::Synthesis).map(|v| v.into())
        })?;

        self.plan.cs.copy(
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
        let dynamic = self.plan.dynamic.lock().unwrap();
        self.plan.cs.assign_fixed(
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
        let mut dynamic = self.plan.dynamic.lock().unwrap();
        dynamic.constants.push((constant, cell));
        Ok(())
    }

    fn constrain_equal(&self, left: Cell, right: Cell) -> Result<(), Error> {
        let dynamic = self.plan.dynamic.lock().unwrap();
        self.plan.cs.copy(
            left.column,
            *dynamic.regions[*left.region_index] + left.row_offset,
            right.column,
            *dynamic.regions[*right.region_index] + right.row_offset,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pairing::bn256::Fr as Scalar;

    use crate::{
        dev::MockProver,
        plonk::{Advice, Circuit, Column, Error},
    };

    #[test]
    fn not_enough_columns_for_constants() {
        struct MyCircuit {}

        impl Circuit<Scalar> for MyCircuit {
            type Config = Column<Advice>;
            type FloorPlanner = super::V1;

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
