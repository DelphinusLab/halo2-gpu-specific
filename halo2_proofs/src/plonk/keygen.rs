#![allow(clippy::int_plus_one)]

use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    ops::Range,
    sync::{Arc, Mutex},
};

use ark_std::{end_timer, start_timer};
use ff::Field;
use group::Curve;
use pairing::arithmetic::FieldExt;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use super::{
    circuit::{
        Advice, Any, Assignment, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner, Instance,
        Selector,
    },
    evaluation::Evaluator,
    permutation, Assigned, Error, LagrangeCoeff, Polynomial, ProvingKey, VerifyingKey,
};
use crate::{arithmetic::CurveAffine, poly::batch_invert_assigned};
use crate::{
    plonk::Expression,
    poly::{
        commitment::{Blind, Params},
        EvaluationDomain, Rotation,
    },
};

use crate::arithmetic::parallelize;

pub(crate) fn create_domain<C, ConcreteCircuit>(
    params: &Params<C>,
) -> (
    EvaluationDomain<C::Scalar>,
    ConstraintSystem<C::Scalar>,
    ConcreteCircuit::Config,
)
where
    C: CurveAffine,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let cs = ConstraintSystem::default();
    let (config, cs) = cs.circuit_configure::<ConcreteCircuit>();

    let degree = cs.degree();
    let domain = EvaluationDomain::new(degree as u32, params.k);

    (domain, cs, config)
}

/// Assembly to be used in circuit synthesis.
#[derive(Debug)]
struct Assembly<F: Field> {
    #[allow(dead_code)]
    k: u32,
    fixed: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    permutation: permutation::keygen::Assembly,
    selectors: Vec<Vec<bool>>,
    // A range of available rows for assignment and copies.
    #[allow(dead_code)]
    usable_rows: Range<usize>,
    _marker: std::marker::PhantomData<F>,
}

/// Assembly to be used in circuit synthesis.
#[derive(Clone, Debug)]
struct AssemblyAssigner<F: Field> {
    k: u32,
    fixed: Arc<Mutex<Vec<Polynomial<Assigned<F>, LagrangeCoeff>>>>,
    permutation: Arc<Mutex<permutation::keygen::ParallelAssembly>>,
    selectors: Arc<Mutex<Vec<Vec<bool>>>>,
    // A range of available rows for assignment and copies.
    usable_rows: Range<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: FieldExt> Into<Assembly<F>> for AssemblyAssigner<F> {
    fn into(self) -> Assembly<F> {
        Assembly {
            k: self.k,
            fixed: Arc::try_unwrap(self.fixed).unwrap().into_inner().unwrap(),
            permutation: permutation::keygen::Assembly::from(
                Arc::try_unwrap(self.permutation)
                    .unwrap()
                    .into_inner()
                    .unwrap(),
            ),
            selectors: Arc::try_unwrap(self.selectors)
                .unwrap()
                .into_inner()
                .unwrap(),
            usable_rows: self.usable_rows,
            _marker: PhantomData,
        }
    }
}

impl<F: Field> Assignment<F> for AssemblyAssigner<F> {
    fn is_in_prove_mode(&self) -> bool {
        false
    }

    fn enter_region<NR, N>(&self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about regions in this context.
    }

    fn exit_region(&self) {
        // Do nothing; we don't care about regions in this context.
    }

    fn enable_selector<A, AR>(&self, _: A, selector: &Selector, row: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        let mut selectors = self.selectors.lock().unwrap();
        selectors[selector.0][row] = true;

        Ok(())
    }

    fn query_instance(&self, _: Column<Instance>, row: usize) -> Result<Option<F>, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        // There is no instance in this context.
        Ok(None)
    }

    fn assign_advice<V, VR, A, AR>(
        &self,
        _: A,
        _: Column<Advice>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Result<VR, Error>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about fixed columns here
        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &self,
        _: A,
        column: Column<Fixed>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Result<VR, Error>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        let mut fixed = self.fixed.lock().unwrap();
        *fixed
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to()?.into();

        Ok(())
    }

    fn copy(
        &self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&left_row) || !self.usable_rows.contains(&right_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        let mut permutation = self.permutation.lock().unwrap();
        permutation.copy(left_column, left_row, right_column, right_row)
    }

    fn fill_from_row(
        &self,
        column: Column<Fixed>,
        from_row: usize,
        to: Option<Assigned<F>>,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&from_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        let mut fixed = self.fixed.lock().unwrap();
        let col = fixed.get_mut(column.index()).ok_or(Error::BoundsFailure)?;

        for row in self.usable_rows.clone().skip(from_row) {
            col[row] = to.ok_or(Error::Synthesis)?;
        }

        Ok(())
    }

    fn push_namespace<NR, N>(&self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&self, _: Option<String>) {
        // Do nothing; we don't care about namespaces in this context.
    }
}

#[derive(Clone, Debug)]
struct PreprocessCollector<'a, F: Field> {
    /// The circuit’s log‑scale size parameter.
    pub k: u32,

    /// Storage for fixed‑column values, indexed by (column_index, real_row).
    pub fixeds: Arc<Mutex<Vec<Polynomial<Assigned<F>, LagrangeCoeff>>>>,

    /// Permutation gadget collecting copy constraints.
    pub permutation: Arc<Mutex<permutation::keygen::Permutation>>,

    /// Boolean selectors, indexed by (selector_index, real_row).
    pub selectors: Arc<Mutex<Vec<Vec<bool>>>>,

    /// Number of instance rows available per instance column.
    pub _num_instances: Vec<usize>,

    /// Mapping from “virtual” row indices (0..n) to real row indices in the above buffers.
    pub row_mapping: &'a Vec<usize>,

    _marker: PhantomData<F>,
}

impl<'a, F: Field> Assignment<F> for PreprocessCollector<'a, F> {
    fn is_in_prove_mode(&self) -> bool {
        false
    }

    fn enter_region<NR, N>(&self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // no-op
    }

    fn exit_region(&self) {
        // no-op
    }
    fn enable_selector<A, AR>(
        &self,
        _: A,
        selector: &Selector,
        virtual_row: usize,
    ) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let real_row = *self
            .row_mapping
            .get(virtual_row)
            .ok_or(Error::NotEnoughRowsAvailable { current_k: self.k })?;

        let mut selectors = self.selectors.lock().unwrap();
        selectors[selector.0][real_row] = true;
        Ok(())
    }

    fn query_instance(&self, _: Column<Instance>, virtual_row: usize) -> Result<Option<F>, Error> {
        let _ = *self
            .row_mapping
            .get(virtual_row)
            .ok_or(Error::NotEnoughRowsAvailable { current_k: self.k })?;

        Ok(None)
    }

    fn assign_advice<V, VR, A, AR>(
        &self,
        _: A,
        _: Column<Advice>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Result<VR, Error>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // nothing to do
        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &self,
        _: A,
        column: Column<Fixed>,
        virtual_row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Result<VR, Error>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let real_row = *self
            .row_mapping
            .get(virtual_row)
            .ok_or(Error::NotEnoughRowsAvailable { current_k: self.k })?;

        let mut fixeds = self.fixeds.lock().unwrap();
        let slot = fixeds
            .get_mut(column.index())
            .and_then(|col| col.get_mut(real_row))
            .ok_or(Error::BoundsFailure)?;

        *slot = to()?.into();
        Ok(())
    }

    fn copy(
        &self,
        lhs_column: Column<Any>,
        lhs_virtual: usize,
        rhs_column: Column<Any>,
        rhs_virtual: usize,
    ) -> Result<(), Error> {
        let lhs = *self
            .row_mapping
            .get(lhs_virtual)
            .ok_or(Error::NotEnoughRowsAvailable { current_k: self.k })?;
        let rhs = *self
            .row_mapping
            .get(rhs_virtual)
            .ok_or(Error::NotEnoughRowsAvailable { current_k: self.k })?;

        let mut perm = self.permutation.lock().unwrap();
        perm.copy(lhs_column, lhs, rhs_column, rhs)
    }

    fn fill_from_row(
        &self,
        column: Column<Fixed>,
        from_virtual: usize,
        to: Option<Assigned<F>>,
    ) -> Result<(), Error> {
        // ensure mapping exists
        let _ = self
            .row_mapping
            .get(from_virtual)
            .ok_or(Error::NotEnoughRowsAvailable { current_k: self.k })?;

        let mut fixeds = self.fixeds.lock().unwrap();
        let col = fixeds.get_mut(column.index()).ok_or(Error::BoundsFailure)?;

        let filler = to.ok_or(Error::Synthesis)?;
        for &real_row in self.row_mapping.iter().skip(from_virtual) {
            col[real_row] = filler.clone();
        }
        Ok(())
    }
    fn push_namespace<NR, N>(&self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // no-op
    }

    fn pop_namespace(&self, _: Option<String>) {
        // no-op
    }
}

impl<'a, F: Field> Into<Assembly<F>> for PreprocessCollector<'a, F> {
    fn into(self) -> Assembly<F> {
        let fixed = Arc::try_unwrap(self.fixeds).unwrap().into_inner().unwrap();
        let perm = Arc::try_unwrap(self.permutation)
            .unwrap()
            .into_inner()
            .unwrap();
        let sels = Arc::try_unwrap(self.selectors)
            .unwrap()
            .into_inner()
            .unwrap();

        Assembly {
            k: self.k,
            fixed,
            permutation: permutation::keygen::Assembly {
                mapping: perm.into_cycles(),
            },
            selectors: sels,
            usable_rows: 0..self.row_mapping.len(),
            _marker: PhantomData,
        }
    }
}

pub fn get_preprocess_polys_and_permutations<'a, C, ConcreteCircuit>(
    k: u32,
    row_mapping: &'a Vec<usize>,
    permutation_idx: HashMap<(Any, usize), usize>,
    circuit: &ConcreteCircuit,
    config: &ConcreteCircuit::Config,
) -> Result<
    (
        Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
        Vec<Vec<(usize, usize)>>,
    ),
    Error,
>
where
    C: CurveAffine,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let mut cs = ConstraintSystem::default();
    let _ = ConcreteCircuit::configure(&mut cs);

    use std::iter::Iterator;
    let permutation_idx: HashMap<(Any, usize), usize> = permutation_idx
        .iter()
        .map(|(&(pany, local_idx), &global_idx)| ((pany.into(), local_idx), global_idx))
        .collect();

    let mut assembly: PreprocessCollector<'a, C::Scalar> = PreprocessCollector {
        k,
        fixeds: Arc::new(Mutex::new(vec![
            Polynomial {
                values: vec![C::Scalar::zero().into(); 1 << k as usize],
                _marker: PhantomData,
            };
            cs.num_fixed_columns
        ])),
        permutation: Arc::new(Mutex::new(permutation::keygen::Permutation::new(
            permutation_idx,
        ))),
        selectors: Arc::new(Mutex::new(vec![
            vec![false; 1 << k as usize];
            cs.num_selectors
        ])),
        _num_instances: vec![],
        row_mapping,
        _marker: PhantomData,
    };
    // Synthesize the circuit to obtain URS
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config.clone(),
        cs.constants.clone(),
    )?;

    let assembly: Assembly<C::Scalar> = assembly.into();

    let mut fixed = batch_invert_assigned(assembly.fixed);
    fixed.extend(assembly.selectors.into_iter().map(|selectors| {
        let values: Vec<_> = selectors
            .into_iter()
            .map(|selector| {
                if selector {
                    C::Scalar::one()
                } else {
                    C::Scalar::zero()
                }
            })
            .collect();
        Polynomial {
            values,
            _marker: PhantomData::<LagrangeCoeff>,
        }
    }));

    let permutations: Vec<Vec<(usize, usize)>> = assembly
        .permutation
        .mapping
        .into_iter()
        .map(|cycle| {
            cycle
                .into_iter()
                .map(|(a, b)| (a as usize, b as usize))
                .collect()
        })
        .collect();
    Ok((fixed, permutations))
}

/// Generate a `VerifyingKey` from an instance of `Circuit`.
pub fn keygen_vk<C, ConcreteCircuit>(
    params: &Params<C>,
    circuit: &ConcreteCircuit,
) -> Result<VerifyingKey<C>, Error>
where
    C: CurveAffine,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let (domain, cs, config) = create_domain::<C, ConcreteCircuit>(params);

    if (params.n as usize) < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(params.k));
    }

    let mut assembly: AssemblyAssigner<C::Scalar> = AssemblyAssigner {
        k: params.k,
        fixed: Arc::new(Mutex::new(vec![
            domain.empty_lagrange_assigned();
            cs.num_fixed_columns
        ])),
        permutation: Arc::new(Mutex::new(permutation::keygen::ParallelAssembly::new(
            params.n as usize,
            &cs.permutation,
        ))),
        selectors: Arc::new(Mutex::new(vec![
            vec![false; params.n as usize];
            cs.num_selectors
        ])),
        usable_rows: 0..params.n as usize - (cs.blinding_factors() + 1),
        _marker: PhantomData,
    };

    // Synthesize the circuit to obtain URS
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants.clone(),
    )?;

    let assembly: Assembly<C::Scalar> = assembly.into();

    let mut fixed = batch_invert_assigned(assembly.fixed);
    let (cs, selector_polys) = cs.compress_selectors(assembly.selectors);
    fixed.extend(
        selector_polys
            .into_iter()
            .map(|poly| domain.lagrange_from_vec(poly)),
    );

    let permutation_vk = assembly
        .permutation
        .build_vk(params, &domain, &cs.permutation);

    let fixed_commitments = fixed
        .iter()
        .map(|poly| params.commit_lagrange(poly).to_affine())
        .collect();

    Ok(VerifyingKey {
        domain,
        fixed_commitments,
        permutation: permutation_vk,
        cs,
    })
}

/// Generate a `ProvingKey` from a `VerifyingKey` and an instance of `Circuit`.
pub fn keygen_pk<C, ConcreteCircuit>(
    params: &Params<C>,
    vk: VerifyingKey<C>,
    circuit: &ConcreteCircuit,
) -> Result<ProvingKey<C>, Error>
where
    C: CurveAffine,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let cs = ConstraintSystem::default();
    let (config, cs) = cs.circuit_configure::<ConcreteCircuit>();

    if (params.n as usize) < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(params.k));
    }

    let mut assembly: AssemblyAssigner<C::Scalar> = AssemblyAssigner {
        k: params.k,
        fixed: Arc::new(Mutex::new(vec![
            vk.domain.empty_lagrange_assigned();
            cs.num_fixed_columns
        ])),
        permutation: Arc::new(Mutex::new(permutation::keygen::ParallelAssembly::new(
            params.n as usize,
            &cs.permutation,
        ))),
        selectors: Arc::new(Mutex::new(vec![
            vec![false; params.n as usize];
            cs.num_selectors
        ])),
        usable_rows: 0..params.n as usize - (cs.blinding_factors() + 1),
        _marker: std::marker::PhantomData,
    };

    // Synthesize the circuit to obtain URS
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants.clone(),
    )?;

    let assembly: Assembly<C::Scalar> = assembly.into();

    let timer = start_timer!(|| "unnecessary part");
    let (cs, fixed) = if false {
        let mut fixed = batch_invert_assigned(assembly.fixed);
        let (cs, selector_polys) = cs.compress_selectors(assembly.selectors);
        fixed.extend(
            selector_polys
                .into_iter()
                .map(|poly| vk.domain.lagrange_from_vec(poly)),
        );
        (cs, fixed)
    } else {
        assert!(assembly.selectors.len() == 0);
        (
            cs,
            assembly
                .fixed
                .into_par_iter()
                .map(|x| Polynomial {
                    values: x
                        .into_iter()
                        .map(|x| {
                            assert!(x.denominator().is_none());
                            x.numerator()
                        })
                        .collect(),
                    _marker: std::marker::PhantomData,
                })
                .collect::<Vec<_>>(),
        )
    };
    end_timer!(timer);

    let timer = start_timer!(|| "fix poly");
    let fixed_polys: Vec<_> = fixed
        .par_iter()
        .map(|poly| vk.domain.lagrange_to_coeff_st(poly.clone()))
        .collect();
    end_timer!(timer);

    #[cfg(not(feature = "cuda"))]
    let fixed_cosets = fixed_polys
        .iter()
        .map(|poly| vk.domain.coeff_to_extended(poly.clone()))
        .collect();

    let timer = start_timer!(|| "assembly build pkey");
    let permutation_pk = assembly
        .permutation
        .build_pk(params, &vk.domain, &cs.permutation);
    end_timer!(timer);

    let timer = start_timer!(|| "l poly");
    // Compute l_0(X)
    // TODO: this can be done more efficiently
    let mut l0 = vk.domain.empty_lagrange();
    l0[0] = C::Scalar::one();
    let l0 = vk.domain.lagrange_to_coeff(l0);
    #[cfg(not(feature = "cuda"))]
    let l0 = vk.domain.coeff_to_extended(l0);

    // Compute l_blind(X) which evaluates to 1 for each blinding factor row
    // and 0 otherwise over the domain.
    let mut l_blind = vk.domain.empty_lagrange();
    for evaluation in l_blind[..].iter_mut().rev().take(cs.blinding_factors()) {
        *evaluation = C::Scalar::one();
    }
    let l_blind = vk.domain.lagrange_to_coeff(l_blind);
    let l_blind_extended = vk.domain.coeff_to_extended(l_blind);

    // Compute l_last(X) which evaluates to 1 on the first inactive row (just
    // before the blinding factors) and 0 otherwise over the domain
    let mut l_last = vk.domain.empty_lagrange();
    l_last[params.n as usize - cs.blinding_factors() - 1] = C::Scalar::one();
    let l_last = vk.domain.lagrange_to_coeff(l_last);
    let l_last_extended = vk.domain.coeff_to_extended(l_last.clone());

    // Compute l_active_row(X)
    let one = C::Scalar::one();

    let mut l_active_row = vk.domain.empty_extended();
    parallelize(&mut l_active_row, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = i + start;
            *value = one - (l_last_extended[idx] + l_blind_extended[idx]);
        }
    });
    end_timer!(timer);

    let timer = start_timer!(|| "prepare ev");
    // Compute the optimized evaluation data structure
    let ev = Evaluator::new(&vk.cs);
    end_timer!(timer);

    #[cfg(not(feature = "cuda"))]
    let l_last = l_last_extended;

    Ok(ProvingKey {
        vk,
        l0,
        l_last,
        l_active_row,
        fixed_values: fixed,
        fixed_polys,

        #[cfg(not(feature = "cuda"))]
        fixed_cosets,
        permutation: permutation_pk,
        ev,
    })
}

/// Generate a `ProvingKey` from a `VerifyingKey` and an instance of `Circuit`.
pub(crate) fn keygen_pk_from_info<C>(
    params: &Params<C>,
    vk: &VerifyingKey<C>,
    fixed: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    permutation: permutation::keygen::Assembly,
) -> Result<ProvingKey<C>, Error>
where
    C: CurveAffine,
{
    let cs = vk.cs.clone();
    assert!(cs.num_selectors == 0);
    //We do not support the case when selectors exists
    //let selectors = vec![vec![false; params.n as usize]; cs.num_selectors];
    let selectors = vec![];
    use ark_std::{end_timer, start_timer};

    let timer = start_timer!(|| "compress selectors ...");
    let (cs, _) = cs.compress_selectors(selectors);
    end_timer!(timer);
    let timer = start_timer!(|| "fixed polys ...");

    let fixed_polys: Vec<_> = fixed
        .iter()
        .map(|poly| vk.domain.lagrange_to_coeff_st(poly.clone()))
        .collect();
    end_timer!(timer);

    #[cfg(not(feature = "cuda"))]
    let fixed_cosets = fixed_polys
        .iter()
        .map(|poly| vk.domain.coeff_to_extended(poly.clone()))
        .collect();

    let timer = start_timer!(|| "build pk time...");
    let permutation_pk = permutation.build_pk(params, &vk.domain, &cs.permutation);
    end_timer!(timer);

    let timer = start_timer!(|| "l poly");
    // Compute l_0(X)
    // TODO: this can be done more efficiently
    let mut l0 = vk.domain.empty_lagrange();
    l0[0] = C::Scalar::one();
    let l0 = vk.domain.lagrange_to_coeff(l0);
    #[cfg(not(feature = "cuda"))]
    let l0 = vk.domain.coeff_to_extended(l0);

    // Compute l_blind(X) which evaluates to 1 for each blinding factor row
    // and 0 otherwise over the domain.
    let mut l_blind = vk.domain.empty_lagrange();
    for evaluation in l_blind[..].iter_mut().rev().take(cs.blinding_factors()) {
        *evaluation = C::Scalar::one();
    }
    let l_blind = vk.domain.lagrange_to_coeff(l_blind);
    let l_blind_extended = vk.domain.coeff_to_extended(l_blind);

    // Compute l_last(X) which evaluates to 1 on the first inactive row (just
    // before the blinding factors) and 0 otherwise over the domain
    let mut l_last = vk.domain.empty_lagrange();
    l_last[params.n as usize - cs.blinding_factors() - 1] = C::Scalar::one();
    let l_last = vk.domain.lagrange_to_coeff(l_last);
    let l_last_extended = vk.domain.coeff_to_extended(l_last.clone());

    // Compute l_active_row(X)
    let one = C::Scalar::one();

    let mut l_active_row = vk.domain.empty_extended();
    parallelize(&mut l_active_row, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = i + start;
            *value = one - (l_last_extended[idx] + l_blind_extended[idx]);
        }
    });
    end_timer!(timer);

    let timer = start_timer!(|| "prepare ev");
    // Compute the optimized evaluation data structure
    let ev = Evaluator::new(&vk.cs);
    end_timer!(timer);

    #[cfg(not(feature = "cuda"))]
    let l_last = l_last_extended;

    Ok(ProvingKey {
        vk: vk.clone(),
        l0,
        l_last,
        l_active_row,
        fixed_values: fixed,
        fixed_polys,

        #[cfg(not(feature = "cuda"))]
        fixed_cosets,
        permutation: permutation_pk,
        ev,
    })
}

/// Generate a `ProvingKey` from a `VerifyingKey` and an instance of `Circuit`.
pub(crate) fn generate_pk_info<C, ConcreteCircuit>(
    params: &Params<C>,
    vk: &VerifyingKey<C>,
    circuit: &ConcreteCircuit,
) -> Result<
    (
        Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
        permutation::keygen::Assembly,
    ),
    Error,
>
where
    C: CurveAffine,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let cs = ConstraintSystem::default();
    let (config, cs) = cs.circuit_configure::<ConcreteCircuit>();

    if (params.n as usize) < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(params.k));
    }

    let mut assembly: AssemblyAssigner<C::Scalar> = AssemblyAssigner {
        k: params.k,
        fixed: Arc::new(Mutex::new(vec![
            vk.domain.empty_lagrange_assigned();
            cs.num_fixed_columns
        ])),
        permutation: Arc::new(Mutex::new(permutation::keygen::ParallelAssembly::new(
            params.n as usize,
            &cs.permutation,
        ))),
        selectors: Arc::new(Mutex::new(vec![
            vec![false; params.n as usize];
            cs.num_selectors
        ])),
        usable_rows: 0..params.n as usize - (cs.blinding_factors() + 1),
        _marker: std::marker::PhantomData,
    };

    // Synthesize the circuit to obtain URS
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants.clone(),
    )?;

    let assembly: Assembly<C::Scalar> = assembly.into();

    let fixed = batch_invert_assigned(assembly.fixed);
    Ok((fixed, assembly.permutation))
}
