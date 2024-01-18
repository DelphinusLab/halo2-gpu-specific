#![allow(clippy::int_plus_one)]

use std::ops::Range;

use ark_std::{end_timer, start_timer};
use ff::Field;
use group::Curve;
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
    let mut cs = ConstraintSystem::default();
    let config = ConcreteCircuit::configure(&mut cs);

    let degree = cs.degree();

    let domain = EvaluationDomain::new(degree as u32, params.k);

    (domain, cs, config)
}

/// Assembly to be used in circuit synthesis.
#[derive(Debug)]
struct Assembly<F: Field> {
    k: u32,
    fixed: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    permutation: permutation::keygen::Assembly,
    selectors: Vec<Vec<bool>>,
    // A range of available rows for assignment and copies.
    usable_rows: Range<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field> Assignment<F> for Assembly<F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about regions in this context.
    }

    fn exit_region(&mut self) {
        // Do nothing; we don't care about regions in this context.
    }

    fn enable_selector<A, AR>(&mut self, _: A, selector: &Selector, row: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.selectors[selector.0][row] = true;

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
        &mut self,
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
        &mut self,
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

        *self
            .fixed
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to()?.into();

        Ok(())
    }

    fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&left_row) || !self.usable_rows.contains(&right_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.permutation
            .copy(left_column, left_row, right_column, right_row)
    }

    fn fill_from_row(
        &mut self,
        column: Column<Fixed>,
        from_row: usize,
        to: Option<Assigned<F>>,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&from_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        let col = self
            .fixed
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;

        for row in self.usable_rows.clone().skip(from_row) {
            col[row] = to.ok_or(Error::Synthesis)?;
        }

        Ok(())
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self, _: Option<String>) {
        // Do nothing; we don't care about namespaces in this context.
    }
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

    let mut assembly: Assembly<C::Scalar> = Assembly {
        k: params.k,
        fixed: vec![domain.empty_lagrange_assigned(); cs.num_fixed_columns],
        permutation: permutation::keygen::Assembly::new(params.n as usize, &cs.permutation),
        selectors: vec![vec![false; params.n as usize]; cs.num_selectors],
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
    let mut cs = ConstraintSystem::default();
    let config = ConcreteCircuit::configure(&mut cs);

    let cs = cs;

    if (params.n as usize) < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(params.k));
    }

    let mut assembly: Assembly<C::Scalar> = Assembly {
        k: params.k,
        fixed: vec![vk.domain.empty_lagrange_assigned(); cs.num_fixed_columns],
        permutation: permutation::keygen::Assembly::new(params.n as usize, &cs.permutation),
        selectors: vec![vec![false; params.n as usize]; cs.num_selectors],
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
    let mut cs = ConstraintSystem::default();
    let config = ConcreteCircuit::configure(&mut cs);

    if (params.n as usize) < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(params.k));
    }

    let mut assembly: Assembly<C::Scalar> = Assembly {
        k: params.k,
        fixed: vec![vk.domain.empty_lagrange_assigned(); cs.num_fixed_columns],
        permutation: permutation::keygen::Assembly::new(params.n as usize, &cs.permutation),
        selectors: vec![vec![false; params.n as usize]; cs.num_selectors],
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
    let fixed = batch_invert_assigned(assembly.fixed);
    Ok((fixed, assembly.permutation))
}
