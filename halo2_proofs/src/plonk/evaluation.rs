use super::{evaluation_gpu, ConstraintSystem, Expression};
use crate::multicore;
use crate::plonk::evaluation_gpu::{LookupProveExpression, ProveExpression};
use crate::plonk::lookup::prover::Committed;
use crate::plonk::permutation::Argument;
use crate::plonk::{lookup, permutation, Any, ProvingKey};
use crate::poly::Basis;
use crate::{
    arithmetic::{eval_polynomial, parallelize, BaseExt, CurveAffine, FieldExt},
    poly::{
        commitment::Params, multiopen::ProverQuery, Coeff, EvaluationDomain, ExtendedLagrangeCoeff,
        LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ark_std::{end_timer, start_timer};
use group::prime::PrimeCurve;
use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use num_bigint::BigUint;
use std::any::TypeId;
use std::collections::BTreeSet;
use std::convert::TryInto;
use std::iter::FromIterator;
use std::num::ParseIntError;
use std::str::FromStr;
use std::{cmp, slice};
use std::{
    collections::BTreeMap,
    iter,
    ops::{Index, Mul, MulAssign},
};

#[cfg(not(feature = "cuda"))]
/// Return the index in the polynomial of size `isize` after rotation `rot`.
fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
    (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
}

/// Value used in a calculation
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum ValueSource {
    /// This is a constant value
    Constant(usize),
    /// This is an intermediate value
    Intermediate(usize),
    /// This is a fixed column
    Fixed(usize, usize),
    /// This is an advice (witness) column
    Advice(usize, usize),
    /// This is an instance (external) column
    Instance(usize, usize),
}

#[cfg(not(feature = "cuda"))]
impl ValueSource {
    /// Get the value for this source
    pub fn get<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
    ) -> F {
        match self {
            ValueSource::Constant(idx) => constants[*idx],
            ValueSource::Intermediate(idx) => intermediates[*idx],
            ValueSource::Fixed(column_index, rotation) => {
                fixed_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Advice(column_index, rotation) => {
                advice_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Instance(column_index, rotation) => {
                instance_values[*column_index][rotations[*rotation]]
            }
        }
    }
}

/// Calculation
#[derive(Clone, Debug, PartialEq)]
pub enum Calculation {
    /// This is an addition
    Add(ValueSource, ValueSource),
    /// This is a subtraction
    Sub(ValueSource, ValueSource),
    /// This is a product
    Mul(ValueSource, ValueSource),
    /// This is a negation
    Negate(ValueSource),
    /// This is `(a + beta) * b`
    LcBeta(ValueSource, ValueSource),
    /// This is `a * theta + b`
    LcTheta(ValueSource, ValueSource),
    /// This is `a + gamma`
    AddGamma(ValueSource),
    /// This is a simple assignment
    Store(ValueSource),
}

#[cfg(not(feature = "cuda"))]
impl Calculation {
    /// Get the resulting value of this calculation
    pub fn evaluate<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        beta: &F,
        gamma: &F,
        theta: &F,
    ) -> F {
        match self {
            Calculation::Add(a, b) => {
                let a = a.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                );
                let b = b.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                );
                a + b
            }
            Calculation::Sub(a, b) => {
                let a = a.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                );
                let b = b.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                );
                a - b
            }
            Calculation::Mul(a, b) => {
                let a = a.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                );
                let b = b.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                );
                a * b
            }
            Calculation::Negate(v) => -v.get(
                rotations,
                constants,
                intermediates,
                fixed_values,
                advice_values,
                instance_values,
            ),
            Calculation::LcBeta(a, b) => {
                let a = a.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                );
                let b = b.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                );
                (a + beta) * b
            }
            Calculation::LcTheta(a, b) => {
                let a = a.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                );
                let b = b.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                );
                a * theta + b
            }
            Calculation::AddGamma(v) => {
                v.get(
                    rotations,
                    constants,
                    intermediates,
                    fixed_values,
                    advice_values,
                    instance_values,
                ) + gamma
            }
            Calculation::Store(v) => v.get(
                rotations,
                constants,
                intermediates,
                fixed_values,
                advice_values,
                instance_values,
            ),
        }
    }
}

/// EvaluationData
#[derive(Default, Debug)]
pub struct Evaluator<C: CurveAffine> {
    /// Constants
    pub constants: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Value parts
    pub value_parts: Vec<ValueSource>,
    /// Lookup results
    pub lookup_results: Vec<Calculation>,
    /// GPU
    pub gpu_gates_expr: Vec<ProveExpression<C::ScalarExt>>,
    pub gpu_lookup_expr: Vec<LookupProveExpression<C::ScalarExt>>,
    pub unit_ref_count: Vec<(usize, u32)>,
}

/// CaluclationInfo
#[derive(Debug)]
pub struct CalculationInfo {
    /// Calculation
    pub calculation: Calculation,
    /// How many times this calculation is used
    pub counter: usize,
}

impl<C: CurveAffine> Evaluator<C> {
    /// Creates a new evaluation structure
    pub fn new(cs: &ConstraintSystem<C::ScalarExt>) -> Self {
        let mut e = ProveExpression::new();

        let mut ev = Evaluator::default();
        ev.add_constant(&C::ScalarExt::zero());
        ev.add_constant(&C::ScalarExt::one());

        // Custom gates

        for gate in cs.gates.iter() {
            for poly in gate.polynomials().iter() {
                let vs = ev.add_expression(poly);
                ev.value_parts.push(vs);
                e = e.add_gate(poly);
            }
        }

        let e_exprs = e.flatten().into_iter().collect::<Vec<_>>();

        let n_gpu = *crate::plonk::N_GPU;
        println!("gpus number is {}", n_gpu);
        let es = e_exprs
            .chunks((e_exprs.len() + n_gpu - 1) / n_gpu)
            .map(|e| ProveExpression::reconstruct(e))
            .collect::<Vec<_>>();

        for (i, e) in es.iter().enumerate() {
            let complexity = e.get_complexity();
            ev.unit_ref_count = complexity.ref_cnt.into_iter().collect();
            ev.unit_ref_count.sort_by(|(_, l), (_, r)| u32::cmp(l, r));
            ev.unit_ref_count.reverse();

            println!("--------- expr part {} ---------", i);
            println!("complexity is {:?}", e.get_complexity());
            println!("sorted ref cnt is {:?}", ev.unit_ref_count);
            println!("r deep is {}", e.get_r_deep());
        }

        // Lookups
        for lookup in cs.lookups.iter() {
            let evaluate_lc = |ev: &mut Evaluator<_>, expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| ev.add_expression(expr))
                    .collect::<Vec<_>>();
                let mut lc = parts[0];
                for part in parts.iter().skip(1) {
                    lc = ev.add_calculation(Calculation::LcTheta(lc, *part));
                }
                lc
            };
            // Input coset
            let compressed_input_coset = evaluate_lc(&mut ev, &lookup.input_expressions);
            // table coset
            let compressed_table_coset = evaluate_lc(&mut ev, &lookup.table_expressions);
            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            let right_gamma = ev.add_calculation(Calculation::AddGamma(compressed_table_coset));
            ev.lookup_results
                .push(Calculation::LcBeta(compressed_input_coset, right_gamma));
        }

        // Lookups in GPU
        for lookup in cs.lookups.iter() {
            let evaluate_lc = |expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| LookupProveExpression::Expression(ProveExpression::from_expr(expr)))
                    .collect::<Vec<_>>();
                let mut lc = parts[0].clone();
                for part in parts.into_iter().skip(1) {
                    lc = LookupProveExpression::LcTheta(Box::new(lc), Box::new(part));
                }
                lc
            };
            // Input coset
            let compressed_input_coset = evaluate_lc(&lookup.input_expressions);
            // table coset
            let compressed_table_coset = evaluate_lc(&lookup.table_expressions);
            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            let right_gamma = LookupProveExpression::AddGamma(Box::new(compressed_table_coset));
            ev.gpu_lookup_expr.push(LookupProveExpression::LcBeta(
                Box::new(compressed_input_coset),
                Box::new(right_gamma),
            ));
        }

        ev.gpu_gates_expr = es;
        ev
    }

    /// Adds a rotation
    fn add_rotation(&mut self, rotation: &Rotation) -> usize {
        let position = self.rotations.iter().position(|&c| c == rotation.0);
        match position {
            Some(pos) => pos,
            None => {
                self.rotations.push(rotation.0);
                self.rotations.len() - 1
            }
        }
    }

    /// Adds a constant
    fn add_constant(&mut self, constant: &C::ScalarExt) -> ValueSource {
        let position = self.constants.iter().position(|&c| c == *constant);
        ValueSource::Constant(match position {
            Some(pos) => pos,
            None => {
                self.constants.push(*constant);
                self.constants.len() - 1
            }
        })
    }

    /// Adds a calculation.
    /// Currently does the simplest thing possible: just stores the
    /// resulting value so the result can be reused  when that calculation
    /// is done multiple times.
    fn add_calculation(&mut self, calculation: Calculation) -> ValueSource {
        let position = self
            .calculations
            .iter()
            .position(|c| c.calculation == calculation);
        match position {
            Some(pos) => {
                self.calculations[pos].counter += 1;
                ValueSource::Intermediate(pos)
            }
            None => {
                self.calculations.push(CalculationInfo {
                    counter: 1,
                    calculation,
                });
                ValueSource::Intermediate(self.calculations.len() - 1)
            }
        }
    }

    /// Generates an optimized evaluation for the expression
    fn add_expression(&mut self, expr: &Expression<C::ScalarExt>) -> ValueSource {
        match expr {
            Expression::Constant(scalar) => self.add_constant(scalar),
            Expression::Selector(_selector) => unreachable!(),
            Expression::Fixed {
                query_index: _,
                column_index,
                rotation,
            } => {
                let rot_idx = self.add_rotation(rotation);
                self.add_calculation(Calculation::Store(ValueSource::Fixed(
                    *column_index,
                    rot_idx,
                )))
            }
            Expression::Advice {
                query_index: _,
                column_index,
                rotation,
            } => {
                let rot_idx = self.add_rotation(rotation);
                self.add_calculation(Calculation::Store(ValueSource::Advice(
                    *column_index,
                    rot_idx,
                )))
            }
            Expression::Instance {
                query_index: _,
                column_index,
                rotation,
            } => {
                let rot_idx = self.add_rotation(rotation);
                self.add_calculation(Calculation::Store(ValueSource::Instance(
                    *column_index,
                    rot_idx,
                )))
            }
            Expression::Negated(a) => match **a {
                Expression::Constant(scalar) => self.add_constant(&-scalar),
                _ => {
                    let result_a = self.add_expression(a);
                    match result_a {
                        ValueSource::Constant(0) => result_a,
                        _ => self.add_calculation(Calculation::Negate(result_a)),
                    }
                }
            },
            Expression::Sum(a, b) => {
                // Undo subtraction stored as a + (-b) in expressions
                match &**b {
                    Expression::Negated(b_int) => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b_int);
                        if result_a == ValueSource::Constant(0) {
                            result_b
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else {
                            self.add_calculation(Calculation::Sub(result_a, result_b))
                        }
                    }
                    _ => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b);
                        if result_a == ValueSource::Constant(0) {
                            result_b
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else if result_a <= result_b {
                            self.add_calculation(Calculation::Add(result_a, result_b))
                        } else {
                            self.add_calculation(Calculation::Add(result_b, result_a))
                        }
                    }
                }
            }
            Expression::Product(a, b) => {
                let result_a = self.add_expression(a);
                let result_b = self.add_expression(b);
                if result_a == ValueSource::Constant(0) || result_b == ValueSource::Constant(0) {
                    ValueSource::Constant(0)
                } else if result_a == ValueSource::Constant(1) {
                    result_b
                } else if result_b == ValueSource::Constant(1) {
                    result_a
                } else if result_a <= result_b {
                    self.add_calculation(Calculation::Mul(result_a, result_b))
                } else {
                    self.add_calculation(Calculation::Mul(result_b, result_a))
                }
            }
            Expression::Scaled(a, f) => {
                if *f == C::ScalarExt::zero() {
                    ValueSource::Constant(0)
                } else if *f == C::ScalarExt::one() {
                    self.add_expression(a)
                } else {
                    let cst = self.add_constant(f);
                    let result_a = self.add_expression(a);
                    self.add_calculation(Calculation::Mul(result_a, cst))
                }
            }
        }
    }

    /// Evaluate h poly
    #[cfg(not(feature = "cuda"))]
    pub(in crate::plonk) fn evaluate_h(
        &self,
        pk: &ProvingKey<C>,
        advice: Vec<&Vec<Polynomial<C::ScalarExt, ExtendedLagrangeCoeff>>>,
        instance: Vec<&Vec<Polynomial<C::ScalarExt, ExtendedLagrangeCoeff>>>,
        y: C::ScalarExt,
        beta: C::ScalarExt,
        gamma: C::ScalarExt,
        theta: C::ScalarExt,
        lookups: &[Vec<lookup::prover::Committed<C>>],
        permutations: &[permutation::prover::Committed<C>],
    ) -> Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> {
        let domain = &pk.vk.domain;
        let size = domain.extended_len();
        let rot_scale = 1 << (domain.extended_k() - domain.k());
        let fixed = &pk.fixed_cosets[..];
        let extended_omega = domain.get_extended_omega();
        let num_lookups = pk.vk.cs.lookups.len();
        let isize = size as i32;
        let one = C::ScalarExt::one();
        let l0 = &pk.l0;
        let l_last = &pk.l_last;
        let l_active_row = &pk.l_active_row;
        let p = &pk.vk.cs.permutation;

        let mut values = domain.empty_extended();
        let mut lookup_values = vec![C::Scalar::zero(); size * num_lookups];

        // Core expression evaluations
        let num_threads = multicore::current_num_threads();
        let mut table_values_box = ThreadBox::wrap(&mut lookup_values);

        for (((advice, instance), lookups), permutation) in advice
            .iter()
            .zip(instance.iter())
            .zip(lookups.iter())
            .zip(permutations.iter())
        {
            let timer = ark_std::start_timer!(|| "expressions");
            multicore::scope(|scope| {
                let chunk_size = (size + num_threads - 1) / num_threads;
                for (thread_idx, values) in values.chunks_mut(chunk_size).enumerate() {
                    let start = thread_idx * chunk_size;
                    scope.spawn(move |_| {
                        let table_values = table_values_box.unwrap();
                        let mut rotations = vec![0usize; self.rotations.len()];
                        let mut intermediates: Vec<C::ScalarExt> =
                            vec![C::ScalarExt::zero(); self.calculations.len()];
                        for (i, value) in values.iter_mut().enumerate() {
                            let idx = start + i;

                            // All rotation index values
                            for (rot_idx, rot) in self.rotations.iter().enumerate() {
                                rotations[rot_idx] = get_rotation_idx(idx, *rot, rot_scale, isize);
                            }

                            // All calculations, with cached intermediate results
                            for (i_idx, calc) in self.calculations.iter().enumerate() {
                                intermediates[i_idx] = calc.calculation.evaluate(
                                    &rotations,
                                    &self.constants,
                                    &intermediates,
                                    fixed,
                                    advice,
                                    instance,
                                    &beta,
                                    &gamma,
                                    &theta,
                                );
                            }

                            // Accumulate value parts
                            for value_part in self.value_parts.iter() {
                                *value = *value * y
                                    + value_part.get(
                                        &rotations,
                                        &self.constants,
                                        &intermediates,
                                        fixed,
                                        advice,
                                        instance,
                                    );
                            }

                            // Values required for the lookups
                            for (t, table_result) in self.lookup_results.iter().enumerate() {
                                table_values[t * size + idx] = table_result.evaluate(
                                    &rotations,
                                    &self.constants,
                                    &intermediates,
                                    fixed,
                                    advice,
                                    instance,
                                    &beta,
                                    &gamma,
                                    &theta,
                                );
                            }
                        }
                    });
                }
            });
            end_timer!(timer);

            let timer = ark_std::start_timer!(|| "permutations");
            // Permutations
            let sets = &permutation.sets;
            if !sets.is_empty() {
                let blinding_factors = pk.vk.cs.blinding_factors();
                let last_rotation = Rotation(-((blinding_factors + 1) as i32));
                let chunk_len = pk.vk.cs.degree() - 2;
                let delta_start = beta * &C::Scalar::ZETA;

                let first_set = sets.first().unwrap();
                let last_set = sets.last().unwrap();

                // Permutation constraints
                parallelize(&mut values, |values, start| {
                    let mut beta_term = extended_omega.pow_vartime(&[start as u64, 0, 0, 0]);
                    for (i, value) in values.iter_mut().enumerate() {
                        let idx = start + i;
                        let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                        let r_last = get_rotation_idx(idx, last_rotation.0, rot_scale, isize);

                        // Enforce only for the first set.
                        // l_0(X) * (1 - z_0(X)) = 0
                        *value = *value * y
                            + ((one - first_set.permutation_product_coset[idx]) * l0[idx]);
                        // Enforce only for the last set.
                        // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
                        *value = *value * y
                            + ((last_set.permutation_product_coset[idx]
                                * last_set.permutation_product_coset[idx]
                                - last_set.permutation_product_coset[idx])
                                * l_last[idx]);
                        // Except for the first set, enforce.
                        // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
                        for (set_idx, set) in sets.iter().enumerate() {
                            if set_idx != 0 {
                                *value = *value * y
                                    + ((set.permutation_product_coset[idx]
                                        - permutation.sets[set_idx - 1].permutation_product_coset
                                            [r_last])
                                        * l0[idx]);
                            }
                        }
                        // And for all the sets we enforce:
                        // (1 - (l_last(X) + l_blind(X))) * (
                        //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
                        // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
                        // )
                        let mut current_delta = delta_start * beta_term;
                        for ((set, columns), cosets) in sets
                            .iter()
                            .zip(p.columns.chunks(chunk_len))
                            .zip(pk.permutation.cosets.chunks(chunk_len))
                        {
                            let mut left = set.permutation_product_coset[r_next];
                            for (values, permutation) in columns
                                .iter()
                                .map(|&column| match column.column_type() {
                                    Any::Advice => &advice[column.index()],
                                    Any::Fixed => &fixed[column.index()],
                                    Any::Instance => &instance[column.index()],
                                })
                                .zip(cosets.iter())
                            {
                                left *= values[idx] + beta * permutation[idx] + gamma;
                            }

                            let mut right = set.permutation_product_coset[idx];
                            for values in columns.iter().map(|&column| match column.column_type() {
                                Any::Advice => &advice[column.index()],
                                Any::Fixed => &fixed[column.index()],
                                Any::Instance => &instance[column.index()],
                            }) {
                                right *= values[idx] + current_delta + gamma;
                                current_delta *= &C::Scalar::DELTA;
                            }

                            *value = *value * y + ((left - right) * l_active_row[idx]);
                        }
                        beta_term *= &extended_omega;
                    }
                });
            }
            end_timer!(timer);

            let timer = ark_std::start_timer!(|| "eval_h_lookups");

            for (lookup_idx, lookup) in lookups.iter().enumerate() {
                // Lookup constraints
                let table = &lookup_values[lookup_idx * size..(lookup_idx + 1) * size];
                // Polynomials required for this lookup.
                // Calculated here so these only have to be kept in memory for the short time
                // they are actually needed.
                let product_coset = pk.vk.domain.coeff_to_extended(lookup.product_poly.clone());
                let permuted_input_coset = pk
                    .vk
                    .domain
                    .coeff_to_extended(lookup.permuted_input_poly.clone());
                let permuted_table_coset = pk
                    .vk
                    .domain
                    .coeff_to_extended(lookup.permuted_table_poly.clone());

                parallelize(&mut values, |values, start| {
                    for (i, value) in values.iter_mut().enumerate() {
                        let idx = start + i;

                        let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                        let r_prev = get_rotation_idx(idx, -1, rot_scale, isize);

                        let a_minus_s = permuted_input_coset[idx] - permuted_table_coset[idx];
                        // l_0(X) * (1 - z(X)) = 0
                        *value = *value * y + ((one - product_coset[idx]) * l0[idx]);
                        // l_last(X) * (z(X)^2 - z(X)) = 0
                        *value = *value * y
                            + ((product_coset[idx] * product_coset[idx] - product_coset[idx])
                                * l_last[idx]);
                        // (1 - (l_last(X) + l_blind(X))) * (
                        //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
                        //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
                        //          (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
                        // ) = 0

                        *value = *value * y
                            + ((product_coset[r_next]
                                * (permuted_input_coset[idx] + beta)
                                * (permuted_table_coset[idx] + gamma)
                                - product_coset[idx] * table[idx])
                                * l_active_row[idx]);

                        // Check that the first values in the permuted input expression and permuted
                        // fixed expression are the same.
                        // l_0(X) * (a'(X) - s'(X)) = 0
                        *value = *value * y + (a_minus_s * l0[idx]);

                        // Check that each value in the permuted lookup input expression is either
                        // equal to the value above it, or the value at the same index in the
                        // permuted table expression.
                        // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
                        *value = *value * y
                            + (a_minus_s
                                * (permuted_input_coset[idx] - permuted_input_coset[r_prev])
                                * l_active_row[idx]);
                    }
                });
            }

            end_timer!(timer);
        }

        values
    }

    #[cfg(feature = "cuda")]
    pub(in crate::plonk) fn evaluate_h(
        &self,
        pk: &ProvingKey<C>,
        advice_poly: Vec<&Vec<Polynomial<C::ScalarExt, Coeff>>>,
        instance_poly: Vec<&Vec<Polynomial<C::ScalarExt, Coeff>>>,
        y: C::ScalarExt,
        beta: C::ScalarExt,
        gamma: C::ScalarExt,
        theta: C::ScalarExt,
        lookups: &[Vec<lookup::prover::Committed<C>>],
        permutations: &[permutation::prover::Committed<C>],
    ) -> Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> {
        use ec_gpu_gen::{fft::FftKernel, rust_gpu_tools::Device, rust_gpu_tools::LocalBuffer};
        use ff::PrimeField;
        use group::ff::Field;
        use pairing::bn256::Fr;
        use rayon::{
            prelude::{
                IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
                ParallelIterator,
            },
            slice::ParallelSlice,
        };
        use std::{collections::LinkedList, marker::PhantomData};
        use crate::arithmetic::acquire_gpu;
        use crate::arithmetic::release_gpu;

        use crate::plonk::evaluation_gpu::{do_extended_fft, gen_do_extended_fft};

        assert!(advice_poly.len() == 1);
        let timer = start_timer!(|| "expressions gpu eval");

        let mut values = pk
            .ev
            .gpu_gates_expr
            .par_iter()
            .map(|x| {
                 let gpu_idx = acquire_gpu();
                 let r = x.eval_gpu(gpu_idx, pk, &advice_poly[0], &instance_poly[0], y);
                 release_gpu(gpu_idx);
                 r
            })
            .collect::<Vec<_>>()
            .into_iter()
            .reduce(|acc, x| acc + &x)
            .unwrap();

        end_timer!(timer);

        let domain = &pk.vk.domain;
        let size = domain.extended_len();
        let rot_scale = 1 << (domain.extended_k() - domain.k());
        let fixed = &pk.fixed_polys[..];
        let extended_omega = domain.get_extended_omega();
        let l0 = &pk.l0;
        let l_last = &pk.l_last;
        let l_active_row = &pk.l_active_row;
        let p = &pk.vk.cs.permutation;

        let timer = ark_std::start_timer!(|| "permutations");
        // Permutations
        let permutation = &permutations[0];
        let sets = &permutation.sets;
        if !sets.is_empty() {
            let blinding_factors = pk.vk.cs.blinding_factors();
            let last_rotation = Rotation(-((blinding_factors + 1) as i32));
            let chunk_len = pk.vk.cs.degree() - 2;
            let delta_start = beta * &C::Scalar::ZETA;

            let first_set = sets.first().unwrap();
            let last_set = sets.last().unwrap();

            let closures = ec_gpu_gen::rust_gpu_tools::program_closures!(
                |program, values: &mut [Fr]| -> ec_gpu_gen::EcResult<()> {
                    macro_rules! create_buffer_from {
                        ($x:ident, $y:expr) => {
                            let $x = program.create_buffer_from_slice($y)?;
                        };
                    }

                    let y_beta_gamma = vec![y, beta, gamma, C::Scalar::DELTA];

                    create_buffer_from!(values_buf, values);
                    create_buffer_from!(y_beta_gamma_buf, &y_beta_gamma[..]);
                    create_buffer_from!(l_active_row_buf, &l_active_row[..]);

                    let mut helper = gen_do_extended_fft(pk, program)?;
                    let mut _allocator = LinkedList::new();
                    let allocator = &mut _allocator;

                    let l0_buf = do_extended_fft(pk, program, &l0, allocator, &mut helper)?;

                    let l_last_buf = do_extended_fft(pk, program, &l_last, allocator, &mut helper)?;

                    let first_set_buf = do_extended_fft(
                        pk,
                        program,
                        &first_set.permutation_product_poly,
                        allocator,
                        &mut helper,
                    )?;
                    let last_set_buf = do_extended_fft(
                        pk,
                        program,
                        &last_set.permutation_product_poly,
                        allocator,
                        &mut helper,
                    )?;

                    let local_work_size = 128;
                    let global_work_size = size / local_work_size;
                    let kernel_name = format!("{}_eval_h_permutation_part1", "Bn256_Fr");
                    let kernel = program.create_kernel(
                        &kernel_name,
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;
                    kernel
                        .arg(&values_buf)
                        .arg(&first_set_buf)
                        .arg(&last_set_buf)
                        .arg(&l0_buf)
                        .arg(&l_last_buf)
                        .arg(&l_active_row_buf)
                        .arg(&y_beta_gamma_buf)
                        .run()?;

                    let mut prev_set_buf = first_set_buf;
                    let sets_len = sets.len();
                    for i in 1..sets_len - 1 {
                        let set = &sets[i];
                        let curr_set_buf = do_extended_fft(
                            pk,
                            program,
                            &set.permutation_product_poly,
                            allocator,
                            &mut helper,
                        )?;
                        let kernel_name = format!("{}_eval_h_permutation_part2", "Bn256_Fr");
                        let kernel = program.create_kernel(
                            &kernel_name,
                            global_work_size as usize,
                            local_work_size as usize,
                        )?;
                        kernel
                            .arg(&values_buf)
                            .arg(&curr_set_buf)
                            .arg(&prev_set_buf)
                            .arg(&l0_buf)
                            .arg(&y_beta_gamma_buf)
                            .arg(&(last_rotation.0 * rot_scale + size as i32))
                            .arg(&(size as u32))
                            .run()?;
                        allocator.push_back(prev_set_buf);
                        prev_set_buf = curr_set_buf;
                    }

                    if sets_len > 1 {
                        let curr_set_buf = last_set_buf;
                        let kernel_name = format!("{}_eval_h_permutation_part2", "Bn256_Fr");
                        let kernel = program.create_kernel(
                            &kernel_name,
                            global_work_size as usize,
                            local_work_size as usize,
                        )?;
                        kernel
                            .arg(&values_buf)
                            .arg(&curr_set_buf)
                            .arg(&prev_set_buf)
                            .arg(&l0_buf)
                            .arg(&y_beta_gamma_buf)
                            .arg(&(last_rotation.0 * rot_scale + size as i32))
                            .arg(&(size as u32))
                            .run()?;
                        allocator.push_back(prev_set_buf);
                    }

                    let left_buf = unsafe { program.create_buffer::<C::ScalarExt>(size)? };

                    let mut beta_term = vec![delta_start];
                    for _ in 1..size {
                        beta_term.push(*beta_term.last().unwrap() * &extended_omega);
                    }
                    create_buffer_from!(beta_term_buf, &beta_term);

                    for ((set, columns), polys) in sets
                        .iter()
                        .zip(p.columns.chunks(chunk_len))
                        .zip(pk.permutation.polys.chunks(chunk_len))
                    {
                        let curr_set_buf = do_extended_fft(
                            pk,
                            program,
                            &set.permutation_product_poly,
                            allocator,
                            &mut helper,
                        )?;
                        let kernel_name = format!("{}_eval_h_permutation_left_prepare", "Bn256_Fr");
                        let kernel = program.create_kernel(
                            &kernel_name,
                            global_work_size as usize,
                            local_work_size as usize,
                        )?;
                        kernel
                            .arg(&left_buf)
                            .arg(&curr_set_buf)
                            .arg(&(rot_scale))
                            .arg(&(size as u32))
                            .run()?;

                        let right_buf = curr_set_buf;

                        for (values, permutation) in columns
                            .iter()
                            .map(|&column| match column.column_type() {
                                Any::Advice => &advice_poly[0][column.index()],
                                Any::Fixed => &fixed[column.index()],
                                Any::Instance => &instance_poly[0][column.index()],
                            })
                            .zip(polys.iter())
                        {
                            let extended_data_buf =
                                do_extended_fft(pk, program, values, allocator, &mut helper)?;
                            let permutation_buf =
                                do_extended_fft(pk, program, permutation, allocator, &mut helper)?;

                            let kernel_name =
                                format!("{}_eval_h_permutation_left_right", "Bn256_Fr");
                            let kernel = program.create_kernel(
                                &kernel_name,
                                global_work_size as usize,
                                local_work_size as usize,
                            )?;

                            kernel
                                .arg(&left_buf)
                                .arg(&right_buf)
                                .arg(&extended_data_buf)
                                .arg(&permutation_buf)
                                .arg(&beta_term_buf)
                                .arg(&y_beta_gamma_buf)
                                .run()?;
                        }

                        let kernel_name = format!("{}_eval_h_permutation_part3", "Bn256_Fr");
                        let kernel = program.create_kernel(
                            &kernel_name,
                            global_work_size as usize,
                            local_work_size as usize,
                        )?;
                        kernel
                            .arg(&values_buf)
                            .arg(&left_buf)
                            .arg(&right_buf)
                            .arg(&l_active_row_buf)
                            .arg(&y_beta_gamma_buf)
                            .run()?;
                    }

                    program.read_into_buffer(&values_buf, values)?;
                    Ok(())
                }
            );

            let devices = Device::all();
            let gpu_idx = acquire_gpu();

            let device = devices[gpu_idx % devices.len()];
            let programs = vec![ec_gpu_gen::program!(device).unwrap()];
            let kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");
            kern.kernels[0]
                .program
                .run(closures, unsafe {
                    std::mem::transmute::<_, &mut [Fr]>(&mut values.values[..])
                })
                .unwrap();
            release_gpu(gpu_idx);
        }
        end_timer!(timer);

        let timer = ark_std::start_timer!(|| "eval_h_lookups");
        let lookups = &lookups[0];

        let n_gpu = *crate::plonk::N_GPU;
        let group_expr_len = (lookups.len() + n_gpu - 1) / n_gpu;

        if group_expr_len > 0 {
            values = lookups
                .par_chunks(group_expr_len)
                .enumerate()
                .map(|(group_idx, lookups)| {
                    // combine fft with eval_h_lookups:
                    // fft code: from ec-gpu lib.
                    let mut buffer = vec![];
                    buffer.resize(domain.extended_len(), C::Scalar::zero());

                    let gpu_idx = acquire_gpu();
                    let devices = Device::all();
                    let device = devices[gpu_idx % devices.len()];

                    let programs = vec![ec_gpu_gen::program!(device).unwrap()];
                    let kern =
                        FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");

                    let closures =
                        ec_gpu_gen::rust_gpu_tools::program_closures!(|program,
                                                                       args: (
                            &mut [Fr],
                            usize,
                            &[Committed<C>]
                        )|
                         -> ec_gpu_gen::EcResult<
                            (),
                        > {
                            let (input, group_idx, lookups) = args;
                            macro_rules! create_buffer_from {
                                ($x:ident, $y:expr) => {
                                    let $x = program.create_buffer_from_slice($y)?;
                                };
                            }

                            let y_beta_gamma = vec![y, beta, gamma];

                            let values_buf =
                                unsafe { program.create_buffer(domain.extended_len())? };
                            create_buffer_from!(y_beta_gamma_buf, &y_beta_gamma[..]);

                            let mut helper = gen_do_extended_fft(pk, program)?;
                            let mut allocator = LinkedList::new();

                            let l0_buf =
                                do_extended_fft(pk, program, &l0, &mut allocator, &mut helper)?;
                            let l_last_buf =
                                do_extended_fft(pk, program, &l_last, &mut allocator, &mut helper)?;
                            create_buffer_from!(l_active_row_buf, &l_active_row[..]);

                            let cache_size = std::env::var("HALO2_PROOF_GPU_EVAL_CACHE")
                                .unwrap_or("5".to_owned());
                            let cache_size = usize::from_str_radix(&cache_size, 10)
                                .expect("Invalid HALO2_PROOF_GPU_EVAL_CACHE");
                            let mut unit_cache = super::evaluation_gpu::Cache::new(cache_size);
                            for (lookup_idx, lookup) in lookups.iter().enumerate() {
                                let mut ys = vec![C::ScalarExt::one(), y];
                                let table_buf = pk.ev.gpu_lookup_expr
                                    [lookup_idx + group_idx * group_expr_len]
                                    ._eval_gpu(
                                        pk,
                                        program,
                                        &advice_poly[0],
                                        &instance_poly[0],
                                        &mut ys,
                                        beta,
                                        theta,
                                        gamma,
                                        &mut unit_cache,
                                        &mut allocator,
                                        &mut helper,
                                    )
                                    .unwrap()
                                    .0;

                                let permuted_input_coset_buf = do_extended_fft(
                                    pk,
                                    program,
                                    &lookup.permuted_input_poly,
                                    &mut allocator,
                                    &mut helper,
                                )?;
                                let permuted_table_coset_buf = do_extended_fft(
                                    pk,
                                    program,
                                    &lookup.permuted_table_poly,
                                    &mut allocator,
                                    &mut helper,
                                )?;
                                let product_coset_buf = do_extended_fft(
                                    pk,
                                    program,
                                    &lookup.product_poly,
                                    &mut allocator,
                                    &mut helper,
                                )?;

                                let local_work_size = 128;
                                let global_work_size = size / local_work_size;
                                let kernel_name = format!("{}_eval_h_lookups", "Bn256_Fr");
                                let kernel = program.create_kernel(
                                    &kernel_name,
                                    global_work_size as usize,
                                    local_work_size as usize,
                                )?;
                                kernel
                                    .arg(&values_buf)
                                    .arg(table_buf.as_ref())
                                    .arg(&permuted_input_coset_buf)
                                    .arg(&permuted_table_coset_buf)
                                    .arg(&product_coset_buf)
                                    .arg(&l0_buf)
                                    .arg(&l_last_buf)
                                    .arg(&l_active_row_buf)
                                    .arg(&y_beta_gamma_buf)
                                    .arg(&(rot_scale as u32))
                                    .arg(&(size as u32))
                                    .run()?;
                            }

                            program.read_into_buffer(&values_buf, input)?;
                            Ok(())
                        });

                    let mut tmp_value = pk.vk.domain.empty_extended();

                    kern.kernels[0]
                        .program
                        .run(closures, unsafe {
                            (
                                std::mem::transmute::<_, &mut [Fr]>(&mut tmp_value.values[..]),
                                group_idx,
                                lookups,
                            )
                        })
                        .unwrap();
                    release_gpu(gpu_idx);
                    (tmp_value, lookups.len())
                })
                .collect::<Vec<_>>()
                .iter()
                .fold(values, |acc, (x, len)| {
                    acc * y.pow_vartime([*len as u64 * 5, 0, 0, 0]) + x
                });
        }

        end_timer!(timer);

        values
    }
}

#[derive(Clone, Copy)]
struct ThreadBox<T>(*mut T, usize);
#[allow(unsafe_code)]
unsafe impl<T> Send for ThreadBox<T> {}
#[allow(unsafe_code)]
unsafe impl<T> Sync for ThreadBox<T> {}

/// Wraps a mutable slice so it can be passed into a thread without
/// hard to fix borrow checks caused by difficult data access patterns.
#[cfg(not(feature = "cuda"))]
impl<T> ThreadBox<T> {
    fn wrap(data: &mut [T]) -> Self {
        Self(data.as_mut_ptr(), data.len())
    }

    fn unwrap(&mut self) -> &mut [T] {
        #[allow(unsafe_code)]
        unsafe {
            slice::from_raw_parts_mut(self.0, self.1)
        }
    }
}

#[cfg(feature = "cuda")]
/// Simple evaluation of an expression
fn _evaluate_gpu<F: FieldExt, B: Basis>(
    program: &ec_gpu_gen::rust_gpu_tools::cuda::Program,
    expression: &Expression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    tmp_buffer: &mut ec_gpu_gen::rust_gpu_tools::cuda::Buffer<F>,
) -> ec_gpu_gen::EcResult<(ec_gpu_gen::rust_gpu_tools::cuda::Buffer<F>, i32)> {
    let local_work_size = 128;
    let global_work_size = size / local_work_size;
    match expression {
        Expression::Constant(c) => {
            let c = vec![*c];
            let c_buffer = program.create_buffer_from_slice(&c[..])?;
            let buffer = unsafe { program.create_buffer::<F>(size)? };
            let kernel_name = format!("{}_eval_constant", "Bn256_Fr");
            let kernel = program.create_kernel(
                &kernel_name,
                global_work_size as usize,
                local_work_size as usize,
            )?;
            kernel.arg(&buffer).arg(&c_buffer).run()?;
            Ok((buffer, 0))
        }
        Expression::Fixed {
            column_index,
            rotation,
            ..
        } => Ok((
            program.create_buffer_from_slice(&fixed[*column_index].values)?,
            rotation.0 * rot_scale,
        )),
        Expression::Advice {
            column_index,
            rotation,
            ..
        } => Ok((
            program.create_buffer_from_slice(&advice[*column_index].values)?,
            rotation.0 * rot_scale,
        )),
        Expression::Instance {
            column_index,
            rotation,
            ..
        } => Ok((
            program.create_buffer_from_slice(&instance[*column_index].values)?,
            rotation.0 * rot_scale,
        )),
        Expression::Negated(l) => {
            let c = vec![-F::one()];
            let c_buffer = program.create_buffer_from_slice(&c[..])?;
            let mut buffer = _evaluate_gpu(
                program, &l, size, rot_scale, fixed, advice, instance, tmp_buffer,
            )?;
            let kernel_name = format!("{}_eval_mul_c", "Bn256_Fr");
            let kernel = program.create_kernel(
                &kernel_name,
                global_work_size as usize,
                local_work_size as usize,
            )?;
            kernel
                .arg(tmp_buffer)
                .arg(&buffer.0)
                .arg(&buffer.1)
                .arg(&c_buffer)
                .arg(&(size as u32))
                .run()?;
            std::mem::swap(tmp_buffer, &mut buffer.0);
            Ok((buffer.0, 0))
        }
        Expression::Sum(l, r) => {
            let mut l = _evaluate_gpu(
                program, &l, size, rot_scale, fixed, advice, instance, tmp_buffer,
            )?;
            let r = _evaluate_gpu(
                program, &r, size, rot_scale, fixed, advice, instance, tmp_buffer,
            )?;
            let kernel_name = format!("{}_eval_sum", "Bn256_Fr");
            let kernel = program.create_kernel(
                &kernel_name,
                global_work_size as usize,
                local_work_size as usize,
            )?;
            kernel
                .arg(tmp_buffer)
                .arg(&l.0)
                .arg(&r.0)
                .arg(&l.1)
                .arg(&r.1)
                .arg(&(size as u32))
                .run()?;
            std::mem::swap(tmp_buffer, &mut l.0);
            Ok((l.0, 0))
        }
        Expression::Product(l, r) => {
            let mut l = _evaluate_gpu(
                program, &l, size, rot_scale, fixed, advice, instance, tmp_buffer,
            )?;
            let r = _evaluate_gpu(
                program, &r, size, rot_scale, fixed, advice, instance, tmp_buffer,
            )?;
            let kernel_name = format!("{}_eval_mul", "Bn256_Fr");
            let kernel = program.create_kernel(
                &kernel_name,
                global_work_size as usize,
                local_work_size as usize,
            )?;
            kernel
                .arg(tmp_buffer)
                .arg(&l.0)
                .arg(&r.0)
                .arg(&l.1)
                .arg(&r.1)
                .arg(&(size as u32))
                .run()?;
            std::mem::swap(tmp_buffer, &mut l.0);
            Ok((l.0, 0))
        }
        Expression::Selector(_) => unreachable!(),
        Expression::Scaled(l, r) => {
            let c = vec![*r];
            let c_buffer = program.create_buffer_from_slice(&c[..])?;
            let mut buffer = _evaluate_gpu(
                program, &l, size, rot_scale, fixed, advice, instance, tmp_buffer,
            )?;
            let kernel_name = format!("{}_eval_mul_c", "Bn256_Fr");
            let kernel = program.create_kernel(
                &kernel_name,
                global_work_size as usize,
                local_work_size as usize,
            )?;
            kernel
                .arg(tmp_buffer)
                .arg(&buffer.0)
                .arg(&buffer.1)
                .arg(&c_buffer)
                .arg(&(size as u32))
                .run()?;
            std::mem::swap(tmp_buffer, &mut buffer.0);
            Ok((buffer.0, 0))
        }
    }
}

#[cfg(feature = "cuda")]
/// Simple evaluation of an expression
fn evaluate_gpu<F: FieldExt, B: Basis>(
    expression: &[Expression<F>],
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    theta: F,
) -> Vec<F> {
    use crate::arithmetic::acquire_gpu;
    use crate::arithmetic::release_gpu;
    use crate::plonk::{GPU_COND_VAR, GPU_LOCK};
    use ec_gpu_gen::rust_gpu_tools::program_closures;
    use ec_gpu_gen::{
        fft::FftKernel, multiexp::SingleMultiexpKernel, rust_gpu_tools::Device, threadpool::Worker,
    };
    use group::Curve;
    use pairing::bn256::Fr;

    let mut values = vec![F::zero(); size];

    let gpu_idx = acquire_gpu();

    let closures = program_closures!(|program, input: &mut [F]| -> ec_gpu_gen::EcResult<()> {
        let local_work_size = 128;
        let global_work_size = size / local_work_size;

        let mut tmp_buffer = unsafe { program.create_buffer(size)? };
        let mut it = expression.iter();
        let mut buffer = _evaluate_gpu(
            program,
            it.next().unwrap(),
            size,
            rot_scale,
            fixed,
            advice,
            instance,
            &mut tmp_buffer,
        )?;

        let c = vec![theta];
        let c_buffer = program.create_buffer_from_slice(&c[..])?;

        for expression in it {
            let buffer_r = _evaluate_gpu(
                program,
                expression,
                size,
                rot_scale,
                fixed,
                advice,
                instance,
                &mut tmp_buffer,
            )?;
            let kernel_name = format!("{}_eval_lctheta", "Bn256_Fr");
            let kernel = program.create_kernel(
                &kernel_name,
                global_work_size as usize,
                local_work_size as usize,
            )?;
            kernel
                .arg(&tmp_buffer)
                .arg(&buffer.0)
                .arg(&buffer_r.0)
                .arg(&buffer.1)
                .arg(&buffer_r.1)
                .arg(&(size as u32))
                .arg(&c_buffer)
                .run()?;
            std::mem::swap(&mut tmp_buffer, &mut buffer.0);
            buffer.1 = 0;
        }

        if buffer.1 != 0 {
            panic!("Evaluate expression failed, find non-zero rotation of pure column");
        }

        program.read_into_buffer(&buffer.0, input)?;
        Ok(())
    });

    let devices = Device::all();
    let device = devices[gpu_idx % devices.len()];
    let program = ec_gpu_gen::program!(device).unwrap();
    program
        .run(closures, unsafe {
            std::mem::transmute::<_, &mut [F]>(&mut values[..])
        })
        .unwrap();

    release_gpu(gpu_idx);

    values
}

/// Simple evaluation of an expression
pub fn evaluate<F: FieldExt, B: Basis>(
    expression: &Expression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    _theta: F,
) -> Vec<F> {
    if let Some(idx) = expression.is_pure_fixed() {
        return fixed[idx].to_vec();
    }

    if let Some(idx) = expression.is_pure_advice() {
        return advice[idx].to_vec();
    }

    if let Some(idx) = expression.is_pure_instance() {
        return instance[idx].to_vec();
    }

    #[cfg(not(feature = "cuda"))]
    {
        let mut values = vec![F::zero(); size];
        let isize = size as i32;
        parallelize(&mut values, |values, start| {
            for (i, value) in values.iter_mut().enumerate() {
                let idx = start + i;
                *value = expression.evaluate(
                    &|scalar| scalar,
                    &|_| panic!("virtual selectors are removed during optimization"),
                    &|_, column_index, rotation| {
                        fixed[column_index][get_rotation_idx(idx, rotation.0, rot_scale, isize)]
                    },
                    &|_, column_index, rotation| {
                        advice[column_index][get_rotation_idx(idx, rotation.0, rot_scale, isize)]
                    },
                    &|_, column_index, rotation| {
                        instance[column_index][get_rotation_idx(idx, rotation.0, rot_scale, isize)]
                    },
                    &|a| -a,
                    &|a, b| a + &b,
                    &|a, b| {
                        let a = a();

                        if a == F::zero() {
                            a
                        } else {
                            a * b()
                        }
                    },
                    &|a, scalar| a * scalar,
                );
            }
        });
        return values;
    }

    #[cfg(feature = "cuda")]
    {
        return evaluate_gpu(
            &[expression.clone()],
            size,
            rot_scale,
            fixed,
            advice,
            instance,
            _theta,
        );
    }
}

/// Simple evaluation of an expression
pub fn evaluate_with_theta<F: FieldExt, B: Basis>(
    expressions: &[Expression<F>],
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    theta: F,
) -> Vec<F> {
    if expressions.len() == 1 {
        evaluate(
            &expressions[0],
            size,
            rot_scale,
            fixed,
            advice,
            instance,
            theta,
        )
    } else {
        #[cfg(not(feature = "cuda"))]
        {
            let mut values = vec![F::zero(); size];
            let isize = size as i32;
            parallelize(&mut values, |values, start| {
                for (i, value) in values.iter_mut().enumerate() {
                    let idx = start + i;
                    for expression in expressions {
                        *value = *value * theta;
                        *value += expression.evaluate(
                            &|scalar| scalar,
                            &|_| panic!("virtual selectors are removed during optimization"),
                            &|_, column_index, rotation| {
                                fixed[column_index]
                                    [get_rotation_idx(idx, rotation.0, rot_scale, isize)]
                            },
                            &|_, column_index, rotation| {
                                advice[column_index]
                                    [get_rotation_idx(idx, rotation.0, rot_scale, isize)]
                            },
                            &|_, column_index, rotation| {
                                instance[column_index]
                                    [get_rotation_idx(idx, rotation.0, rot_scale, isize)]
                            },
                            &|a| -a,
                            &|a, b| a + &b,
                            &|a, b| {
                                let a = a();

                                if a == F::zero() {
                                    a
                                } else {
                                    a * b()
                                }
                            },
                            &|a, scalar| a * scalar,
                        );
                    }
                }
            });
            return values;
        }

        #[cfg(feature = "cuda")]
        {
            return evaluate_gpu(expressions, size, rot_scale, fixed, advice, instance, theta);
        }
    }
}
