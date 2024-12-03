use super::circuit::Expression;
use ff::Field;

pub(crate) mod prover;
pub(crate) mod verifier;

#[derive(Clone, Debug)]
pub struct InputExpressionSet<F: Field>(pub Vec<Vec<Expression<F>>>);

// logup: logarithmic derivative lookup feature support multiple inputs set map to one shared table
// limited by required degree, inputs were chunked to multiple sets.
#[derive(Clone, Debug)]
pub struct Argument<F: Field> {
    pub name: &'static str,
    pub table_expressions: Vec<Expression<F>>,
    // collect all inputs sets chunked by required degree
    // specially, the first inputs set combined with table, degree sum<= required degree
    // the extra inputs set excluding table and degree sum<= required degree
    // [[inputs],[inputs,inputs..],..]
    pub input_expressions_sets: Vec<InputExpressionSet<F>>,
}

impl<F: Field> Argument<F> {
    /// Constructs a new lookup argument.
    pub fn new(
        name: &'static str,
        table_expressions: &Vec<Expression<F>>,
        input_expressions: &InputExpressionSet<F>,
    ) -> Self {
        Argument {
            name,
            input_expressions_sets: vec![input_expressions.to_owned()],
            table_expressions: table_expressions.to_owned(),
        }
    }

    pub(crate) fn required_degree(&self) -> usize {
        let mut input_degree = 1;
        for inputs_set in self.input_expressions_sets.iter() {
            for inputs in inputs_set.0.iter() {
                assert_eq!(inputs.len(), self.table_expressions.len());
                for expr in inputs {
                    input_degree = std::cmp::max(input_degree, expr.degree());
                }
            }
        }
        let mut table_degree = 1;
        for expr in self.table_expressions.iter() {
            table_degree = std::cmp::max(table_degree, expr.degree());
        }

        std::cmp::max(
            // (1 - (l_last + l_blind)) τ(X) * f_i(X) * ((ϕ(gX) - ϕ(X))-( 1/f_i(X) - m(X) / τ(X)))
            4,
            2 + input_degree + table_degree,
        )
    }
}

#[derive(Clone, Debug)]
pub struct ArgumentTracer<F: Field> {
    pub name: &'static str,
    pub table_expressions: Vec<Expression<F>>,
    pub input_expression_set: Vec<Vec<Expression<F>>>,
}

impl<F: Field> ArgumentTracer<F> {
    pub fn new(
        name: &'static str,
        input_expressions: Vec<Expression<F>>,
        table_expressions: Vec<Expression<F>>,
    ) -> Self {
        ArgumentTracer {
            name,
            input_expression_set: vec![input_expressions],
            table_expressions,
        }
    }

    //group the argument tracer's input expressions by required global degree
    pub fn chunks(&self, global_degree: usize) -> Argument<F> {
        //reserve degree 2: (1 - (l_last + l_blind)) (z(wx)-z(x))
        assert!(global_degree > 2);
        let max_degree = global_degree - 2;
        //the degree covered table+max(inputs), degree(inputs[0])<=max(inputs) so the input[0] is ok
        let mut argument = Argument {
            name: self.name,
            table_expressions: self.table_expressions.clone(),
            input_expressions_sets: vec![InputExpressionSet(vec![
                self.input_expression_set[0].clone()
            ])],
        };
        let mut extra_input_expressions_sets: Vec<InputExpressionSet<F>> = vec![];
        let table_degree = self
            .table_expressions
            .iter()
            .map(|e| e.degree())
            .max()
            .unwrap();

        for input in self.input_expression_set.iter().skip(1) {
            let new_input_degree = input.iter().map(|e| e.degree()).max().unwrap();
            let mut indicator = false;

            // 1. table + input_expressions_set case
            let sum: usize = argument.input_expressions_sets[0]
                .0
                .iter()
                .map(|e| e.iter().map(|v| v.degree()).max().unwrap())
                .sum();
            if table_degree + sum + new_input_degree <= max_degree {
                argument.input_expressions_sets[0].0.push(input.clone());
                continue;
            }

            // 2. extra input_expressions_set case
            for set in extra_input_expressions_sets.iter_mut() {
                let sum: usize = set
                    .0
                    .iter()
                    .map(|e| e.iter().map(|v| v.degree()).max().unwrap())
                    .sum();
                // inputs set extension degree only care the inputs degree, no table degree
                if sum + new_input_degree <= max_degree {
                    set.0.push(input.clone());
                    indicator = true;
                    break;
                }
            }
            // 3. new InputExpressionSet to extra input_expressions_set
            if !indicator {
                extra_input_expressions_sets.push(InputExpressionSet(vec![input.clone()]));
            }
        }
        argument
            .input_expressions_sets
            .append(&mut extra_input_expressions_sets);

        // argument's chunked input_expressions count == tracer's input_expressions count
        assert_eq!(
            argument
                .input_expressions_sets
                .iter()
                .map(|set| set.0.len())
                .sum::<usize>(),
            self.input_expression_set.len()
        );

        // each single set's degree sum <= max_degree
        assert!(argument.input_expressions_sets.iter().all(|set| set
            .0
            .iter()
            .map(|e| e.iter().map(|v| v.degree()).max().unwrap())
            .sum::<usize>()
            <= max_degree));

        argument
    }

    pub(crate) fn required_degree(&self) -> usize {
        for input in self.input_expression_set.iter() {
            assert_eq!(input.len(), self.table_expressions.len());
        }

        let mut input_degree = 1;
        for inputs in self.input_expression_set.iter() {
            for expr in inputs {
                input_degree = std::cmp::max(input_degree, expr.degree());
            }
        }
        let mut table_degree = 1;
        for expr in self.table_expressions.iter() {
            table_degree = std::cmp::max(table_degree, expr.degree());
        }

        std::cmp::max(
            // (1 - (l_last + l_blind)) τ(X) * f_i(X) * ((ϕ(gX) - ϕ(X))-( 1/f_i(X) - m(X) / τ(X)))
            4,
            2 + input_degree + table_degree,
        )
    }
}

#[test]
fn test_chunks_normal() {
    use super::circuit::{ConstraintSystem, VirtualCells};
    use crate::poly::Rotation;
    use pairing::bn256::Fr;

    let mut cs = ConstraintSystem::<Fr>::default();
    let [input_0, input_1, lookup_0, _lookup_1] = [(); 4].map(|_| cs.advice_column());
    let [s_0, s_1, table_0, table_1] = [(); 4].map(|_| cs.fixed_column());

    cs.lookup_any("table1", |meta| {
        let input0 = meta.query_advice(input_0, Rotation::cur());
        let table0 = meta.query_fixed(table_0, Rotation::cur());
        [(input0, table0)].to_vec()
    });
    cs.lookup_any("table2", |meta| {
        let input1 = meta.query_advice(input_1, Rotation::cur());
        let table0 = meta.query_fixed(table_0, Rotation::cur());
        [(input1, table0)].to_vec()
    });

    //degree=4
    let mut cs = cs.chunk_lookups();
    assert_eq!(cs.lookups.len(), 1);
    assert_eq!(cs.lookups[0].table_expressions.len(), 1);
    assert_eq!(cs.lookups[0].input_expressions_sets.len(), 2);

    //degree=5
    cs.lookup_any("table3", |meta| {
        let input0 = meta.query_advice(input_1, Rotation::cur());
        let input1 = meta.query_advice(input_1, Rotation::cur());
        let table0 = meta.query_fixed(table_0, Rotation::cur());
        let s0 = meta.query_fixed(s_0, Rotation::cur());
        [(input1 * s0, table0.clone()), (input0, table0)].to_vec()
    });

    let mut cs = cs.chunk_lookups();
    assert_eq!(cs.lookups.len(), 2);
    assert_eq!(cs.lookups[0].table_expressions.len(), 1);
    assert_eq!(cs.lookups[0].input_expressions_sets.len(), 1);
    assert_eq!(cs.lookups[1].table_expressions.len(), 2);
    assert_eq!(cs.lookups[1].input_expressions_sets.len(), 1);
}

//test the big degree expression head
#[test]
fn test_chunks_order() {
    use super::circuit::{ConstraintSystem, VirtualCells};
    use crate::poly::Rotation;
    use pairing::bn256::Fr;

    let mut cs = ConstraintSystem::<Fr>::default();
    let [input_0, input_1, lookup_0, _lookup_1] = [(); 4].map(|_| cs.advice_column());
    let [s_0, s_1, table_0, table_1] = [(); 4].map(|_| cs.fixed_column());

    cs.lookup_any("table1", |meta| {
        let input0 = meta.query_advice(input_0, Rotation::cur());
        let table0 = meta.query_fixed(table_0, Rotation::cur());
        [(input0, table0)].to_vec()
    });

    //degree=5
    cs.lookup_any("table2", |meta| {
        let input1 = meta.query_advice(input_1, Rotation::cur());
        let table0 = meta.query_fixed(table_0, Rotation::cur());
        let s0 = meta.query_fixed(s_0, Rotation::cur());
        [(input1 * s0, table0)].to_vec()
    });

    cs.lookup_any("table3", |meta| {
        let input1 = meta.query_advice(input_1, Rotation::cur());
        let table0 = meta.query_fixed(table_0, Rotation::cur());
        [(input1, table0)].to_vec()
    });

    let mut cs = cs.chunk_lookups();
    assert_eq!(cs.lookups.len(), 1);
    assert_eq!(cs.lookups[0].table_expressions.len(), 1);
    assert_eq!(cs.lookups[0].input_expressions_sets.len(), 2);
}
