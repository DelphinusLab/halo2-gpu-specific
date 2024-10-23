use super::circuit::Expression;
use ff::Field;

pub(crate) mod prover;
pub(crate) mod verifier;

#[derive(Clone, Debug)]
pub struct InputExpressionSet<F: Field>(pub Vec<Vec<Expression<F>>>);

#[derive(Clone, Debug)]
pub struct Argument<F: Field> {
    pub name: &'static str,
    pub table_expressions: Vec<Expression<F>>,
    // pub input_expressions: Vec<Vec<Vec<Expression<F>>>>,
    //only [table+[inputs[0]..inputs[n]]] case
    pub input_expressions_set: InputExpressionSet<F>,
    //one table map to multiple inputs, inputs are chunked by degree.
    // pub input_expressions_ext: Option<Vec<InputExpressionSet<F>>>,
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
            input_expressions_set: input_expressions.to_owned(),
            table_expressions: table_expressions.to_owned(),
        }
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

    pub fn chunks(&self, degree: usize) -> Vec<Argument<F>> {
        //reserve degree 2: (1 - (l_last + l_blind)) (z(wx)-z(x))
        assert!(degree > 2);
        let degree = degree - 2;
        //the degree covered table+max(inputs), so the input[0] is ok
        let mut args = vec![Argument {
            name: self.name,
            table_expressions: self.table_expressions.clone(),
            input_expressions_set: InputExpressionSet(vec![self.input_expression_set[0].clone()]),
        }];
        let table_degree = self
            .table_expressions
            .iter()
            .map(|e| e.degree())
            .max()
            .unwrap();

        for input in self.input_expression_set.iter().skip(1) {
            let new_input_degree = input.iter().map(|e| e.degree()).max().unwrap();
            let mut indicator = false;
            for arg in args.iter_mut() {
                let sum: usize = arg
                    .input_expressions_set
                    .0
                    .iter()
                    .map(|e| e.iter().map(|v| v.degree()).max().unwrap())
                    .sum();
                if table_degree + sum + new_input_degree <= degree {
                    arg.input_expressions_set.0.push(input.clone());
                    indicator = true;
                    break;
                }
            }
            if !indicator {
                args.push(Argument {
                    name: self.name,
                    table_expressions: self.table_expressions.clone(),
                    input_expressions_set: InputExpressionSet(vec![input.clone()]),
                })
            }
        }

        args
    }

    pub(crate) fn required_degree(&self) -> usize {
        for input in self.input_expression_set.iter() {
            assert_eq!(input.len(), self.table_expressions.len());
        }

        let mut input_degree = 1;
        for chunks in self.input_expression_set.iter() {
            for expr in chunks {
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
