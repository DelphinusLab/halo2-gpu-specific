use super::circuit::Expression;
use ff::Field;

pub(crate) mod prover;
pub(crate) mod verifier;

#[derive(Clone, Debug)]
pub struct Argument<F: Field> {
    pub name: &'static str,
    pub input_expressions: Vec<Expression<F>>,
    pub shuffle_expressions: Vec<Expression<F>>,
}

impl<F: Field> Argument<F> {
    /// Constructs a new shuffle lookup argument.
    ///
    /// `shuffle_map` is a sequence of `(input, shuffle)` tuples.
    pub fn new(name: &'static str, shuffle_map: Vec<(Expression<F>, Expression<F>)>) -> Self {
        let (input_expressions, shuffle_expressions) = shuffle_map.into_iter().unzip();
        Argument {
            name,
            input_expressions,
            shuffle_expressions,
        }
    }

    pub(crate) fn required_degree(&self) -> usize {
        assert_eq!(self.input_expressions.len(), self.shuffle_expressions.len());
        // degree 2+input or 2+shuffle degree:
        // (1 - (l_last + l_blind)) (z(\omega X) (s(X) + \gamma) - z(X) (a(X) + \gamma))
        let mut input_degree = 1;
        for expr in self.input_expressions.iter() {
            input_degree = std::cmp::max(input_degree, expr.degree());
        }
        let mut shuffle_degree = 1;
        for expr in self.shuffle_expressions.iter() {
            shuffle_degree = std::cmp::max(shuffle_degree, expr.degree());
        }
        std::cmp::max(2 + shuffle_degree, 2 + input_degree)
    }
}
