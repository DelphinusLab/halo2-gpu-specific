use super::circuit::Expression;
use ff::Field;
use std::collections::BTreeMap;
pub(crate) mod prover;
pub(crate) mod verifier;

// group of shuffle ArgumentUnit
#[derive(Clone, Debug)]
pub struct Argument<F: Field>(pub Vec<ArgumentUnit<F>>);

impl<F: Field> Argument<F> {
    //get the degree sum to group's all elements
    pub(crate) fn degree_sum(&self) -> usize {
        self.0.iter().map(|arg| arg.degree()).sum::<usize>()
    }
}

#[derive(Clone, Debug)]
pub struct ArgumentUnit<F: Field> {
    pub name: &'static str,
    pub input_expressions: Vec<Expression<F>>,
    pub shuffle_expressions: Vec<Expression<F>>,
}

impl<F: Field> ArgumentUnit<F> {
    /// Constructs a new shuffle lookup argument.
    /// `shuffle_map` is a sequence of `(input, shuffle)` tuples.
    pub fn new(name: &'static str, shuffle_map: Vec<(Expression<F>, Expression<F>)>) -> Self {
        let (input_expressions, shuffle_expressions) = shuffle_map.into_iter().unzip();
        ArgumentUnit {
            name,
            input_expressions,
            shuffle_expressions,
        }
    }

    //get expressions max degree
    pub(crate) fn degree(&self) -> usize {
        assert_eq!(self.input_expressions.len(), self.shuffle_expressions.len());
        let mut input_degree = 1;
        for expr in self.input_expressions.iter() {
            input_degree = std::cmp::max(input_degree, expr.degree());
        }
        let mut shuffle_degree = 1;
        for expr in self.shuffle_expressions.iter() {
            shuffle_degree = std::cmp::max(shuffle_degree, expr.degree());
        }
        std::cmp::max(shuffle_degree, input_degree)
    }

    //get shuffle gate's max degree
    pub(crate) fn required_degree(&self) -> usize {
        assert_eq!(self.input_expressions.len(), self.shuffle_expressions.len());
        // degree 2+input or 2+shuffle degree:
        // (1 - (l_last + l_blind)) (z(\omega X) (s1(X) + \beta) - z(X) (a(X) + \beta))
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

// compact little degree shuffle argument to one group according to max_degree and reduce the final shuffle poly amount.
// (1 - (l_last + l_blind)) (z(\omega X)*(s1(X) + \beta)*(s2(X) + \beta^2).. - z(X)*(a1(X) + \beta)*(a2(X) + \beta^2)..)
pub(crate) fn chunk<F: Field>(
    tracer: &[ArgumentUnit<F>],
    global_degree: usize,
) -> Vec<Argument<F>> {
    assert!(tracer.len() > 0, "shuffle tracer is 0");
    assert!(global_degree > 2, "Invalid degree");
    //(1 - (l_last + l_blind)) * z(\omega X) has 2 degree
    let max_degree = global_degree - 2;
    let mut groups = vec![Argument(vec![tracer[0].clone()])];
    for arg in tracer.iter().skip(1) {
        let new_deg = arg.degree();
        let mut hit = false;
        for group in groups.iter_mut() {
            if group.degree_sum() + new_deg <= max_degree {
                group.0.push(arg.clone());
                hit = true;
                break;
            }
        }
        //not hit, create new group
        if !hit {
            groups.push(Argument(vec![arg.clone()]));
        }
    }
    assert_eq!(
        groups.iter().map(|group| group.0.len()).sum::<usize>(),
        tracer.len()
    );
    assert_eq!(
        groups.iter().all(|group| group.degree_sum() <= max_degree),
        true
    );
    groups
}
