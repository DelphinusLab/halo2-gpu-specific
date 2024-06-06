use super::circuit::Expression;
use ff::Field;
use std::collections::BTreeMap;
pub(crate) mod prover;
pub(crate) mod verifier;

#[derive(Clone, Debug)]
pub struct Argument<F: Field>(pub Vec<ArgumentElement<F>>);

impl<F: Field> Argument<F> {
    pub(crate) fn new() -> Self {
        Argument(vec![])
    }

    pub fn group(&self, degree: usize) -> Vec<ArgumentGroup<F>> {
        assert!(degree > 2, "Invalid degree");
        //(1 - (l_last + l_blind)) * z(\omega X) has 2 degree
        let reserve_degree = 2;
        let mut degree_arguments: BTreeMap<usize, Vec<ArgumentElement<F>>> = BTreeMap::default();

        self.0.iter().for_each(|v| {
            degree_arguments
                .entry(v.degree())
                .or_default()
                .push(v.clone())
        });

        let mut degree_set = vec![];
        degree_arguments
            .iter()
            .for_each(|(k, v)| degree_set.extend(vec![*k; v.len()]));
        degree_set.reverse();

        let buckets = min_buckets(&degree_set, degree - reserve_degree);
        let mut group: Vec<ArgumentGroup<F>> = Vec::new();
        for bucket in buckets.into_iter() {
            let mut argument = vec![];
            bucket.into_iter().for_each(|k| {
                argument.push(degree_arguments.get_mut(&k).unwrap().pop().unwrap());
            });
            group.push(ArgumentGroup(argument));
        }
        degree_arguments
            .keys()
            .for_each(|k| assert!(degree_arguments.get(k).unwrap().is_empty()));
        group
    }
}

fn min_buckets(sets: &[usize], volume: usize) -> Vec<Vec<usize>> {
    let mut buckets: Vec<Vec<usize>> = Vec::new();

    for &num in sets {
        let mut placed = false;

        for bucket in buckets.iter_mut() {
            if bucket.iter().sum::<usize>() + num <= volume {
                bucket.push(num);
                placed = true;
                break;
            }
        }
        if !placed {
            buckets.push(vec![num]);
        }
    }
    buckets
}

#[derive(Clone, Debug)]
pub struct ArgumentGroup<F: Field>(pub Vec<ArgumentElement<F>>);

#[derive(Clone, Debug)]
pub struct ArgumentElement<F: Field> {
    pub name: &'static str,
    pub input_expressions: Vec<Expression<F>>,
    pub shuffle_expressions: Vec<Expression<F>>,
}

impl<F: Field> ArgumentElement<F> {
    /// Constructs a new shuffle lookup argument.
    /// `shuffle_map` is a sequence of `(input, shuffle)` tuples.
    pub fn new(name: &'static str, shuffle_map: Vec<(Expression<F>, Expression<F>)>) -> Self {
        let (input_expressions, shuffle_expressions) = shuffle_map.into_iter().unzip();
        ArgumentElement {
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
