use super::circuit::Expression;
use ff::Field;

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

        let (mut low, mut high): (Vec<ArgumentElement<F>>, Vec<ArgumentElement<F>>) =
            self.0.iter().cloned().partition(|s| s.degree() == 1);

        //get all degree list, and put the bigger degree in advance for algorithm
        let mut degree_set = high.iter().map(|s| s.degree()).collect::<Vec<usize>>();
        degree_set.extend(vec![1; low.len()]);
        let buckets = min_buckets(&degree_set, degree - reserve_degree);
        let mut group: Vec<ArgumentGroup<F>> = Vec::new();
        for bucket in buckets.into_iter() {
            let mut argument = vec![];
            bucket.into_iter().for_each(|num| {
                if num == 1 {
                    argument.push(low.pop().unwrap());
                } else {
                    let i = high.iter().position(|h| h.degree() == num).unwrap();
                    argument.push(high.remove(i));
                }
            });
            group.push(ArgumentGroup(argument));
        }
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
    ///
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
