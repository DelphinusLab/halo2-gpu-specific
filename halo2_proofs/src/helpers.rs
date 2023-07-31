use std::io;
use pairing::arithmetic::{CurveAffine, FieldExt};
use num_derive::FromPrimitive;
use num;
use crate::{
    plonk::{
        VerifyingKey, permutation, Column, Any, self, ColumnType, VirtualCell, Fixed, ConstraintSystem, Advice, Instance, Expression, Gate
    },
    poly::{
        EvaluationDomain, Rotation
    }
};

pub(crate) trait CurveRead: CurveAffine {
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let mut compressed = Self::Repr::default();
        reader.read_exact(compressed.as_mut())?;
        Option::from(Self::from_bytes(&compressed))
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "invalid point encoding in proof"))
    }
}

impl<C: CurveAffine> CurveRead for C {}

pub fn write_vkey<C: CurveAffine, W: io::Write>(
    vkey: &VerifyingKey<C>,
    writer: &mut W,
) -> io::Result<()> {
    let j = (vkey.domain.get_quotient_poly_degree() + 1) as u32; // quotient_poly_degree is j-1
    let k = vkey.domain.k() as u32;
    writer.write(&mut j.to_le_bytes())?;
    writer.write(&mut k.to_le_bytes())?;
    write_cs::<C, W>(&vkey.cs, writer)?;

    //println!("write cs {:?}", &vkey.cs);
    vkey.write(writer)?;
    Ok(())
}

pub fn read_vkey<C: CurveAffine, R: io::Read>(
    reader: &mut R,
) -> io::Result<VerifyingKey<C>> {
    let j = read_u32(reader)?;
    let k = read_u32(reader)?;
    let domain: EvaluationDomain<C::Scalar> = EvaluationDomain::new(j, k);
    let cs = read_cs::<C, R>(reader)?;

    println!("read cs {:?}", cs);
    let fixed_commitments: Vec<_> = (0..cs.num_fixed_columns)
        .map(|_| C::read(reader))
        .collect::<Result<_, _>>()?;

    let permutation = permutation::VerifyingKey::read(reader, &cs.permutation)?;

    Ok(VerifyingKey {
        domain,
        cs,
        fixed_commitments,
        permutation,
    })
}

fn write_argument<W: std::io::Write>(column: &Column<Any>, writer: &mut W) -> std::io::Result<()> {
    writer.write(&mut (column.index as u32).to_le_bytes())?;
    writer.write(&mut (*column.column_type() as u32).to_le_bytes())?;
    Ok(())
}

fn read_argument<R: std::io::Read>(reader: &mut R) -> std::io::Result<Column<Any>> {
    let index = read_u32(reader)?;
    let typ = read_u32(reader)?;
    let typ = if typ == Any::Advice as u32 {
        Any::Advice
    } else if typ == Any::Instance as u32 {
        Any::Instance
    } else if typ == Any::Fixed as u32 {
        Any::Instance
    } else {
        unreachable!()
    };
    Ok(Column {
        index: index as usize,
        column_type: typ,
    })
}

fn write_arguments<W: std::io::Write>(
    columns: &Vec<Column<Any>>,
    writer: &mut W,
) -> std::io::Result<()> {
    writer.write(&mut (columns.len() as u32).to_le_bytes())?;
    for c in columns.iter() {
        write_argument(c, writer)?;
    }
    Ok(())
}

fn read_arguments<R: std::io::Read>(
    reader: &mut R,
) -> std::io::Result<plonk::permutation::Argument> {
    let len = read_u32(reader)?;
    let mut cols = vec![];
    for _ in 0..len {
        cols.push(read_argument(reader)?);
    }
    Ok(plonk::permutation::Argument { columns: cols })
}

fn write_column<T: ColumnType, W: std::io::Write>(
    column: &Column<T>,
    writer: &mut W,
) -> std::io::Result<()> {
    writer.write(&mut (column.index as u32).to_le_bytes())?;
    Ok(())
}

fn read_column<T: ColumnType, R: std::io::Read>(
    reader: &mut R,
    t: T,
) -> std::io::Result<Column<T>> {
    let index = read_u32(reader)? as usize;
    Ok(Column {
        index,
        column_type: t,
    })
}

fn write_queries<T: ColumnType, W: std::io::Write>(
    columns: &Vec<(Column<T>, Rotation)>,
    writer: &mut W,
) -> std::io::Result<()> {
    writer.write(&mut (columns.len() as u32).to_le_bytes())?;
    for (c, rotation) in columns.iter() {
        write_column(c, writer)?;
        writer.write(&mut (rotation.0 as u32).to_le_bytes())?;
    }
    Ok(())
}

fn write_virtual_cells <W: std::io::Write>(
    columns: &Vec<VirtualCell>,
    writer: &mut W,
) -> std::io::Result<()> {
    writer.write(&mut (columns.len() as u32).to_le_bytes())?;
    for cell in columns.iter() {
        write_argument(&cell.column, writer)?;
        writer.write(&mut (cell.rotation.0 as u32).to_le_bytes())?;
    }
    Ok(())
}

fn read_queries<T: ColumnType, R: std::io::Read>(
    reader: &mut R,
    t: T,
) -> std::io::Result<Vec<(Column<T>, Rotation)>> {
    let mut queries = vec![];
    let len = read_u32(reader)?;
    for _ in 0..len {
        let column = read_column(reader, t)?;
        let rotation = read_u32(reader)?;
        let rotation = Rotation(rotation as i32); //u32 to i32??
        queries.push((column, rotation))
    }
    Ok(queries)
}

fn read_virtual_cells<R: std::io::Read>(
    reader: &mut R,
) -> std::io::Result<Vec<VirtualCell>> {
    let mut vcells = vec![];
    let len = read_u32(reader)?;
    for _ in 0..len {
        let column = read_argument(reader)?;
        let rotation = read_u32(reader)?;
        let rotation = Rotation(rotation as i32); //u32 to i32??
        vcells.push(VirtualCell {column, rotation})
    }
    Ok(vcells)
}

fn write_fixed_column<W: std::io::Write>(
    column: &Column<Fixed>,
    writer: &mut W,
) -> std::io::Result<()> {
    writer.write(&mut (column.index as u32).to_le_bytes())?;
    Ok(())
}

fn read_fixed_column<R: std::io::Read>(reader: &mut R) -> std::io::Result<Column<Fixed>> {
    let index = read_u32(reader)?;
    Ok(Column::<Fixed>::new(index as usize, Fixed))
}

fn write_fixed_columns<W: std::io::Write>(
    columns: &Vec<Column<Fixed>>,
    writer: &mut W,
) -> std::io::Result<()> {
    writer.write(&mut (columns.len() as u32).to_le_bytes())?;
    for c in columns.iter() {
        write_fixed_column(c, writer)?;
    }
    Ok(())
}

fn read_fixed_columns<R: std::io::Read>(reader: &mut R) -> std::io::Result<Vec<Column<Fixed>>> {
    let len = read_u32(reader)?;
    let mut columns = vec![];
    for _ in 0..len {
        columns.push(read_fixed_column(reader)?);
    }
    Ok(columns)
}

fn write_cs<C: CurveAffine, W: io::Write>(
    cs: &ConstraintSystem<C::Scalar>,
    writer: &mut W,
) -> io::Result<()> {
    writer.write(&mut (cs.num_advice_columns as u32).to_le_bytes())?;
    writer.write(&mut (cs.num_instance_columns as u32).to_le_bytes())?;
    writer.write(&mut (cs.num_selectors as u32).to_le_bytes())?;
    writer.write(&mut (cs.num_fixed_columns as u32).to_le_bytes())?;
    writer.write(&mut (cs.num_advice_queries.len() as u32).to_le_bytes())?;
    for n in cs.num_advice_queries.iter() {
        writer.write(&mut (*n as u32).to_le_bytes())?;
    }
    write_fixed_columns(&cs.selector_map, writer)?;
    write_fixed_columns(&cs.constants, writer)?;
    write_queries::<Advice, W>(&cs.advice_queries, writer)?;
    write_queries::<Instance, W>(&cs.instance_queries, writer)?;
    write_queries::<Fixed, W>(&cs.fixed_queries, writer)?;
    write_arguments(&cs.permutation.columns, writer)?;
    writer.write(&(cs.lookups.len() as u32).to_le_bytes())?;
    for p in cs.lookups.iter() {
        write_expressions::<C, W>(&p.input_expressions, writer)?;
        write_expressions::<C, W>(&p.table_expressions, writer)?;
    }
    write_gates::<C, W>(&cs.gates, writer)?;
    Ok(())
}

fn read_u32<R: io::Read>(reader: &mut R) -> io::Result<u32> {
    let mut r = [0u8; 4];
    reader.read(&mut r)?;
    Ok(u32::from_le_bytes(r))
}

fn read_cs<C: CurveAffine, R: io::Read>(reader: &mut R) -> io::Result<ConstraintSystem<C::Scalar>> {
    let num_advice_columns = read_u32(reader)? as usize;
    let num_instance_columns = read_u32(reader)? as usize;
    let num_selectors = read_u32(reader)? as usize;
    let num_fixed_columns = read_u32(reader)? as usize;

    let num_advice_queries_len = read_u32(reader)?;
    let mut num_advice_queries = vec![];
    for _ in 0..num_advice_queries_len {
        num_advice_queries.push(read_u32(reader)? as usize);
    }

    let selector_map = read_fixed_columns(reader)?;
    let constants = read_fixed_columns(reader)?;

    let advice_queries = read_queries::<Advice, R>(reader, Advice)?;
    let instance_queries = read_queries::<Instance, R>(reader, Instance)?;
    let fixed_queries = read_queries::<Fixed, R>(reader, Fixed)?;
    let permutation = read_arguments(reader)?;

    let mut lookups = vec![];
    let nb_lookup = read_u32(reader)?;
    for _ in 0..nb_lookup {
        let input_expressions = read_expressions::<C, R>(reader)?;
        let table_expressions = read_expressions::<C, R>(reader)?;
        lookups.push(plonk::lookup::Argument {
            name: "",
            input_expressions,
            table_expressions,
        });
    }
    let gates = read_gates::<C, R>(reader)?;
    Ok(ConstraintSystem {
        num_fixed_columns,
        num_advice_columns,
        num_instance_columns,
        num_selectors,
        selector_map,
        gates,
        advice_queries,
        num_advice_queries,
        instance_queries,
        fixed_queries,
        permutation,
        lookups,
        constants,
        minimum_degree: None,
    })
}

fn write_expressions<C: CurveAffine, W: std::io::Write>(
    expressions: &Vec<Expression<C::Scalar>>,
    writer: &mut W,
) -> std::io::Result<()> {
    writer.write(&mut (expressions.len() as u32).to_le_bytes())?;
    for e in expressions.iter() {
        encode_expression(&e, writer)?;
    }
    Ok(())
}

fn read_expressions<C: CurveAffine, R: std::io::Read>(
    reader: &mut R,
) -> std::io::Result<Vec<Expression<C::Scalar>>> {
    let nb_expr = read_u32(reader)?;
    let mut exps = vec![];
    for _ in 0..nb_expr {
        exps.push(decode_expression(reader)?)
    }
    Ok(exps)
}

fn write_gates<C: CurveAffine, W: std::io::Write>(
    gates: &Vec<Gate<C::Scalar>>,
    writer: &mut W,
) -> std::io::Result<()> {
    writer.write(&mut (gates.len() as u32).to_le_bytes())?;
    for gate in gates.iter() {
        write_expressions::<C, W>(&gate.polys, writer)?;
        write_virtual_cells(&gate.queried_cells, writer)?;
    }
    Ok(())
}

fn read_gates<C: CurveAffine, R: std::io::Read>(
    reader: &mut R,
) -> std::io::Result<Vec<Gate<C::Scalar>>> {
    let nb_gates = read_u32(reader)?;
    let mut gates = vec![];
    for _ in 0..nb_gates {
        gates.push(Gate::new_with_polys_and_queries(read_expressions::<C, R>(reader)?, read_virtual_cells(reader)?));
    }
    Ok(gates)
}

#[derive(FromPrimitive)]
enum ExpressionCode {
    Constant = 0,
    Fixed,
    Advice,
    Instance,
    Negated,
    Sum,
    Product,
    Scaled,
}

fn expression_code<F: FieldExt>(e: &Expression<F>) -> ExpressionCode {
    match e {
        Expression::Constant(_) => ExpressionCode::Constant,
        Expression::Fixed {
            query_index: _,
            column_index: _,
            rotation: _,
        } => ExpressionCode::Fixed,
        Expression::Advice {
            query_index: _,
            column_index: _,
            rotation: _,
        } => ExpressionCode::Advice,
        Expression::Instance {
            query_index: _,
            column_index: _,
            rotation: _,
        } => ExpressionCode::Instance,
        Expression::Negated(_) => ExpressionCode::Negated,
        Expression::Sum(_, _) => ExpressionCode::Sum,
        Expression::Product(_, _) => ExpressionCode::Product,
        Expression::Scaled(_, _) => ExpressionCode::Scaled,
        Expression::Selector(_) => unreachable!(),
    }
}

fn decode_expression<F: FieldExt, R: io::Read>(reader: &mut R) -> io::Result<Expression<F>> {
    let code = read_u32(reader)?;
    match num::FromPrimitive::from_u32(code).unwrap() {
        ExpressionCode::Constant => {
            let scalar = F::read(reader)?;
            Ok(Expression::Constant(scalar))
        }
        ExpressionCode::Fixed => {
            let query_index = read_u32(reader)? as usize;
            let column_index = read_u32(reader)? as usize;
            let rotation = Rotation(read_u32(reader)? as i32);
            Ok(Expression::Fixed {
                query_index,
                column_index,
                rotation,
            })
        }
        ExpressionCode::Advice => {
            let query_index = read_u32(reader)? as usize;
            let column_index = read_u32(reader)? as usize;
            let rotation = Rotation(read_u32(reader)? as i32);
            Ok(Expression::Advice {
                query_index,
                column_index,
                rotation,
            })
        }
        ExpressionCode::Instance => {
            let query_index = read_u32(reader)? as usize;
            let column_index = read_u32(reader)? as usize;
            let rotation = Rotation(read_u32(reader)? as i32);
            Ok(Expression::Instance {
                query_index,
                column_index,
                rotation,
            })
        }
        ExpressionCode::Negated => Ok(Expression::Negated(Box::new(decode_expression(reader)?))),

        ExpressionCode::Sum => {
            let a = decode_expression(reader)?;
            let b = decode_expression(reader)?;
            Ok(Expression::Sum(Box::new(a), Box::new(b)))
        }

        ExpressionCode::Product => {
            let a = decode_expression(reader)?;
            let b = decode_expression(reader)?;
            Ok(Expression::Product(Box::new(a), Box::new(b)))
        }

        ExpressionCode::Scaled => {
            let a = decode_expression(reader)?;
            let f = F::read(reader)?;
            Ok(Expression::Scaled(Box::new(a), f))
        }
    }
}

fn encode_expression<F: FieldExt, W: io::Write>(
    e: &Expression<F>,
    writer: &mut W,
) -> io::Result<()> {
    writer.write(&mut (expression_code(e) as u32).to_le_bytes())?;
    match e {
        Expression::Constant(scalar) => {
            writer.write(&mut scalar.to_repr().as_ref())?;
            Ok(())
        }
        Expression::Fixed {
            query_index,
            column_index,
            rotation,
        } => {
            writer.write(&(*query_index as u32).to_le_bytes())?;
            writer.write(&(*column_index as u32).to_le_bytes())?;
            writer.write(&(rotation.0 as u32).to_le_bytes())?;
            Ok(())
        }
        Expression::Advice {
            query_index,
            column_index,
            rotation,
        } => {
            writer.write(&(*query_index as u32).to_le_bytes())?;
            writer.write(&(*column_index as u32).to_le_bytes())?;
            writer.write(&(rotation.0 as u32).to_le_bytes())?;
            Ok(())
        }
        Expression::Instance {
            query_index,
            column_index,
            rotation,
        } => {
            writer.write(&(*query_index as u32).to_le_bytes())?;
            writer.write(&(*column_index as u32).to_le_bytes())?;
            writer.write(&(rotation.0 as u32).to_le_bytes())?;
            Ok(())
        }
        Expression::Negated(a) => encode_expression(&a, writer),
        Expression::Sum(a, b) => {
            encode_expression(&a, writer)?;
            encode_expression(&b, writer)?;
            Ok(())
        }
        Expression::Product(a, b) => {
            encode_expression(&a, writer)?;
            encode_expression(&b, writer)?;
            Ok(())
        }
        Expression::Scaled(a, f) => {
            encode_expression(&a, writer)?;
            writer.write(&mut f.to_repr().as_ref())?;
            Ok(())
        }

        Expression::Selector(_) => unreachable!(),
    }
}
