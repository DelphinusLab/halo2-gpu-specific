use crate::plonk::circuit::FloorPlanner;
use crate::plonk::range_check::RangeCheckRel;
use crate::{
    arithmetic::{CurveAffine, FieldExt},
    plonk::{generate_pk_info, keygen_pk_from_info},
    poly::batch_invert_assigned,
};
use crate::{
    plonk::{
        self,
        permutation::{self, keygen::Assembly},
        Advice, Any, Assigned, Assignment, Circuit, Column, ColumnType, ConstraintSystem, Error,
        Expression, Fixed, Gate, Instance, ProvingKey, Selector, VerifyingKey, VirtualCell,
    },
    poly::{commitment::Params, EvaluationDomain, LagrangeCoeff, Polynomial, Rotation},
    transcript::EncodedChallenge,
};
use ff::Field;
use memmap::{MmapMut, MmapOptions};
use num;
use num::FromPrimitive;
use num_derive::FromPrimitive;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::io::Seek;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::{
    fs::{File, OpenOptions},
    io,
    ops::RangeTo,
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

pub trait Serializable: Clone {
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn fetch<R: io::Read>(reader: &mut R) -> io::Result<Self>;
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn store<W: io::Write>(&self, writer: &mut W) -> io::Result<()>;
}

pub trait ParaSerializable: Clone {
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn vec_fetch(fd: &mut File) -> io::Result<Self>;
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn vec_store(&self, fd: &mut File) -> io::Result<()>;
}

fn write_u32<W: io::Write>(v: u32, writer: &mut W) -> io::Result<()> {
    writer.write(&v.to_le_bytes())?;

    Ok(())
}

pub(crate) fn read_u32<R: io::Read>(reader: &mut R) -> io::Result<u32> {
    let mut r = [0u8; 4];
    reader.read(&mut r)?;
    Ok(u32::from_le_bytes(r))
}

impl Serializable for u32 {
    fn fetch<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let u = read_u32(reader)?;
        Ok(u)
    }
    fn store<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write(&mut (*self).to_le_bytes())?;
        Ok(())
    }
}

impl Serializable for String {
    fn fetch<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let len = read_u32(reader)?;
        let mut s = vec![0; len as usize];
        reader.read_exact(&mut s)?;
        let u = String::from_utf8(s.to_vec()).unwrap();
        Ok(u)
    }
    fn store<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let bytes = self.as_bytes();
        writer.write(&mut (bytes.len() as u32).to_le_bytes())?;
        writer.write(bytes)?;
        Ok(())
    }
}

impl Serializable for (String, u32) {
    fn fetch<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        Ok((String::fetch(reader)?, u32::fetch(reader)?))
    }
    fn store<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.0.store(writer)?;
        self.1.store(writer)?;
        Ok(())
    }
}

impl ParaSerializable for Vec<Vec<(u32, u32)>> {
    fn vec_fetch(fd: &mut File) -> io::Result<Self> {
        let columns = read_u32(fd)?;
        let mut offset = 0;
        let mut offsets = vec![];
        for _ in 0..columns {
            let l = read_u32(fd)?;
            offsets.push((offset, l));
            offset = offset + l;
        }
        let position = fd.stream_position()?;
        let res: Vec<Vec<(u32, u32)>> = (0..columns)
            .into_par_iter()
            .map(|i| {
                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(position + (offsets[i as usize].0 as u64 * 8))
                        .len(offsets[i as usize].1 as usize * 8)
                        .map(&fd)
                        .unwrap()
                };
                let s: &[(u32, u32)] = unsafe {
                    std::slice::from_raw_parts(
                        mmap.as_ptr() as *const (u32, u32),
                        offsets[i as usize].1 as usize,
                    )
                };
                let mut s2 = vec![];
                s2.extend_from_slice(s);
                s2
            })
            .collect();
        Ok(res)
    }

    fn vec_store(&self, fd: &mut File) -> io::Result<()> {
        let u = self.len() as u32;
        u.store(fd)?;
        let mut offset = 0;
        let mut offsets = vec![];
        for i in 0..u {
            let l = self[i as usize].len();
            offsets.push((offset, l));
            offset = offset + l;
            (l as u32).store(fd)?;
        }
        let position = fd.stream_position()?;
        fd.set_len(position + (offset as u64 * 8)).unwrap();
        self.into_par_iter().enumerate().for_each(|(i, s2)| {
            let mut mmap = unsafe {
                MmapOptions::new()
                    .offset(position + (offsets[i as usize].0 as u64 * 8))
                    .len(offsets[i as usize].1 as usize * 8)
                    .map_mut(&fd)
                    .unwrap()
            };
            let s: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    s2.as_ptr() as *const u8,
                    offsets[i as usize].1 as usize * 8,
                )
            };
            (&mut mmap).copy_from_slice(s);
        });
        Ok(())
    }
}

impl<B: Clone, F: FieldExt> Serializable for Polynomial<F, B> {
    fn fetch<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let u = read_u32(reader)?;
        let mut buf = vec![0u8; u as usize * 32];
        reader.read_exact(&mut buf)?;
        let s: &[F] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const F, u as usize) };
        Ok(Polynomial::new(s.to_vec()))
    }
    fn store<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let u = self.values.len() as u32;
        u.store(writer)?;
        let s: &[u8] = unsafe {
            std::slice::from_raw_parts(self.values.as_ptr() as *const u8, u as usize * 32)
        };
        writer.write(s)?;
        Ok(())
    }
}

impl<C: CurveAffine> Serializable for VerifyingKey<C> {
    /// deprecated, use CircuitData instead
    fn store<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let j = (self.domain.get_quotient_poly_degree() + 1) as u32; // quotient_poly_degree is j-1
        let k = self.domain.k() as u32;
        writer.write(&mut j.to_le_bytes())?;
        writer.write(&mut k.to_le_bytes())?;
        write_cs::<C, W>(&self.cs, writer)?;

        self.write(writer)?;

        Ok(())
    }

    /// deprecated, use CircuitData instead
    fn fetch<R: io::Read>(reader: &mut R) -> io::Result<VerifyingKey<C>> {
        let j = read_u32(reader)?;
        let k = read_u32(reader)?;
        let domain: EvaluationDomain<C::Scalar> = EvaluationDomain::new(j, k);
        let cs = read_cs::<C, R>(reader)?;

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
}

impl<T: Serializable> Serializable for Vec<T> {
    fn store<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write(&mut (self.len() as u32).to_le_bytes())?;
        for c in self.iter() {
            c.store(writer)?;
        }
        Ok(())
    }
    fn fetch<R: io::Read>(reader: &mut R) -> io::Result<Vec<T>> {
        let len = read_u32(reader)?;
        let mut v = vec![];
        for _ in 0..len {
            v.push(T::fetch(reader)?);
        }
        Ok(v)
    }
}

impl Serializable for Column<Any> {
    fn store<W: io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write(&mut (self.index as u32).to_le_bytes())?;
        writer.write(&mut (*self.column_type() as u32).to_le_bytes())?;
        Ok(())
    }

    fn fetch<R: io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let index = read_u32(reader)?;
        let typ = read_u32(reader)?;
        let typ = if typ == Any::Advice as u32 {
            Any::Advice
        } else if typ == Any::Instance as u32 {
            Any::Instance
        } else if typ == Any::Fixed as u32 {
            Any::Fixed
        } else {
            unreachable!()
        };
        Ok(Column {
            index: index as usize,
            column_type: typ,
        })
    }
}

fn write_arguments<W: std::io::Write>(
    columns: &Vec<Column<Any>>,
    writer: &mut W,
) -> std::io::Result<()> {
    writer.write(&mut (columns.len() as u32).to_le_bytes())?;
    for c in columns.iter() {
        c.store(writer)?
    }
    Ok(())
}

fn read_arguments<R: std::io::Read>(
    reader: &mut R,
) -> std::io::Result<plonk::permutation::Argument> {
    let len = read_u32(reader)?;
    let mut cols = vec![];
    for _ in 0..len {
        cols.push(Column::<Any>::fetch(reader)?);
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

fn write_virtual_cells<W: std::io::Write>(
    columns: &Vec<VirtualCell>,
    writer: &mut W,
) -> std::io::Result<()> {
    writer.write(&mut (columns.len() as u32).to_le_bytes())?;
    for cell in columns.iter() {
        cell.column.store(writer)?;
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

fn read_virtual_cells<R: std::io::Read>(reader: &mut R) -> std::io::Result<Vec<VirtualCell>> {
    let mut vcells = vec![];
    let len = read_u32(reader)?;
    for _ in 0..len {
        let column = Column::<Any>::fetch(reader)?;
        let rotation = read_u32(reader)?;
        let rotation = Rotation(rotation as i32); //u32 to i32??
        vcells.push(VirtualCell { column, rotation })
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

pub(crate) fn write_cs<C: CurveAffine, W: io::Write>(
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
        p.input_expressions.store(writer)?;
        p.table_expressions.store(writer)?;
    }
    writer.write(&(cs.shuffles.0.len() as u32).to_le_bytes())?;
    for p in cs.shuffles.0.iter() {
        p.input_expressions.store(writer)?;
        p.shuffle_expressions.store(writer)?;
    }

    write_u32(cs.range_check.0.len() as u32, writer)?;
    for argument in cs.range_check.0.iter() {
        write_column(&argument.origin, writer)?;
        write_column(&argument.sort, writer)?;
        write_u32(argument.min.0 as u32, writer)?;
        write_u32(argument.max.0 as u32, writer)?;
        write_u32(argument.step.0 as u32, writer)?;
    }

    cs.named_advices.store(writer)?;
    write_gates::<C, W>(&cs.gates, writer)?;
    Ok(())
}

pub(crate) fn read_cs<C: CurveAffine, R: io::Read>(
    reader: &mut R,
) -> io::Result<ConstraintSystem<C::Scalar>> {
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
        let input_expressions = Vec::<Expression<C::Scalar>>::fetch(reader)?;
        let table_expressions = Vec::<Expression<C::Scalar>>::fetch(reader)?;
        lookups.push(plonk::lookup::Argument {
            name: "",
            input_expressions,
            table_expressions,
        });
    }
    let mut shuffles = plonk::shuffle::Argument(vec![]);
    let nb_shuffle = read_u32(reader)?;
    for _ in 0..nb_shuffle {
        let input_expressions = Vec::<Expression<C::Scalar>>::fetch(reader)?;
        let shuffle_expressions = Vec::<Expression<C::Scalar>>::fetch(reader)?;
        shuffles.0.push(plonk::shuffle::ArgumentElement {
            name: "",
            input_expressions,
            shuffle_expressions,
        });
    }

    let mut range_check = plonk::range_check::Argument::new();
    let range_check_count = read_u32(reader)?;
    for _ in 0..range_check_count {
        let origin = read_column(reader, Advice)?;
        let sort = read_column(reader, Advice)?;
        let min = read_u32(reader)?;
        let max = read_u32(reader)?;
        let step = read_u32(reader)?;

        range_check.0.push(RangeCheckRel {
            origin,
            sort,
            min: (min, C::ScalarExt::from(min as u64)),
            max: (max, C::ScalarExt::from(max as u64)),
            step: (step, C::ScalarExt::from(step as u64)),
        })
    }

    let named_advices = Vec::fetch(reader)?;
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
        named_advices,
        permutation,
        lookups,
        shuffles,
        range_check,
        constants,
        minimum_degree: None,
    })
}

fn write_gates<C: CurveAffine, W: std::io::Write>(
    gates: &Vec<Gate<C::Scalar>>,
    writer: &mut W,
) -> std::io::Result<()> {
    writer.write(&mut (gates.len() as u32).to_le_bytes())?;
    for gate in gates.iter() {
        gate.polys.store(writer)?;
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
        gates.push(Gate::new_with_polys_and_queries(
            Vec::<Expression<C::Scalar>>::fetch(reader)?,
            read_virtual_cells(reader)?,
        ));
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

impl<F: FieldExt> Serializable for Expression<F> {
    fn fetch<R: io::Read>(reader: &mut R) -> io::Result<Expression<F>> {
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
            ExpressionCode::Negated => Ok(Expression::Negated(Box::new(Self::fetch(reader)?))),

            ExpressionCode::Sum => {
                let a = Self::fetch(reader)?;
                let b = Self::fetch(reader)?;
                Ok(Expression::Sum(Box::new(a), Box::new(b)))
            }

            ExpressionCode::Product => {
                let a = Self::fetch(reader)?;
                let b = Self::fetch(reader)?;
                Ok(Expression::Product(Box::new(a), Box::new(b)))
            }

            ExpressionCode::Scaled => {
                let a = Self::fetch(reader)?;
                let f = F::read(reader)?;
                Ok(Expression::Scaled(Box::new(a), f))
            }
        }
    }

    fn store<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write(&mut (expression_code(self) as u32).to_le_bytes())?;
        match self {
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
            Expression::Negated(a) => a.store(writer),
            Expression::Sum(a, b) => {
                a.store(writer)?;
                b.store(writer)?;
                Ok(())
            }
            Expression::Product(a, b) => {
                a.store(writer)?;
                b.store(writer)?;
                Ok(())
            }
            Expression::Scaled(a, f) => {
                a.store(writer)?;
                writer.write(&mut f.to_repr().as_ref())?;
                Ok(())
            }
            Expression::Selector(_) => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct AssignWitnessCollection<'a, C: CurveAffine> {
    pub k: u32,
    pub advice: Vec<Polynomial<Assigned<C::Scalar>, LagrangeCoeff>>,
    pub instances: &'a [&'a [C::Scalar]],
    pub usable_rows: RangeTo<usize>,
}

#[derive(Clone, Debug)]
pub struct AssignWitnessCollectionAssigner<'a, C: CurveAffine> {
    pub k: u32,
    pub advice: Arc<Mutex<Vec<Polynomial<Assigned<C::Scalar>, LagrangeCoeff>>>>,
    pub instances: &'a [&'a [C::Scalar]],
    pub usable_rows: RangeTo<usize>,
}

impl<'a, C: CurveAffine> Into<AssignWitnessCollection<'a, C>>
    for AssignWitnessCollectionAssigner<'a, C>
{
    fn into(self) -> AssignWitnessCollection<'a, C> {
        AssignWitnessCollection {
            k: self.k,
            advice: Arc::try_unwrap(self.advice).unwrap().into_inner().unwrap(),
            instances: self.instances,
            usable_rows: self.usable_rows,
        }
    }
}

impl<'a, C: CurveAffine> Assignment<C::Scalar> for AssignWitnessCollectionAssigner<'a, C> {
    fn is_in_prove_mode(&self) -> bool {
        true
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

    fn enable_selector<A, AR>(&self, _: A, _: &Selector, _: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about advice columns here

        Ok(())
    }

    fn query_instance(
        &self,
        column: Column<Instance>,
        row: usize,
    ) -> Result<Option<C::Scalar>, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.instances
            .get(column.index())
            .and_then(|column| column.get(row))
            .map(|v| Some(*v))
            .ok_or(Error::BoundsFailure)
    }

    fn assign_advice<V, VR, A, AR>(
        &self,
        _: A,
        column: Column<Advice>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Result<VR, Error>,
        VR: Into<Assigned<C::Scalar>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        let mut advice = self.advice.lock().unwrap();

        *advice
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to()?.into();

        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &self,
        _: A,
        _: Column<Fixed>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Result<VR, Error>,
        VR: Into<Assigned<C::Scalar>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about advice columns here

        Ok(())
    }

    fn copy(&self, _: Column<Any>, _: usize, _: Column<Any>, _: usize) -> Result<(), Error> {
        // We only care about advice columns here

        Ok(())
    }

    fn fill_from_row(
        &self,
        _: Column<Fixed>,
        _: usize,
        _: Option<Assigned<C::Scalar>>,
    ) -> Result<(), Error> {
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

impl<B: Clone, F: FieldExt> Serializable for Polynomial<Assigned<F>, B> {
    fn fetch<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        Ok(Polynomial::new(Vec::<Assigned<F>>::fetch(reader)?))
    }
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn store<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.values.store(writer)?;
        Ok(())
    }
}

impl ParaSerializable for Assembly {
    fn vec_fetch(fd: &mut File) -> io::Result<Self> {
        let assembly = Assembly {
            mapping: Vec::<Vec<(u32, u32)>>::vec_fetch(fd)?,
        };
        Ok(assembly)
    }
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn vec_store(&self, fd: &mut File) -> io::Result<()> {
        //self.columns.store(writer)?;
        self.mapping.vec_store(fd)?;

        Ok(())
    }
}

impl<'a, C: CurveAffine> AssignWitnessCollection<'a, C> {
    pub fn store_witness<ConcreteCircuit: Circuit<C::Scalar>>(
        params: &Params<C>,
        pk: &ProvingKey<C>,
        instances: &[&[C::Scalar]],
        unusable_rows_start: usize,
        circuit: &ConcreteCircuit,
        fd: &mut File,
    ) -> Result<(), Error> {
        use std::io::prelude::*;
        let mut meta = ConstraintSystem::default();
        let config = ConcreteCircuit::configure(&mut meta);

        let domain = &pk.get_vk().domain;
        let meta = &pk.get_vk().cs;
        let mut witness = AssignWitnessCollectionAssigner::<C> {
            k: params.k,
            advice: Arc::new(Mutex::new(vec![
                domain.empty_lagrange_assigned();
                meta.num_advice_columns
            ])),
            instances,
            // The prover will not be allowed to assign values to advice
            // cells that exist within inactive rows, which include some
            // number of blinding factors and an extra row for use in the
            // permutation argument.
            usable_rows: ..unusable_rows_start,
        };

        // Synthesize the circuit to obtain the witness and other information.
        ConcreteCircuit::FloorPlanner::synthesize(
            &mut witness,
            circuit,
            config.clone(),
            meta.constants.clone(),
        )?;

        let witness: AssignWitnessCollection<_> = witness.into();

        let bundlesize = params.k + 5;
        let advice = batch_invert_assigned(witness.advice);
        fd.set_len(4 + (1u64 << bundlesize)).unwrap();
        fd.write(&(advice.len() as u32).to_le_bytes())?;
        fd.set_len(4 + ((advice.len() as u64) << bundlesize))
            .unwrap();
        {
            advice.into_par_iter().enumerate().for_each(|(i, s2)| {
                let mut mmap = unsafe {
                    MmapOptions::new()
                        .offset(4 + ((i as u64) << bundlesize))
                        .len(1 << bundlesize)
                        .map_mut(&fd)
                        .unwrap()
                };
                let s: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        s2.as_ptr() as *const C::Scalar as *const u8,
                        1 << bundlesize,
                    )
                };
                (&mut mmap).copy_from_slice(s);
            });
        }
        println!("witness stored!");

        //witness.advice.store(writer)?;
        Ok(())
    }

    pub fn fetch_witness(
        params: &Params<C>,
        fd: &mut File,
    ) -> Result<Vec<Polynomial<C::Scalar, LagrangeCoeff>>, Error> {
        let len = read_u32(fd)?;
        let bundlesize = params.k + 5;
        let advice: Vec<Polynomial<_, LagrangeCoeff>> = (0..len)
            .into_par_iter()
            .map(|i| {
                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(4 + ((i as u64) << bundlesize))
                        .len(1 << bundlesize)
                        .map(&fd)
                        .unwrap()
                };
                let s: &[C::Scalar] = unsafe {
                    std::slice::from_raw_parts(mmap.as_ptr() as *const C::Scalar, 1 << params.k)
                };
                let mut s2 = vec![];
                s2.extend_from_slice(s);
                Polynomial::new(s2)
            })
            .collect();

        Ok(advice)
    }
}

#[derive(FromPrimitive)]
enum AssignedCode {
    Zero = 0,
    Trivial,
    Rational,
}

fn assigned_code<F: FieldExt>(e: &Assigned<F>) -> AssignedCode {
    match e {
        Assigned::Zero => AssignedCode::Zero,
        Assigned::Trivial(_) => AssignedCode::Trivial,
        Assigned::Rational(_, _) => AssignedCode::Rational,
    }
}

impl<F: FieldExt> Serializable for Assigned<F> {
    fn fetch<R: io::Read>(reader: &mut R) -> io::Result<Assigned<F>> {
        let code = read_u32(reader)?;
        match num::FromPrimitive::from_u32(code).unwrap() {
            AssignedCode::Zero => Ok(Assigned::Zero),
            AssignedCode::Trivial => {
                let scalar = F::read(reader)?;
                Ok(Assigned::Trivial(scalar))
            }
            AssignedCode::Rational => {
                let p = F::read(reader)?;
                let q = F::read(reader)?;
                Ok(Assigned::Rational(p, q))
            }
        }
    }

    fn store<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write(&mut (assigned_code(self) as u32).to_le_bytes())?;
        match self {
            Assigned::Zero => Ok(()),
            Assigned::Trivial(f) => {
                writer.write(&mut f.to_repr().as_ref())?;
                Ok(())
            }
            Assigned::Rational(p, q) => {
                writer.write(&mut p.to_repr().as_ref())?;
                writer.write(&mut q.to_repr().as_ref())?;
                Ok(())
            }
        }
    }
}

#[deprecated = "use CircuitData::new instead"]
pub fn store_pk_info<C: CurveAffine, ConcreteCircuit>(
    params: &Params<C>,
    vk: &VerifyingKey<C>,
    circuit: &ConcreteCircuit,
    //writer: &mut W,
    fd: &mut File,
) -> io::Result<()>
where
    ConcreteCircuit: Circuit<C::Scalar>,
{
    use ark_std::{end_timer, start_timer};
    let timer = start_timer!(|| "test generate_pk_info ...");
    let (fixed, permutation) = generate_pk_info(params, vk, circuit).unwrap();
    end_timer!(timer);
    let timer = start_timer!(|| "test store fixed ...");
    fixed.store(fd)?;
    end_timer!(timer);
    let timer = start_timer!(|| "test store permutation ...");
    permutation.vec_store(fd)?;
    end_timer!(timer);
    Ok(())
}

#[deprecated = "use CircuitData::read and CircuitData::into_proving_key instead"]
pub fn fetch_pk_info<C: CurveAffine>(
    params: &Params<C>,
    vk: &VerifyingKey<C>,
    reader: &mut File,
) -> io::Result<ProvingKey<C>> {
    use ark_std::{end_timer, start_timer};
    let timer = start_timer!(|| "test fetch fixed...");
    let fixed = Vec::fetch(reader)?;
    end_timer!(timer);
    let timer = start_timer!(|| "test fetch permutation ...");
    let permutation = Assembly::vec_fetch(reader)?;
    end_timer!(timer);
    let pkey = keygen_pk_from_info(params, vk, fixed, permutation).unwrap();
    Ok(pkey)
}
