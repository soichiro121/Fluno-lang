// src/ad/tensor.rs

use crate::ad::backend::{TensorBackend, TensorStorage};
use crate::ad::cpu_backend::{CpuBackend, NdarrayStorage};
use crate::ad::graph::{ADNode, BinaryOp, ReduceOp, Tape};
use crate::ad::types::ADFloat;
use crate::ad::with_tape;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub enum ADTensor {
    Concrete(NdarrayStorage),
    Dual {
        tape_id: usize,
        node_id: usize,
        value: Option<NdarrayStorage>,
    },
}

impl ADTensor {
    pub fn new_input(value: NdarrayStorage, tape_id: usize) -> Self {
        let node = ADNode::TensorInput {
            value: value.clone(),
        };
        let node_id = with_tape(tape_id, |tape| tape.push(node));
        ADTensor::Dual {
            tape_id,
            node_id,
            value: Some(value),
        }
    }

    pub fn from_elem(shape: &[usize], value: f64, tape_id: usize) -> Self {
        let storage = CpuBackend::from_elem(shape, value);
        Self::new_input(storage, tape_id)
    }

    pub fn eval(&self) -> NdarrayStorage {
        match self {
            ADTensor::Concrete(v) => v.clone(),
            ADTensor::Dual {
                tape_id,
                node_id,
                value,
            } => {
                if let Some(v) = value {
                    return v.clone();
                }
                with_tape(*tape_id, |tape| evaluate_forward(tape, *node_id))
            }
        }
    }

    pub fn sum(&self) -> ADFloat {
        let val = self.eval();
        let sum_val = val.sum();

        match self {
            ADTensor::Concrete(_) => ADFloat::Concrete(sum_val),
            ADTensor::Dual {
                tape_id, node_id, ..
            } => {
                let id = with_tape(*tape_id, |tape| {
                    tape.push(ADNode::TensorReduce {
                        op: ReduceOp::Sum,
                        arg: *node_id,
                        value: sum_val,
                    })
                });
                ADFloat::Dual {
                    value: sum_val,
                    tape_id: *tape_id,
                    node_id: id,
                }
            }
        }
    }
}

fn push_tensor_binary(lhs: &ADTensor, rhs: &ADTensor, op: BinaryOp) -> ADTensor {
    match (lhs, rhs) {
        (
            ADTensor::Dual {
                tape_id: t1,
                node_id: n1,
                ..
            },
            ADTensor::Dual {
                tape_id: t2,
                node_id: n2,
                ..
            },
        ) => {
            assert_eq!(t1, t2, "Cannot operate on tensors from different tapes");
            let node_id = with_tape(*t1, |tape| {
                tape.push(ADNode::TensorBinary {
                    op,
                    lhs: *n1,
                    rhs: *n2,
                    value: None,
                })
            });
            ADTensor::Dual {
                tape_id: *t1,
                node_id,
                value: None,
            }
        }
        _ => panic!("Mixed Concrete/Dual or Cross-Tape tensor ops not fully implemented"),
    }
}

impl Add for ADTensor {
    type Output = ADTensor;
    fn add(self, rhs: Self) -> Self::Output {
        push_tensor_binary(&self, &rhs, BinaryOp::Add)
    }
}

impl Sub for ADTensor {
    type Output = ADTensor;
    fn sub(self, rhs: Self) -> Self::Output {
        push_tensor_binary(&self, &rhs, BinaryOp::Sub)
    }
}

impl Mul for ADTensor {
    type Output = ADTensor;
    fn mul(self, rhs: Self) -> Self::Output {
        push_tensor_binary(&self, &rhs, BinaryOp::Mul)
    }
}

impl Div for ADTensor {
    type Output = ADTensor;
    fn div(self, rhs: Self) -> Self::Output {
        push_tensor_binary(&self, &rhs, BinaryOp::Div)
    }
}

pub fn evaluate_forward(tape: &Tape, node_id: usize) -> NdarrayStorage {
    {
        let nodes = tape.nodes.borrow();
        if let Some(val) = check_value(&nodes[node_id]) {
            return val.clone();
        }
    }

    let op_info = {
        let nodes = tape.nodes.borrow();
        match &nodes[node_id] {
            ADNode::TensorBinary { op, lhs, rhs, .. } => Some((Some((*op, *lhs, *rhs)), None)),
            ADNode::TensorFusedMulAdd { a, b, c, .. } => Some((None, Some((*a, *b, *c)))),
            ADNode::TensorInput { value } => return value.clone(),
            _ => panic!("Eval: unsupported node type or not a tensor node"),
        }
    };

    let (binary_info, fused_info) = op_info.unwrap();

    if let Some((a, b, c)) = fused_info {
        return eval_fused_mul_add(tape, a, b, c);
    }

    let (op, lhs_id, rhs_id) = binary_info.unwrap();

    let lhs_val = evaluate_forward(tape, lhs_id);
    let rhs_val = evaluate_forward(tape, rhs_id);

    let result = match op {
        BinaryOp::Add => CpuBackend::add(&lhs_val, &rhs_val),
        BinaryOp::Sub => CpuBackend::sub(&lhs_val, &rhs_val),
        BinaryOp::Mul => CpuBackend::mul(&lhs_val, &rhs_val),
        BinaryOp::Div => CpuBackend::div(&lhs_val, &rhs_val),
        BinaryOp::MatMul => CpuBackend::matmul(&lhs_val, &rhs_val),
        _ => panic!("Eval: op not implemented"),
    };

    let mut nodes = tape.nodes.borrow_mut();
    match &mut nodes[node_id] {
        ADNode::TensorBinary { value, .. } => *value = Some(result.clone()),
        ADNode::TensorFusedMulAdd { value, .. } => *value = Some(result.clone()),
        _ => {}
    }

    result
}

fn eval_fused_mul_add(tape: &Tape, a: usize, b: usize, c: usize) -> NdarrayStorage {
    let val_a = evaluate_forward(tape, a);
    let val_b = evaluate_forward(tape, b);
    let val_c = evaluate_forward(tape, c);

    let ab = CpuBackend::mul(&val_a, &val_b);
    CpuBackend::add(&ab, &val_c)
}

fn check_value(node: &ADNode) -> Option<&NdarrayStorage> {
    match node {
        ADNode::TensorInput { value } | ADNode::TensorConstant { value } => Some(value),
        ADNode::TensorBinary { value, .. }
        | ADNode::TensorUnary { value, .. }
        | ADNode::TensorFusedMulAdd { value, .. } => value.as_ref(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ad::create_tape;

    #[test]
    fn test_lazy_tensor_eval() {
        let tape_id = create_tape();
        let a = ADTensor::from_elem(&[2, 2], 1.0, tape_id);
        let b = ADTensor::from_elem(&[2, 2], 2.0, tape_id);

        let c = a + b;

        match &c {
            ADTensor::Dual { value, .. } => assert!(value.is_none(), "Tensor should be lazy!"),
            _ => panic!("Expected Dual tensor"),
        }

        let result = c.eval();
        assert_eq!(result.0[[0, 0]], 3.0);

        if let ADTensor::Dual { node_id, .. } = c {
            crate::ad::with_tape(tape_id, |tape| {
                let nodes = tape.nodes.borrow();
                match &nodes[node_id] {
                    ADNode::TensorBinary { value, .. } => {
                        assert!(value.is_some(), "Node value in Tape should be cached")
                    }
                    _ => panic!("Wrong node type"),
                }
            });
        }
    }
}
