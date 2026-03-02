// src/ad/optimizer.rs

use crate::ad::graph::{ADNode, BinaryOp, ReduceOp, Tape, UnaryOp};
use std::collections::HashMap;

#[derive(Hash, PartialEq, Eq, Clone)]
enum NodeKey {
    Binary(BinaryOp, usize, usize),
    Unary(UnaryOp, usize),
    TensorBinary(BinaryOp, usize, usize),
    TensorUnary(UnaryOp, usize),
    TensorReduce(ReduceOp, usize),
    TensorFusedMulAdd(usize, usize, usize),
}

impl NodeKey {
    fn from_node(node: &ADNode) -> Option<Self> {
        match node {
            ADNode::Binary { op, lhs, rhs, .. } => Some(NodeKey::Binary(*op, *lhs, *rhs)),
            ADNode::Unary { op, arg, .. } => Some(NodeKey::Unary(*op, *arg)),
            ADNode::TensorBinary { op, lhs, rhs, .. } => {
                Some(NodeKey::TensorBinary(*op, *lhs, *rhs))
            }
            ADNode::TensorUnary { op, arg, .. } => Some(NodeKey::TensorUnary(*op, *arg)),
            ADNode::TensorReduce { op, arg, .. } => Some(NodeKey::TensorReduce(*op, *arg)),
            ADNode::TensorFusedMulAdd { a, b, c, .. } => {
                Some(NodeKey::TensorFusedMulAdd(*a, *b, *c))
            }
            _ => None,
        }
    }
}

pub fn optimize_graph(tape: &Tape) {
    let mut nodes = tape.nodes.borrow_mut();
    let len = nodes.len();
    let mut remap: Vec<usize> = (0..len).collect();
    let mut seen: HashMap<NodeKey, usize> = HashMap::new();

    for i in 0..len {
        rewire_inputs(&mut nodes[i], &remap);

        if constant_fold(&mut nodes, i) {
            continue;
        }

        if algebraic_simplify(&nodes, &mut remap, i) {
            continue;
        }

        apply_fusion(&mut nodes, i);

        if let Some(key) = NodeKey::from_node(&nodes[i]) {
            if let Some(&canonical_idx) = seen.get(&key) {
                remap[i] = canonical_idx;
            } else {
                seen.insert(key, i);
            }
        }
    }
}

fn rewire_inputs(node: &mut ADNode, remap: &[usize]) {
    match node {
        ADNode::Binary { lhs, rhs, .. } => {
            *lhs = remap[*lhs];
            *rhs = remap[*rhs];
        }
        ADNode::Unary { arg, .. } => {
            *arg = remap[*arg];
        }
        ADNode::TensorBinary { lhs, rhs, .. } => {
            *lhs = remap[*lhs];
            *rhs = remap[*rhs];
        }
        ADNode::TensorUnary { arg, .. } => {
            *arg = remap[*arg];
        }
        ADNode::TensorReduce { arg, .. } => {
            *arg = remap[*arg];
        }
        ADNode::TensorFusedMulAdd { a, b, c, .. } => {
            *a = remap[*a];
            *b = remap[*b];
            *c = remap[*c];
        }
        _ => {}
    }
}

fn constant_fold(nodes: &mut [ADNode], idx: usize) -> bool {
    let folded_value: Option<f64> = match &nodes[idx] {
        ADNode::Binary { op, lhs, rhs, .. } => {
            let l = get_scalar_constant(nodes, *lhs);
            let r = get_scalar_constant(nodes, *rhs);
            l.and_then(|lv| r.and_then(|rv| apply_binary_op(*op, lv, rv)))
        }
        ADNode::Unary { op, arg, .. } => {
            get_scalar_constant(nodes, *arg).and_then(|v| apply_unary_op(*op, v))
        }
        _ => None,
    };

    if let Some(value) = folded_value {
        nodes[idx] = ADNode::Constant { value };
        true
    } else {
        false
    }
}

fn get_scalar_constant(nodes: &[ADNode], idx: usize) -> Option<f64> {
    match &nodes[idx] {
        ADNode::Constant { value } => Some(*value),
        _ => None,
    }
}

fn apply_binary_op(op: BinaryOp, l: f64, r: f64) -> Option<f64> {
    Some(match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div if r != 0.0 => l / r,
        BinaryOp::Pow => l.powf(r),
        BinaryOp::Atan2 => l.atan2(r),
        _ => return None,
    })
}

fn apply_unary_op(op: UnaryOp, v: f64) -> Option<f64> {
    Some(match op {
        UnaryOp::Neg => -v,
        UnaryOp::Exp => v.exp(),
        UnaryOp::Log if v > 0.0 => v.ln(),
        UnaryOp::Sin => v.sin(),
        UnaryOp::Cos => v.cos(),
        UnaryOp::Tan => v.tan(),
        UnaryOp::Sqrt if v >= 0.0 => v.sqrt(),
        UnaryOp::Abs => v.abs(),
        UnaryOp::Tanh => v.tanh(),
        UnaryOp::Sigmoid => 1.0 / (1.0 + (-v).exp()),
        UnaryOp::Softplus => (1.0 + v.exp()).ln(),
        _ => return None,
    })
}

fn algebraic_simplify(nodes: &[ADNode], remap: &mut [usize], idx: usize) -> bool {
    match &nodes[idx] {
        ADNode::Binary {
            op: BinaryOp::Add,
            lhs,
            rhs,
            ..
        } => {
            if is_zero(nodes, *lhs) {
                remap[idx] = *rhs;
                return true;
            }
            if is_zero(nodes, *rhs) {
                remap[idx] = *lhs;
                return true;
            }
        }
        ADNode::Binary {
            op: BinaryOp::Sub,
            lhs,
            rhs,
            ..
        } => {
            if is_zero(nodes, *rhs) {
                remap[idx] = *lhs;
                return true;
            }
        }
        ADNode::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
            ..
        } => {
            if is_zero(nodes, *lhs) {
                remap[idx] = *lhs;
                return true;
            }
            if is_zero(nodes, *rhs) {
                remap[idx] = *rhs;
                return true;
            }
            if is_one(nodes, *lhs) {
                remap[idx] = *rhs;
                return true;
            }
            if is_one(nodes, *rhs) {
                remap[idx] = *lhs;
                return true;
            }
        }
        ADNode::Binary {
            op: BinaryOp::Div,
            lhs,
            rhs,
            ..
        } => {
            if is_one(nodes, *rhs) {
                remap[idx] = *lhs;
                return true;
            }
        }
        ADNode::Binary {
            op: BinaryOp::Pow,
            lhs,
            rhs,
            ..
        } => {
            if is_zero(nodes, *lhs) {
                remap[idx] = *lhs;
                return true;
            }
            if is_one(nodes, *rhs) {
                remap[idx] = *lhs;
                return true;
            }
        }
        ADNode::Unary {
            op: UnaryOp::Neg,
            arg,
            ..
        } => {
            if let ADNode::Unary {
                op: UnaryOp::Neg,
                arg: inner_arg,
                ..
            } = &nodes[*arg]
            {
                remap[idx] = *inner_arg;
                return true;
            }
        }
        _ => {}
    }
    false
}

fn is_zero(nodes: &[ADNode], idx: usize) -> bool {
    matches!(&nodes[idx], ADNode::Constant { value } if *value == 0.0)
}

fn is_one(nodes: &[ADNode], idx: usize) -> bool {
    matches!(&nodes[idx], ADNode::Constant { value } if *value == 1.0)
}

fn apply_fusion(nodes: &mut [ADNode], idx: usize) {
    let fused_node = if let ADNode::TensorBinary {
        op: BinaryOp::Add,
        lhs,
        rhs,
        ..
    } = &nodes[idx]
    {
        if let Some((a, b)) = check_mul(nodes, *lhs) {
            Some(ADNode::TensorFusedMulAdd {
                a,
                b,
                c: *rhs,
                value: None,
            })
        } else if let Some((a, b)) = check_mul(nodes, *rhs) {
            Some(ADNode::TensorFusedMulAdd {
                a,
                b,
                c: *lhs,
                value: None,
            })
        } else {
            None
        }
    } else {
        None
    };

    if let Some(new_node) = fused_node {
        nodes[idx] = new_node;
    }
}

fn check_mul(nodes: &[ADNode], idx: usize) -> Option<(usize, usize)> {
    match &nodes[idx] {
        ADNode::TensorBinary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
            ..
        } => Some((*lhs, *rhs)),
        _ => None,
    }
}

pub fn mark_live_nodes(nodes: &[ADNode], output_nodes: &[usize]) -> Vec<bool> {
    let mut live = vec![false; nodes.len()];
    let mut worklist: Vec<usize> = output_nodes.to_vec();

    while let Some(idx) = worklist.pop() {
        if idx >= nodes.len() || live[idx] {
            continue;
        }
        live[idx] = true;

        match &nodes[idx] {
            ADNode::Binary { lhs, rhs, .. } => {
                worklist.push(*lhs);
                worklist.push(*rhs);
            }
            ADNode::Unary { arg, .. } => worklist.push(*arg),
            ADNode::TensorBinary { lhs, rhs, .. } => {
                worklist.push(*lhs);
                worklist.push(*rhs);
            }
            ADNode::TensorUnary { arg, .. } => worklist.push(*arg),
            ADNode::TensorReduce { arg, .. } => worklist.push(*arg),
            ADNode::TensorFusedMulAdd { a, b, c, .. } => {
                worklist.push(*a);
                worklist.push(*b);
                worklist.push(*c);
            }
            ADNode::CustomVjp { args, .. } => {
                for &arg in args {
                    worklist.push(arg);
                }
            }
            ADNode::Input { .. }
            | ADNode::Constant { .. }
            | ADNode::TensorInput { .. }
            | ADNode::TensorConstant { .. } => {}
        }
    }
    live
}

pub fn compute_ref_counts(nodes: &[ADNode]) -> Vec<usize> {
    let mut counts = vec![0usize; nodes.len()];

    for node in nodes {
        match node {
            ADNode::Binary { lhs, rhs, .. } => {
                counts[*lhs] += 1;
                counts[*rhs] += 1;
            }
            ADNode::Unary { arg, .. } => counts[*arg] += 1,
            ADNode::TensorBinary { lhs, rhs, .. } => {
                counts[*lhs] += 1;
                counts[*rhs] += 1;
            }
            ADNode::TensorUnary { arg, .. } => counts[*arg] += 1,
            ADNode::TensorReduce { arg, .. } => counts[*arg] += 1,
            ADNode::TensorFusedMulAdd { a, b, c, .. } => {
                counts[*a] += 1;
                counts[*b] += 1;
                counts[*c] += 1;
            }
            _ => {}
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ad::cpu_backend::NdarrayStorage;
    use crate::ad::create_tape;
    use crate::ad::tensor::ADTensor;
    use crate::ad::with_tape;

    #[test]
    fn test_tensor_fusion() {
        let tape_id = create_tape();
        let a = ADTensor::from_elem(&[2, 2], 2.0, tape_id);
        let b = ADTensor::from_elem(&[2, 2], 3.0, tape_id);
        let c = ADTensor::from_elem(&[2, 2], 1.0, tape_id);

        let d = (a * b) + c;

        with_tape(tape_id, |tape| {
            optimize_graph(tape);
            let nodes = tape.nodes.borrow();
            if let ADTensor::Dual { node_id, .. } = d {
                match &nodes[node_id] {
                    ADNode::TensorFusedMulAdd { .. } => {}
                    n => panic!("Expected FusedMulAdd, found {:?}", n),
                }
            }
        });

        let result = d.eval();
        assert_eq!(result.0[[0, 0]], 7.0);
    }

    #[test]
    fn test_cse() {
        let tape_id = create_tape();
        let a = ADTensor::from_elem(&[2, 2], 2.0, tape_id);
        let b = ADTensor::from_elem(&[2, 2], 2.0, tape_id);

        let x = a.clone() * b.clone();
        let y = a * b;

        let x_id = match &x {
            ADTensor::Dual { node_id, .. } => *node_id,
            _ => panic!(),
        };
        let y_id = match &y {
            ADTensor::Dual { node_id, .. } => *node_id,
            _ => panic!(),
        };

        let z = x + y;

        with_tape(tape_id, |tape| {
            optimize_graph(tape);
            let nodes = tape.nodes.borrow();

            let z_id = match &z {
                ADTensor::Dual { node_id, .. } => *node_id,
                _ => panic!(),
            };

            match &nodes[z_id] {
                ADNode::TensorBinary { lhs, rhs, .. } => {
                    assert_eq!(*lhs, *rhs, "Inputs to Z should be identical due to CSE");
                    assert_eq!(*lhs, x_id);
                }
                ADNode::TensorFusedMulAdd {
                    a: fa,
                    b: fb,
                    c: fc,
                    ..
                } => {
                    assert_eq!(*fc, x_id, "Fused c should be x_id (CSE remapped y->x)");
                    if let ADNode::TensorBinary { lhs, rhs, .. } = &nodes[x_id] {
                        assert_eq!(*fa, *lhs);
                        assert_eq!(*fb, *rhs);
                    }
                }
                _ => panic!(
                    "Expected TensorBinary or TensorFusedMulAdd for Z, found {:?}",
                    nodes[z_id]
                ),
            }
        });

        let res = z.eval();
        assert_eq!(res.0[[0, 0]], 8.0);
    }

    #[test]
    fn test_constant_folding() {
        use crate::ad::types::ADFloat;

        let tape_id = create_tape();

        with_tape(tape_id, |tape| {
            let c1_id = tape.push(ADNode::Constant { value: 2.0 });
            let c2_id = tape.push(ADNode::Constant { value: 3.0 });
            let add_id = tape.push(ADNode::Binary {
                op: BinaryOp::Add,
                lhs: c1_id,
                rhs: c2_id,
                value: 0.0,
            });

            optimize_graph(tape);

            let nodes = tape.nodes.borrow();
            match &nodes[add_id] {
                ADNode::Constant { value } => {
                    assert!((value - 5.0).abs() < 1e-10, "Expected 5.0, got {}", value);
                }
                n => panic!("Expected Constant, found {:?}", n),
            }
        });
    }

    #[test]
    fn test_algebraic_simplification_add_zero() {
        let tape_id = create_tape();

        with_tape(tape_id, |tape| {
            let x_id = tape.push(ADNode::Input { value: 42.0 });
            let zero_id = tape.push(ADNode::Constant { value: 0.0 });
            let add_id = tape.push(ADNode::Binary {
                op: BinaryOp::Add,
                lhs: x_id,
                rhs: zero_id,
                value: 0.0,
            });

            let result_id = tape.push(ADNode::Unary {
                op: UnaryOp::Neg,
                arg: add_id,
                value: 0.0,
            });

            optimize_graph(tape);

            let nodes = tape.nodes.borrow();
            match &nodes[result_id] {
                ADNode::Unary { arg, .. } => {
                    assert_eq!(*arg, x_id, "Expected arg to be x_id after simplification");
                }
                n => panic!("Expected Unary, found {:?}", n),
            }
        });
    }

    #[test]
    fn test_algebraic_simplification_mul_one() {
        let tape_id = create_tape();

        with_tape(tape_id, |tape| {
            let x_id = tape.push(ADNode::Input { value: 42.0 });
            let one_id = tape.push(ADNode::Constant { value: 1.0 });
            let mul_id = tape.push(ADNode::Binary {
                op: BinaryOp::Mul,
                lhs: x_id,
                rhs: one_id,
                value: 0.0,
            });

            let result_id = tape.push(ADNode::Unary {
                op: UnaryOp::Neg,
                arg: mul_id,
                value: 0.0,
            });

            optimize_graph(tape);

            let nodes = tape.nodes.borrow();
            match &nodes[result_id] {
                ADNode::Unary { arg, .. } => {
                    assert_eq!(*arg, x_id, "Expected arg to be x_id after simplification");
                }
                n => panic!("Expected Unary, found {:?}", n),
            }
        });
    }

    #[test]
    fn test_algebraic_simplification_mul_zero() {
        let tape_id = create_tape();

        with_tape(tape_id, |tape| {
            let x_id = tape.push(ADNode::Input { value: 42.0 });
            let zero_id = tape.push(ADNode::Constant { value: 0.0 });
            let mul_id = tape.push(ADNode::Binary {
                op: BinaryOp::Mul,
                lhs: x_id,
                rhs: zero_id,
                value: 0.0,
            });

            let result_id = tape.push(ADNode::Unary {
                op: UnaryOp::Neg,
                arg: mul_id,
                value: 0.0,
            });

            optimize_graph(tape);

            let nodes = tape.nodes.borrow();
            match &nodes[result_id] {
                ADNode::Constant { value } => {
                    assert!(
                        (*value == 0.0 || *value == -0.0),
                        "Expected 0.0 or -0.0 after constant folding, got {}",
                        value
                    );
                }
                ADNode::Unary { arg, .. } => {
                    assert_eq!(
                        *arg, zero_id,
                        "Expected arg to be zero_id after simplification"
                    );
                }
                n => panic!("Expected Constant or Unary, found {:?}", n),
            }
        });
    }

    #[test]
    fn test_double_negation() {
        let tape_id = create_tape();

        with_tape(tape_id, |tape| {
            let x_id = tape.push(ADNode::Input { value: 42.0 });
            let neg1_id = tape.push(ADNode::Unary {
                op: UnaryOp::Neg,
                arg: x_id,
                value: 0.0,
            });
            let neg2_id = tape.push(ADNode::Unary {
                op: UnaryOp::Neg,
                arg: neg1_id,
                value: 0.0,
            });

            let result_id = tape.push(ADNode::Unary {
                op: UnaryOp::Exp,
                arg: neg2_id,
                value: 0.0,
            });

            optimize_graph(tape);

            let nodes = tape.nodes.borrow();
            match &nodes[result_id] {
                ADNode::Unary { arg, .. } => {
                    assert_eq!(
                        *arg, x_id,
                        "Expected arg to be x_id after double negation elimination"
                    );
                }
                n => panic!("Expected Unary, found {:?}", n),
            }
        });
    }
}
