// src/ad/backward.rs

use std::collections::HashMap;
use crate::ad::graph::{ADNode, BinaryOp, UnaryOp, Tape};
use crate::ad::types::ADGradient;
use crate::ad::cpu_backend::{NdarrayStorage, CpuBackend};
use crate::ad::backend::TensorBackend;

pub fn backward(tape: &Tape, output_node_id: usize) -> HashMap<usize, ADGradient> {
    ensure_tensor_values_computed(tape, output_node_id);
    
    let nodes = tape.nodes.borrow();
    let mut grads = HashMap::new();
    
    grads.insert(output_node_id, ADGradient::Scalar(1.0));
    
    for node_id in (0..=output_node_id).rev() {
        let grad = match grads.get(&node_id) {
            Some(g) => g.clone(),
            None => continue,
        };
        
        match &nodes[node_id] {
            ADNode::Input { .. } => {}
            ADNode::TensorInput { .. } => {}
            ADNode::Constant { .. } | ADNode::TensorConstant { .. } => {}
            
            ADNode::Unary { op, arg, value } => {
                if let ADGradient::Scalar(g) = grad {
                    let input_val = get_node_scalar(&nodes, *arg);
                    let grad_arg = compute_unary_grad(*op, input_val, *value, g);
                    add_grad(&mut grads, *arg, ADGradient::Scalar(grad_arg));
                }
            }
            ADNode::Binary { op, lhs, rhs, value } => {
                if let ADGradient::Scalar(g) = grad {
                    let lhs_val = get_node_scalar(&nodes, *lhs);
                    let rhs_val = get_node_scalar(&nodes, *rhs);
                    let (grad_lhs, grad_rhs) = compute_binary_grad(*op, lhs_val, rhs_val, *value, g);
                    
                    add_grad(&mut grads, *lhs, ADGradient::Scalar(grad_lhs));
                    add_grad(&mut grads, *rhs, ADGradient::Scalar(grad_rhs));
                }
            }
            
            ADNode::TensorBinary { op, lhs, rhs, value: _ } => {
                if let ADGradient::Tensor(g) = grad {
                    let lhs_val = get_node_tensor(&nodes, *lhs);
                    let rhs_val = get_node_tensor(&nodes, *rhs);
                    let (grad_lhs, grad_rhs) = compute_tensor_binary_grad(*op, lhs_val, rhs_val, &g);
                    
                    add_grad(&mut grads, *lhs, ADGradient::Tensor(grad_lhs));
                    add_grad(&mut grads, *rhs, ADGradient::Tensor(grad_rhs));
                }
            }
            ADNode::TensorUnary { op, arg, value: _ } => {
                if let ADGradient::Tensor(g) = grad {
                    let input_val = get_node_tensor(&nodes, *arg);
                    let output_val = get_node_tensor(&nodes, node_id);
                    
                    let grad_input = compute_tensor_unary_grad(*op, input_val, output_val, &g);
                    add_grad(&mut grads, *arg, ADGradient::Tensor(grad_input));
                }
            }
            ADNode::TensorReduce { op, arg, value: _ } => {
                if let ADGradient::Scalar(g) = grad {
                    let input_val = get_node_tensor(&nodes, *arg);
                    match op {
                        crate::ad::graph::ReduceOp::Sum => {
                            let grad_input = CpuBackend::from_elem(input_val.shape(), g);
                            add_grad(&mut grads, *arg, ADGradient::Tensor(grad_input));
                        }
                        crate::ad::graph::ReduceOp::Mean => {
                            let n = input_val.len() as f64;
                            let grad_elem = g / n;
                            let grad_input = CpuBackend::from_elem(input_val.shape(), grad_elem);
                            add_grad(&mut grads, *arg, ADGradient::Tensor(grad_input));
                        }
                        _ => panic!("Unsupported reduce op {:?}", op),
                    }
                }
            }
            ADNode::TensorFusedMulAdd { a, b, c, value: _ } => {
                if let ADGradient::Tensor(g) = grad {
                    let a_val = get_node_tensor(&nodes, *a);
                    let b_val = get_node_tensor(&nodes, *b);

                    let grad_a = CpuBackend::mul(b_val, &g);
                    let grad_b = CpuBackend::mul(a_val, &g);
                    let grad_c = g.clone();
                    
                    add_grad(&mut grads, *a, ADGradient::Tensor(grad_a));
                    add_grad(&mut grads, *b, ADGradient::Tensor(grad_b));
                    add_grad(&mut grads, *c, ADGradient::Tensor(grad_c));
                }
            }
            
            ADNode::CustomVjp { name, args, value } => {
                if let ADGradient::Scalar(g) = grad {
                    let input_vals: Vec<f64> = args.iter()
                        .map(|id| get_node_scalar(&nodes, *id))
                        .collect();
                    
                    if let Some(grads_out) = crate::ad::call_vjp(name, &input_vals, *value, g) {
                        for (i, &arg_id) in args.iter().enumerate() {
                            if i < grads_out.len() {
                                add_grad(&mut grads, arg_id, ADGradient::Scalar(grads_out[i]));
                            }
                        }
                    }
                }
            }
        }
    }
    
    grads
}

fn add_grad(grads: &mut HashMap<usize, ADGradient>, id: usize, update: ADGradient) {
    match grads.get_mut(&id) {
        Some(grad) => match (grad, update) {
            (ADGradient::Scalar(g), ADGradient::Scalar(u)) => *g += u,
            (ADGradient::Tensor(g), ADGradient::Tensor(u)) => *g = CpuBackend::add(g, &u),
            (g, u) => panic!("Gradient type mismatch at node {}: {:?} vs {:?}", id, g, u),
        },
        None => {
            grads.insert(id, update);
        }
    }
}

fn get_node_scalar(nodes: &[ADNode], id: usize) -> f64 {
    match &nodes[id] {
        ADNode::Input { value } | ADNode::Constant { value } |
        ADNode::Unary { value, .. } | ADNode::Binary { value, .. } |
        ADNode::TensorReduce { value, .. } => *value,
        _ => panic!("Expected scalar node at {}, found {:?}", id, nodes[id]),
    }
}

fn get_node_tensor(nodes: &[ADNode], id: usize) -> &NdarrayStorage {
    match &nodes[id] {
        ADNode::TensorInput { value } | ADNode::TensorConstant { value } => value,
        ADNode::TensorBinary { value, .. } | ADNode::TensorUnary { value, .. } | ADNode::TensorFusedMulAdd { value, .. } => {
            value.as_ref().expect("Internal Error: Tensor value not evaluated during backward pass")
        }
        _ => panic!("Expected tensor node at {}, found {:?}", id, nodes[id]),
    }
}

fn compute_unary_grad(op: UnaryOp, input_val: f64, output_val: f64, grad_output: f64) -> f64 {
    match op {
        UnaryOp::Neg => -grad_output,
        UnaryOp::Exp => output_val * grad_output,
        UnaryOp::Log => grad_output / input_val,
        UnaryOp::Sin => input_val.cos() * grad_output,
        UnaryOp::Cos => -input_val.sin() * grad_output,
        UnaryOp::Tan => {
            let c = input_val.cos();
            (1.0 / (c * c)) * grad_output
        }
        UnaryOp::Sqrt => {
            grad_output / (2.0 * output_val)
        }
        UnaryOp::Abs => {
            if input_val > 0.0 { grad_output }
            else if input_val < 0.0 { -grad_output }
            else { 0.0 }
        }
        UnaryOp::Tanh => {
            (1.0 - output_val * output_val) * grad_output
        }
        UnaryOp::Sigmoid => {
            output_val * (1.0 - output_val) * grad_output
        }
        UnaryOp::LGamma => {
             use statrs::function::gamma::digamma;
             digamma(input_val) * grad_output
        }
        UnaryOp::Softplus => {
             let s = 1.0 / (1.0 + (-input_val).exp());
             s * grad_output
        }
    }
}

fn compute_binary_grad(
    op: BinaryOp,
    lhs_val: f64,
    rhs_val: f64,
    output_val: f64,
    grad_output: f64,
) -> (f64, f64) {
    match op {
        BinaryOp::Add => (grad_output, grad_output),
        BinaryOp::Sub => (grad_output, -grad_output),
        BinaryOp::Mul => {
            (rhs_val * grad_output, lhs_val * grad_output)
        }
        BinaryOp::Div => {
            (grad_output / rhs_val, -lhs_val * grad_output / (rhs_val * rhs_val))
        }
        BinaryOp::Pow => {
            let result = lhs_val.powf(rhs_val);
            let grad_lhs = rhs_val * lhs_val.powf(rhs_val - 1.0) * grad_output;
            let grad_rhs = result * lhs_val.ln() * grad_output;
            (grad_lhs, grad_rhs)
        }
        BinaryOp::Atan2 => {
            let denom = lhs_val * lhs_val + rhs_val * rhs_val;
            (rhs_val / denom * grad_output, -lhs_val / denom * grad_output)
        }
        BinaryOp::BetaSample => {
            use statrs::function::beta::beta_reg;
            use statrs::function::gamma::ln_gamma;
             
            let alpha = lhs_val;
            let beta_param = rhs_val;
            let z = output_val;
             
            if z <= 0.0 || z >= 1.0 {
                return (0.0, 0.0);
            }

            let ln_b = ln_gamma(alpha) + ln_gamma(beta_param) - ln_gamma(alpha + beta_param);
            let ln_pdf = (alpha - 1.0) * z.ln() + (beta_param - 1.0) * (1.0 - z).ln() - ln_b;
            let pdf = ln_pdf.exp();

            let h = 1e-4;
             
            let cdf_a_plus = beta_reg(alpha + h, beta_param, z);
            let cdf_a_minus = beta_reg(alpha - h, beta_param, z);
            let d_cdf_d_alpha = (cdf_a_plus - cdf_a_minus) / (2.0 * h);
             
            let cdf_b_plus = beta_reg(alpha, beta_param + h, z);
            let cdf_b_minus = beta_reg(alpha, beta_param - h, z);
            let d_cdf_d_beta = (cdf_b_plus - cdf_b_minus) / (2.0 * h);
             
            let dz_da = -d_cdf_d_alpha / pdf;
            let dz_db = -d_cdf_d_beta / pdf;

            (dz_da * grad_output, dz_db * grad_output)
        }
        BinaryOp::MatMul => panic!("MatMul is not a scalar operation"),
    }
}

fn compute_tensor_unary_grad(
    op: UnaryOp,
    input_val: &NdarrayStorage,
    output_val: &NdarrayStorage,
    grad_output: &NdarrayStorage
) -> NdarrayStorage {
    let mut grad_input = input_val.0.clone();
    
    for ((gin, &vin), (&vout, &gout)) in grad_input.iter_mut().zip(input_val.0.iter()).zip(output_val.0.iter().zip(grad_output.0.iter())) {
        *gin = compute_unary_grad(op, vin, vout, gout);
    }
    
    NdarrayStorage(grad_input)
}

fn compute_tensor_binary_grad(
    op: BinaryOp,
    lhs: &NdarrayStorage,
    rhs: &NdarrayStorage,
    grad_output: &NdarrayStorage
) -> (NdarrayStorage, NdarrayStorage) {
    match op {
        BinaryOp::Add => (grad_output.clone(), grad_output.clone()),
        BinaryOp::Sub => (grad_output.clone(), NdarrayStorage(-&grad_output.0)),
        BinaryOp::Mul => (
            NdarrayStorage(&rhs.0 * &grad_output.0),
            NdarrayStorage(&lhs.0 * &grad_output.0)
        ),
        BinaryOp::Div => {
            let grad_lhs = NdarrayStorage(&grad_output.0 / &rhs.0);
            let grad_rhs = NdarrayStorage(-(&grad_output.0 * &lhs.0) / (&rhs.0 * &rhs.0));
            (grad_lhs, grad_rhs)
        }
        BinaryOp::MatMul => {
            let lhs_2d = lhs.0.view().into_dimensionality::<ndarray::Ix2>().expect("MatMul lhs must be 2D");
            let rhs_2d = rhs.0.view().into_dimensionality::<ndarray::Ix2>().expect("MatMul rhs must be 2D");
            let grad_2d = grad_output.0.view().into_dimensionality::<ndarray::Ix2>().expect("MatMul grad must be 2D");
             
            let da = grad_2d.dot(&rhs_2d.t());
            let db = lhs_2d.t().dot(&grad_2d);
             
            (NdarrayStorage(da.into_dyn()), NdarrayStorage(db.into_dyn()))
        }
        _ => panic!("Unsupported tensor binary op {:?}", op),
    }
}

fn ensure_tensor_values_computed(tape: &Tape, up_to_node_id: usize) {
    let nodes_to_eval: Vec<usize> = {
        let nodes = tape.nodes.borrow();
        (0..=up_to_node_id)
            .filter(|&id| {
                matches!(
                    &nodes[id],
                    ADNode::TensorBinary { value: None, .. } |
                    ADNode::TensorUnary { value: None, .. } |
                    ADNode::TensorFusedMulAdd { value: None, .. }
                )
            })
            .collect()
    };
    
    for node_id in nodes_to_eval {
        crate::ad::tensor::evaluate_forward(tape, node_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ad::types::ADFloat;
    use crate::ad::create_tape;
    
    #[test]
    fn test_simple_gradient() {
        let tape_id = create_tape();
        let x = ADFloat::new_input(3.0, tape_id);
        let y = x.clone() * x.clone();
        
        if let ADFloat::Dual { node_id, tape_id, .. } = y {
            let grads = crate::ad::with_tape(tape_id, |tape| {
                backward(tape, node_id)
            });
            
            if let ADFloat::Dual { node_id: x_id, .. } = x {
                let grad = grads.get(&x_id).and_then(|g| g.as_scalar()).unwrap_or(0.0);
                assert_eq!(grad, 6.0);
            }
        }
    }

    #[test]
    fn test_lgamma_grad() {
        use statrs::function::gamma::digamma;
        let tape_id = create_tape();
        let x_val = 2.5;
        let x = ADFloat::new_input(x_val, tape_id);
        let y = x.clone().lgamma();
        
        if let ADFloat::Dual { node_id, tape_id, .. } = y {
            let grads = crate::ad::with_tape(tape_id, |tape| {
                backward(tape, node_id)
            });
            if let ADFloat::Dual { node_id: x_id, .. } = x {
                let grad = grads.get(&x_id).and_then(|g| g.as_scalar()).unwrap_or(0.0);
                let expected = digamma(x_val);
                assert!((grad - expected).abs() < 1e-6);
            }
        }
    }
    
    #[test]
    fn test_custom_vjp() {
        crate::ad::register_custom_vjp(
            "cube",
            |args| args[0].powi(3),
            |args, _output, grad| vec![3.0 * args[0].powi(2) * grad]
        );
        
        let tape_id = create_tape();
        let x = ADFloat::new_input(2.0, tape_id);
        let y = ADFloat::apply_custom_vjp("cube", vec![x.clone()]);
        
        assert!((y.value() - 8.0).abs() < 1e-10);
        
        if let ADFloat::Dual { node_id, tape_id, .. } = y {
            let grads = crate::ad::with_tape(tape_id, |tape| {
                backward(tape, node_id)
            });
            
            if let ADFloat::Dual { node_id: x_id, .. } = x {
                let grad = grads.get(&x_id).and_then(|g| g.as_scalar()).unwrap_or(0.0);
                assert!((grad - 12.0).abs() < 1e-10);
            }
        }
    }
}
