#[cfg(test)]
mod tests {
    use fluno::ad::{create_tape, with_tape};
    use fluno::ad::graph::{ADNode, ReduceOp, BinaryOp, UnaryOp};
    use fluno::ad::backward::backward;
    use fluno::ad::types::ADGradient;
    use fluno::ad::cpu_backend::NdarrayStorage;
    use ndarray::{array, ArrayD};

    #[test]
    fn test_tensor_sum_grad() {
        let tape_id = create_tape();
        
        // x = [1.0, 2.0, 3.0]
        let x_val = array![1.0, 2.0, 3.0].into_dyn();
        
        let (x_id, sum_id) = with_tape(tape_id, |tape| {
            let x_id = tape.push(ADNode::new_tensor_input(NdarrayStorage(x_val)));
            
            // y = sum(x)
            let sum_id = tape.push(ADNode::TensorReduce {
                op: ReduceOp::Sum,
                arg: x_id,
                value: 6.0,
            });
            
            (x_id, sum_id)
        });
        
        // backward(y)
        let grads = with_tape(tape_id, |tape| {
            backward(tape, sum_id)
        });
        
        // dSum/dx = [1, 1, 1]
        if let Some(ADGradient::Tensor(g)) = grads.get(&x_id) {
            assert_eq!(g.0.shape(), &[3]);
            assert_eq!(g.0, array![1.0, 1.0, 1.0].into_dyn());
        } else {
            panic!("Gradient for x not found or not a Tensor");
        }
    }

    #[test]
    fn test_tensor_mean_grad() {
        let tape_id = create_tape();
        
        // x = [1.0, 2.0, 3.0, 4.0]
        let x_val = array![1.0, 2.0, 3.0, 4.0].into_dyn();
        
        let (x_id, mean_id) = with_tape(tape_id, |tape| {
            let x_id = tape.push(ADNode::new_tensor_input(NdarrayStorage(x_val)));
            
            // y = mean(x)
            let mean_id = tape.push(ADNode::TensorReduce {
                op: ReduceOp::Mean,
                arg: x_id,
                value: 2.5,
            });
            
            (x_id, mean_id)
        });
        
        let grads = with_tape(tape_id, |tape| {
            backward(tape, mean_id)
        });
        
        // dMean/dx = [0.25, 0.25, 0.25, 0.25]
        if let Some(ADGradient::Tensor(g)) = grads.get(&x_id) {
            assert_eq!(g.0, array![0.25, 0.25, 0.25, 0.25].into_dyn());
        } else {
            panic!("Gradient for x not found or not a Tensor");
        }
    }
    
    #[test]
    fn test_tensor_binary_add() {
        let tape_id = create_tape();
        
        // A = [1, 2], B = [3, 4]
        let a_val = array![1.0, 2.0].into_dyn();
        let b_val = array![3.0, 4.0].into_dyn();
        
        let (a_id, b_id, sum_node_id) = with_tape(tape_id, |tape| {
             let a_id = tape.push(ADNode::new_tensor_input(NdarrayStorage(a_val)));
             let b_id = tape.push(ADNode::new_tensor_input(NdarrayStorage(b_val)));
             
             // C = A + B
             let c_val = array![4.0, 6.0].into_dyn();
             let c_id = tape.push(ADNode::TensorBinary {
                 op: BinaryOp::Add,
                 lhs: a_id,
                 rhs: b_id,
                 value: Some(NdarrayStorage(c_val)),
             });
             
             // Loss = Sum(C)
             let loss_id = tape.push(ADNode::TensorReduce {
                 op: ReduceOp::Sum,
                 arg: c_id,
                 value: 10.0,
             });
             
             (a_id, b_id, loss_id)
        });
        
        let grads = with_tape(tape_id, |tape| {
            backward(tape, sum_node_id)
        });
        
        // dLoss/dA = dSum/dC * dC/dA = 1 * 1 = 1
        if let Some(ADGradient::Tensor(g)) = grads.get(&a_id) {
            assert_eq!(g.0, array![1.0, 1.0].into_dyn());
        }
        if let Some(ADGradient::Tensor(g)) = grads.get(&b_id) {
             assert_eq!(g.0, array![1.0, 1.0].into_dyn());
        }
    }

    #[test]
    fn test_tensor_unary_exp() {
        let tape_id = create_tape();
        
        // x = [0.0, 1.0]
        let x_val = array![0.0, 1.0].into_dyn();
        
        let (x_id, sum_id) = with_tape(tape_id, |tape| {
            let x_id = tape.push(ADNode::new_tensor_input(NdarrayStorage(x_val)));
            
            // y = exp(x) = [1.0, e]
            let y_val = array![1.0, std::f64::consts::E].into_dyn();
            let y_id = tape.push(ADNode::TensorUnary {
                op: UnaryOp::Exp,
                arg: x_id,
                value: Some(NdarrayStorage(y_val)),
            });
            
            // Loss = Sum(y)
            let loss_id = tape.push(ADNode::TensorReduce {
                op: ReduceOp::Sum,
                arg: y_id,
                value: 1.0 + std::f64::consts::E,
            });
            
            (x_id, loss_id)
        });
        
        let grads = with_tape(tape_id, |tape| {
            backward(tape, sum_id)
        });
        
        // dLoss/dx = dy/dx = exp(x)
        if let Some(ADGradient::Tensor(g)) = grads.get(&x_id) {
            let expected = array![1.0, std::f64::consts::E].into_dyn();
            // Allow small error
            let diff = &g.0 - &expected;
            assert!(diff.iter().all(|&v| v.abs() < 1e-6));
        }
    }

    #[test]
    fn test_tensor_matmul() {
        let tape_id = create_tape();
        
        // A (2x3)
        let a_val = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        // B (3x2)
        let b_val = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]].into_dyn();
        
        let (a_id, b_id, sum_id) = with_tape(tape_id, |tape| {
            let a_id = tape.push(ADNode::new_tensor_input(NdarrayStorage(a_val)));
            let b_id = tape.push(ADNode::new_tensor_input(NdarrayStorage(b_val)));
            
            // C = A . B (2x2)
            // Value is precomputed for the graph (forward pass simulation)
            // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
            // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
            let c_val = array![[58.0, 64.0], [139.0, 154.0]].into_dyn();
            
            let c_id = tape.push(ADNode::TensorBinary {
                op: BinaryOp::MatMul,
                lhs: a_id,
                rhs: b_id,
                value: Some(NdarrayStorage(c_val)),
            });
            
            // Loss = Sum(C)
            let loss_id = tape.push(ADNode::TensorReduce {
                op: ReduceOp::Sum,
                arg: c_id,
                value: 415.0,
            });
            
            (a_id, b_id, loss_id)
        });
        
        let grads = with_tape(tape_id, |tape| {
            backward(tape, sum_id)
        });
        
        // Loss = Sum(A . B) = Sum_ij (Sum_k A_ik B_kj)
        // dLoss/dA_ik = Sum_j B_kj
        // dLoss/dB_kj = Sum_i A_ik
        
        // dLoss/dA should be (2x3). 
        // Row 0: Sum of cols of B => [7+8, 9+10, 11+12] = [15, 19, 23]
        // Row 1: Sum of cols of B => [15, 19, 23]
        
        if let Some(ADGradient::Tensor(g)) = grads.get(&a_id) {
            let expected = array![[15.0, 19.0, 23.0], [15.0, 19.0, 23.0]].into_dyn();
            assert_eq!(g.0, expected);
        } else {
             panic!("Gradient for A not found");
        }
        
        // dLoss/dB should be (3x2).
        // Col 0: Sum of rows of A => [1+4, 2+5, 3+6]T = [5, 7, 9]T - Wait
        // dL/dB_kj = Sum_i A_ik.
        // k=0 (Row 0 of B): coeff is Sum_i A_i0 = A_00 + A_10 = 1 + 4 = 5.
        // B_01 (8): coeff is Sum_i A_i0 = 5.
        // B_10 (9): coeff is Sum_i A_i1 = 2 + 5 = 7.
        // B_11 (10): coeff is 7.
        // B_20 (11): coeff is Sum_i A_i2 = 3 + 6 = 9.
        // B_21 (12): coeff is 9.
        
        if let Some(ADGradient::Tensor(g)) = grads.get(&b_id) {
            let expected = array![[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]].into_dyn();
            assert_eq!(g.0, expected);
        } else {
             panic!("Gradient for B not found");
        }
    }
}
