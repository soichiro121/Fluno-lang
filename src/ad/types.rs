// src/ad/types.rs

use crate::ad::HashMap;
use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::ad::graph::{ADNode, BinaryOp, UnaryOp};
use crate::ad::with_tape;

#[derive(Debug, Clone, Copy)]
pub enum ADFloat {
    Concrete(f64),

    Dual {
        value: f64,
        tape_id: usize,
        node_id: usize,
    },
}

#[derive(Debug, Clone)]
pub enum ADGradient {
    Scalar(f64),
    Tensor(crate::ad::cpu_backend::NdarrayStorage),
}

impl ADGradient {
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            ADGradient::Scalar(v) => Some(*v),
            _ => None,
        }
    }
}

impl ADFloat {
    pub fn value(&self) -> f64 {
        match self {
            ADFloat::Concrete(v) => *v,
            ADFloat::Dual { value, .. } => *value,
        }
    }

    pub fn new_input(value: f64, tape_id: usize) -> Self {
        let node_id = with_tape(tape_id, |tape| tape.push(ADNode::Input { value }));
        ADFloat::Dual {
            value,
            tape_id,
            node_id,
        }
    }

    fn unary_op(self, op: UnaryOp, f: fn(f64) -> f64) -> Self {
        match self {
            ADFloat::Concrete(v) => ADFloat::Concrete(f(v)),
            ADFloat::Dual {
                value,
                tape_id,
                node_id,
            } => {
                let new_val = f(value);
                let new_id = with_tape(tape_id, |tape| {
                    tape.push(ADNode::Unary {
                        op,
                        arg: node_id,
                        value: new_val,
                    })
                });
                ADFloat::Dual {
                    value: new_val,
                    tape_id,
                    node_id: new_id,
                }
            }
        }
    }

    pub fn node_id(&self) -> Option<usize> {
        match self {
            ADFloat::Dual { node_id, .. } => Some(*node_id),
            _ => None,
        }
    }

    pub fn sqrt(self) -> Self {
        self.unary_op(UnaryOp::Sqrt, |x| x.sqrt())
    }

    pub fn exp(self) -> Self {
        self.unary_op(UnaryOp::Exp, |x| x.exp())
    }

    pub fn ln(self) -> Self {
        self.unary_op(UnaryOp::Log, |x| x.ln())
    }

    pub fn sin(self) -> Self {
        self.unary_op(UnaryOp::Sin, |x| x.sin())
    }

    pub fn cos(self) -> Self {
        self.unary_op(UnaryOp::Cos, |x| x.cos())
    }

    pub fn tan(self) -> Self {
        self.unary_op(UnaryOp::Tan, |x| x.tan())
    }

    pub fn abs(self) -> Self {
        self.unary_op(UnaryOp::Abs, |x| x.abs())
    }

    pub fn tanh(self) -> Self {
        self.unary_op(UnaryOp::Tanh, |x| x.tanh())
    }

    pub fn sigmoid(self) -> Self {
        self.unary_op(UnaryOp::Sigmoid, |x| 1.0 / (1.0 + (-x).exp()))
    }

    pub fn lgamma(self) -> Self {
        use statrs::function::gamma::ln_gamma;
        self.unary_op(UnaryOp::LGamma, |x| ln_gamma(x))
    }

    pub fn softplus(self) -> Self {
        self.unary_op(UnaryOp::Softplus, |x| {
            if x > 20.0 {
                x
            } else {
                (1.0 + x.exp()).ln()
            }
        })
    }

    pub fn ceil(self) -> Self {
        ADFloat::Concrete(self.value().ceil())
    }

    pub fn floor(self) -> Self {
        ADFloat::Concrete(self.value().floor())
    }

    pub fn round(self) -> Self {
        ADFloat::Concrete(self.value().round())
    }

    pub fn powf(self, exp: f64) -> Self {
        let base = self.value();
        ADFloat::Concrete(base.powf(exp))
    }

    pub fn beta_sample(self, rhs: Self, z: f64) -> Self {
        use crate::ad::graph::{ADNode, BinaryOp};
        match (self, rhs) {
            (ADFloat::Concrete(_), ADFloat::Concrete(_)) => ADFloat::Concrete(z),

            (
                ADFloat::Dual {
                    tape_id: ta,
                    node_id: lhs_id,
                    ..
                },
                ADFloat::Concrete(b),
            ) => {
                let id = with_tape(ta, |tape| {
                    let rhs_id = tape.push(ADNode::Constant { value: b });
                    tape.push(ADNode::Binary {
                        op: BinaryOp::BetaSample,
                        lhs: lhs_id,
                        rhs: rhs_id,
                        value: z,
                    })
                });
                ADFloat::Dual {
                    value: z,
                    tape_id: ta,
                    node_id: id,
                }
            }

            (
                ADFloat::Concrete(a),
                ADFloat::Dual {
                    tape_id: tb,
                    node_id: rhs_id,
                    ..
                },
            ) => {
                let id = with_tape(tb, |tape| {
                    let lhs_id = tape.push(ADNode::Constant { value: a });
                    tape.push(ADNode::Binary {
                        op: BinaryOp::BetaSample,
                        lhs: lhs_id,
                        rhs: rhs_id,
                        value: z,
                    })
                });
                ADFloat::Dual {
                    value: z,
                    tape_id: tb,
                    node_id: id,
                }
            }

            (
                ADFloat::Dual {
                    tape_id: ta,
                    node_id: lhs_id,
                    ..
                },
                ADFloat::Dual {
                    tape_id: tb,
                    node_id: rhs_id,
                    ..
                },
            ) => {
                if ta != tb {
                    panic!("ADFloat: beta_sample across different tapes");
                }
                let id = with_tape(ta, |tape| {
                    tape.push(ADNode::Binary {
                        op: BinaryOp::BetaSample,
                        lhs: lhs_id,
                        rhs: rhs_id,
                        value: z,
                    })
                });
                ADFloat::Dual {
                    value: z,
                    tape_id: ta,
                    node_id: id,
                }
            }
        }
    }

    pub fn backward(&self) -> HashMap<usize, ADGradient> {
        match self {
            ADFloat::Concrete(_) => HashMap::new(),
            ADFloat::Dual {
                node_id, tape_id, ..
            } => crate::ad::with_tape(*tape_id, |tape| {
                crate::ad::backward::backward(tape, *node_id)
            }),
        }
    }

    pub fn apply_custom_vjp(name: &str, args: Vec<ADFloat>) -> Self {
        let input_vals: Vec<f64> = args.iter().map(|a| a.value()).collect();

        let output_val = crate::ad::call_forward(name, &input_vals)
            .expect(&format!("Custom VJP '{}' not registered", name));

        let tape_id_opt = args.iter().find_map(|a| match a {
            ADFloat::Dual { tape_id, .. } => Some(*tape_id),
            _ => None,
        });

        match tape_id_opt {
            None => ADFloat::Concrete(output_val),
            Some(tape_id) => {
                let arg_node_ids: Vec<usize> = args
                    .iter()
                    .map(|a| match a {
                        ADFloat::Dual { node_id, .. } => *node_id,
                        ADFloat::Concrete(v) => crate::ad::with_tape(tape_id, |tape| {
                            tape.push(ADNode::Constant { value: *v })
                        }),
                    })
                    .collect();

                let node_id = crate::ad::with_tape(tape_id, |tape| {
                    tape.push(ADNode::CustomVjp {
                        name: name.to_string(),
                        args: arg_node_ids,
                        value: output_val,
                    })
                });

                ADFloat::Dual {
                    value: output_val,
                    tape_id,
                    node_id,
                }
            }
        }
    }
}

impl PartialEq for ADFloat {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl PartialOrd for ADFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value().partial_cmp(&other.value())
    }
}

fn binary_op(lhs: ADFloat, rhs: ADFloat, op: BinaryOp, f: fn(f64, f64) -> f64) -> ADFloat {
    match (lhs, rhs) {
        (ADFloat::Concrete(a), ADFloat::Concrete(b)) => ADFloat::Concrete(f(a, b)),

        (
            ADFloat::Dual {
                value: a,
                tape_id,
                node_id: lhs_id,
            },
            ADFloat::Concrete(b),
        ) => {
            let v = f(a, b);
            let id = with_tape(tape_id, |tape| {
                let rhs_id = tape.push(ADNode::Constant { value: b });
                tape.push(ADNode::Binary {
                    op,
                    lhs: lhs_id,
                    rhs: rhs_id,
                    value: v,
                })
            });
            ADFloat::Dual {
                value: v,
                tape_id,
                node_id: id,
            }
        }

        (
            ADFloat::Concrete(a),
            ADFloat::Dual {
                value: b,
                tape_id,
                node_id: rhs_id,
            },
        ) => {
            let v = f(a, b);
            let id = with_tape(tape_id, |tape| {
                let lhs_id = tape.push(ADNode::Constant { value: a });
                tape.push(ADNode::Binary {
                    op,
                    lhs: lhs_id,
                    rhs: rhs_id,
                    value: v,
                })
            });
            ADFloat::Dual {
                value: v,
                tape_id,
                node_id: id,
            }
        }

        (
            ADFloat::Dual {
                value: a,
                tape_id: ta,
                node_id: lhs_id,
            },
            ADFloat::Dual {
                value: b,
                tape_id: tb,
                node_id: rhs_id,
            },
        ) => {
            if ta != tb {
                panic!("ADFloat: binary op across different tapes is unsupported");
            }
            let v = f(a, b);
            let id = with_tape(ta, |tape| {
                tape.push(ADNode::Binary {
                    op,
                    lhs: lhs_id,
                    rhs: rhs_id,
                    value: v,
                })
            });
            ADFloat::Dual {
                value: v,
                tape_id: ta,
                node_id: id,
            }
        }
    }
}

impl Add for ADFloat {
    type Output = ADFloat;

    fn add(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, BinaryOp::Add, |a, b| a + b)
    }
}

impl Sub for ADFloat {
    type Output = ADFloat;

    fn sub(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, BinaryOp::Sub, |a, b| a - b)
    }
}

impl Mul for ADFloat {
    type Output = ADFloat;

    fn mul(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, BinaryOp::Mul, |a, b| a * b)
    }
}

impl Div for ADFloat {
    type Output = ADFloat;

    fn div(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, BinaryOp::Div, |a, b| a / b)
    }
}

impl Neg for ADFloat {
    type Output = ADFloat;

    fn neg(self) -> Self::Output {
        self.unary_op(UnaryOp::Neg, |x| -x)
    }
}

impl From<f64> for ADFloat {
    fn from(v: f64) -> Self {
        ADFloat::Concrete(v)
    }
}

impl Add<f64> for ADFloat {
    type Output = ADFloat;

    fn add(self, rhs: f64) -> Self::Output {
        self + ADFloat::Concrete(rhs)
    }
}

impl Add<ADFloat> for f64 {
    type Output = ADFloat;

    fn add(self, rhs: ADFloat) -> Self::Output {
        ADFloat::Concrete(self) + rhs
    }
}

impl Sub<f64> for ADFloat {
    type Output = ADFloat;

    fn sub(self, rhs: f64) -> Self::Output {
        self - ADFloat::Concrete(rhs)
    }
}

impl Sub<ADFloat> for f64 {
    type Output = ADFloat;

    fn sub(self, rhs: ADFloat) -> Self::Output {
        ADFloat::Concrete(self) - rhs
    }
}

impl Mul<f64> for ADFloat {
    type Output = ADFloat;

    fn mul(self, rhs: f64) -> Self::Output {
        self * ADFloat::Concrete(rhs)
    }
}

impl Mul<ADFloat> for f64 {
    type Output = ADFloat;

    fn mul(self, rhs: ADFloat) -> Self::Output {
        ADFloat::Concrete(self) * rhs
    }
}

impl Div<f64> for ADFloat {
    type Output = ADFloat;

    fn div(self, rhs: f64) -> Self::Output {
        self / ADFloat::Concrete(rhs)
    }
}

impl Div<ADFloat> for f64 {
    type Output = ADFloat;

    fn div(self, rhs: ADFloat) -> Self::Output {
        ADFloat::Concrete(self) / rhs
    }
}

impl std::fmt::Display for ADFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ad::create_tape;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_concrete_arithmetic() {
        let a = ADFloat::Concrete(3.0);
        let b = ADFloat::Concrete(2.0);

        assert!(approx_eq((a + b).value(), 5.0, 1e-10));
        assert!(approx_eq((a - b).value(), 1.0, 1e-10));
        assert!(approx_eq((a * b).value(), 6.0, 1e-10));
        assert!(approx_eq((a / b).value(), 1.5, 1e-10));
        assert!(approx_eq((-a).value(), -3.0, 1e-10));
    }

    #[test]
    fn test_unary_ops_forward() {
        let x = ADFloat::Concrete(2.0);

        assert!(approx_eq(x.sqrt().value(), 2.0_f64.sqrt(), 1e-10));
        assert!(approx_eq(x.exp().value(), 2.0_f64.exp(), 1e-10));
        assert!(approx_eq(x.ln().value(), 2.0_f64.ln(), 1e-10));
        assert!(approx_eq(x.sin().value(), 2.0_f64.sin(), 1e-10));
        assert!(approx_eq(x.cos().value(), 2.0_f64.cos(), 1e-10));
        assert!(approx_eq(x.tan().value(), 2.0_f64.tan(), 1e-10));
        assert!(approx_eq(x.abs().value(), 2.0_f64.abs(), 1e-10));
        assert!(approx_eq(x.tanh().value(), 2.0_f64.tanh(), 1e-10));

        // Sigmoid: 1 / (1 + exp(-x))
        let expected_sigmoid = 1.0 / (1.0 + (-2.0_f64).exp());
        assert!(approx_eq(x.sigmoid().value(), expected_sigmoid, 1e-10));
    }

    #[test]
    fn test_gradient_add() {
        // f(x, y) = x + y, df/dx = 1, df/dy = 1
        let tape_id = create_tape();
        let x = ADFloat::new_input(3.0, tape_id);
        let y = ADFloat::new_input(4.0, tape_id);
        let z = x + y;

        assert!(approx_eq(z.value(), 7.0, 1e-10));

        let grads = z.backward();
        let x_id = x.node_id().unwrap();
        let y_id = y.node_id().unwrap();

        let dx = grads.get(&x_id).unwrap().as_scalar().unwrap();
        let dy = grads.get(&y_id).unwrap().as_scalar().unwrap();

        assert!(approx_eq(dx, 1.0, 1e-10));
        assert!(approx_eq(dy, 1.0, 1e-10));
    }

    #[test]
    fn test_gradient_mul() {
        // f(x, y) = x * y, df/dx = y, df/dy = x
        let tape_id = create_tape();
        let x = ADFloat::new_input(3.0, tape_id);
        let y = ADFloat::new_input(4.0, tape_id);
        let z = x * y;

        assert!(approx_eq(z.value(), 12.0, 1e-10));

        let grads = z.backward();
        let x_id = x.node_id().unwrap();
        let y_id = y.node_id().unwrap();

        let dx = grads.get(&x_id).unwrap().as_scalar().unwrap();
        let dy = grads.get(&y_id).unwrap().as_scalar().unwrap();

        assert!(approx_eq(dx, 4.0, 1e-10));
        assert!(approx_eq(dy, 3.0, 1e-10));
    }

    #[test]
    fn test_gradient_chain_rule() {
        // f(x) = (x * x) + x = x^2 + x, df/dx = 2x + 1
        let tape_id = create_tape();
        let x = ADFloat::new_input(5.0, tape_id);
        let x2 = x * x;
        let z = x2 + x;

        assert!(approx_eq(z.value(), 30.0, 1e-10));

        let grads = z.backward();
        let x_id = x.node_id().unwrap();
        let dx = grads.get(&x_id).unwrap().as_scalar().unwrap();

        assert!(approx_eq(dx, 11.0, 1e-10));
    }

    #[test]
    fn test_gradient_exp() {
        // f(x) = exp(x), df/dx = exp(x)
        let tape_id = create_tape();
        let x = ADFloat::new_input(2.0, tape_id);
        let z = x.exp();

        let expected = 2.0_f64.exp();
        assert!(approx_eq(z.value(), expected, 1e-10));

        let grads = z.backward();
        let x_id = x.node_id().unwrap();
        let dx = grads.get(&x_id).unwrap().as_scalar().unwrap();

        assert!(approx_eq(dx, expected, 1e-10));
    }

    #[test]
    fn test_gradient_sin_cos() {
        // f(x) = sin(x), df/dx = cos(x)
        let tape_id = create_tape();
        let x = ADFloat::new_input(1.0, tape_id);
        let z = x.sin();

        assert!(approx_eq(z.value(), 1.0_f64.sin(), 1e-10));

        let grads = z.backward();
        let x_id = x.node_id().unwrap();
        let dx = grads.get(&x_id).unwrap().as_scalar().unwrap();

        assert!(approx_eq(dx, 1.0_f64.cos(), 1e-10));
    }

    #[test]
    fn test_gradient_div() {
        // f(x, y) = x / y, df/dx = 1/y, df/dy = -x/y^2
        let tape_id = create_tape();
        let x = ADFloat::new_input(6.0, tape_id);
        let y = ADFloat::new_input(2.0, tape_id);
        let z = x / y;

        assert!(approx_eq(z.value(), 3.0, 1e-10));

        let grads = z.backward();
        let x_id = x.node_id().unwrap();
        let y_id = y.node_id().unwrap();

        let dx = grads.get(&x_id).unwrap().as_scalar().unwrap();
        let dy = grads.get(&y_id).unwrap().as_scalar().unwrap();

        assert!(approx_eq(dx, 0.5, 1e-10));
        assert!(approx_eq(dy, -1.5, 1e-10));
    }

    #[test]
    fn test_gradient_sqrt() {
        // f(x) = sqrt(x), df/dx = 1/(2*sqrt(x))
        let tape_id = create_tape();
        let x = ADFloat::new_input(4.0, tape_id);
        let z = x.sqrt();

        assert!(approx_eq(z.value(), 2.0, 1e-10));

        let grads = z.backward();
        let x_id = x.node_id().unwrap();
        let dx = grads.get(&x_id).unwrap().as_scalar().unwrap();

        assert!(approx_eq(dx, 0.25, 1e-10));
    }

    #[test]
    fn test_gradient_neg() {
        // f(x) = -x, df/dx = -1
        let tape_id = create_tape();
        let x = ADFloat::new_input(5.0, tape_id);
        let z = -x;

        assert!(approx_eq(z.value(), -5.0, 1e-10));

        let grads = z.backward();
        let x_id = x.node_id().unwrap();
        let dx = grads.get(&x_id).unwrap().as_scalar().unwrap();

        assert!(approx_eq(dx, -1.0, 1e-10));
    }

    #[test]
    fn test_gradient_tanh() {
        // f(x) = tanh(x), df/dx = 1 - tanh(x)^2
        let tape_id = create_tape();
        let x = ADFloat::new_input(1.0, tape_id);
        let z = x.tanh();

        let tanh_val = 1.0_f64.tanh();
        assert!(approx_eq(z.value(), tanh_val, 1e-10));

        let grads = z.backward();
        let x_id = x.node_id().unwrap();
        let dx = grads.get(&x_id).unwrap().as_scalar().unwrap();

        let expected_grad = 1.0 - tanh_val * tanh_val;
        assert!(approx_eq(dx, expected_grad, 1e-10));
    }

    #[test]
    fn test_gradient_sigmoid() {
        // f(x) = sigmoid(x), df/dx = sigmoid(x) * (1 - sigmoid(x))
        let tape_id = create_tape();
        let x = ADFloat::new_input(1.0, tape_id);
        let z = x.sigmoid();

        let sig_val = 1.0 / (1.0 + (-1.0_f64).exp());
        assert!(approx_eq(z.value(), sig_val, 1e-10));

        let grads = z.backward();
        let x_id = x.node_id().unwrap();
        let dx = grads.get(&x_id).unwrap().as_scalar().unwrap();

        let expected_grad = sig_val * (1.0 - sig_val);
        assert!(approx_eq(dx, expected_grad, 1e-10));
    }

    #[test]
    fn test_complex_expression() {
        // f(x) = x^2 * exp(x) - sin(x)
        let tape_id = create_tape();
        let x = ADFloat::new_input(1.0, tape_id);
        let z = (x * x) * x.exp() - x.sin();

        let _grads = z.backward();
    }
}
