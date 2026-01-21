// src/ad/graph.rs

use std::cell::RefCell;
use crate::ad::types::ADFloat;
use crate::ad::cpu_backend::NdarrayStorage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Pow, Atan2, BetaSample, MatMul
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg, Exp, Log, Sin, Cos, Tan, Sqrt, Abs, Tanh, Sigmoid, LGamma, Softplus
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum, Mean, Max, Min
}

#[derive(Debug, Clone)]
pub enum ADNode {
    Input { value: f64 },
    Constant { value: f64 },
    Binary {
        op: BinaryOp,
        lhs: usize,
        rhs: usize,
        value: f64,
    },

    Unary {
        op: UnaryOp,
        arg: usize,
        value: f64,
    },

    TensorInput { value: NdarrayStorage },
    TensorConstant { value: NdarrayStorage },
    TensorBinary {
        op: BinaryOp,
        lhs: usize,
        rhs: usize,
        value: Option<NdarrayStorage>,
    },
    TensorUnary {
        op: UnaryOp,
        arg: usize,
        value: Option<NdarrayStorage>,
    },
    TensorReduce {
        op: ReduceOp,
        arg: usize,
        value: f64, 
    },
    TensorFusedMulAdd {
        a: usize,
        b: usize,
        c: usize,
        value: Option<NdarrayStorage>,
    },
    
    CustomVjp {
        name: String,
        args: Vec<usize>,
        value: f64,
    },
}

impl ADNode {
    pub fn new_input(value: f64) -> Self {
        ADNode::Input { value }
    }
    
    pub fn new_tensor_input(value: NdarrayStorage) -> Self {
        ADNode::TensorInput { value }
    }
}

#[derive(Debug)] 
pub struct Tape {
    pub nodes: RefCell<Vec<ADNode>>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
        }
    }

    pub fn push(&self, node: ADNode) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let id = nodes.len();
        nodes.push(node);
        id
    }

    pub fn param(&self, value: f64) -> ADFloat {
        let mut nodes = self.nodes.borrow_mut();
        let id = nodes.len();
        nodes.push(ADNode::new_input(value));
        ADFloat::new_input(value, id)
    }
}
