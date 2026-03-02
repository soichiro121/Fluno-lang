// src/ast/operator.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorTrait {
    Add,
    Sub,
    Mul,
    Div,
}

impl OperatorTrait {
    pub fn method_name(self) -> &'static str {
        match self {
            OperatorTrait::Add => "add",
            OperatorTrait::Sub => "sub",
            OperatorTrait::Mul => "mul",
            OperatorTrait::Div => "div",
        }
    }
    pub fn trait_name(self) -> &'static str {
        match self {
            OperatorTrait::Add => "Add",
            OperatorTrait::Sub => "Sub",
            OperatorTrait::Mul => "Mul",
            OperatorTrait::Div => "Div",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OverloadNode {
    pub trait_kind: OperatorTrait,
    pub lhs: Box<Expression>,
    pub rhs: Box<Expression>,
    pub span: Span,
}
