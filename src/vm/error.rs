// src/vm/error.rs

use crate::ast::node::Span;
use thiserror::Error;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Error, Debug, Clone)]
pub enum RuntimeError {
    #[error("Undefined variable '{name}' at line {}, column {}", span.line, span.column)]
    UndefinedVariable { name: String, span: Span },

    #[error("Undefined field '{field}'")]
    UndefinedField { field: String },

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Index out of bounds: index {index}, length {length}")]
    IndexOutOfBounds { index: usize, length: usize },

    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },

    #[error("Type mismatch: {message}")]
    TypeMismatch { message: String },

    #[error("Arity mismatch: expected {expected} arguments, found {found} at line {}, column {}", span.line, span.column)]
    ArityMismatch {
        expected: usize,
        found: usize,
        span: Span,
    },

    #[error("Not callable: {value} at line {}, column {}", span.line, span.column)]
    NotCallable { value: String, span: Span },

    #[error("Stack overflow")]
    StackOverflow,

    #[error("Feature not yet implemented: {0}")]
    Unimplemented(String),

    #[error("Early return from function")]
    EarlyReturn,

    #[error("Argument mismatch: expected {expected}, found {found} at line {}, column {}", span.line, span.column)]
    ArgumentMismatch {
        expected: usize,
        found: usize,
        span: crate::ast::node::Span,
    },

    #[error("Distribution error: {message}")]
    DistributionError { message: String },
}
