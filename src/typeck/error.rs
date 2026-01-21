// Type checking error definitions and formatting.
//
// This module provides comprehensive error types for type checking,
// including position information and user-friendly error messages.

use crate::ast::node::{Span, Type, DefId};
use std::fmt;

// Type checking error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    // E2001: Type mismatch
    TypeMismatch,
    // E2002: Undefined variable
    UndefinedVariable,
    // E2003: Duplicate definition
    DuplicateDefinition,
    // E2004: Arity mismatch (wrong number of arguments)
    ArityMismatch,
    // E2005: Cannot infer type
    CannotInfer,
    // E2006: Invalid unary operation
    InvalidUnaryOp,
    // E2007: Invalid binary operation
    InvalidBinaryOp,
    // E2008: Non-exhaustive patterns
    NonExhaustivePatterns,
    // E2009: Unreachable pattern
    UnreachablePattern,
    // E2010: Invalid field access
    InvalidFieldAccess,
    // E2011: Missing field in struct initialization
    MissingField,
    // E2012: Unknown field in struct initialization
    UnknownField,
    // E2013: Invalid index operation
    InvalidIndex,
    // E2014: Return type mismatch
    ReturnTypeMismatch,
    // E2015: Break outside loop
    BreakOutsideLoop,
    // E2016: Continue outside loop
    ContinueOutsideLoop,
    // E2017: Not callable (not a function)
    NotCallable,
    // E2018: Undefined type
    UndefinedType,
    // E2019: Recursive type definition
    RecursiveType,
    // E2020: Invalid assignment target
    InvalidAssignmentTarget,
    // E2021: undefined field
    UndefinedField,
    // E2022: no matching field
    NoMatchingImpl,
    // E2023
    AmbiguousImpl,
    // E2024: Invalid distribution parameter
    InvalidDistributionParameter,
}

impl ErrorCode {
    // Get the error code as a string (e.g., "E2001")
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCode::TypeMismatch => "E2001",
            ErrorCode::UndefinedVariable => "E2002",
            ErrorCode::DuplicateDefinition => "E2003",
            ErrorCode::ArityMismatch => "E2004",
            ErrorCode::CannotInfer => "E2005",
            ErrorCode::InvalidUnaryOp => "E2006",
            ErrorCode::InvalidBinaryOp => "E2007",
            ErrorCode::NonExhaustivePatterns => "E2008",
            ErrorCode::UnreachablePattern => "E2009",
            ErrorCode::InvalidFieldAccess => "E2010",
            ErrorCode::MissingField => "E2011",
            ErrorCode::UnknownField => "E2012",
            ErrorCode::InvalidIndex => "E2013",
            ErrorCode::ReturnTypeMismatch => "E2014",
            ErrorCode::BreakOutsideLoop => "E2015",
            ErrorCode::ContinueOutsideLoop => "E2016",
            ErrorCode::NotCallable => "E2017",
            ErrorCode::UndefinedType => "E2018",
            ErrorCode::RecursiveType => "E2019",
            ErrorCode::InvalidAssignmentTarget => "E2020",
            ErrorCode::UndefinedField => "E2021",
            ErrorCode::NoMatchingImpl => "E2022",
            ErrorCode::AmbiguousImpl => "E2023",
            ErrorCode::InvalidDistributionParameter => "E2024",
        }
    }

    // Get a short description of the error
    pub fn description(&self) -> &'static str {
        match self {
            ErrorCode::TypeMismatch => "type mismatch",
            ErrorCode::UndefinedVariable => "undefined variable",
            ErrorCode::DuplicateDefinition => "duplicate definition",
            ErrorCode::ArityMismatch => "wrong number of arguments",
            ErrorCode::CannotInfer => "cannot infer type",
            ErrorCode::InvalidUnaryOp => "invalid unary operation",
            ErrorCode::InvalidBinaryOp => "invalid binary operation",
            ErrorCode::NonExhaustivePatterns => "non-exhaustive patterns",
            ErrorCode::UnreachablePattern => "unreachable pattern",
            ErrorCode::InvalidFieldAccess => "invalid field access",
            ErrorCode::MissingField => "missing field",
            ErrorCode::UnknownField => "unknown field",
            ErrorCode::InvalidIndex => "invalid index operation",
            ErrorCode::ReturnTypeMismatch => "return type mismatch",
            ErrorCode::BreakOutsideLoop => "break outside loop",
            ErrorCode::ContinueOutsideLoop => "continue outside loop",
            ErrorCode::NotCallable => "not callable",
            ErrorCode::UndefinedType => "undefined type",
            ErrorCode::RecursiveType => "recursive type",
            ErrorCode::InvalidAssignmentTarget => "invalid assignment target",
            ErrorCode::UndefinedField => "undefined field",
            ErrorCode::NoMatchingImpl => "no matching Impl",
            ErrorCode::AmbiguousImpl => "ambiguos Impl",
            ErrorCode::InvalidDistributionParameter => "invalid distribution parameter",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.as_str())
    }
}

// Type checking errors with detailed context
#[derive(Debug, Clone)]
pub enum TypeError {
    // Type mismatch: expected one type but found another
    TypeMismatch {
        expected: Type,
        found: Type,
        span: Span,
    },

    // Variable not defined in any scope
    UndefinedVariable { name: String, span: Span },

    // Variable already defined in current scope
    DuplicateDefinition { name: String, span: Span },

    // Wrong number of function arguments
    ArityMismatch {
        expected: usize,
        found: usize,
        span: Span,
    },

    // Cannot infer type from context
    CannotInfer { span: Span },

    // Invalid unary operation for the given type
    InvalidUnaryOp {
        op: String,
        operand_type: Type,
        span: Span,
    },

    // Invalid binary operation for the given types
    InvalidBinaryOp {
        op: String,
        left_type: Type,
        right_type: Type,
        span: Span,
    },

    // Match expression has non-exhaustive patterns
    NonExhaustivePatterns { span: Span },

    // Pattern is unreachable (covered by previous patterns)
    UnreachablePattern { span: Span },

    // Invalid field access on non-struct type
    InvalidFieldAccess {
        field: String,
        base_type: Type,
        span: Span,
    },

    // Missing required field in struct initialization
    MissingField {
        field: String,
        struct_name: String,
        span: Span,
    },

    // Unknown field in struct initialization
    UnknownField {
        field: String,
        struct_name: String,
        span: Span,
    },

    // Invalid index operation on non-indexable type
    InvalidIndex { base_type: Type, span: Span },

    // Return type doesn't match function signature
    ReturnTypeMismatch {
        expected: Type,
        found: Type,
        span: Span,
    },

    // Break statement outside of loop
    BreakOutsideLoop { span: Span },

    // Continue statement outside of loop
    ContinueOutsideLoop { span: Span },

    // Attempted to call a non-function value
    NotCallable { value_type: Type, span: Span },

    // Type name is not defined
    UndefinedType { name: String, span: Span },

    // Recursive type definition without indirection
    RecursiveType { name: String, span: Span },

    // Invalid assignment target (not a variable or field)
    InvalidAssignmentTarget { span: Span },

    UndefinedField {
        structname: String,
        field: String,
        span: Span,
    },
    NoMatchingImpl {
        trait_def: DefId,
        receiver: Type,
        span: Span,
    },
    AmbiguousImpl {
        trait_def: DefId,
        receiver: Type,
        span: Span,
    },

    // Invalid distribution parameter (e.g., negative std for Gaussian)
    InvalidDistributionParameter {
        distribution: String,
        param_name: String,
        reason: String,
        span: Span,
    },
}

impl TypeError {
    // Get the error code for this error
    pub fn code(&self) -> ErrorCode {
        match self {
            TypeError::TypeMismatch { .. } => ErrorCode::TypeMismatch,
            TypeError::UndefinedVariable { .. } => ErrorCode::UndefinedVariable,
            TypeError::DuplicateDefinition { .. } => ErrorCode::DuplicateDefinition,
            TypeError::ArityMismatch { .. } => ErrorCode::ArityMismatch,
            TypeError::CannotInfer { .. } => ErrorCode::CannotInfer,
            TypeError::InvalidUnaryOp { .. } => ErrorCode::InvalidUnaryOp,
            TypeError::InvalidBinaryOp { .. } => ErrorCode::InvalidBinaryOp,
            TypeError::NonExhaustivePatterns { .. } => ErrorCode::NonExhaustivePatterns,
            TypeError::UnreachablePattern { .. } => ErrorCode::UnreachablePattern,
            TypeError::InvalidFieldAccess { .. } => ErrorCode::InvalidFieldAccess,
            TypeError::MissingField { .. } => ErrorCode::MissingField,
            TypeError::UnknownField { .. } => ErrorCode::UnknownField,
            TypeError::InvalidIndex { .. } => ErrorCode::InvalidIndex,
            TypeError::ReturnTypeMismatch { .. } => ErrorCode::ReturnTypeMismatch,
            TypeError::BreakOutsideLoop { .. } => ErrorCode::BreakOutsideLoop,
            TypeError::ContinueOutsideLoop { .. } => ErrorCode::ContinueOutsideLoop,
            TypeError::NotCallable { .. } => ErrorCode::NotCallable,
            TypeError::UndefinedType { .. } => ErrorCode::UndefinedType,
            TypeError::RecursiveType { .. } => ErrorCode::RecursiveType,
            TypeError::InvalidAssignmentTarget { .. } => ErrorCode::InvalidAssignmentTarget,
            TypeError::UndefinedField { .. } => ErrorCode::UndefinedField,
            TypeError::NoMatchingImpl { .. } => ErrorCode::NoMatchingImpl,
            TypeError::AmbiguousImpl { .. } => ErrorCode::AmbiguousImpl,
            TypeError::InvalidDistributionParameter { .. } => ErrorCode::InvalidDistributionParameter,
        }
    }

    // Get the span where this error occurred
    pub fn span(&self) -> Span {
        match self {
            TypeError::TypeMismatch { span, .. }
            | TypeError::UndefinedVariable { span, .. }
            | TypeError::DuplicateDefinition { span, .. }
            | TypeError::ArityMismatch { span, .. }
            | TypeError::CannotInfer { span }
            | TypeError::InvalidUnaryOp { span, .. }
            | TypeError::InvalidBinaryOp { span, .. }
            | TypeError::NonExhaustivePatterns { span }
            | TypeError::UnreachablePattern { span }
            | TypeError::InvalidFieldAccess { span, .. }
            | TypeError::MissingField { span, .. }
            | TypeError::UnknownField { span, .. }
            | TypeError::InvalidIndex { span, .. }
            | TypeError::ReturnTypeMismatch { span, .. }
            | TypeError::BreakOutsideLoop { span }
            | TypeError::ContinueOutsideLoop { span }
            | TypeError::NotCallable { span, .. }
            | TypeError::UndefinedType { span, .. }
            | TypeError::RecursiveType { span, .. }
            | TypeError::InvalidAssignmentTarget { span } => *span,
            | TypeError::UndefinedField { span, .. } => *span,
            | TypeError::NoMatchingImpl { span, .. } => *span,
            | TypeError::AmbiguousImpl { span, .. } => *span,
            | TypeError::InvalidDistributionParameter { span, .. } => *span,
        }
    }

    // Get a detailed error message
    pub fn message(&self) -> String {
        match self {
            TypeError::TypeMismatch { expected, found, .. } => {
                format!("Expected type '{}', but found '{}'", expected, found)
            }
            TypeError::UndefinedVariable { name, .. } => {
                format!("Variable '{}' is not defined", name)
            }
            TypeError::DuplicateDefinition { name, .. } => {
                format!("Variable '{}' is already defined in this scope", name)
            }
            TypeError::ArityMismatch { expected, found, .. } => {
                format!(
                    "Expected {} argument{}, but found {}",
                    expected,
                    if *expected == 1 { "" } else { "s" },
                    found
                )
            }
            TypeError::CannotInfer { .. } => {
                "Cannot infer type from context".to_string()
            }
            TypeError::InvalidUnaryOp { op, operand_type, .. } => {
                format!("Cannot apply unary operator '{}' to type '{}'", op, operand_type)
            }
            TypeError::InvalidBinaryOp { op, left_type, right_type, .. } => {
                format!(
                    "Cannot apply binary operator '{}' to types '{}' and '{}'",
                    op, left_type, right_type
                )
            }
            TypeError::NonExhaustivePatterns { .. } => {
                "Match expression has non-exhaustive patterns".to_string()
            }
            TypeError::UnreachablePattern { .. } => {
                "This pattern is unreachable".to_string()
            }
            TypeError::InvalidFieldAccess { field, base_type, .. } => {
                format!("Type '{}' does not have a field '{}'", base_type, field)
            }
            TypeError::MissingField { field, struct_name, .. } => {
                format!("Missing required field '{}' in struct '{}'", field, struct_name)
            }
            TypeError::UnknownField { field, struct_name, .. } => {
                format!("Unknown field '{}' for struct '{}'", field, struct_name)
            }
            TypeError::InvalidIndex { base_type, .. } => {
                format!("Cannot index into type '{}'", base_type)
            }
            TypeError::ReturnTypeMismatch { expected, found, .. } => {
                format!(
                    "Function returns '{}', but expected '{}'",
                    found, expected
                )
            }
            TypeError::BreakOutsideLoop { .. } => {
                "Break statement can only be used inside a loop".to_string()
            }
            TypeError::ContinueOutsideLoop { .. } => {
                "Continue statement can only be used inside a loop".to_string()
            }
            TypeError::NotCallable { value_type, .. } => {
                format!("Type '{}' is not callable", value_type)
            }
            TypeError::UndefinedType { name, .. } => {
                format!("Type '{}' is not defined", name)
            }
            TypeError::RecursiveType { name, .. } => {
                format!("Type '{}' is recursive without indirection", name)
            }
            TypeError::InvalidAssignmentTarget { .. } => {
                "Invalid assignment target".to_string()
            }
            TypeError::UndefinedField { structname, field, .. } => {
                format!("Undefined field '{field}' for struct '{structname}'")
            }
            TypeError::NoMatchingImpl { trait_def, receiver, .. } => {
                format!("No implementation of trait '{:?}' found for type '{}'", trait_def, receiver)
            }
            TypeError::AmbiguousImpl { trait_def, receiver, .. } => {
                format!("Multiple implementations of trait '{:?}' found for type '{}'", trait_def, receiver)
            }
            TypeError::InvalidDistributionParameter { distribution, param_name, reason, .. } => {
                format!("Invalid parameter '{}' for distribution '{}': {}", param_name, distribution, reason)
            }
        }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let span = self.span();
        write!(
            f,
            "{} {} at line {}, column {}: {}",
            self.code(),
            self.code().description(),
            span.line,
            span.column,
            self.message()
        )
    }
}

impl std::error::Error for TypeError {}

// Result type for type checking operations
pub type TypeResult<T> = Result<T, TypeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_display() {
        assert_eq!(ErrorCode::TypeMismatch.as_str(), "E2001");
        assert_eq!(ErrorCode::UndefinedVariable.as_str(), "E2002");
        assert_eq!(format!("{}", ErrorCode::TypeMismatch), "[E2001]");
    }

    #[test]
    fn test_type_error_display() {
        let err = TypeError::TypeMismatch {
            expected: Type::Int,
            found: Type::Float,
            span: Span::initial(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("E2001"));
        assert!(msg.contains("Expected type 'Int'"));
    }

    #[test]
    fn test_error_code() {
        let err = TypeError::UndefinedVariable {
            name: "x".to_string(),
            span: Span::initial(),
        };
        assert_eq!(err.code(), ErrorCode::UndefinedVariable);
    }
}
