//src/diagnostics/mod.rs

use std::fmt;

// Error code categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    // Runtime Errors (R1000-R1999)
    // Division by zero
    R1001,
    // Type mismatch at runtime
    R1002,
    // Undefined variable
    R1003,
    // Arity mismatch in function call
    R1004,
    // Index out of bounds
    R1005,
    // Null pointer dereference
    R1006,
    // Stack overflow
    R1007,
    // Early return (internal)
    R1008,

    // Type Errors (T2000-T2999)
    // Type mismatch in expression
    T2001,
    // Cannot infer type
    T2002,
    // Invalid binary operation
    T2003,
    // Return type mismatch
    T2004,
    // Undefined type
    T2005,

    // Parse Errors (P3000-P3999)
    // Unexpected token
    P3001,
    // Expected token not found
    P3002,
    // Invalid syntax
    P3003,
    // Unterminated string
    P3004,

    // I/O Errors (I4000-I4999)
    // File not found
    I4001,
    // Permission denied
    I4002,
    // Read error
    I4003,
    // Write error
    I4004,

    // Panic Errors (P5000-P5999)
    // Unrecoverable error
    P5001,
    // Assertion failed
    P5002,
    // Out of memory
    P5003,
}

impl ErrorCode {
    // Get the numeric code
    pub fn code(&self) -> u32 {
        match self {
            ErrorCode::R1001 => 1001,
            ErrorCode::R1002 => 1002,
            ErrorCode::R1003 => 1003,
            ErrorCode::R1004 => 1004,
            ErrorCode::R1005 => 1005,
            ErrorCode::R1006 => 1006,
            ErrorCode::R1007 => 1007,
            ErrorCode::R1008 => 1008,
            
            ErrorCode::T2001 => 2001,
            ErrorCode::T2002 => 2002,
            ErrorCode::T2003 => 2003,
            ErrorCode::T2004 => 2004,
            ErrorCode::T2005 => 2005,
            
            ErrorCode::P3001 => 3001,
            ErrorCode::P3002 => 3002,
            ErrorCode::P3003 => 3003,
            ErrorCode::P3004 => 3004,
            
            ErrorCode::I4001 => 4001,
            ErrorCode::I4002 => 4002,
            ErrorCode::I4003 => 4003,
            ErrorCode::I4004 => 4004,
            
            ErrorCode::P5001 => 5001,
            ErrorCode::P5002 => 5002,
            ErrorCode::P5003 => 5003,
        }
    }
    
    pub fn category(&self) -> &'static str {
        match self {
            ErrorCode::R1001 | ErrorCode::R1002 | ErrorCode::R1003 | 
            ErrorCode::R1004 | ErrorCode::R1005 | ErrorCode::R1006 | 
            ErrorCode::R1007 | ErrorCode::R1008 => "Runtime",
            
            ErrorCode::T2001 | ErrorCode::T2002 | ErrorCode::T2003 | 
            ErrorCode::T2004 | ErrorCode::T2005 => "Type",
            
            ErrorCode::P3001 | ErrorCode::P3002 | ErrorCode::P3003 | 
            ErrorCode::P3004 => "Parse",
            
            ErrorCode::I4001 | ErrorCode::I4002 | ErrorCode::I4003 | 
            ErrorCode::I4004 => "I/O",
            
            ErrorCode::P5001 | ErrorCode::P5002 | ErrorCode::P5003 => "Panic",
        }
    }
    
    // Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            ErrorCode::R1001 => "Division by zero",
            ErrorCode::R1002 => "Type mismatch at runtime",
            ErrorCode::R1003 => "Undefined variable",
            ErrorCode::R1004 => "Function arity mismatch",
            ErrorCode::R1005 => "Index out of bounds",
            ErrorCode::R1006 => "Null pointer dereference",
            ErrorCode::R1007 => "Stack overflow",
            ErrorCode::R1008 => "Early return",
            
            ErrorCode::T2001 => "Type mismatch",
            ErrorCode::T2002 => "Cannot infer type",
            ErrorCode::T2003 => "Invalid binary operation",
            ErrorCode::T2004 => "Return type mismatch",
            ErrorCode::T2005 => "Undefined type",
            
            ErrorCode::P3001 => "Unexpected token",
            ErrorCode::P3002 => "Expected token",
            ErrorCode::P3003 => "Invalid syntax",
            ErrorCode::P3004 => "Unterminated string",
            
            ErrorCode::I4001 => "File not found",
            ErrorCode::I4002 => "Permission denied",
            ErrorCode::I4003 => "Read error",
            ErrorCode::I4004 => "Write error",
            
            ErrorCode::P5001 => "Unrecoverable error",
            ErrorCode::P5002 => "Assertion failed",
            ErrorCode::P5003 => "Out of memory",
            
            _ => "Unknown error",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[E{}]", self.code())
    }
}

// Diagnostic message with error code
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub code: ErrorCode,
    pub message: String,
    pub span: Option<crate::ast::node::Span>,
}

impl Diagnostic {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Diagnostic {
            code,
            message: message.into(),
            span: None,
        }
    }
    
    pub fn with_span(mut self, span: crate::ast::node::Span) -> Self {
        self.span = Some(span);
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}: {}", self.code, self.code.category(), self.message)?;
        if let Some(span) = self.span {
            write!(f, " at line {}, column {}", span.line, span.column)?;
        }
        Ok(())
    }
}
