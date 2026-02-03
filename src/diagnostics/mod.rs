//src/diagnostics/mod.rs

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    R1001,
    R1002,
    R1003,
    R1004,
    R1005,
    R1006,
    R1007,
    R1008,

    T2001,
    T2002,
    T2003,
    T2004,
    T2005,

    P3001,
    P3002,
    P3003,
    P3004,

    I4001,
    I4002,
    I4003,
    I4004,

    P5001,
    P5002,
    P5003,
}

impl ErrorCode {
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
            

        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[E{}]", self.code())
    }
}

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
