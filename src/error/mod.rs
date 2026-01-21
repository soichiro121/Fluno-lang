// Unified error handling for Fluno language
//
// This module provides a centralized error type that encompasses
// all error categories in the Fluno compiler and runtime.

use crate::typeck::error::TypeError;
use crate::vm::RuntimeError;
use crate::parser::ParseError;
use std::fmt;
use std::io;

// Unified error type for Fluno language
#[derive(Debug, Clone)]
pub enum Error {
    // Runtime execution errors
    RuntimeError(RuntimeError),
    
    // Type checking errors
    TypeError(TypeError),
    
    // Parsing errors
    ParseError(ParseError),
    
    // I/O errors (file operations, etc.)
    IoError(String),
    
    // Panic errors (unrecoverable runtime failures)
    Panic(String),
    
    // Generic error with message
    Generic(String),

    CompilationError(String), 
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::RuntimeError(e) => write!(f, "Runtime Error: {}", e),
            Error::TypeError(e) => write!(f, "Type Error: {}", e),
            Error::ParseError(e) => write!(f, "Parse Error: {}", e),
            Error::IoError(msg) => write!(f, "I/O Error: {}", msg),
            Error::Panic(msg) => write!(f, "Panic: {}", msg),
            Error::Generic(msg) => write!(f, "Error: {}", msg),
            Error::CompilationError(msg) => write!(f, "Compilation Error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

// ========================================
// From implementations for automatic conversion
// ========================================

impl From<RuntimeError> for Error {
    fn from(e: RuntimeError) -> Self {
        Error::RuntimeError(e)
    }
}

impl From<TypeError> for Error {
    fn from(e: TypeError) -> Self {
        Error::TypeError(e)
    }
}

impl From<ParseError> for Error {
    fn from(e: ParseError) -> Self {
        Error::ParseError(e)
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::IoError(e.to_string())
    }
}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::Generic(msg)
    }
}

impl From<&str> for Error {
    fn from(msg: &str) -> Self {
        Error::Generic(msg.to_string())
    }
}

// Result type alias with unified Error
pub type FlunoResult<T> = Result<T, Error>;
