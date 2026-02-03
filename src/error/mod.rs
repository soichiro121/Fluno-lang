use crate::typeck::error::TypeError;
use crate::vm::RuntimeError;
use crate::parser::ParseError;
use std::fmt;
use std::io;

#[derive(Debug, Clone)]
pub enum Error {
    RuntimeError(RuntimeError),
    
    TypeError(TypeError),
    
    ParseError(ParseError),
    
    IoError(String),
    
    Panic(String),
    
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

pub type FlunoResult<T> = Result<T, Error>;
