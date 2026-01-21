// Fluno - Fluno Programming Language Compiler/Interpreter

pub mod error;
pub mod diagnostics;
pub mod lexer;
pub mod parser;
pub mod ast;
pub mod vm;
pub mod typeck;
pub mod gc;
pub mod resolve;
pub mod ad; 
pub mod compiler;
pub mod manifest;
pub mod prelude;

// Re-export main types for convenience
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;
pub use ast::node::{Program, Expression, Statement, Type};
pub use vm::{Interpreter, Value};
pub use error::{Error, FlunoResult};
pub use ad::types::ADFloat;

// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
