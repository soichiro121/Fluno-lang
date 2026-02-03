// src/lib.rs

pub mod error;
pub mod diagnostics;
pub mod lexer;
pub mod parser;
pub mod ast;
pub mod vm;
pub mod bytecode;
pub mod typeck;
pub mod gc;
pub mod resolve;
pub mod ad; 
pub mod compiler;
pub mod manifest;
pub mod prelude;
pub mod lsp;

#[cfg(test)]
mod gc_tests;
#[cfg(test)]
mod gc_cycle_tests;

pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;
pub use ast::node::{Program, Expression, Statement, Type};
pub use vm::{Interpreter, Value};
pub use error::{Error, FlunoResult};
pub use ad::types::ADFloat;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
