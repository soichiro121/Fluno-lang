// src/lib.rs

pub mod ad;
pub mod ast;
pub mod bytecode;
pub mod compiler;
pub mod diagnostics;
pub mod error;
pub mod gc;
pub mod lexer;
pub mod lsp;
pub mod manifest;
pub mod parser;
pub mod prelude;
pub mod resolve;
pub mod typeck;
pub mod vm;

#[cfg(test)]
mod gc_cycle_tests;
#[cfg(test)]
mod gc_store_tests;
#[cfg(test)]
mod gc_tests;

pub use ad::types::ADFloat;
pub use ast::node::{Expression, Program, Statement, Type};
pub use error::{Error, FlunoResult};
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;
pub use vm::{Interpreter, Value};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
