// src/typeck/mod.rs

use crate::ast::node::Program;
use crate::typeck::error::TypeError;
use crate::typeck::infer::TypeInfer;

pub mod env;
pub mod error;
pub mod infer;
pub mod prob;

#[derive(Debug)]
pub struct TypeChecker {
    infer: TypeInfer,
    errors: Vec<TypeError>,
}

impl TypeChecker {
    pub fn new() -> Self {
        let infer = TypeInfer::new();
        Self {
            infer,
            errors: Vec::new(),
        }
    }

    pub fn check_program(&mut self, program: &mut Program) -> Result<(), Vec<TypeError>> {
        self.errors.clear();
        if let Err(e) = self.infer.check_program(program) {
            self.errors.push(e);
        }
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}
