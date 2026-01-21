// Flux Type Checking Entry


use crate::ast::node::Program;
use crate::typeck::infer::TypeInfer;
use crate::typeck::error::TypeError;


pub mod infer;
pub mod env;
pub mod error;


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


    // 型検査エントリポイント
    pub fn check_program(&mut self, program: &mut Program) -> Result<(), Vec<TypeError>> {
        self.errors.clear();


        // infer.rs に実装した check_program を呼び出す
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
