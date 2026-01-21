// Environment management for variable bindings.

use crate::vm::Value;
use std::collections::HashMap;

// Environment for variable bindings with lexical scoping.
//
// The environment maintains a stack of scopes, where each scope
// is a HashMap of variable names to values.
#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    // Stack of scopes (innermost scope is at the end)
    scopes: Vec<HashMap<String, Value>>,
}

impl Environment {
    // Create a new environment with a single global scope.
    pub fn new() -> Self {
        Environment {
            scopes: vec![HashMap::new()],
        }
    }

    // Push a new scope onto the stack.
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    // Pop the current scope from the stack.
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    // Set a variable in the current (innermost) scope.
    pub fn set(&mut self, name: String, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, value);
        }
    }

    // Get a variable, searching from innermost to outermost scope.
    pub fn get(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value.clone());
            }
        }
        None
    }

    // Update an existing variable (searching all scopes).
    pub fn update(&mut self, name: &str, value: Value) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), value);
                return true;
            }
        }
        false
    }

    // Check if a variable exists in any scope.
    pub fn contains(&self, name: &str) -> bool {
        self.scopes.iter().any(|scope| scope.contains_key(name))
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_scope() {
        let mut env = Environment::new();
        env.set("x".to_string(), Value::Int(42));
        
        assert_eq!(env.get("x"), Some(Value::Int(42)));
        assert_eq!(env.get("y"), None);
    }

    #[test]
    fn test_nested_scopes() {
        let mut env = Environment::new();
        
        env.set("x".to_string(), Value::Int(1));
        env.push_scope();
        env.set("y".to_string(), Value::Int(2));
        
        assert_eq!(env.get("x"), Some(Value::Int(1)));
        assert_eq!(env.get("y"), Some(Value::Int(2)));
        
        env.pop_scope();
        assert_eq!(env.get("x"), Some(Value::Int(1)));
        assert_eq!(env.get("y"), None);
    }

    #[test]
    fn test_shadowing() {
        let mut env = Environment::new();
        
        env.set("x".to_string(), Value::Int(1));
        env.push_scope();
        env.set("x".to_string(), Value::Int(2));
        
        assert_eq!(env.get("x"), Some(Value::Int(2)));
        
        env.pop_scope();
        assert_eq!(env.get("x"), Some(Value::Int(1)));
    }
}
