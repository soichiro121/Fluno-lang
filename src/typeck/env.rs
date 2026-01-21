// Type environment for type checking.

use crate::ast::node::{DefId, Span, Type, TypeParameter, WherePredicate};
use crate::typeck::error::{TypeError, TypeResult};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TypeEnv {
    scopes: Vec<HashMap<String, Type>>,
    next_def: u32,
    symbols: HashMap<String, DefId>,
    pub defs: HashMap<DefId, ItemDef>,
    pub methods: HashMap<DefId, HashMap<String, (Type, DefId)>>,
    pub impls: Vec<ImplDef>,
}


#[derive(Debug, Clone)]
pub struct ImplDef {
  pub impl_id: usize,
  pub trait_def: DefId,
  pub typeparams: Vec<TypeParameter>,
  pub self_ty: Type,
  pub where_preds: Vec<WherePredicate>,
  pub assoc_bindings: HashMap<String, Type>,
  pub methods: HashMap<String, (Type, DefId)>,
  pub span: Span,
}

#[derive(Debug, Clone)]
pub enum WherePred {
  TraitBound { ty: Type, trait_def: DefId },
}

#[derive(Debug, Clone)]
pub struct TypeAliasDef {
    pub typeparams: Vec<crate::ast::node::TypeParameter>,
    pub ty: Type,
}


#[derive(Debug, Clone)]
pub struct TraitDefInfo {
    pub typeparams: Vec<crate::ast::node::TypeParameter>,
    pub assoc_types: HashMap<String, Type>,
    pub methods: HashMap<String, Type>,
}


#[derive(Debug, Clone)]
pub struct EnumDefInfo {
    pub typeparams: Vec<crate::ast::node::TypeParameter>,
    pub variants: HashMap<String, VariantInfo>,
}

#[derive(Debug, Clone)]
pub enum VariantInfo {
    Unit,
    Tuple(Vec<Type>),
    Struct(Vec<(String, Type)>),
}

#[derive(Debug, Clone)]
pub struct StructDefInfo {
    pub typeparams: Vec<crate::ast::node::TypeParameter>,
    pub fields: HashMap<String, Type>,
}

#[derive(Debug, Clone)]
pub enum ItemDef {
    Function(Type),
    Struct(StructDefInfo),
    Enum(EnumDefInfo),
    Trait(TraitDefInfo),
    TypeAlias(TypeAliasDef),
}

impl TypeEnv {
    pub fn new() -> Self {
        let env = TypeEnv {
            scopes: vec![HashMap::new()],
            next_def: 0,
            symbols: HashMap::new(),
            defs: HashMap::new(),
            methods: HashMap::new(),
            impls: Vec::new(),
        };
        env
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() <= 1 {
            panic!("Cannot pop global scope");
        }
        self.scopes.pop();
    }

    pub fn define(&mut self, name: String, ty: Type) -> TypeResult<()> {
        let current_scope = self.scopes.last_mut().expect("No scope available");
        if current_scope.contains_key(&name) {
            return Err(TypeError::DuplicateDefinition {
                name,
                span: Span::initial(),
            });
        }
        current_scope.insert(name, ty);
        Ok(())
    }


    pub fn lookup_trait_method(&self, traitid: DefId, name: &str) -> Option<Type> {
        match self.get_def(traitid)? {
            ItemDef::Trait(info) => info.methods.get(name).cloned(),
            _ => None,
        }
    }

    pub fn lookup_trait_assoc_type(&self, traitid: DefId, name: &str) -> Option<Type> {
        match self.get_def(traitid)? {
            ItemDef::Trait(info) => info.assoc_types.get(name).cloned(),
            _ => None,
        }
    }
    
    pub fn lookup(&self, name: &str) -> Option<Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty.clone());
            }
        }
        None
    }

    pub fn update(&mut self, name: &str, ty: Type) -> TypeResult<()> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), ty);
                return Ok(());
            }
        }
        Err(TypeError::UndefinedVariable {
            name: name.to_string(),
            span: Span::initial(),
        })
    }

    pub fn depth(&self) -> usize {
        self.scopes.len()
    }

    pub fn is_defined_in_current_scope(&self, name: &str) -> bool {
        self.scopes
            .last()
            .map(|scope| scope.contains_key(name))
            .unwrap_or(false)
    }

    pub fn current_scope_variables(&self) -> Vec<&str> {
        self.scopes
            .last()
            .map(|scope| scope.keys().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    fn alloc_defid(&mut self) -> DefId {
        let id = self.next_def;
        self.next_def += 1;
        DefId(id)
    }

    pub fn insert_def(&mut self, fq: String, def: ItemDef, span: Span) -> TypeResult<DefId> {
        if self.symbols.contains_key(&fq) {
            return Err(TypeError::DuplicateDefinition { name: fq, span });
        }
        let id = self.alloc_defid();
        self.symbols.insert(fq, id);
        self.defs.insert(id, def);
        Ok(id)
    }

    pub fn insert_trait_def(
        &mut self,
        name: String,
        info: TraitDefInfo,
        span: Span,
    ) -> TypeResult<DefId> {
        if let Some(existing) = self.resolve_def(&name) {
            match self.get_def(existing) {
                Some(ItemDef::Trait(_)) => return Ok(existing),
                _ => {
                    return Err(TypeError::DuplicateDefinition { name, span });
                }
            }
        }
        self.insert_def(name, ItemDef::Trait(info), span)
    }

    pub fn resolve_def(&self, fq: &str) -> Option<DefId> {
        self.symbols.get(fq).copied()
    }

    pub fn get_def(&self, id: DefId) -> Option<&ItemDef> {
        self.defs.get(&id)
    }
    pub fn alias_def(&mut self, local: String, target: DefId, span: Span) -> TypeResult<()> {
        if self.symbols.contains_key(&local) {
            return Err(TypeError::DuplicateDefinition { name: local, span });
        }
        self.symbols.insert(local, target);
        Ok(())
    }
    pub fn insert_method(
        &mut self,
        self_type_def: DefId,
        method_name: String,
        method_ty: Type,
        method_def_id: DefId,
        span: Span,
    ) -> TypeResult<DefId> {
        let entry = self.methods.entry(self_type_def).or_default();
        if entry.contains_key(&method_name) {
            return Err(TypeError::DuplicateDefinition {
                name: method_name,
                span,
            });
        }
        entry.insert(method_name, (method_ty, method_def_id)); 
        Ok(method_def_id)
    }

    pub fn alloc_def_id(&mut self) -> DefId {
        let id = self.next_def;
        self.next_def += 1;
        DefId(id)
    }


    pub fn lookup_method_defid(&self, self_type_def: DefId, method_name: &str) -> Option<DefId> {
        self.lookup_method(self_type_def, method_name)
            .map(|(_, defid)| *defid)
    }

    pub fn lookup_method(
        &self,
        self_type_def: DefId,
        method_name: &str,
    ) -> Option<&(Type, DefId)> {
        self.methods
            .get(&self_type_def)
            .and_then(|m| m.get(method_name))
    }


}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_env() {
        let env = TypeEnv::new();
        assert_eq!(env.depth(), 1);
    }

    #[test]
    fn test_define_and_lookup() {
        let mut env = TypeEnv::new();
        env.define("x".to_string(), Type::Int).unwrap();
        assert_eq!(env.lookup("x"), Some(Type::Int));
    }

    #[test]
    fn test_duplicate_definition() {
        let mut env = TypeEnv::new();
        env.define("x".to_string(), Type::Int).unwrap();
        let result = env.define("x".to_string(), Type::Float);
        assert!(result.is_err());
    }

    #[test]
    fn test_scopes() {
        let mut env = TypeEnv::new();
        env.define("x".to_string(), Type::Int).unwrap();

        env.push_scope();
        env.define("x".to_string(), Type::Float).unwrap();
        assert_eq!(env.lookup("x"), Some(Type::Float));

        env.pop_scope();
        assert_eq!(env.lookup("x"), Some(Type::Int));
    }

    #[test]
    fn test_update() {
        let mut env = TypeEnv::new();
        env.define("x".to_string(), Type::Int).unwrap();
        env.update("x", Type::Float).unwrap();
        assert_eq!(env.lookup("x"), Some(Type::Float));
    }

    #[test]
    fn test_update_undefined() {
        let mut env = TypeEnv::new();
        let result = env.update("x", Type::Int);
        assert!(result.is_err());
    }
}
