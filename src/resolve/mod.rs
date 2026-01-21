// Name resolution / module expansion for Fluno.
//
// Phase1: flatten nested modules into top-level "qualified names".
// - `mod a { fn f() { ... } }`  ==> creates a top-level `fn a::f() { ... }`
// - also flattens nested modules recursively: `a::b::g`
//
// Notes:
// - Current AST represents `ImportStmt.path` as `Vec<Identifier>` (not `Path`),
//   so we don't attach resolution IDs yet.  We only normalize the item namespace
//   so later phases (typeck/env) can look up `a::f` easily.

use crate::ast::node::{Identifier, Item, ModuleDef, Program, Span};

#[derive(Debug, Clone)]
pub struct ResolveError {
    pub message: String,
    pub span: Span,
}

pub type ResolveResult<T> = Result<T, Vec<ResolveError>>;

// Phase1 resolver: collect/flatten module items into top-level qualified items.
//
// This function mutates `program.items`:
// - keeps original non-module items
// - expands module contents into additional items with qualified names
// - removes the original `Item::Module` nodes (so the rest of pipeline sees a flat namespace)
pub fn resolve_program(program: &mut Program) -> ResolveResult<()> {
    let mut errors = Vec::<ResolveError>::new();

    // output items after flattening
    let mut out = Vec::<Item>::new();

    // walk original items
    for item in program.items.clone() {
        match item {
            Item::Module(m) => {
                flatten_module(&m, &mut Vec::new(), &mut out, &mut errors);
            }
            other => out.push(other),
        }
    }

    if errors.is_empty() {
        program.items = out;
        Ok(())
    } else {
        Err(errors)
    }
}

// Recursively flatten a module into top-level items, prefixing names with `prefix + module.name`.
fn flatten_module(
    module: &ModuleDef,
    prefix: &mut Vec<Identifier>,
    out: &mut Vec<Item>,
    errors: &mut Vec<ResolveError>,
) {
    // push this module name onto prefix
    prefix.push(module.name.clone());

    for item in module.items.clone() {
        match item {
            Item::Module(inner) => {
                flatten_module(&inner, prefix, out, errors);
            }

            Item::Function(mut f) => {
                qualify_ident(&mut f.name, prefix);
                out.push(Item::Function(f));
            }

            Item::Struct(mut s) => {
                qualify_ident(&mut s.name, prefix);
                out.push(Item::Struct(s));
            }

            Item::Enum(mut e) => {
                qualify_ident(&mut e.name, prefix);
                out.push(Item::Enum(e));
            }

            Item::TypeAlias(mut t) => {
                qualify_ident(&mut t.name, prefix);
                out.push(Item::TypeAlias(t));
            }

            Item::Trait(mut t) => {
                qualify_ident(&mut t.name, prefix);
                out.push(Item::Trait(t));
            }

            Item::Impl(i) => {
                // For Phase1: keep impl as-is (method lookup/typeck may key off self type).
                // If later you want impls to live in module namespaces, this needs redesign.
                out.push(Item::Impl(i));
            }

            Item::Import(imp) => {
                // Imports are left as-is for now. (Typeck already has handling.)
                out.push(Item::Import(imp));
            }

            Item::Extern(mut ext) => {
                for f in &mut ext.functions {
                    qualify_ident(&mut f.name, prefix);
                }
                out.push(Item::Extern(ext));
            }
        }
    }

    // pop this module name
    prefix.pop();
}

// Rewrite an Identifier like `f` into `a::b::f` by embedding `::` in `name`.
//
// This is a Phase1 hack to avoid changing the AST to a proper Path type.
// Typeck currently builds keys using `a::b::f` style strings, so this matches that world.
fn qualify_ident(ident: &mut Identifier, prefix: &[Identifier]) {
    if prefix.is_empty() {
        return;
    }

    let mut parts: Vec<String> = prefix.iter().map(|i| i.name.clone()).collect();
    parts.push(ident.name.clone());

    ident.name = parts.join("::");
}
