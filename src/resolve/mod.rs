// src/resolve/mod.rs

use crate::ast::node::{Identifier, Item, ModuleDef, Program, Span};

#[derive(Debug, Clone)]
pub struct ResolveError {
    pub message: String,
    pub span: Span,
}

pub type ResolveResult<T> = Result<T, Vec<ResolveError>>;

pub fn resolve_program(program: &mut Program) -> ResolveResult<()> {
    let mut errors = Vec::<ResolveError>::new();

    let mut out = Vec::<Item>::new();

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

fn flatten_module(
    module: &ModuleDef,
    prefix: &mut Vec<Identifier>,
    out: &mut Vec<Item>,
    errors: &mut Vec<ResolveError>,
) {
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
                out.push(Item::Impl(i));
            }

            Item::Import(imp) => {
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

    prefix.pop();
}

fn qualify_ident(ident: &mut Identifier, prefix: &[Identifier]) {
    if prefix.is_empty() {
        return;
    }

    let mut parts: Vec<String> = prefix.iter().map(|i| i.name.clone()).collect();
    parts.push(ident.name.clone());

    ident.name = parts.join("::");
}