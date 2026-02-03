// src/lsp/symbol.rs

#[derive(Debug, Clone)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    TypeAlias,
    Module,
    Variable,
    Field,
}

#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: SymbolKind,
    pub span: crate::ast::node::Span, // Original anchor span
    pub range: tower_lsp::lsp_types::Range, // Full range of the item
    pub selection_range: tower_lsp::lsp_types::Range, // Range of the identifier
    pub detail: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub symbols: Vec<SymbolInfo>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self { symbols: Vec::new() }
    }

    pub fn add(&mut self, name: String, kind: SymbolKind, span: crate::ast::node::Span, range: tower_lsp::lsp_types::Range, selection_range: tower_lsp::lsp_types::Range, detail: Option<String>) {
        self.symbols.push(SymbolInfo { name, kind, span, range, selection_range, detail });
    }
}
