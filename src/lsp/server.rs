// src/lsp/server.rs

use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::typeck::TypeChecker;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use crate::lsp::symbol::{SymbolKind as FlunoSymbolKind, SymbolTable};

pub struct Backend {
    pub client: Client,
    pub document_state: Arc<Mutex<DocumentState>>,
}

pub struct DocumentState {
    pub source: String,
    pub symbols: SymbolTable,
}

impl Backend {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            document_state: Arc::new(Mutex::new(DocumentState {
                source: String::new(),
                symbols: SymbolTable::new(),
            })),
        }
    }

    async fn validate_document(&self, uri: Url, source: &str) {
        let mut diagnostics = Vec::new();
        let mut symbols = SymbolTable::new();

        let lexer = Lexer::new(source);
        let mut parser = match Parser::new(lexer) {
            Ok(p) => p,
            Err(e) => {
                diagnostics.push(Diagnostic {
                    range: Range::new(
                        Position::new(
                            e.line().unwrap_or(1).saturating_sub(1) as u32,
                            e.column().unwrap_or(1).saturating_sub(1) as u32,
                        ),
                        Position::new(
                            e.line().unwrap_or(1).saturating_sub(1) as u32,
                            e.column().unwrap_or(1).saturating_sub(1) as u32 + 1,
                        ),
                    ),
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: Some(NumberOrString::String(e.error_code().to_string())),
                    message: e.to_string(),
                    ..Default::default()
                });
                self.client
                    .publish_diagnostics(uri, diagnostics, None)
                    .await;
                return;
            }
        };

        match parser.parse_program() {
            Ok(mut program) => {
                self.collect_symbols(&program.items, &mut symbols);

                let mut type_checker = TypeChecker::new();
                if let Err(errors) = type_checker.check_program(&mut program) {
                    for err in errors {
                        let span = err.span();
                        diagnostics.push(Diagnostic {
                            range: Range::new(
                                Position::new(
                                    span.line.saturating_sub(1) as u32,
                                    span.column.saturating_sub(1) as u32,
                                ),
                                Position::new(
                                    span.line.saturating_sub(1) as u32,
                                    (span.column.saturating_sub(1) + span.length) as u32,
                                ),
                            ),
                            severity: Some(DiagnosticSeverity::ERROR),
                            code: Some(NumberOrString::String(err.code().as_str().to_string())),
                            message: err.message(),
                            ..Default::default()
                        });
                    }
                }
            }
            Err(e) => {
                diagnostics.push(Diagnostic {
                    range: Range::new(
                        Position::new(
                            e.line().unwrap_or(1).saturating_sub(1) as u32,
                            e.column().unwrap_or(1).saturating_sub(1) as u32,
                        ),
                        Position::new(
                            e.line().unwrap_or(1).saturating_sub(1) as u32,
                            e.column().unwrap_or(1).saturating_sub(1) as u32 + 1,
                        ),
                    ),
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: Some(NumberOrString::String(e.error_code().to_string())),
                    message: e.to_string(),
                    ..Default::default()
                });
            }
        }

        {
            let mut state = self.document_state.lock().await;
            state.symbols = symbols;
        }

        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }

    fn collect_symbols(&self, items: &[crate::ast::node::Item], symbols: &mut SymbolTable) {
        for item in items {
            match item {
                crate::ast::node::Item::Function(f) => {
                    let range = self.span_to_range(f.span);
                    let sel_range = self.span_to_range(f.name.span);
                    symbols.add(
                        f.name.name.clone(),
                        FlunoSymbolKind::Function,
                        f.name.span,
                        range,
                        sel_range,
                        Some(format!("fn {}", f.name.name)),
                    );
                }
                crate::ast::node::Item::Struct(s) => {
                    let range = self.span_to_range(s.span);
                    let sel_range = self.span_to_range(s.name.span);
                    symbols.add(
                        s.name.name.clone(),
                        FlunoSymbolKind::Struct,
                        s.name.span,
                        range,
                        sel_range,
                        Some(format!("struct {}", s.name.name)),
                    );
                }
                crate::ast::node::Item::Enum(e) => {
                    let range = self.span_to_range(e.span);
                    let sel_range = self.span_to_range(e.name.span);
                    symbols.add(
                        e.name.name.clone(),
                        FlunoSymbolKind::Enum,
                        e.name.span,
                        range,
                        sel_range,
                        Some(format!("enum {}", e.name.name)),
                    );
                }
                crate::ast::node::Item::Trait(t) => {
                    let range = self.span_to_range(t.span);
                    let sel_range = self.span_to_range(t.name.span);
                    symbols.add(
                        t.name.name.clone(),
                        FlunoSymbolKind::Trait,
                        t.name.span,
                        range,
                        sel_range,
                        Some(format!("trait {}", t.name.name)),
                    );
                }
                crate::ast::node::Item::TypeAlias(t) => {
                    let range = self.span_to_range(t.span);
                    let sel_range = self.span_to_range(t.name.span);
                    symbols.add(
                        t.name.name.clone(),
                        FlunoSymbolKind::TypeAlias,
                        t.name.span,
                        range,
                        sel_range,
                        Some(format!("type {}", t.name.name)),
                    );
                }
                crate::ast::node::Item::Module(m) => {
                    let range = self.span_to_range(m.span);
                    let sel_range = self.span_to_range(m.name.span);
                    symbols.add(
                        m.name.name.clone(),
                        FlunoSymbolKind::Module,
                        m.name.span,
                        range,
                        sel_range,
                        Some(format!("mod {}", m.name.name)),
                    );
                    self.collect_symbols(&m.items, symbols);
                }
                _ => {}
            }
        }
    }

    fn span_to_range(&self, span: crate::ast::node::Span) -> Range {
        Range::new(
            Position::new(
                span.line.saturating_sub(1) as u32,
                span.column.saturating_sub(1) as u32,
            ),
            Position::new(
                span.line.saturating_sub(1) as u32,
                (span.column.saturating_sub(1) + span.length) as u32,
            ),
        )
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Fluno Language Server initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let source = params.text_document.text;
        let uri = params.text_document.uri;
        {
            let mut state = self.document_state.lock().await;
            state.source = source.clone();
        }
        self.validate_document(uri, &source).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.first() {
            let source = change.text.clone();
            let uri = params.text_document.uri;
            {
                let mut state = self.document_state.lock().await;
                state.source = source.clone();
            }
            self.validate_document(uri, &source).await;
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let position = params.text_document_position_params.position;
        let state = self.document_state.lock().await;

        for sym in &state.symbols.symbols {
            if sym.span.line == (position.line + 1) as usize
                && sym.span.column <= (position.character + 1) as usize
                && (sym.span.column + sym.name.len()) >= (position.character + 1) as usize
            {
                return Ok(Some(Hover {
                    contents: HoverContents::Scalar(MarkedString::String(
                        sym.detail.clone().unwrap_or_else(|| sym.name.clone()),
                    )),
                    range: Some(sym.selection_range),
                }));
            }
        }
        Ok(None)
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let position = params.text_document_position_params.position;
        let uri = params.text_document_position_params.text_document.uri;
        let state = self.document_state.lock().await;

        for sym in &state.symbols.symbols {
            if sym.span.line == (position.line + 1) as usize
                && sym.span.column <= (position.character + 1) as usize
                && (sym.span.column + sym.name.len()) >= (position.character + 1) as usize
            {
                return Ok(Some(GotoDefinitionResponse::Scalar(Location::new(
                    uri,
                    sym.selection_range,
                ))));
            }
        }
        Ok(None)
    }

    async fn document_symbol(
        &self,
        _params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let state = self.document_state.lock().await;
        let mut symbols = Vec::new();

        for sym in &state.symbols.symbols {
            #[allow(deprecated)]
            symbols.push(DocumentSymbol {
                name: sym.name.clone(),
                detail: sym.detail.clone(),
                kind: match sym.kind {
                    FlunoSymbolKind::Function => tower_lsp::lsp_types::SymbolKind::FUNCTION,
                    FlunoSymbolKind::Struct => tower_lsp::lsp_types::SymbolKind::STRUCT,
                    FlunoSymbolKind::Enum => tower_lsp::lsp_types::SymbolKind::ENUM,
                    FlunoSymbolKind::Trait => tower_lsp::lsp_types::SymbolKind::INTERFACE,
                    FlunoSymbolKind::TypeAlias => tower_lsp::lsp_types::SymbolKind::TYPE_PARAMETER,
                    FlunoSymbolKind::Module => tower_lsp::lsp_types::SymbolKind::MODULE,
                    _ => tower_lsp::lsp_types::SymbolKind::VARIABLE,
                },
                tags: None,
                deprecated: None,
                range: sym.range,
                selection_range: sym.selection_range,
                children: None,
            });
        }

        Ok(Some(DocumentSymbolResponse::Nested(symbols)))
    }

    async fn completion(&self, _params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let state = self.document_state.lock().await;
        let mut items = Vec::new();

        let keywords = vec![
            "fn", "let", "if", "else", "match", "while", "for", "loop", "break", "continue",
            "return", "struct", "enum", "impl", "trait", "type", "mod", "use", "import", "async",
            "await",
        ];
        for kw in keywords {
            items.push(CompletionItem {
                label: kw.to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                ..Default::default()
            });
        }

        for sym in &state.symbols.symbols {
            items.push(CompletionItem {
                label: sym.name.clone(),
                kind: Some(match sym.kind {
                    FlunoSymbolKind::Function => CompletionItemKind::FUNCTION,
                    FlunoSymbolKind::Struct => CompletionItemKind::STRUCT,
                    FlunoSymbolKind::Enum => CompletionItemKind::ENUM,
                    FlunoSymbolKind::Trait => CompletionItemKind::INTERFACE,
                    FlunoSymbolKind::TypeAlias => CompletionItemKind::CLASS,
                    FlunoSymbolKind::Module => CompletionItemKind::MODULE,
                    _ => CompletionItemKind::VARIABLE,
                }),
                detail: sym.detail.clone(),
                ..Default::default()
            });
        }

        Ok(Some(CompletionResponse::Array(items)))
    }
}
