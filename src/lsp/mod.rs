// src/lsp/mod.rs

pub mod server;
pub mod symbol;

use crate::lsp::server::Backend;
use tower_lsp::{LspService, Server};

pub async fn run_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend::new(client));
    Server::new(stdin, stdout, socket).serve(service).await;
}
