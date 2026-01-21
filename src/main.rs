// Fluno Programming Language Compiler/Interpreter
//
// This is the main entry point for the Fluno language toolchain.
// It provides a command-line interface for lexing, parsing, type checking,
// and executing Fluno programs.

use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::process;

use fluno::{Lexer, Parser as FluxParser, Interpreter, ast, Token, TokenKind};
use fluno::typeck::TypeChecker;

// Fluno Programming Language Compiler/Interpreter
#[derive(Parser)]
#[command(name = "fluno")]
#[command(author = "Fluno Development Team")]
#[command(version = "0.1.0")]
#[command(about = "Fluno language compiler and interpreter", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Run a Fluno source file
    Run {
        // Path to the Fluno source file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        // Show lexer output
        #[arg(long)]
        show_tokens: bool,

        // Show parser output (AST)
        #[arg(long)]
        show_ast: bool,

        // Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    // Perform lexical analysis only
    Lex {
        // Path to the Fluno source file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        // Pretty-print tokens
        #[arg(long)]
        pretty: bool,
    },

    // Perform parsing only
    Parse {
        // Path to the Fluno source file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        // Pretty-print AST
        #[arg(long)]
        pretty: bool,
    },

    // Type check a Fluno source file (not yet implemented)
    Check {
        // Path to the Fluno source file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    // Compile a Fluno source file to native code
    Compile {
        // Input Fluno source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        // Output file path
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        // Optimization level (0-3)
        #[arg(short = 'O', default_value = "0")]
        opt_level: u8,
    },

    // Start an interactive REPL (not yet implemented)
    Repl {
        // Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    // Build a Fluno project using fluno.toml
    Build {
        // Path to the project directory (defaults to current directory)
        #[arg(value_name = "DIR", default_value = ".")]
        path: PathBuf,

        // Optimization level (0-3)
        #[arg(short = 'O', default_value = "0")]
        opt_level: u8,
    },
}

fn main() {
    let cli = Cli::parse();

    let exit_code = match cli.command {
        Commands::Run { file, show_tokens, show_ast, verbose } => {
            run_file(&file, show_tokens, show_ast, verbose)
        }
        Commands::Lex { file, pretty } => {
            lex_file(&file, pretty)
        }
        Commands::Parse { file, pretty } => {
            parse_file(&file, pretty)
        }
        Commands::Check { file } => {
            check_file(&file)
        }
        Commands::Compile { input, output, opt_level } => {
            compile_file(&input, output.as_deref(), opt_level)
        }
        Commands::Repl { verbose } => {
            start_repl(verbose)
        }
        Commands::Build { path, opt_level } => {
            build_project(&path, opt_level)
        }
    };

    process::exit(exit_code);
}

fn run_file(path: &PathBuf, show_tokens: bool, show_ast: bool, verbose: bool) -> i32 {
    if verbose {
        println!("Running file: {}", path.display());
    }


    let source = match read_file(path) {
        Ok(s) => s,
        Err(code) => return code,
    };

    if verbose {
        println!("Source length: {} bytes", source.len());
    }


    let tokens = match lex_source(&source, path, show_tokens) {
        Ok(t) => t,
        Err(code) => return code,
    };

    if verbose {
        println!("Tokens: {} found", tokens.len());
    }


    let mut program = match parse_source(&source, path, show_ast) {
        Ok(p) => p,
        Err(code) => return code,
    };

    if verbose {
        println!("AST: {} top-level items", program.items.len());
    }

    // Type Checking
    if verbose {
        println!("\n=== Type Checking ===");
    }

    use fluno::typeck::TypeChecker;

    let mut type_checker = TypeChecker::new();
    match type_checker.check_program(&mut program) {
        Ok(_) => {
            if verbose {
                println!("Type check passed");
            }
        }
        Err(errors) => {
            eprintln!("\nType checking failed with {} error(s):", errors.len());
            for err in errors {
                eprintln!(
                    "{} at line {}, column {}",
                    err.code(),
                    err.span().line,
                    err.span().column
                );
                eprintln!("  → {}", err.message());
            }
            return 1; // 型エラーモード終了
        }
    }

    // Execution
    if verbose {
        println!("\n=== Execution ===");
    }

    let mut interpreter = Interpreter::new();

    interpreter.load_program_defs(&program);
    
    match interpreter.execute(program) {
        Ok(()) => {
            if verbose {
                println!("\nProgram executed successfully");
            }
            0
        }
        Err(e) => {
            eprintln!("\nRuntime Error:");
            eprintln!("{}", e);
            1
        }
    }
}

// Perform lexical analysis only.
fn lex_file(path: &PathBuf, pretty: bool) -> i32 {
    println!("Lexical analysis: {}", path.display());

    let source = match read_file(path) {
        Ok(s) => s,
        Err(code) => return code,
    };

    match lex_source(&source, path, true) {
        Ok(tokens) => {
            if pretty {
                print_tokens_pretty(&tokens);
            } else {
                print_tokens(&tokens);
            }
            0
        }
        Err(code) => code,
    }
}

// Perform parsing only.
fn parse_file(path: &PathBuf, pretty: bool) -> i32 {
    println!("Parsing: {}", path.display());

    let source = match read_file(path) {
        Ok(s) => s,
        Err(code) => return code,
    };

    match parse_source(&source, path, true) {
        Ok(program) => {
            if pretty {
                print_ast_pretty(&program);
            } else {
                println!("{:#?}", program);
            }
            0
        }
        Err(code) => code,
    }
}

// Type check a Fluno source file.
fn check_file(path: &PathBuf) -> i32 {
    println!("Type Checking: {}", path.display());

    // ファイル読込
    let source = match read_file(path) {
        Ok(s) => s,
        Err(code) => return code,
    };

    // パース（AST構築）
    let mut program = match parse_source(&source, path, false) {
        Ok(p) => p,
        Err(code) => return code,
    };

    // 型検査
    let mut type_checker = TypeChecker::new();
    match type_checker.check_program(&mut program) {
        Ok(_) => {
            println!("\nType check passed ✅");
            0
        }
        Err(errors) => {
            eprintln!("\nType checking failed with {} error(s):", errors.len());
            for err in errors {
                eprintln!(
                    "{} at line {}, column {}",
                    err.code(),
                    err.span().line,
                    err.span().column
                );
                eprintln!("  → {}", err.message());
            }
            1
        }
    }
}

// Start an interactive REPL.
fn start_repl(verbose: bool) -> i32 {
    eprintln!("REPL not yet implemented");
    if verbose {
        eprintln!("Verbose mode enabled");
    }
    1
}



// Read a source file from disk.
fn read_file(path: &PathBuf) -> Result<String, i32> {
    fs::read_to_string(path).map_err(|e| {
        eprintln!("Error reading file '{}': {}", path.display(), e);
        1
    })
}

// Perform lexical analysis on source code.
fn lex_source(source: &str, path: &PathBuf, print_output: bool) -> Result<Vec<Token>, i32> {
    let mut lexer = Lexer::new(source);

    match lexer.tokenize() {
        Ok(tokens) => {
            if print_output {
                println!("\n=== Tokens ===");
                for token in &tokens {
                    if token.kind != TokenKind::Eof {
                        println!("{:?}", token);
                    }
                }
                println!();
            }
            Ok(tokens)
        }
        Err(e) => {
            eprintln!("\nLexical Error in '{}':", path.display());
            eprintln!("{}", e);
            Err(1)
        }
    }
}

// Perform parsing on source code.
fn parse_source(source: &str, path: &PathBuf, print_output: bool) -> Result<ast::node::Program, i32> {
    let lexer = Lexer::new(source);
    let mut parser = FluxParser::new(lexer).map_err(|e| {
        eprintln!("\nParser Initialization Error in '{}':", path.display());
        eprintln!("{}", e);
        1
    })?;

    match parser.parse_program() {
        Ok(program) => {
            if print_output {
                println!("\n=== AST ===");
                println!("{:#?}", program);
                println!();
            }
            Ok(program)
        }
        Err(e) => {
            eprintln!("\nParse Error in '{}':", path.display());
            eprintln!("{}", e);
            Err(1)
        }
    }
}

// Print tokens in a simple format.
fn print_tokens(tokens: &[Token]) {
    for (i, token) in tokens.iter().enumerate() {
        if token.kind == TokenKind::Eof {
            continue;
        }
        println!("{:4}: {:?}", i, token);
    }
}

// Print tokens in a pretty format.
fn print_tokens_pretty(tokens: &[Token]) {
    println!("\n{:<6} {:<4} {:<4} {:<20} {:<30}", "Index", "Line", "Col", "Kind", "Text");
    println!("{}", "-".repeat(70));

    for (i, token) in tokens.iter().enumerate() {
        if token.kind == TokenKind::Eof {
            continue;
        }

        let text = token.text().unwrap_or("");
        let kind_str = format!("{:?}", token.kind);

        println!(
            "{:<6} {:<4} {:<4} {:<20} {:<30}",
            i,
            token.line,
            token.column,
            kind_str,
            if text.len() > 28 {
                format!("{}...", &text[..25])
            } else {
                text.to_string()
            }
        );
    }
    println!();
}

// Print AST in a pretty format.
fn print_ast_pretty(program: &ast::node::Program) {
    println!("\n=== Program Structure ===\n");
    println!("Top-level items: {}", program.items.len());
    println!();

    for (i, item) in program.items.iter().enumerate() {
        println!("Item {}: {}", i + 1, describe_item(item));
    }

    println!("\n=== Detailed AST ===\n");
    println!("{:#?}", program);
}

// Get a human-readable description of an AST item.
fn describe_item(item: &ast::node::Item) -> String {
    match item {
        ast::node::Item::Function(func) => {
            format!(
                "Function '{}' with {} parameter(s)",
                func.name.name,
                func.params.len()
            )
        }
        ast::node::Item::Struct(s) => {
            format!("Struct '{}' with {} field(s)", s.name.name, s.fields.len())
        }
        ast::node::Item::Enum(e) => {
            format!("Enum '{}' with {} variant(s)", e.name.name, e.variants.len())
        }
        ast::node::Item::TypeAlias(t) => {
            format!("Type alias '{}'", t.name.name)
        }
        ast::node::Item::Trait(t) => {
            format!("Trait '{}' with {} method(s)", t.name.name, t.methods.len())
        }
        ast::node::Item::Impl(i) => {
            format!("Impl block with {} item(s)", i.items.len())
        }
        ast::node::Item::Module(m) => {
            format!("Module '{}' with {} item(s)", m.name.name, m.items.len())
        }
        ast::node::Item::Import(imp) => {
            format!("Import: {}", imp.path.iter_idents()
                .map(|id| id.name.as_str())
                .collect::<Vec<_>>()
                .join("::"))
        }
        ast::node::Item::Extern(ext) => {
            format!("Extern block ({}) with {} function(s)", ext.abi, ext.functions.len())
        }
    }
}

fn compile_file(input: &PathBuf, output: Option<&std::path::Path>, opt_level: u8) -> i32 {
    println!("Compiling: {}", input.display());


    let source = match read_file(input) {
        Ok(s) => s,
        Err(code) => return code,
    };


    let mut program = match parse_source(&source, input, false) {
        Ok(p) => p,
        Err(code) => return code,
    };

    use fluno::typeck::TypeChecker;
    let mut type_checker = TypeChecker::new();
    if let Err(errors) = type_checker.check_program(&mut program) {
        eprintln!("Type checking warnings (Phase 1/2 limitations):");
        for err in errors {
            eprintln!(" - {}", err.message());
        }
        // Proceeding to compilation despite type errors.
        // This allows using prelude functions (like `clone`) not yet known to TypeChecker.
    }


    let output_path = if let Some(path) = output {
        path.to_path_buf()
    } else {
        // 入力が "hello.fluno" なら "hello" (拡張子なし) をデフォルト出力に
        if cfg!(windows) {
            input.with_extension("exe")
        } else {
            input.with_extension("")
        }
    };


    use fluno::compiler::Compiler;
    let compiler = Compiler::new(opt_level);
    match compiler.compile(&program, &output_path) {
        Ok(_) => {
            println!("Build successful.");
            0
        }
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            1
        }
    }
}

fn build_project(project_path: &PathBuf, opt_level: u8) -> i32 {
    use fluno::manifest::FlunoManifest;
    use fluno::compiler::Compiler;
    use fluno::typeck::TypeChecker;
    
    let manifest_path = project_path.join("fluno.toml");
    
    if !manifest_path.exists() {
        eprintln!("Error: fluno.toml not found in {}", project_path.display());
        eprintln!("Hint: Create a fluno.toml file or use 'fluno compile <file>' for single-file compilation.");
        return 1;
    }
    
    println!("Building project: {}", project_path.display());
    
    // Parse manifest
    let manifest = match FlunoManifest::from_file(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error reading fluno.toml: {}", e);
            return 1;
        }
    };
    
    println!("  Package: {} v{}", manifest.package.name, manifest.package.version);
    
    // Find main source file
    let src_dir = project_path.join("src");
    let main_file = src_dir.join("main.fln");
    
    if !main_file.exists() {
        eprintln!("Error: src/main.fln not found");
        eprintln!("Hint: Create src/main.fln with your main function.");
        return 1;
    }
    
    // Read source
    let source = match read_file(&main_file) {
        Ok(s) => s,
        Err(code) => return code,
    };
    
    // Parse
    let mut program = match parse_source(&source, &main_file, false) {
        Ok(p) => p,
        Err(code) => return code,
    };
    
    // Type check
    let mut type_checker = TypeChecker::new();
    if let Err(errors) = type_checker.check_program(&mut program) {
        eprintln!("Type checking warnings:");
        for err in errors {
            eprintln!(" - {}", err.message());
        }
    }
    
    // Setup build directory
    let build_dir = project_path.join(format!("{}_build", manifest.package.name));
    let src_build_dir = build_dir.join("src");
    std::fs::create_dir_all(&src_build_dir).ok();
    
    // Generate Cargo.toml from manifest
    let current_dir = std::env::current_dir().unwrap_or_default();
    let fluno_path = current_dir.to_string_lossy().replace("\\", "/");
    let cargo_toml_content = manifest.generate_cargo_toml(&fluno_path);
    
    let cargo_toml_path = build_dir.join("Cargo.toml");
    if let Err(e) = std::fs::write(&cargo_toml_path, cargo_toml_content) {
        eprintln!("Failed to write Cargo.toml: {}", e);
        return 1;
    }
    
    // Generate Rust code
    let mut codegen = fluno::compiler::codegen::CodeGenerator::new();
    let (rust_code, _source_map) = match codegen.generate(&program) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Code generation failed: {}", e);
            return 1;
        }
    };
    
    let main_rs_path = src_build_dir.join("main.rs");
    if let Err(e) = std::fs::write(&main_rs_path, rust_code) {
        eprintln!("Failed to write main.rs: {}", e);
        return 1;
    }
    
    println!("Generated Cargo project at: {}", build_dir.display());
    
    // Run cargo build
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build").current_dir(&build_dir);
    
    if opt_level > 0 {
        println!("Running: cargo build --release");
        cmd.arg("--release");
    } else {
        println!("Running: cargo build");
    }
    
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Failed to run cargo: {}", e);
            return 1;
        }
    };
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("{}", stderr);
        return 1;
    }
    
    // Copy binary to project root
    let bin_name = if cfg!(windows) { 
        format!("{}.exe", manifest.package.name) 
    } else { 
        manifest.package.name.clone() 
    };
    let target_subdir = if opt_level > 0 { "release" } else { "debug" };
    let src_bin_name = if cfg!(windows) { "generated_app.exe" } else { "generated_app" };
    
    // We need the package name in Cargo.toml matches
    let target_bin = build_dir.join("target").join(target_subdir).join(&bin_name);
    let output_bin = project_path.join(&bin_name);
    
    if let Err(e) = std::fs::copy(&target_bin, &output_bin) {
        eprintln!("Failed to copy binary: {}", e);
        eprintln!("  From: {}", target_bin.display());
        eprintln!("  To: {}", output_bin.display());
        return 1;
    }
    
    println!("Build successful: {}", output_bin.display());
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_file() {
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(temp_file, "fn main() {{}}").unwrap();

        let path = temp_file.path().to_path_buf();
        let content = read_file(&path);

        assert!(content.is_ok());
        assert!(content.unwrap().contains("fn main"));
    }

    #[test]
    fn test_lex_source() {
        let source = "fn main() { let x = 42; }";
        let path = PathBuf::from("test.flux");

        let result = lex_source(source, &path, false);
        assert!(result.is_ok());

        let tokens = result.unwrap();
        assert!(tokens.len() > 0);
    }

    #[test]
    fn test_parse_source() {
        let source = "fn main() { let x = 42; }";
        let path = PathBuf::from("test.flux");

        let result = parse_source(source, &path, false);
        assert!(result.is_ok());
    }
}
