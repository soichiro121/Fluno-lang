// src/compiler/mod.rs

pub mod codegen;

use crate::ast::node::Program;
use crate::error::{Error, FlunoResult};
use std::path::Path;
use std::process::Command;

use crate::compiler::codegen::SourceMap;

pub struct Compiler {
    opt_level: u8,
}

impl Compiler {
    pub fn new(opt_level: u8) -> Self {
        Self { opt_level }
    }

    pub fn compile(&self, program: &Program, output_path: &Path) -> FlunoResult<()> {
        println!("Compiling to Native Binary (via Rust)...");

        let build_dir = output_path.parent().unwrap().join(format!(
            "{}_build",
            output_path.file_stem().unwrap().to_string_lossy()
        ));
        let src_dir = build_dir.join("src");
        std::fs::create_dir_all(&src_dir).map_err(|e| Error::IoError(e.to_string()))?;
        let cargo_toml_path = build_dir.join("Cargo.toml");
        let current_dir = std::env::current_dir().map_err(|e| Error::IoError(e.to_string()))?;
        let fluno_path = current_dir.to_string_lossy().replace("\\", "/");

        let cargo_toml_content = format!(
            r#"
            [package]
            name = "generated_app"
            version = "0.1.0"
            edition = "2021"

            [dependencies]
            rand = "0.8"
            tokio = {{ version = "1.0", features = ["full"] }}
            fluno = {{ path = "{}" }}

            [profile.dev]
            debug = false
        "#,
            fluno_path
        );

        std::fs::write(&cargo_toml_path, cargo_toml_content)
            .map_err(|e| Error::IoError(e.to_string()))?;

        let mut codegen = codegen::CodeGenerator::new();
        let (rust_code, source_map) = codegen.generate(program)?;
        let main_rs_path = src_dir.join("main.rs");
        std::fs::write(&main_rs_path, rust_code).map_err(|e| Error::IoError(e.to_string()))?;

        println!("Generated Cargo project at: {}", build_dir.display());

        let mut cmd = Command::new("cargo");
        cmd.arg("build").current_dir(&build_dir);

        let is_release = self.opt_level > 0;
        if is_release {
            println!("Running: cargo build --release");
            cmd.arg("--release");
        } else {
            println!("Running: cargo build (debug)");
        }

        let output = cmd.output().map_err(|e| Error::IoError(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);

            let translation = Self::format_translation(&stderr, &source_map, output_path);

            if !translation.is_empty() {
                eprintln!("\n=== Compilation Error ===\n{}", translation);
            } else {
                eprintln!("{}", stderr);
            }

            return Err(Error::CompilationError("cargo build failed".to_string()));
        }

        let bin_name = if cfg!(windows) {
            "generated_app.exe"
        } else {
            "generated_app"
        };
        let target_subdir = if is_release { "release" } else { "debug" };
        let target_bin = build_dir.join("target").join(target_subdir).join(bin_name);

        std::fs::copy(&target_bin, output_path)
            .map_err(|e| Error::IoError(format!("Failed to copy binary: {}", e)))?;

        println!(
            "Compilation finished successfully: {}",
            output_path.display()
        );
        Ok(())
    }

    fn format_translation(stderr: &str, source_map: &SourceMap, original_path: &Path) -> String {
        let mut output = String::new();
        let mut seen_lines = std::collections::HashSet::new();

        for line in stderr.lines() {
            if let Some(idx) = line.find("main.rs:") {
                let rest = &line[idx + "main.rs:".len()..];

                let mut parts = rest.split(':');
                if let Some(line_str) = parts.next() {
                    if let Ok(rust_line) = line_str.parse::<usize>() {
                        let col_str = parts.next().unwrap_or("0");
                        let _rust_col = col_str.parse::<usize>().unwrap_or(0);

                        let mapping = source_map
                            .iter()
                            .filter(|(l, _)| *l <= rust_line)
                            .max_by_key(|(l, _)| l);

                        if let Some((_, span)) = mapping {
                            if seen_lines.contains(&span.line) {
                                continue;
                            }
                            seen_lines.insert(span.line);

                            output.push_str(&format!(
                                "\nError in {} at line {}:{}\n",
                                original_path.display(),
                                span.line,
                                span.column
                            ));
                            output.push_str(&format!(
                                "  -> (Internal Rust Error at main.rs:{})\n",
                                rust_line
                            ));

                            if let Ok(content) = std::fs::read_to_string(original_path) {
                                if let Some(code_line) = content.lines().nth(span.line - 1) {
                                    output.push_str("     | \n");
                                    output.push_str(&format!(
                                        "{:4} | {}\n",
                                        span.line,
                                        code_line.trim()
                                    ));
                                    output.push_str("     | \n");
                                }
                            }
                        }
                    }
                }
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::node::Span;
    use std::path::PathBuf;

    #[test]
    fn test_error_translation() {
        let source_map = vec![(
            10,
            Span {
                line: 2,
                column: 5,
                length: 10,
            },
        )];

        let stderr = "error[E0308]: mismatched types\n  --> src/main.rs:10:20\n   |\n10 |     let x: Int = \"error\";\n   |                  ^^^^^^^ expected integer, found `&str`";

        let path = PathBuf::from("test.fln");

        let output = Compiler::format_translation(stderr, &source_map, &path);

        assert!(output.contains("Error in test.fln at line 2:5"));
        assert!(output.contains("Internal Rust Error at main.rs:10"));
    }
}
