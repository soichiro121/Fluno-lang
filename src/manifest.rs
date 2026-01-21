use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct FlunoManifest {
    pub package: PackageInfo,
    #[serde(default)]
    pub dependencies: Dependencies,
}

#[derive(Debug, Deserialize, Default)]
pub struct Dependencies {
    #[serde(default)]
    pub rust: HashMap<String, toml::Value>,
    #[serde(default)]
    pub flux: HashMap<String, DependencySpec>,
}

#[derive(Debug, Deserialize)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default = "default_edition")]
    pub edition: String,
}

fn default_edition() -> String {
    "2021".to_string()
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum DependencySpec {
    Simple(String),
    Detailed {
        version: Option<String>,
        path: Option<String>,
        git: Option<String>,
        branch: Option<String>,
    },
}

impl FlunoManifest {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read manifest: {}", e))?;
        
        toml::from_str(&content)
            .map_err(|e| format!("Failed to parse manifest: {}", e))
    }
    
    pub fn generate_cargo_toml(&self, fluno_path: &str) -> String {
        let mut deps = String::new();
        
        deps.push_str("rand = \"0.8\"\n");
        deps.push_str("tokio = { version = \"1.0\", features = [\"full\"] }\n");
        deps.push_str(&format!("fluno = {{ path = \"{}\" }}\n", fluno_path));
        
        for (name, spec) in &self.dependencies.rust {
            match spec {
                toml::Value::String(version) => {
                    deps.push_str(&format!("{} = \"{}\"\n", name, version));
                }
                toml::Value::Table(table) => {
                    let table_str = toml::to_string(table).unwrap_or_default();
                    deps.push_str(&format!("{} = {{ {} }}\n", name, table_str.trim()));
                }
                _ => {}
            }
        }
        
        format!(r#"[package]
name = "{}"
version = "{}"
edition = "{}"

[dependencies]
{}
[profile.dev]
debug = false
"#, self.package.name, self.package.version, self.package.edition, deps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_manifest() {
        let toml_str = r#"
[package]
name = "test_project"
version = "0.1.0"

[dependencies.rust]
serde = "1.0"
"#;
        let manifest: FlunoManifest = toml::from_str(toml_str).unwrap();
        assert_eq!(manifest.package.name, "test_project");
        assert_eq!(manifest.package.version, "0.1.0");
        assert!(manifest.dependencies.rust.contains_key("serde"));
    }
    
    #[test]
    fn test_generate_cargo_toml() {
        let toml_str = r#"
[package]
name = "my_app"
version = "1.0.0"

[dependencies.rust]
serde = "1.0"
"#;
        let manifest: FlunoManifest = toml::from_str(toml_str).unwrap();
        let cargo = manifest.generate_cargo_toml("/path/to/fluno");
        
        assert!(cargo.contains("name = \"my_app\""));
        assert!(cargo.contains("serde = \"1.0\""));
        assert!(cargo.contains("fluno = { path = \"/path/to/fluno\" }"));
    }
}
