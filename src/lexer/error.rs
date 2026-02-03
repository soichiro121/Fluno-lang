// src/lexer/error.rs

use thiserror::Error;

pub type LexResult<T> = Result<T, LexError>;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum LexError {
    #[error("[E0001] Invalid character '{character}' at line {line}, column {column}")]
    InvalidCharacter {
        character: char,
        line: usize,
        column: usize,
    },

    #[error("[E0002] Unterminated string literal starting at line {line}, column {column}")]
    UnterminatedString {
        line: usize,
        column: usize,
    },

    #[error("[E0003] Invalid escape sequence '\\{sequence}' in string at line {line}, column {column}")]
    InvalidEscapeSequence {
        sequence: String,
        line: usize,
        column: usize,
    },

    #[error("[E0004] Invalid Unicode escape sequence '\\u{{{sequence}}}' at line {line}, column {column}: {reason}")]
    InvalidUnicodeEscape {
        sequence: String,
        reason: String,
        line: usize,
        column: usize,
    },

    #[error("[E0005] Invalid integer literal '{literal}' at line {line}, column {column}: {reason}")]
    InvalidIntegerLiteral {
        literal: String,
        reason: String,
        line: usize,
        column: usize,
    },

    #[error("[E0006] Invalid float literal '{literal}' at line {line}, column {column}: {reason}")]
    InvalidFloatLiteral {
        literal: String,
        reason: String,
        line: usize,
        column: usize,
    },

    #[error("[E0007] Integer literal '{literal}' overflows at line {line}, column {column}")]
    IntegerOverflow {
        literal: String,
        line: usize,
        column: usize,
    },

    #[error("[E0008] Invalid digit '{digit}' for base-{base} number at line {line}, column {column}")]
    InvalidDigitForBase {
        digit: char,
        base: u32,
        line: usize,
        column: usize,
    },

    #[error("[E0009] Unterminated block comment starting at line {line}, column {column}")]
    UnterminatedBlockComment {
        line: usize,
        column: usize,
    },

    #[error("[E0010] Invalid UTF-8 encoding at line {line}, column {column}")]
    InvalidUtf8 {
        line: usize,
        column: usize,
    },

    #[error("[E0011] Invalid identifier '{identifier}' at line {line}, column {column}: {reason}")]
    InvalidIdentifier {
        identifier: String,
        reason: String,
        line: usize,
        column: usize,
    },

    #[error("[E0012] Expected digits after '{prefix}' prefix at line {line}, column {column}")]
    ExpectedDigitsAfterPrefix {
        prefix: String,
        line: usize,
        column: usize,
    },

    #[error("[E0013] Multiple decimal points in number literal at line {line}, column {column}")]
    MultipleDecimalPoints {
        line: usize,
        column: usize,
    },

    #[error("[E0014] Invalid exponent in float literal at line {line}, column {column}: {reason}")]
    InvalidExponent {
        reason: String,
        line: usize,
        column: usize,
    },

    #[error("[E0015] Unexpected end of file at line {line}, column {column}")]
    UnexpectedEof {
        line: usize,
        column: usize,
    },

    #[error("[E0016] Byte Order Mark (BOM) is not allowed in Flux source files at line {line}, column {column}")]
    BomNotAllowed {
        line: usize,
        column: usize,
    },

    #[error("[E0017] Tab character found at line {line}, column {column}; use spaces for indentation")]
    TabInIndentation {
        line: usize,
        column: usize,
    },

    #[error("[E0018] Invalid suffix '{suffix}' on numeric literal at line {line}, column {column}")]
    InvalidNumberSuffix {
        suffix: String,
        line: usize,
        column: usize,
    },

    #[error("[E0019] Bare carriage return (CR) without line feed (LF) at line {line}, column {column}")]
    BareCarriageReturn {
        line: usize,
        column: usize,
    },

    #[error("[E0020] Lexical error at line {line}, column {column}: {message}")]
    Generic {
        message: String,
        line: usize,
        column: usize,
    },
}

impl LexError {
    pub fn error_code(&self) -> &'static str {
        match self {
            LexError::InvalidCharacter { .. } => "E0001",
            LexError::UnterminatedString { .. } => "E0002",
            LexError::InvalidEscapeSequence { .. } => "E0003",
            LexError::InvalidUnicodeEscape { .. } => "E0004",
            LexError::InvalidIntegerLiteral { .. } => "E0005",
            LexError::InvalidFloatLiteral { .. } => "E0006",
            LexError::IntegerOverflow { .. } => "E0007",
            LexError::InvalidDigitForBase { .. } => "E0008",
            LexError::UnterminatedBlockComment { .. } => "E0009",
            LexError::InvalidUtf8 { .. } => "E0010",
            LexError::InvalidIdentifier { .. } => "E0011",
            LexError::ExpectedDigitsAfterPrefix { .. } => "E0012",
            LexError::MultipleDecimalPoints { .. } => "E0013",
            LexError::InvalidExponent { .. } => "E0014",
            LexError::UnexpectedEof { .. } => "E0015",
            LexError::BomNotAllowed { .. } => "E0016",
            LexError::TabInIndentation { .. } => "E0017",
            LexError::InvalidNumberSuffix { .. } => "E0018",
            LexError::BareCarriageReturn { .. } => "E0019",
            LexError::Generic { .. } => "E0020",
        }
    }

    pub fn line(&self) -> usize {
        match self {
            LexError::InvalidCharacter { line, .. }
            | LexError::UnterminatedString { line, .. }
            | LexError::InvalidEscapeSequence { line, .. }
            | LexError::InvalidUnicodeEscape { line, .. }
            | LexError::InvalidIntegerLiteral { line, .. }
            | LexError::InvalidFloatLiteral { line, .. }
            | LexError::IntegerOverflow { line, .. }
            | LexError::InvalidDigitForBase { line, .. }
            | LexError::UnterminatedBlockComment { line, .. }
            | LexError::InvalidUtf8 { line, .. }
            | LexError::InvalidIdentifier { line, .. }
            | LexError::ExpectedDigitsAfterPrefix { line, .. }
            | LexError::MultipleDecimalPoints { line, .. }
            | LexError::InvalidExponent { line, .. }
            | LexError::UnexpectedEof { line, .. }
            | LexError::BomNotAllowed { line, .. }
            | LexError::TabInIndentation { line, .. }
            | LexError::InvalidNumberSuffix { line, .. }
            | LexError::BareCarriageReturn { line, .. }
            | LexError::Generic { line, .. } => *line,
        }
    }

    pub fn column(&self) -> usize {
        match self {
            LexError::InvalidCharacter { column, .. }
            | LexError::UnterminatedString { column, .. }
            | LexError::InvalidEscapeSequence { column, .. }
            | LexError::InvalidUnicodeEscape { column, .. }
            | LexError::InvalidIntegerLiteral { column, .. }
            | LexError::InvalidFloatLiteral { column, .. }
            | LexError::IntegerOverflow { column, .. }
            | LexError::InvalidDigitForBase { column, .. }
            | LexError::UnterminatedBlockComment { column, .. }
            | LexError::InvalidUtf8 { column, .. }
            | LexError::InvalidIdentifier { column, .. }
            | LexError::ExpectedDigitsAfterPrefix { column, .. }
            | LexError::MultipleDecimalPoints { column, .. }
            | LexError::InvalidExponent { column, .. }
            | LexError::UnexpectedEof { column, .. }
            | LexError::BomNotAllowed { column, .. }
            | LexError::TabInIndentation { column, .. }
            | LexError::InvalidNumberSuffix { column, .. }
            | LexError::BareCarriageReturn { column, .. }
            | LexError::Generic { column, .. } => *column,
        }
    }

    pub fn format_with_source(&self, source: &str) -> String {
        let line_num = self.line();
        let col_num = self.column();
        
        let lines: Vec<&str> = source.lines().collect();
        let mut output = String::new();
        
        output.push_str(&format!("error[{}]: {}\n", self.error_code(), self));
        output.push_str(&format!("  --> line {}, column {}\n", line_num, col_num));
        
        if line_num > 0 && line_num <= lines.len() {
            if line_num > 1 {
                output.push_str(&format!("{:4} | {}\n", line_num - 1, lines[line_num - 2]));
            }
            
            let error_line = lines[line_num - 1];
            output.push_str(&format!("{:4} | {}\n", line_num, error_line));
            
            output.push_str(&format!("     | {}{}\n", " ".repeat(col_num.saturating_sub(1)), "^"));
            
            if line_num < lines.len() {
                output.push_str(&format!("{:4} | {}\n", line_num + 1, lines[line_num]));
            }
        }
        
        if let Some(hint) = self.hint() {
            output.push_str(&format!("\nHelp: {}\n", hint));
        }
        
        output
    }

    pub fn hint(&self) -> Option<String> {
        match self {
            LexError::InvalidEscapeSequence { sequence, .. } => {
                Some(format!(
                    "Valid escape sequences are: \\n (newline), \\r (carriage return), \\t (tab), \
                     \\\\ (backslash), \\\" (quote), \\u{{xxxx}} (Unicode). Did you mean to escape '{}'?",
                    sequence
                ))
            }
            LexError::UnterminatedString { .. } => {
                Some("Make sure all string literals are closed with a matching quote (\")".to_string())
            }
            LexError::InvalidIntegerLiteral { .. } => {
                Some("Integer literals can be decimal (42), hexadecimal (0xFF), octal (0o755), or binary (0b1010)".to_string())
            }
            LexError::InvalidFloatLiteral { .. } => {
                Some("Float literals must have digits before and/or after the decimal point (e.g., 3.14, 1.0e-10)".to_string())
            }
            LexError::IntegerOverflow { .. } => {
                Some("Integer literals must be in the range -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807".to_string())
            }
            LexError::TabInIndentation { .. } => {
                Some("Flux style guidelines require spaces for indentation (4 spaces per level)".to_string())
            }
            LexError::BomNotAllowed { .. } => {
                Some("Save your file with UTF-8 encoding without BOM".to_string())
            }
            LexError::UnterminatedBlockComment { .. } => {
                Some("Make sure all block comments /* ... */ are properly closed".to_string())
            }
            LexError::InvalidIdentifier { .. } => {
                Some("Identifiers must start with a letter or underscore, followed by letters, digits, or underscores".to_string())
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LexErrorContext {
    pub file_path: Option<String>,
    pub source: String,
}

impl LexErrorContext {
    pub fn new(source: String) -> Self {
        LexErrorContext {
            file_path: None,
            source,
        }
    }

    pub fn with_file(source: String, file_path: String) -> Self {
        LexErrorContext {
            file_path: Some(file_path),
            source,
        }
    }

    pub fn format_error(&self, error: &LexError) -> String {
        let mut output = String::new();
        
        if let Some(path) = &self.file_path {
            output.push_str(&format!("In file: {}\n", path));
        }
        
        output.push_str(&error.format_with_source(&self.source));
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        let err = LexError::InvalidCharacter {
            character: '@',
            line: 1,
            column: 5,
        };
        assert_eq!(err.error_code(), "E0001");

        let err = LexError::UnterminatedString { line: 2, column: 10 };
        assert_eq!(err.error_code(), "E0002");

        let err = LexError::InvalidEscapeSequence {
            sequence: "x".to_string(),
            line: 3,
            column: 15,
        };
        assert_eq!(err.error_code(), "E0003");
    }

    #[test]
    fn test_position_getters() {
        let err = LexError::InvalidCharacter {
            character: '@',
            line: 10,
            column: 25,
        };
        assert_eq!(err.line(), 10);
        assert_eq!(err.column(), 25);
    }

    #[test]
    fn test_error_display() {
        let err = LexError::InvalidCharacter {
            character: '@',
            line: 1,
            column: 5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("E0001"));
        assert!(msg.contains("Invalid character"));
        assert!(msg.contains("'@'"));
    }

    #[test]
    fn test_unterminated_string() {
        let err = LexError::UnterminatedString { line: 5, column: 10 };
        assert_eq!(err.error_code(), "E0002");
        assert!(err.hint().is_some());
    }

    #[test]
    fn test_invalid_escape_sequence() {
        let err = LexError::InvalidEscapeSequence {
            sequence: "x".to_string(),
            line: 3,
            column: 8,
        };
        let hint = err.hint().unwrap();
        assert!(hint.contains("Valid escape sequences"));
    }

    #[test]
    fn test_integer_overflow() {
        let err = LexError::IntegerOverflow {
            literal: "99999999999999999999".to_string(),
            line: 2,
            column: 5,
        };
        assert_eq!(err.error_code(), "E0007");
    }

    #[test]
    fn test_format_with_source() {
        let source = "fn main() {\n    let x = @invalid;\n}\n";
        let err = LexError::InvalidCharacter {
            character: '@',
            line: 2,
            column: 13,
        };
        
        let formatted = err.format_with_source(source);
        assert!(formatted.contains("E0001"));
        assert!(formatted.contains("line 2"));
        assert!(formatted.contains("^")); // Pointer
    }

    #[test]
    fn test_error_context() {
        let source = "fn main() {\n    println(\"Hello, World!\n}\n".to_string();
        let context = LexErrorContext::with_file(source, "main.flux".to_string());
        
        let err = LexError::UnterminatedString { line: 2, column: 13 };
        let formatted = context.format_error(&err);
        
        assert!(formatted.contains("main.flux"));
        assert!(formatted.contains("E0002"));
    }

    #[test]
    fn test_all_error_variants_have_codes() {
        let errors = vec![
            LexError::InvalidCharacter { character: 'x', line: 1, column: 1 },
            LexError::UnterminatedString { line: 1, column: 1 },
            LexError::InvalidEscapeSequence { sequence: "x".into(), line: 1, column: 1 },
            LexError::InvalidUnicodeEscape { sequence: "x".into(), reason: "test".into(), line: 1, column: 1 },
            LexError::InvalidIntegerLiteral { literal: "x".into(), reason: "test".into(), line: 1, column: 1 },
            LexError::InvalidFloatLiteral { literal: "x".into(), reason: "test".into(), line: 1, column: 1 },
            LexError::IntegerOverflow { literal: "x".into(), line: 1, column: 1 },
            LexError::InvalidDigitForBase { digit: 'x', base: 10, line: 1, column: 1 },
            LexError::UnterminatedBlockComment { line: 1, column: 1 },
            LexError::InvalidUtf8 { line: 1, column: 1 },
            LexError::InvalidIdentifier { identifier: "x".into(), reason: "test".into(), line: 1, column: 1 },
            LexError::ExpectedDigitsAfterPrefix { prefix: "0x".into(), line: 1, column: 1 },
            LexError::MultipleDecimalPoints { line: 1, column: 1 },
            LexError::InvalidExponent { reason: "test".into(), line: 1, column: 1 },
            LexError::UnexpectedEof { line: 1, column: 1 },
            LexError::BomNotAllowed { line: 1, column: 1 },
            LexError::TabInIndentation { line: 1, column: 1 },
            LexError::InvalidNumberSuffix { suffix: "x".into(), line: 1, column: 1 },
            LexError::BareCarriageReturn { line: 1, column: 1 },
            LexError::Generic { message: "test".into(), line: 1, column: 1 },
        ];

        for err in errors {
            assert!(err.error_code().starts_with("E0"));
            assert_eq!(err.line(), 1);
            assert_eq!(err.column(), 1);
        }
    }
}
