// src/parser/error.rs

use thiserror::Error;
use crate::lexer::{LexError, TokenKind};

pub type ParseResult<T> = Result<T, ParseError>;

#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("[E0021] Unexpected token at line {line}, column {column}: expected {expected}, found {found}")]
    UnexpectedToken {
        expected: TokenKind,
        found: TokenKind,
        line: usize,
        column: usize,
    },

    #[error("[E0022] Unexpected end of file at line {line}, column {column}: expected {expected}")]
    UnexpectedEof {
        expected: String,
        line: usize,
        column: usize,
    },

    #[error("[E0023] Expected expression at line {line}, column {column}, found {found}")]
    ExpectedExpression {
        found: TokenKind,
        line: usize,
        column: usize,
    },

    #[error("[E0024] Expected statement at line {line}, column {column}, found {found}")]
    ExpectedStatement {
        found: TokenKind,
        line: usize,
        column: usize,
    },

    #[error("[E0025] Expected type at line {line}, column {column}, found {found}")]
    ExpectedType {
        found: TokenKind,
        line: usize,
        column: usize,
    },

    #[error("[E0026] Expected pattern at line {line}, column {column}, found {found}")]
    ExpectedPattern {
        found: TokenKind,
        line: usize,
        column: usize,
    },

    #[error("[E0027] Expected identifier at line {line}, column {column}, found {found}")]
    ExpectedIdentifier {
        found: TokenKind,
        line: usize,
        column: usize,
    },

    #[error("[E0028] Mismatched parentheses at line {line}, column {column}: expected '{expected}', found '{found}'")]
    MismatchedParens {
        expected: char,
        found: char,
        line: usize,
        column: usize,
    },

    #[error("[E0029] Mismatched braces at line {line}, column {column}: expected '}}', found {found}")]
    MismatchedBraces {
        found: TokenKind,
        line: usize,
        column: usize,
    },

    #[error("[E0030] Mismatched brackets at line {line}, column {column}: expected ']', found {found}")]
    MismatchedBrackets {
        found: TokenKind,
        line: usize,
        column: usize,
    },

    #[error("[E0031] Invalid literal '{literal}' at line {line}, column {column}: {reason}")]
    InvalidLiteral {
        literal: String,
        reason: String,
        line: usize,
        column: usize,
    },

    #[error("[E0032] Invalid operator at line {line}, column {column}: {message}")]
    InvalidOperator {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0033] Expected type parameter for '{name}' at line {line}, column {column}")]
    ExpectedTypeParameter {
        name: String,
        line: usize,
        column: usize,
    },

    #[error("[E0034] Invalid type parameter at line {line}, column {column}: {message}")]
    InvalidTypeParameter {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0035] Too many arguments at line {line}, column {column}: expected {expected}, found {found}")]
    TooManyArguments {
        expected: usize,
        found: usize,
        line: usize,
        column: usize,
    },

    #[error("[E0036] Too few arguments at line {line}, column {column}: expected {expected}, found {found}")]
    TooFewArguments {
        expected: usize,
        found: usize,
        line: usize,
        column: usize,
    },

    #[error("[E0037] Duplicate field '{field}' in struct literal at line {line}, column {column}")]
    DuplicateField {
        field: String,
        line: usize,
        column: usize,
    },

    #[error("[E0038] Missing field '{field}' in struct literal at line {line}, column {column}")]
    MissingField {
        field: String,
        line: usize,
        column: usize,
    },

    #[error("[E0039] Invalid pattern in match arm at line {line}, column {column}: {message}")]
    InvalidMatchPattern {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0040] Unreachable pattern at line {line}, column {column}")]
    UnreachablePattern {
        line: usize,
        column: usize,
    },

    #[error("[E0041] Non-exhaustive patterns at line {line}, column {column}: missing {missing}")]
    NonExhaustivePatterns {
        missing: String,
        line: usize,
        column: usize,
    },

    #[error("[E0042] Invalid assignment target at line {line}, column {column}: {message}")]
    InvalidAssignmentTarget {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0043] 'break' statement outside loop at line {line}, column {column}")]
    BreakOutsideLoop {
        line: usize,
        column: usize,
    },

    #[error("[E0044] 'continue' statement outside loop at line {line}, column {column}")]
    ContinueOutsideLoop {
        line: usize,
        column: usize,
    },

    #[error("[E0045] 'return' statement outside function at line {line}, column {column}")]
    ReturnOutsideFunction {
        line: usize,
        column: usize,
    },

    #[error("[E0046] 'await' expression outside async function at line {line}, column {column}")]
    AwaitOutsideAsync {
        line: usize,
        column: usize,
    },

    #[error("[E0047] Invalid function signature at line {line}, column {column}: {message}")]
    InvalidFunctionSignature {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0048] Invalid struct definition at line {line}, column {column}: {message}")]
    InvalidStructDefinition {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0049] Invalid enum definition at line {line}, column {column}: {message}")]
    InvalidEnumDefinition {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0050] Invalid trait definition at line {line}, column {column}: {message}")]
    InvalidTraitDefinition {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0051] Invalid impl block at line {line}, column {column}: {message}")]
    InvalidImplBlock {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0052] Duplicate parameter name '{name}' at line {line}, column {column}")]
    DuplicateParameter {
        name: String,
        line: usize,
        column: usize,
    },

    #[error("[E0053] Invalid visibility modifier at line {line}, column {column}")]
    InvalidVisibility {
        line: usize,
        column: usize,
    },

    #[error("[E0054] Expected semicolon at line {line}, column {column}, found {found}")]
    ExpectedSemicolon {
        found: TokenKind,
        line: usize,
        column: usize,
    },

    #[error("[E0055] Expected comma at line {line}, column {column}, found {found}")]
    ExpectedComma {
        found: TokenKind,
        line: usize,
        column: usize,
    },

    #[error("[E0056] Invalid range expression at line {line}, column {column}: {message}")]
    InvalidRange {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0057] Invalid array size at line {line}, column {column}: {message}")]
    InvalidArraySize {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0058] Invalid cast at line {line}, column {column}: cannot cast from {from} to {to}")]
    InvalidCast {
        from: String,
        to: String,
        line: usize,
        column: usize,
    },

    #[error("[E0059] Ambiguous expression at line {line}, column {column}: {message}")]
    AmbiguousExpression {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("[E0060] Parse error at line {line}, column {column}: {message}")]
    Generic {
        message: String,
        context: Option<String>,
        line: usize,
        column: usize,
    },
    #[error("Invalid syntax: {message} at line {line}, column {column}")]
    InvalidSyntax {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("Lexical error: {0}")]
    LexError(#[from] LexError),

    #[error("Feature not yet implemented: {0}")]
    Unimplemented(String),

    #[error("Multiple parse errors occurred")]
    MultipleErrors,
}

impl ParseError {
    pub fn error_code(&self) -> &'static str {
        match self {
            ParseError::UnexpectedToken { .. } => "E0021",
            ParseError::UnexpectedEof { .. } => "E0022",
            ParseError::ExpectedExpression { .. } => "E0023",
            ParseError::ExpectedStatement { .. } => "E0024",
            ParseError::ExpectedType { .. } => "E0025",
            ParseError::ExpectedPattern { .. } => "E0026",
            ParseError::ExpectedIdentifier { .. } => "E0027",
            ParseError::MismatchedParens { .. } => "E0028",
            ParseError::MismatchedBraces { .. } => "E0029",
            ParseError::MismatchedBrackets { .. } => "E0030",
            ParseError::InvalidLiteral { .. } => "E0031",
            ParseError::InvalidOperator { .. } => "E0032",
            ParseError::ExpectedTypeParameter { .. } => "E0033",
            ParseError::InvalidTypeParameter { .. } => "E0034",
            ParseError::TooManyArguments { .. } => "E0035",
            ParseError::TooFewArguments { .. } => "E0036",
            ParseError::DuplicateField { .. } => "E0037",
            ParseError::MissingField { .. } => "E0038",
            ParseError::InvalidMatchPattern { .. } => "E0039",
            ParseError::UnreachablePattern { .. } => "E0040",
            ParseError::NonExhaustivePatterns { .. } => "E0041",
            ParseError::InvalidAssignmentTarget { .. } => "E0042",
            ParseError::BreakOutsideLoop { .. } => "E0043",
            ParseError::ContinueOutsideLoop { .. } => "E0044",
            ParseError::ReturnOutsideFunction { .. } => "E0045",
            ParseError::AwaitOutsideAsync { .. } => "E0046",
            ParseError::InvalidFunctionSignature { .. } => "E0047",
            ParseError::InvalidStructDefinition { .. } => "E0048",
            ParseError::InvalidEnumDefinition { .. } => "E0049",
            ParseError::InvalidTraitDefinition { .. } => "E0050",
            ParseError::InvalidImplBlock { .. } => "E0051",
            ParseError::DuplicateParameter { .. } => "E0052",
            ParseError::InvalidVisibility { .. } => "E0053",
            ParseError::ExpectedSemicolon { .. } => "E0054",
            ParseError::ExpectedComma { .. } => "E0055",
            ParseError::InvalidRange { .. } => "E0056",
            ParseError::InvalidArraySize { .. } => "E0057",
            ParseError::InvalidCast { .. } => "E0058",
            ParseError::AmbiguousExpression { .. } => "E0059",
            ParseError::Generic { .. } => "E0060",
            ParseError::InvalidSyntax { .. } => "P3005",
            ParseError::LexError(e) => e.error_code(),
            ParseError::Unimplemented(_) => "E9999",
            ParseError::MultipleErrors => "E9998",
        }
    }

    pub fn line(&self) -> Option<usize> {
        match self {
            ParseError::UnexpectedToken { line, .. }
            | ParseError::UnexpectedEof { line, .. }
            | ParseError::ExpectedExpression { line, .. }
            | ParseError::ExpectedStatement { line, .. }
            | ParseError::ExpectedType { line, .. }
            | ParseError::ExpectedPattern { line, .. }
            | ParseError::ExpectedIdentifier { line, .. }
            | ParseError::MismatchedParens { line, .. }
            | ParseError::MismatchedBraces { line, .. }
            | ParseError::MismatchedBrackets { line, .. }
            | ParseError::InvalidLiteral { line, .. }
            | ParseError::InvalidOperator { line, .. }
            | ParseError::ExpectedTypeParameter { line, .. }
            | ParseError::InvalidTypeParameter { line, .. }
            | ParseError::TooManyArguments { line, .. }
            | ParseError::TooFewArguments { line, .. }
            | ParseError::DuplicateField { line, .. }
            | ParseError::MissingField { line, .. }
            | ParseError::InvalidMatchPattern { line, .. }
            | ParseError::UnreachablePattern { line, .. }
            | ParseError::NonExhaustivePatterns { line, .. }
            | ParseError::InvalidAssignmentTarget { line, .. }
            | ParseError::BreakOutsideLoop { line, .. }
            | ParseError::ContinueOutsideLoop { line, .. }
            | ParseError::ReturnOutsideFunction { line, .. }
            | ParseError::AwaitOutsideAsync { line, .. }
            | ParseError::InvalidFunctionSignature { line, .. }
            | ParseError::InvalidStructDefinition { line, .. }
            | ParseError::InvalidEnumDefinition { line, .. }
            | ParseError::InvalidTraitDefinition { line, .. }
            | ParseError::InvalidImplBlock { line, .. }
            | ParseError::DuplicateParameter { line, .. }
            | ParseError::InvalidVisibility { line, .. }
            | ParseError::ExpectedSemicolon { line, .. }
            | ParseError::ExpectedComma { line, .. }
            | ParseError::InvalidRange { line, .. }
            | ParseError::InvalidArraySize { line, .. }
            | ParseError::InvalidCast { line, .. }
            | ParseError::AmbiguousExpression { line, .. }
            | ParseError::Generic { line, .. } => Some(*line),
            ParseError::LexError(e) => Some(e.line()),
            _ => None,
        }
    }

    pub fn column(&self) -> Option<usize> {
        match self {
            ParseError::UnexpectedToken { column, .. }
            | ParseError::UnexpectedEof { column, .. }
            | ParseError::ExpectedExpression { column, .. }
            | ParseError::ExpectedStatement { column, .. }
            | ParseError::ExpectedType { column, .. }
            | ParseError::ExpectedPattern { column, .. }
            | ParseError::ExpectedIdentifier { column, .. }
            | ParseError::MismatchedParens { column, .. }
            | ParseError::MismatchedBraces { column, .. }
            | ParseError::MismatchedBrackets { column, .. }
            | ParseError::InvalidLiteral { column, .. }
            | ParseError::InvalidOperator { column, .. }
            | ParseError::ExpectedTypeParameter { column, .. }
            | ParseError::InvalidTypeParameter { column, .. }
            | ParseError::TooManyArguments { column, .. }
            | ParseError::TooFewArguments { column, .. }
            | ParseError::DuplicateField { column, .. }
            | ParseError::MissingField { column, .. }
            | ParseError::InvalidMatchPattern { column, .. }
            | ParseError::UnreachablePattern { column, .. }
            | ParseError::NonExhaustivePatterns { column, .. }
            | ParseError::InvalidAssignmentTarget { column, .. }
            | ParseError::BreakOutsideLoop { column, .. }
            | ParseError::ContinueOutsideLoop { column, .. }
            | ParseError::ReturnOutsideFunction { column, .. }
            | ParseError::AwaitOutsideAsync { column, .. }
            | ParseError::InvalidFunctionSignature { column, .. }
            | ParseError::InvalidStructDefinition { column, .. }
            | ParseError::InvalidEnumDefinition { column, .. }
            | ParseError::InvalidTraitDefinition { column, .. }
            | ParseError::InvalidImplBlock { column, .. }
            | ParseError::DuplicateParameter { column, .. }
            | ParseError::InvalidVisibility { column, .. }
            | ParseError::ExpectedSemicolon { column, .. }
            | ParseError::ExpectedComma { column, .. }
            | ParseError::InvalidRange { column, .. }
            | ParseError::InvalidArraySize { column, .. }
            | ParseError::InvalidCast { column, .. }
            | ParseError::AmbiguousExpression { column, .. }
            | ParseError::Generic { column, .. } => Some(*column),
            ParseError::LexError(e) => Some(e.column()),
            _ => None,
        }
    }

    pub fn hint(&self) -> Option<String> {
        match self {
            ParseError::UnexpectedToken { expected, .. } => {
                Some(format!("Try adding or changing the token to '{}'", expected))
            }
            ParseError::UnexpectedEof { expected, .. } => {
                Some(format!("The parser expected {} before the end of the file", expected))
            }
            ParseError::MismatchedParens { expected, .. } => {
                Some(format!("Make sure all parentheses are properly matched. Expected '{}'", expected))
            }
            ParseError::MismatchedBraces { .. } => {
                Some("Make sure all braces '{{' and '}}' are properly matched".to_string())
            }
            ParseError::MismatchedBrackets { .. } => {
                Some("Make sure all brackets '[' and ']' are properly matched".to_string())
            }
            ParseError::BreakOutsideLoop { .. } => {
                Some("'break' can only be used inside 'while', 'for', or 'loop' statements".to_string())
            }
            ParseError::ContinueOutsideLoop { .. } => {
                Some("'continue' can only be used inside 'while', 'for', or 'loop' statements".to_string())
            }
            ParseError::ReturnOutsideFunction { .. } => {
                Some("'return' can only be used inside a function body".to_string())
            }
            ParseError::AwaitOutsideAsync { .. } => {
                Some("'await' can only be used inside an async function".to_string())
            }
            ParseError::DuplicateField { field, .. } => {
                Some(format!("Remove the duplicate field '{}' or rename it", field))
            }
            ParseError::ExpectedSemicolon { .. } => {
                Some("Statements in Fluno typically end with a semicolon ';'".to_string())
            }
            ParseError::InvalidAssignmentTarget { .. } => {
                Some("Only variables, fields, and index expressions can be assigned to".to_string())
            }
            ParseError::NonExhaustivePatterns { missing, .. } => {
                Some(format!("Add a pattern for: {}", missing))
            }
            ParseError::InvalidSyntax { .. } => None,
            _ => None,
        }
    }

    pub fn format_with_source(&self, source: &str) -> String {
        let mut output = String::new();
        
        output.push_str(&format!("error[{}]: {}\n", self.error_code(), self));
        
        if let (Some(line), Some(column)) = (self.line(), self.column()) {
            output.push_str(&format!("  --> line {}, column {}\n", line, column));
            
            let lines: Vec<&str> = source.lines().collect();
            if line > 0 && line <= lines.len() {
                if line > 1 {
                    output.push_str(&format!("{:4} | {}\n", line - 1, lines[line - 2]));
                }
                
                let error_line = lines[line - 1];
                output.push_str(&format!("{:4} | {}\n", line, error_line));
                
                output.push_str(&format!("     | {}{}\n", " ".repeat(column.saturating_sub(1)), "^"));
                
                if line < lines.len() {
                    output.push_str(&format!("{:4} | {}\n", line + 1, lines[line]));
                }
            }
        }
        
        if let Some(hint) = self.hint() {
            output.push_str(&format!("\nHelp: {}\n", hint));
        }
        
        output
    }
}

#[derive(Debug, Clone)]
pub struct _ParseErrorContext {
    pub file_path: Option<String>,
    pub source: String,
    pub errors: Vec<ParseError>,
}

impl _ParseErrorContext {
    pub fn _new(source: String) -> Self {
        _ParseErrorContext {
            file_path: None,
            source,
            errors: Vec::new(),
        }
    }

    pub fn _with_file(source: String, file_path: String) -> Self {
        _ParseErrorContext {
            file_path: Some(file_path),
            source,
            errors: Vec::new(),
        }
    }

    pub fn _add_error(&mut self, error: ParseError) -> () {
        self.errors.push(error);
    }
    
    pub fn _format_errors(&self) -> String {
        let mut output = String::new();
        
        if let Some(path) = &self.file_path {
            output.push_str(&format!("In file: {}\n\n", path));
        }
        
        for (i, error) in self.errors.iter().enumerate() {
            if i > 0 {
                output.push_str("\n");
            }
            output.push_str(&error.format_with_source(&self.source));
        }
        
        output.push_str(&format!("\n{} error(s) found\n", self.errors.len()));
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        let err = ParseError::UnexpectedToken {
            expected: TokenKind::Semicolon,
            found: TokenKind::Comma,
            line: 1,
            column: 10,
        };
        assert_eq!(err.error_code(), "E0021");

        let err = ParseError::BreakOutsideLoop { line: 5, column: 3 };
        assert_eq!(err.error_code(), "E0043");
    }

    #[test]
    fn test_position_getters() {
        let err = ParseError::UnexpectedToken {
            expected: TokenKind::Semicolon,
            found: TokenKind::Comma,
            line: 10,
            column: 25,
        };
        assert_eq!(err.line(), Some(10));
        assert_eq!(err.column(), Some(25));
    }

    #[test]
    fn test_hints() {
        let err = ParseError::BreakOutsideLoop { line: 1, column: 1 };
        assert!(err.hint().is_some());
        assert!(err.hint().unwrap().contains("loop"));

        let err = ParseError::ExpectedSemicolon {
            found: TokenKind::Comma,
            line: 1,
            column: 1,
        };
        assert!(err.hint().is_some());
    }

    #[test]
    fn test_error_display() {
        let err = ParseError::UnexpectedToken {
            expected: TokenKind::Semicolon,
            found: TokenKind::Comma,
            line: 1,
            column: 10,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("E0021"));
        assert!(msg.contains("Unexpected token"));
    }

    #[test]
    fn test_format_with_source() {
        let source = "fn main() {\n    let x = 10\n    return x;\n}";
        let err = ParseError::ExpectedSemicolon {
            found: TokenKind::Return,
            line: 3,
            column: 5,
        };
        
        let formatted = err.format_with_source(source);
        assert!(formatted.contains("E0054"));
        assert!(formatted.contains("line 3"));
        assert!(formatted.contains("^"));
    }

    #[test]
    fn test_error_context() {
        let source = "fn test() { break; }".to_string();
        let mut context = _ParseErrorContext::_with_file(source, "test.fln".to_string());
        
        context._add_error(ParseError::BreakOutsideLoop { line: 1, column: 13 });
        
        let formatted = context._format_errors();
        assert!(formatted.contains("test.fln"));
        assert!(formatted.contains("E0043"));
        assert!(formatted.contains("1 error(s) found"));
    }

    #[test]
    fn test_multiple_errors() {
        let source = "fn test() {}".to_string();
        let mut context = _ParseErrorContext::_new(source);
        
        context._add_error(ParseError::ExpectedSemicolon {
            found: TokenKind::RBrace,
            line: 1,
            column: 12,
        });
        context._add_error(ParseError::UnexpectedEof {
            expected: "statement".to_string(),
            line: 1,
            column: 13,
        });
        
        assert_eq!(context.errors.len(), 2);
    }
}
