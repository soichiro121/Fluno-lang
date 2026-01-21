// Token definitions for the Flux programming language lexer.
//
// This module defines all token types used in lexical analysis, including:
// - Keywords (fn, let, if, else, etc.)
// - Literals (integers, floats, booleans, strings)
// - Operators (arithmetic, comparison, logical, bitwise, assignment)
// - Delimiters (parentheses, braces, brackets, punctuation)
// - Identifiers and special tokens (EOF, comments)

use std::fmt;
use crate::ast::node::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub column: usize,
    pub text: Option<String>,
}

impl Token {
    // Create a new token with position information
    pub fn new(kind: TokenKind, line: usize, column: usize) -> Self {
        Token {
            kind,
            line,
            column,
            text: None,
        }
    }

    // Create a new token with text content
    pub fn with_text(kind: TokenKind, line: usize, column: usize, text: String) -> Self {
        Token {
            kind,
            line,
            column,
            text: Some(text),
        }
    }

    // Get the text content of this token, if available
    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }
    
    pub fn span(&self) -> Span {
        let len = self.text.as_ref().map(|s| s.len()).unwrap_or(0);
        Span::new(self.line, self.column, len)
    }

    // Check if this token is a keyword
    pub fn is_keyword(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Fn
                | TokenKind::Let
                | TokenKind::If
                | TokenKind::Else
                | TokenKind::Match
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Loop
                | TokenKind::Break
                | TokenKind::Continue
                | TokenKind::Return
                | TokenKind::Struct
                | TokenKind::Enum
                | TokenKind::Impl
                | TokenKind::Trait
                | TokenKind::Type
                | TokenKind::Pub
                | TokenKind::Priv
                | TokenKind::Mod
                | TokenKind::Use
                | TokenKind::Import
                | TokenKind::As
                | TokenKind::Async
                | TokenKind::Await
                | TokenKind::Spawn
                | TokenKind::True
                | TokenKind::False
                | TokenKind::SelfLower
                | TokenKind::SelfUpper
        )
    }

    // Check if this token is a literal
    pub fn is_literal(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::IntLiteral
                | TokenKind::FloatLiteral
                | TokenKind::BoolLiteral
                | TokenKind::StringLiteral
                | TokenKind::UnitLiteral
        )
    }

    // Check if this token is an operator
    pub fn is_operator(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Star
                | TokenKind::Slash
                | TokenKind::Percent
                | TokenKind::Eq
                | TokenKind::Ne
                | TokenKind::Lt
                | TokenKind::Le
                | TokenKind::Gt
                | TokenKind::Ge
                | TokenKind::And
                | TokenKind::Or
                | TokenKind::Not
                | TokenKind::BitAnd
                | TokenKind::BitOr
                | TokenKind::BitXor
                | TokenKind::BitNot
                | TokenKind::Shl
                | TokenKind::Shr
                | TokenKind::Assign
                | TokenKind::PlusAssign
                | TokenKind::MinusAssign
                | TokenKind::StarAssign
                | TokenKind::SlashAssign
                | TokenKind::PercentAssign
                | TokenKind::BitAndAssign
                | TokenKind::BitOrAssign
                | TokenKind::BitXorAssign
                | TokenKind::ShlAssign
                | TokenKind::ShrAssign
        )
    }
}

// Enumeration of all possible token types in Flux.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // ========== Keywords ==========
    Fn,
    Let,
    If,
    Else,
    Match,
    While,
    For,
    Loop,
    Break,
    Continue,
    Return,
    // `struct` - Structure definition keyword
    Struct,
    // `enum` - Enumeration definition keyword
    Enum,
    // `impl` - Implementation block keyword
    Impl,
    // `trait` - Trait definition keyword
    Trait,
    // `type` - Type alias keyword
    Type,
    // `pub` - Public visibility modifier
    Pub,
    // `priv` - Private visibility modifier
    Priv,
    // `mod` - Module keyword
    Mod,
    // `use` - Import statement keyword
    Use,
    // `import` - Alternative import keyword
    Import,
    // `as` - Renaming keyword
    As,
    // `async` - Asynchronous function keyword
    Async,
    // `await` - Await expression keyword
    Await,
    // `spawn` - Thread spawn keyword
    Spawn,
    // `true` - Boolean true literal
    True,
    // `false` - Boolean false literal
    False,
    // `self` - Self reference (lowercase)
    SelfLower,
    // `Self` - Self type (uppercase)
    SelfUpper,

    // ========== Literals ==========
    // Integer literal (e.g., 42, 0xFF, 0o755, 0b1010)
    IntLiteral,
    // Floating-point literal (e.g., 3.14, 1.0e-10)
    FloatLiteral,
    // Boolean literal (true or false)
    BoolLiteral,
    // String literal (e.g., "Hello, World!")
    StringLiteral,
    // Unit literal `()`
    UnitLiteral,

    // ========== Identifiers ==========
    // User-defined identifier (variable names, function names, etc.)
    Identifier,

    // ========== Arithmetic Operators ==========
    // `+` - Addition operator
    Plus,
    // `-` - Subtraction or negation operator
    Minus,
    // `*` - Multiplication operator
    Star,
    // `/` - Division operator
    Slash,
    // `%` - Modulo operator
    Percent,

    // ========== Comparison Operators ==========
    // `==` - Equality operator
    Eq,
    // `!=` - Inequality operator
    Ne,
    // `<` - Less than operator
    Lt,
    // `<=` - Less than or equal operator
    Le,
    // `>` - Greater than operator
    Gt,
    // `>=` - Greater than or equal operator
    Ge,

    // ========== Logical Operators ==========
    // `&&` - Logical AND operator
    And,
    // `||` - Logical OR operator
    Or,
    // `!` - Logical NOT operator
    Not,

    // ========== Bitwise Operators ==========
    // `&` - Bitwise AND operator
    BitAnd,
    // `|` - Bitwise OR operator
    BitOr,
    // `^` - Bitwise XOR operator
    BitXor,
    // `~` - Bitwise NOT operator
    BitNot,
    // `<<` - Left shift operator
    Shl,
    // `>>` - Right shift operator
    Shr,

    // ========== Assignment Operators ==========
    // `=` - Assignment operator
    Assign,
    // `+=` - Addition assignment
    PlusAssign,
    // `-=` - Subtraction assignment
    MinusAssign,
    // `*=` - Multiplication assignment
    StarAssign,
    // `/=` - Division assignment
    SlashAssign,
    // `%=` - Modulo assignment
    PercentAssign,
    // `&=` - Bitwise AND assignment
    BitAndAssign,
    // `|=` - Bitwise OR assignment
    BitOrAssign,
    // `^=` - Bitwise XOR assignment
    BitXorAssign,
    // `<<=` - Left shift assignment
    ShlAssign,
    // `>>=` - Right shift assignment
    ShrAssign,

    // ========== Delimiters ==========
    // `(` - Left parenthesis
    LParen,
    // `)` - Right parenthesis
    RParen,
    // `{` - Left brace
    LBrace,
    // `}` - Right brace
    RBrace,
    // `[` - Left bracket
    LBracket,
    // `]` - Right bracket
    RBracket,

    // ========== Punctuation ==========
    // `,` - Comma
    Comma,
    // `;` - Semicolon
    Semicolon,
    // `:` - Colon
    Colon,
    // `::` - Double colon (path separator)
    ColonColon,
    // `.` - Dot (member access)
    Dot,
    // `..` - Range operator
    DotDot,
    // `...` - Variadic operator
    DotDotDot,
    // `->` - Arrow (function return type)
    Arrow,
    // `=>` - Fat arrow (match arms, lambdas)
    FatArrow,
    // `?` - Question mark (error propagation)
    Question,

    // ========== Special Tokens ==========
    // End of file
    Eof,
    // Line comment (// ...)
    LineComment,
    // Block comment (/* ... */)
    BlockComment,
    // Documentation comment (/// ...)
    DocComment,
    // Whitespace (typically filtered out)
    Whitespace,
    // Invalid/unrecognized token
    Invalid,

    Pound, // または Hash
    With,
    Extern,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            // Keywords
            TokenKind::Fn => "fn",
            TokenKind::Let => "let",
            TokenKind::If => "if",
            TokenKind::Else => "else",
            TokenKind::Match => "match",
            TokenKind::While => "while",
            TokenKind::For => "for",
            TokenKind::Loop => "loop",
            TokenKind::Break => "break",
            TokenKind::Continue => "continue",
            TokenKind::Return => "return",
            TokenKind::Struct => "struct",
            TokenKind::Enum => "enum",
            TokenKind::Impl => "impl",
            TokenKind::Trait => "trait",
            TokenKind::Type => "type",
            TokenKind::Pub => "pub",
            TokenKind::Priv => "priv",
            TokenKind::Mod => "mod",
            TokenKind::Use => "use",
            TokenKind::Import => "import",
            TokenKind::As => "as",
            TokenKind::Async => "async",
            TokenKind::Await => "await",
            TokenKind::Spawn => "spawn",
            TokenKind::True => "true",
            TokenKind::False => "false",
            TokenKind::SelfLower => "self",
            TokenKind::SelfUpper => "Self",

            // Literals
            TokenKind::IntLiteral => "integer literal",
            TokenKind::FloatLiteral => "float literal",
            TokenKind::BoolLiteral => "boolean literal",
            TokenKind::StringLiteral => "string literal",
            TokenKind::UnitLiteral => "unit literal",

            // Identifiers
            TokenKind::Identifier => "identifier",

            // Operators
            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Star => "*",
            TokenKind::Slash => "/",
            TokenKind::Percent => "%",
            TokenKind::Eq => "==",
            TokenKind::Ne => "!=",
            TokenKind::Lt => "<",
            TokenKind::Le => "<=",
            TokenKind::Gt => ">",
            TokenKind::Ge => ">=",
            TokenKind::And => "&&",
            TokenKind::Or => "||",
            TokenKind::Not => "!",
            TokenKind::BitAnd => "&",
            TokenKind::BitOr => "|",
            TokenKind::BitXor => "^",
            TokenKind::BitNot => "~",
            TokenKind::Shl => "<<",
            TokenKind::Shr => ">>",
            TokenKind::Assign => "=",
            TokenKind::PlusAssign => "+=",
            TokenKind::MinusAssign => "-=",
            TokenKind::StarAssign => "*=",
            TokenKind::SlashAssign => "/=",
            TokenKind::PercentAssign => "%=",
            TokenKind::BitAndAssign => "&=",
            TokenKind::BitOrAssign => "|=",
            TokenKind::BitXorAssign => "^=",
            TokenKind::ShlAssign => "<<=",
            TokenKind::ShrAssign => ">>=",

            // Delimiters
            TokenKind::LParen => "(",
            TokenKind::RParen => ")",
            TokenKind::LBrace => "{",
            TokenKind::RBrace => "}",
            TokenKind::LBracket => "[",
            TokenKind::RBracket => "]",

            // Punctuation
            TokenKind::Comma => ",",
            TokenKind::Semicolon => ";",
            TokenKind::Colon => ":",
            TokenKind::ColonColon => "::",
            TokenKind::Dot => ".",
            TokenKind::DotDot => "..",
            TokenKind::DotDotDot => "...",
            TokenKind::Arrow => "->",
            TokenKind::FatArrow => "=>",
            TokenKind::Question => "?",

            // Special
            TokenKind::Eof => "EOF",
            TokenKind::LineComment => "line comment",
            TokenKind::BlockComment => "block comment",
            TokenKind::DocComment => "doc comment",
            TokenKind::Whitespace => "whitespace",
            TokenKind::Invalid => "invalid token",

            TokenKind::Pound => "#",
            TokenKind::With => "with",
            TokenKind::Extern => "extern",
        };
        write!(f, "{}", s)
    }
}

impl TokenKind {
    // Convert a string slice to a keyword token if it matches,
    // otherwise return None.
    pub fn from_keyword(s: &str) -> Option<Self> {
        match s {
            "fn" => Some(TokenKind::Fn),
            "let" => Some(TokenKind::Let),
            "if" => Some(TokenKind::If),
            "else" => Some(TokenKind::Else),
            "match" => Some(TokenKind::Match),
            "while" => Some(TokenKind::While),
            "for" => Some(TokenKind::For),
            "loop" => Some(TokenKind::Loop),
            "break" => Some(TokenKind::Break),
            "continue" => Some(TokenKind::Continue),
            "return" => Some(TokenKind::Return),
            "struct" => Some(TokenKind::Struct),
            "enum" => Some(TokenKind::Enum),
            "impl" => Some(TokenKind::Impl),
            "trait" => Some(TokenKind::Trait),
            "type" => Some(TokenKind::Type),
            "pub" => Some(TokenKind::Pub),
            "priv" => Some(TokenKind::Priv),
            "mod" => Some(TokenKind::Mod),
            "use" => Some(TokenKind::Use),
            "import" => Some(TokenKind::Import),
            "as" => Some(TokenKind::As),
            "async" => Some(TokenKind::Async),
            "await" => Some(TokenKind::Await),
            "spawn" => Some(TokenKind::Spawn),
            "true" => Some(TokenKind::True),
            "false" => Some(TokenKind::False),
            "self" => Some(TokenKind::SelfLower),
            "Self" => Some(TokenKind::SelfUpper),
            "with" => Some(TokenKind::With),
            "extern" => Some(TokenKind::Extern),
            _ => None,
        }
    }

    // Check if this token kind is a reserved keyword
    pub fn is_reserved_keyword(&self) -> bool {
        matches!(
            self,
            TokenKind::Fn
                | TokenKind::Let
                | TokenKind::If
                | TokenKind::Else
                | TokenKind::Match
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Loop
                | TokenKind::Break
                | TokenKind::Continue
                | TokenKind::Return
                | TokenKind::Struct
                | TokenKind::Enum
                | TokenKind::Impl
                | TokenKind::Trait
                | TokenKind::Type
                | TokenKind::Pub
                | TokenKind::Priv
                | TokenKind::Mod
                | TokenKind::Use
                | TokenKind::Import
                | TokenKind::As
                | TokenKind::Async
                | TokenKind::Await
                | TokenKind::Spawn
                | TokenKind::True
                | TokenKind::False
                | TokenKind::SelfLower
                | TokenKind::SelfUpper
        )
    }

    // Get the precedence level for binary operators (higher = tighter binding).
    // Returns None for non-operator tokens.
    pub fn precedence(&self) -> Option<u8> {
        match self {
            // Assignment operators (lowest precedence)
            TokenKind::Assign
            | TokenKind::PlusAssign
            | TokenKind::MinusAssign
            | TokenKind::StarAssign
            | TokenKind::SlashAssign
            | TokenKind::PercentAssign
            | TokenKind::BitAndAssign
            | TokenKind::BitOrAssign
            | TokenKind::BitXorAssign
            | TokenKind::ShlAssign
            | TokenKind::ShrAssign => Some(1),

            // Logical OR
            TokenKind::Or => Some(2),

            // Logical AND
            TokenKind::And => Some(3),

            // Equality and comparison operators
            TokenKind::Eq | TokenKind::Ne => Some(4),
            TokenKind::Lt | TokenKind::Le | TokenKind::Gt | TokenKind::Ge => Some(5),

            // Bitwise OR
            TokenKind::BitOr => Some(6),

            // Bitwise XOR
            TokenKind::BitXor => Some(7),

            // Bitwise AND
            TokenKind::BitAnd => Some(8),

            // Bit shifts
            TokenKind::Shl | TokenKind::Shr => Some(9),

            // Addition and subtraction
            TokenKind::Plus | TokenKind::Minus => Some(10),

            // Multiplication, division, modulo (highest precedence)
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => Some(11),

            // Not an operator
            _ => None,
        }
    }

    // Check if this operator is right-associative
    pub fn is_right_associative(&self) -> bool {
        matches!(
            self,
            TokenKind::Assign
                | TokenKind::PlusAssign
                | TokenKind::MinusAssign
                | TokenKind::StarAssign
                | TokenKind::SlashAssign
                | TokenKind::PercentAssign
                | TokenKind::BitAndAssign
                | TokenKind::BitOrAssign
                | TokenKind::BitXorAssign
                | TokenKind::ShlAssign
                | TokenKind::ShrAssign
        )
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation() {
        let token = Token::new(TokenKind::Fn, 1, 1);
        assert_eq!(token.kind, TokenKind::Fn);
        assert_eq!(token.line, 1);
        assert_eq!(token.column, 1);
        assert!(token.text.is_none());
    }

    #[test]
    fn test_token_with_text() {
        let token = Token::with_text(TokenKind::Identifier, 2, 5, "myVar".to_string());
        assert_eq!(token.kind, TokenKind::Identifier);
        assert_eq!(token.text(), Some("myVar"));
    }

    #[test]
    fn test_keyword_recognition() {
        assert_eq!(TokenKind::from_keyword("fn"), Some(TokenKind::Fn));
        assert_eq!(TokenKind::from_keyword("let"), Some(TokenKind::Let));
        assert_eq!(TokenKind::from_keyword("if"), Some(TokenKind::If));
        assert_eq!(TokenKind::from_keyword("true"), Some(TokenKind::True));
        assert_eq!(TokenKind::from_keyword("false"), Some(TokenKind::False));
        assert_eq!(TokenKind::from_keyword("self"), Some(TokenKind::SelfLower));
        assert_eq!(TokenKind::from_keyword("Self"), Some(TokenKind::SelfUpper));
        assert_eq!(TokenKind::from_keyword("notakeyword"), None);
    }

    #[test]
    fn test_token_kind_checks() {
        let token = Token::new(TokenKind::Fn, 1, 1);
        assert!(token.is_keyword());
        assert!(!token.is_literal());
        assert!(!token.is_operator());

        let token = Token::new(TokenKind::IntLiteral, 1, 1);
        assert!(!token.is_keyword());
        assert!(token.is_literal());
        assert!(!token.is_operator());

        let token = Token::new(TokenKind::Plus, 1, 1);
        assert!(!token.is_keyword());
        assert!(!token.is_literal());
        assert!(token.is_operator());
    }

    #[test]
    fn test_operator_precedence() {
        assert_eq!(TokenKind::Assign.precedence(), Some(1));
        assert_eq!(TokenKind::Or.precedence(), Some(2));
        assert_eq!(TokenKind::And.precedence(), Some(3));
        assert_eq!(TokenKind::Eq.precedence(), Some(4));
        assert_eq!(TokenKind::Lt.precedence(), Some(5));
        assert_eq!(TokenKind::Plus.precedence(), Some(10));
        assert_eq!(TokenKind::Star.precedence(), Some(11));
        assert_eq!(TokenKind::Fn.precedence(), None);
    }

    #[test]
    fn test_associativity() {
        assert!(TokenKind::Assign.is_right_associative());
        assert!(TokenKind::PlusAssign.is_right_associative());
        assert!(!TokenKind::Plus.is_right_associative());
        assert!(!TokenKind::Star.is_right_associative());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", TokenKind::Fn), "fn");
        assert_eq!(format!("{}", TokenKind::Plus), "+");
        assert_eq!(format!("{}", TokenKind::Arrow), "->");
        assert_eq!(format!("{}", TokenKind::Eq), "==");
    }
}
