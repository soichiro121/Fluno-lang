// src/lexer/token.rs

use crate::ast::node::Span;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub column: usize,
    pub text: Option<String>,
}

impl Token {
    pub fn new(kind: TokenKind, line: usize, column: usize) -> Self {
        Token {
            kind,
            line,
            column,
            text: None,
        }
    }

    pub fn with_text(kind: TokenKind, line: usize, column: usize, text: String) -> Self {
        Token {
            kind,
            line,
            column,
            text: Some(text),
        }
    }

    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    pub fn span(&self) -> Span {
        let len = self.text.as_ref().map(|s| s.len()).unwrap_or(0);
        Span::new(self.line, self.column, len)
    }

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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TokenKind {
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
    Struct,
    Enum,
    Impl,
    Trait,
    Type,
    Pub,
    Priv,
    Mod,
    Use,
    Import,
    As,
    Async,
    Await,
    Spawn,
    True,
    False,
    SelfLower,
    SelfUpper,

    IntLiteral,
    FloatLiteral,
    BoolLiteral,
    StringLiteral,
    UnitLiteral,
    Identifier,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Not,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,
    Assign,
    PlusAssign,
    MinusAssign,
    StarAssign,
    SlashAssign,
    PercentAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,

    ShlAssign,
    ShrAssign,

    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,

    Comma,
    Semicolon,
    Colon,
    ColonColon,
    Dot,
    DotDot,
    DotDotDot,
    Arrow,
    FatArrow,
    Question,

    Eof,
    LineComment,
    BlockComment,
    DocComment,
    Whitespace,
    Invalid,

    Pound,
    With,
    Extern,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
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

            TokenKind::IntLiteral => "integer literal",
            TokenKind::FloatLiteral => "float literal",
            TokenKind::BoolLiteral => "boolean literal",
            TokenKind::StringLiteral => "string literal",
            TokenKind::UnitLiteral => "unit literal",

            TokenKind::Identifier => "identifier",

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

            TokenKind::LParen => "(",
            TokenKind::RParen => ")",
            TokenKind::LBrace => "{",
            TokenKind::RBrace => "}",
            TokenKind::LBracket => "[",
            TokenKind::RBracket => "]",

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

    pub fn precedence(&self) -> Option<u8> {
        match self {
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

            TokenKind::Or => Some(2),

            TokenKind::And => Some(3),

            TokenKind::Eq | TokenKind::Ne => Some(4),
            TokenKind::Lt | TokenKind::Le | TokenKind::Gt | TokenKind::Ge => Some(5),

            TokenKind::BitOr => Some(6),

            TokenKind::BitXor => Some(7),

            TokenKind::BitAnd => Some(8),

            TokenKind::Shl | TokenKind::Shr => Some(9),

            TokenKind::Plus | TokenKind::Minus => Some(10),

            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => Some(11),
            _ => None,
        }
    }

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
