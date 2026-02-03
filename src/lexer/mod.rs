//src/lexer/mod.rs

mod token;
mod error;

pub use token::{Token, TokenKind};
pub use error::{LexError, LexResult, LexErrorContext};

use std::str::Chars;
use std::iter::Peekable;

pub struct Lexer<'a> {
    input: &'a str,
    chars: Peekable<Chars<'a>>,
    position: usize,
    line: usize,
    column: usize,
    token_start: usize,
    token_line: usize,
    token_column: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Lexer {
            input,
            chars: input.chars().peekable(),
            position: 0,
            line: 1,
            column: 1,
            token_start: 0,
            token_line: 1,
            token_column: 1,
        }
    }

    pub fn next_token(&mut self) -> LexResult<Token> {
        self.skip_whitespace_and_comments()?;

        self.mark_token_start();

        let ch = match self.peek_char() {
            Some(c) => c,
            None => return Ok(self._make_token(TokenKind::Eof)),
        };
        let token = match ch {
            'a'..='z' | 'A'..='Z' | '_' => self.read_identifier(),
            '0'..='9' => self.read_number(),
            '"' => self.read_string(),
            '+' => self.read_plus(),
            '-' => self.read_minus(),
            '*' => self.read_star(),
            '/' => self.read_slash(),
            '%' => self.read_percent(),
            '=' => self.read_equals(),
            '!' => self.read_exclamation(),
            '<' => self.read_less_than(),
            '>' => self.read_greater_than(),
            '&' => self.read_ampersand(),
            '|' => self.read_pipe(),
            '^' => self.read_caret(),
            '~' => Ok(self.make_simple_token(TokenKind::BitNot)),
            '#' => Ok(self.make_simple_token(TokenKind::Pound)),
            '(' => Ok(self.make_simple_token(TokenKind::LParen)),
            ')' => Ok(self.make_simple_token(TokenKind::RParen)),
            '{' => Ok(self.make_simple_token(TokenKind::LBrace)),
            '}' => Ok(self.make_simple_token(TokenKind::RBrace)),
            '[' => Ok(self.make_simple_token(TokenKind::LBracket)),
            ']' => Ok(self.make_simple_token(TokenKind::RBracket)),
            ',' => Ok(self.make_simple_token(TokenKind::Comma)),
            ';' => Ok(self.make_simple_token(TokenKind::Semicolon)),
            ':' => self.read_colon(),
            '.' => self.read_dot(),
            '?' => Ok(self.make_simple_token(TokenKind::Question)),
            _ => Err(LexError::InvalidCharacter {
                character: ch,
                line: self.line,
                column: self.column,
            }),
        }?;

        Ok(token)
    }

    pub fn tokenize(&mut self) -> LexResult<Vec<Token>> {
        let mut tokens = Vec::new();
        
        loop {
            let token = self.next_token()?;
            let is_eof = token.kind == TokenKind::Eof;
            tokens.push(token);
            
            if is_eof {
                break;
            }
        }
        
        Ok(tokens)
    }

    fn peek_char(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    fn peek_char_n(&self, n: usize) -> Option<char> {
        self.input[self.position..].chars().nth(n)
    }

    fn advance(&mut self) -> Option<char> {
        if let Some(ch) = self.chars.next() {
            self.position += ch.len_utf8();
            
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            
            Some(ch)
        } else {
            None
        }
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn skip_whitespace_and_comments(&mut self) -> LexResult<()> {
        loop {
            match self.peek_char() {
                Some(' ') | Some('\t') | Some('\n') | Some('\r') => {
                    self.advance();
                }
                
                Some('/') => {
                    if self.peek_char_n(1) == Some('/') {
                        self.skip_line_comment();
                    } else if self.peek_char_n(1) == Some('*') {
                        self.skip_block_comment()?;
                    } else {
                        break;
                    }
                }
                
                _ => break,
            }
        }
        
        Ok(())
    }

    fn skip_line_comment(&mut self) {
        self.advance();
        self.advance();
        
        while let Some(ch) = self.peek_char() {
            if ch == '\n' {
                break;
            }
            self.advance();
        }
    }

    fn skip_block_comment(&mut self) -> LexResult<()> {
        let start_line = self.line;
        let start_column = self.column;

        self.advance();
        self.advance();
        
        let mut depth = 1;
        
        while depth > 0 {
            match self.peek_char() {
                None => {
                    return Err(LexError::UnterminatedBlockComment {
                        line: start_line,
                        column: start_column,
                    });
                }
                Some('*') => {
                    self.advance();
                    if self.match_char('/') {
                        depth -= 1;
                    }
                }
                Some('/') => {
                    self.advance();
                    if self.match_char('*') {
                        depth += 1;
                    }
                }
                Some(_) => {
                    self.advance();
                }
            }
        }
        
        Ok(())
    }

    fn mark_token_start(&mut self) {
        self.token_start = self.position;
        self.token_line = self.line;
        self.token_column = self.column;
    }

    fn _make_token(&self, kind: TokenKind) -> Token {
        Token::new(kind, self.token_line, self.token_column)
    }
    fn make_token_with_text(&self, kind: TokenKind, text: String) -> Token {
        Token::with_text(kind, self.token_line, self.token_column, text)
    }

    fn make_simple_token(&mut self, kind: TokenKind) -> Token {
        self.advance();
        self._make_token(kind)
    }

    fn _current_token_text(&self) -> &str {
        &self.input[self.token_start..self.position]
    }

    fn read_identifier(&mut self) -> LexResult<Token> {
        let start = self.position;
        
        self.advance();
        
        while let Some(ch) = self.peek_char() {
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }
        
        let text = &self.input[start..self.position];
        
        let kind = TokenKind::from_keyword(text).unwrap_or(TokenKind::Identifier);
        
        Ok(self.make_token_with_text(kind, text.to_string()))
    }

    fn read_number(&mut self) -> LexResult<Token> {
        let start = self.position;
        
        if self.peek_char() == Some('0') {
            self.advance();
            
            match self.peek_char() {
                Some('x') | Some('X') => {
                    self.advance();
                    return self.read_hex_number(start);
                }
                Some('o') | Some('O') => {
                    self.advance();
                    return self.read_octal_number(start);
                }
                Some('b') | Some('B') => {
                    self.advance();
                    return self.read_binary_number(start);
                }
                Some('.') => {
                    return self.read_float_number(start);
                }
                Some('0'..='9') => {
                }
                _ => {
                    let text = &self.input[start..self.position];
                    return Ok(self.make_token_with_text(TokenKind::IntLiteral, text.to_string()));
                }
            }
        }
        
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }
        
        if self.peek_char() == Some('.') && self.peek_char_n(1).map_or(false, |c| c.is_ascii_digit()) {
            return self.read_float_number(start);
        }
        
        if matches!(self.peek_char(), Some('e') | Some('E')) {
            return self.read_float_number(start);
        }
        
        let text = &self.input[start..self.position];
        Ok(self.make_token_with_text(TokenKind::IntLiteral, text.to_string()))
    }

    fn read_hex_number(&mut self, start: usize) -> LexResult<Token> {
        let _start_pos = self.position;
        
        let mut has_digits = false;
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_hexdigit() {
                self.advance();
                has_digits = true;
            } else {
                break;
            }
        }
        
        if !has_digits {
            return Err(LexError::ExpectedDigitsAfterPrefix {
                prefix: "0x".to_string(),
                line: self.token_line,
                column: self.token_column,
            });
        }
        
        let text = &self.input[start..self.position];
        Ok(self.make_token_with_text(TokenKind::IntLiteral, text.to_string()))
    }

    fn read_octal_number(&mut self, start: usize) -> LexResult<Token> {
        let mut has_digits = false;
        
        while let Some(ch) = self.peek_char() {
            if ('0'..='7').contains(&ch) {
                self.advance();
                has_digits = true;
            } else if ch.is_ascii_digit() {
                return Err(LexError::InvalidDigitForBase {
                    digit: ch,
                    base: 8,
                    line: self.line,
                    column: self.column,
                });
            } else {
                break;
            }
        }
        
        if !has_digits {
            return Err(LexError::ExpectedDigitsAfterPrefix {
                prefix: "0o".to_string(),
                line: self.token_line,
                column: self.token_column,
            });
        }
        
        let text = &self.input[start..self.position];
        Ok(self.make_token_with_text(TokenKind::IntLiteral, text.to_string()))
    }

    fn read_binary_number(&mut self, start: usize) -> LexResult<Token> {
        let mut has_digits = false;
        
        while let Some(ch) = self.peek_char() {
            if ch == '0' || ch == '1' {
                self.advance();
                has_digits = true;
            } else if ch.is_ascii_digit() {
                return Err(LexError::InvalidDigitForBase {
                    digit: ch,
                    base: 2,
                    line: self.line,
                    column: self.column,
                });
            } else {
                break;
            }
        }
        
        if !has_digits {
            return Err(LexError::ExpectedDigitsAfterPrefix {
                prefix: "0b".to_string(),
                line: self.token_line,
                column: self.token_column,
            });
        }
        
        let text = &self.input[start..self.position];
        Ok(self.make_token_with_text(TokenKind::IntLiteral, text.to_string()))
    }

    fn read_float_number(&mut self, start: usize) -> LexResult<Token> {
        if self.peek_char() == Some('.') {
            self.advance();
            
            while let Some(ch) = self.peek_char() {
                if ch.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        
        if matches!(self.peek_char(), Some('e') | Some('E')) {
            self.advance();
            
            if matches!(self.peek_char(), Some('+') | Some('-')) {
                self.advance();
            }
            
            let mut has_exp_digits = false;
            while let Some(ch) = self.peek_char() {
                if ch.is_ascii_digit() {
                    self.advance();
                    has_exp_digits = true;
                } else {
                    break;
                }
            }
            
            if !has_exp_digits {
                return Err(LexError::InvalidExponent {
                    reason: "Expected digits after exponent".to_string(),
                    line: self.line,
                    column: self.column,
                });
            }
        }
        
        let text = &self.input[start..self.position];
        Ok(self.make_token_with_text(TokenKind::FloatLiteral, text.to_string()))
    }

    fn read_string(&mut self) -> LexResult<Token> {
        let start_line = self.line;
        let start_column = self.column;
        
        self.advance();
        
        let mut string_content = String::new();
        
        loop {
            match self.peek_char() {
                None | Some('\n') => {
                    return Err(LexError::UnterminatedString {
                        line: start_line,
                        column: start_column,
                    });
                }
                Some('"') => {
                    self.advance();
                    break;
                }
                Some('\\') => {
                    self.advance();
                    let escaped = self.read_escape_sequence()?;
                    string_content.push(escaped);
                }
                Some(ch) => {
                    string_content.push(ch);
                    self.advance();
                }
            }
        }
        
        Ok(self.make_token_with_text(TokenKind::StringLiteral, string_content))
    }

    fn read_escape_sequence(&mut self) -> LexResult<char> {
        let line = self.line;
        let column = self.column;
        
        match self.peek_char() {
            Some('n') => {
                self.advance();
                Ok('\n')
            }
            Some('r') => {
                self.advance();
                Ok('\r')
            }
            Some('t') => {
                self.advance();
                Ok('\t')
            }
            Some('\\') => {
                self.advance();
                Ok('\\')
            }
            Some('"') => {
                self.advance();
                Ok('"')
            }
            Some('u') => {
                self.advance();
                self.read_unicode_escape()
            }
            Some(ch) => {
                let seq = ch.to_string();
                self.advance();
                Err(LexError::InvalidEscapeSequence {
                    sequence: seq,
                    line,
                    column,
                })
            }
            None => Err(LexError::UnexpectedEof { line, column }),
        }
    }

    fn read_unicode_escape(&mut self) -> LexResult<char> {
        let line = self.line;
        let column = self.column;
        
        if !self.match_char('{') {
            return Err(LexError::InvalidUnicodeEscape {
                sequence: String::new(),
                reason: "Expected '{' after '\\u'".to_string(),
                line,
                column,
            });
        }
        
        let mut hex_str = String::new();
        
        loop {
            match self.peek_char() {
                Some('}') => {
                    self.advance();
                    break;
                }
                Some(ch) if ch.is_ascii_hexdigit() => {
                    hex_str.push(ch);
                    self.advance();
                }
                _ => {
                    return Err(LexError::InvalidUnicodeEscape {
                        sequence: hex_str,
                        reason: "Invalid hex digit in Unicode escape".to_string(),
                        line,
                        column,
                    });
                }
            }
        }
        
        if hex_str.is_empty() {
            return Err(LexError::InvalidUnicodeEscape {
                sequence: String::new(),
                reason: "Empty Unicode escape sequence".to_string(),
                line,
                column,
            });
        }
        
        let code_point = u32::from_str_radix(&hex_str, 16).map_err(|_| {
            LexError::InvalidUnicodeEscape {
                sequence: hex_str.clone(),
                reason: "Invalid hex number".to_string(),
                line,
                column,
            }
        })?;
        
        char::from_u32(code_point).ok_or_else(|| LexError::InvalidUnicodeEscape {
            sequence: hex_str,
            reason: "Invalid Unicode code point".to_string(),
            line,
            column,
        })
    }

    fn read_plus(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('=') {
            Ok(self._make_token(TokenKind::PlusAssign))
        } else {
            Ok(self._make_token(TokenKind::Plus))
        }
    }

    fn read_minus(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('=') {
            Ok(self._make_token(TokenKind::MinusAssign))
        } else if self.match_char('>') {
            Ok(self._make_token(TokenKind::Arrow))
        } else {
            Ok(self._make_token(TokenKind::Minus))
        }
    }

    fn read_star(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('=') {
            Ok(self._make_token(TokenKind::StarAssign))
        } else {
            Ok(self._make_token(TokenKind::Star))
        }
    }

    fn read_slash(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('=') {
            Ok(self._make_token(TokenKind::SlashAssign))
        } else {
            Ok(self._make_token(TokenKind::Slash))
        }
    }

    fn read_percent(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('=') {
            Ok(self._make_token(TokenKind::PercentAssign))
        } else {
            Ok(self._make_token(TokenKind::Percent))
        }
    }

    fn read_equals(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('=') {
            Ok(self._make_token(TokenKind::Eq))
        } else if self.match_char('>') {
            Ok(self._make_token(TokenKind::FatArrow))
        } else {
            Ok(self._make_token(TokenKind::Assign))
        }
    }

    fn read_exclamation(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('=') {
            Ok(self._make_token(TokenKind::Ne))
        } else {
            Ok(self._make_token(TokenKind::Not))
        }
    }

    fn read_less_than(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('=') {
            Ok(self._make_token(TokenKind::Le))
        } else if self.match_char('<') {
            if self.match_char('=') {
                Ok(self._make_token(TokenKind::ShlAssign))
            } else {
                Ok(self._make_token(TokenKind::Shl))
            }
        } else {
            Ok(self._make_token(TokenKind::Lt))
        }
    }

    fn read_greater_than(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('=') {
            Ok(self._make_token(TokenKind::Ge))
        } else if self.match_char('>') {
            if self.match_char('=') {
                Ok(self._make_token(TokenKind::ShrAssign))
            } else {
                Ok(self._make_token(TokenKind::Shr))
            }
        } else {
            Ok(self._make_token(TokenKind::Gt))
        }
    }

    fn read_ampersand(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('&') {
            Ok(self._make_token(TokenKind::And))
        } else if self.match_char('=') {
            Ok(self._make_token(TokenKind::BitAndAssign))
        } else {
            Ok(self._make_token(TokenKind::BitAnd))
        }
    }

    fn read_pipe(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('|') {
            Ok(self._make_token(TokenKind::Or))
        } else if self.match_char('=') {
            Ok(self._make_token(TokenKind::BitOrAssign))
        } else {
            Ok(self._make_token(TokenKind::BitOr))
        }
    }

    fn read_caret(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('=') {
            Ok(self._make_token(TokenKind::BitXorAssign))
        } else {
            Ok(self._make_token(TokenKind::BitXor))
        }
    }

    fn read_colon(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char(':') {
            Ok(self._make_token(TokenKind::ColonColon))
        } else {
            Ok(self._make_token(TokenKind::Colon))
        }
    }

    fn read_dot(&mut self) -> LexResult<Token> {
        self.advance();
        if self.match_char('.') {
            if self.match_char('.') {
                Ok(self._make_token(TokenKind::DotDotDot))
            } else {
                Ok(self._make_token(TokenKind::DotDot))
            }
        } else {
            Ok(self._make_token(TokenKind::Dot))
        }
    }
}

pub fn tokenize(input: &str) -> LexResult<Vec<Token>> {
    let mut lexer = Lexer::new(input);
    lexer.tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let mut lexer = Lexer::new("");
        let token = lexer.next_token().unwrap();
        assert_eq!(token.kind, TokenKind::Eof);
    }

    #[test]
    fn test_keywords() {
        let input = "fn let if else while for return";
        let tokens = tokenize(input).unwrap();
        
        assert_eq!(tokens[0].kind, TokenKind::Fn);
        assert_eq!(tokens[1].kind, TokenKind::Let);
        assert_eq!(tokens[2].kind, TokenKind::If);
        assert_eq!(tokens[3].kind, TokenKind::Else);
        assert_eq!(tokens[4].kind, TokenKind::While);
        assert_eq!(tokens[5].kind, TokenKind::For);
        assert_eq!(tokens[6].kind, TokenKind::Return);
    }

    #[test]
    fn test_identifiers() {
        let input = "myVar _private add123";
        let tokens = tokenize(input).unwrap();
        
        assert_eq!(tokens[0].kind, TokenKind::Identifier);
        assert_eq!(tokens[0].text(), Some("myVar"));
        assert_eq!(tokens[1].kind, TokenKind::Identifier);
        assert_eq!(tokens[1].text(), Some("_private"));
        assert_eq!(tokens[2].kind, TokenKind::Identifier);
        assert_eq!(tokens[2].text(), Some("add123"));
    }

    #[test]
    fn test_integer_literals() {
        let input = "42 0xFF 0o755 0b1010";
        let tokens = tokenize(input).unwrap();
        
        assert_eq!(tokens[0].kind, TokenKind::IntLiteral);
        assert_eq!(tokens[0].text(), Some("42"));
        assert_eq!(tokens[1].kind, TokenKind::IntLiteral);
        assert_eq!(tokens[1].text(), Some("0xFF"));
        assert_eq!(tokens[2].kind, TokenKind::IntLiteral);
        assert_eq!(tokens[2].text(), Some("0o755"));
        assert_eq!(tokens[3].kind, TokenKind::IntLiteral);
        assert_eq!(tokens[3].text(), Some("0b1010"));
    }

    #[test]
    fn test_float_literals() {
        let input = "3.14 1.0e-10 6.022e23";
        let tokens = tokenize(input).unwrap();
        
        assert_eq!(tokens[0].kind, TokenKind::FloatLiteral);
        assert_eq!(tokens[0].text(), Some("3.14"));
        assert_eq!(tokens[1].kind, TokenKind::FloatLiteral);
        assert_eq!(tokens[1].text(), Some("1.0e-10"));
        assert_eq!(tokens[2].kind, TokenKind::FloatLiteral);
        assert_eq!(tokens[2].text(), Some("6.022e23"));
    }

    #[test]
    fn test_string_literals() {
        let input = r#""Hello, World!" "Line 1\nLine 2""#;
        let tokens = tokenize(input).unwrap();
        
        assert_eq!(tokens[0].kind, TokenKind::StringLiteral);
        assert_eq!(tokens[0].text(), Some("Hello, World!"));
        assert_eq!(tokens[1].kind, TokenKind::StringLiteral);
        assert_eq!(tokens[1].text(), Some("Line 1\nLine 2"));
    }
}
