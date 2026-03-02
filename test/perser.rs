// ==========================
// トークン・位置情報定義
// ==========================
#[derive(Debug, PartialEq, Clone)]
pub struct Position {
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, PartialEq, Clone)]
pub enum TokenKind {
    Ident(String), Int(i64), Float(f64), Bool(bool), String(String),
    LParen, RParen, LBrace, RBrace, Comma, Semicolon, Colon, Arrow,
    Plus, Minus, Star, Slash, Percent, Bang, Eq, EqEq, Neq, Lt, Gt, Leq, Geq,
    Assign, If, Else, While, For, Match, Fn, Let, Return,
    Pipe, FatArrow, Underscore, Comment(String),
    EOF,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub pos: Position,
}

// ==========================
// 字句解析器
// ==========================
pub struct Lexer {
    input: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    pub fn new(input: String) -> Self {
        Lexer { input: input.chars().collect(), pos: 0, line: 1, col: 1 }
    }
    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }
    fn next(&mut self) -> Option<char> {
        if let Some(c) = self.input.get(self.pos).copied() {
            self.pos += 1;
            if c == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            Some(c)
        } else { None }
    }
    fn make_token(&self, kind: TokenKind, start_col: usize) -> Token {
        Token { kind, pos: Position { line: self.line, col: start_col } }
    }
    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) { self.next(); }
    }
    fn ident_or_keyword(&mut self, start_col: usize) -> Token {
        let start = self.pos - 1;
        while matches!(self.peek(), Some(c) if c.is_alphanumeric() || c == '_') { self.next(); }
        let s: String = self.input[start..self.pos].iter().collect();
        let kind = match s.as_str() {
            "fn" => TokenKind::Fn, "let" => TokenKind::Let, "if" => TokenKind::If, "else" => TokenKind::Else,
            "while" => TokenKind::While, "for" => TokenKind::For, "match" => TokenKind::Match,
            "true" => TokenKind::Bool(true), "false" => TokenKind::Bool(false), "return" => TokenKind::Return,
            "_" => TokenKind::Underscore, _ => TokenKind::Ident(s),
        };
        self.make_token(kind, start_col)
    }
    fn number(&mut self, start_col: usize, first: char) -> Token {
        let mut is_float = false;
        let start = self.pos - 1;
        while matches!(self.peek(), Some('0'..='9')) { self.next(); }
        if self.peek() == Some('.') {
            is_float = true;
            self.next();
            while matches!(self.peek(), Some('0'..='9')) { self.next(); }
        }
        let s: String = self.input[start..self.pos].iter().collect();
        let kind = if is_float { TokenKind::Float(s.parse().unwrap()) }
        else { TokenKind::Int(s.parse().unwrap()) };
        self.make_token(kind, start_col)
    }
    fn string_lit(&mut self, start_col: usize) -> Token {
        let mut s = String::new();
        self.next(); // skip "
        while let Some(c) = self.peek() {
            if c == '"' { break; }
            s.push(self.next().unwrap());
        }
        self.next(); // skip "
        self.make_token(TokenKind::String(s), start_col)
    }
    fn comment(&mut self, start_col: usize) -> Token {
        self.next(); // skip /
        if self.peek() == Some('/') {
            self.next();
            let mut s = String::new();
            while let Some(c) = self.peek() {
                if c == '\n' { break; }
                s.push(self.next().unwrap());
            }
            self.make_token(TokenKind::Comment(s), start_col)
        } else {
            self.make_token(TokenKind::Slash, start_col)
        }
    }
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        let start_col = self.col;
        let ch = match self.next() { Some(c) => c, None => return self.make_token(TokenKind::EOF, start_col) };
        match ch {
            c if c.is_alphabetic() || c == '_' => self.ident_or_keyword(start_col),
            c if c.is_digit(10) => self.number(start_col, c),
            '"' => self.string_lit(start_col),
            '(' => self.make_token(TokenKind::LParen, start_col),
            ')' => self.make_token(TokenKind::RParen, start_col),
            '{' => self.make_token(TokenKind::LBrace, start_col),
            '}' => self.make_token(TokenKind::RBrace, start_col),
            ',' => self.make_token(TokenKind::Comma, start_col),
            ';' => self.make_token(TokenKind::Semicolon, start_col),
            ':' => self.make_token(TokenKind::Colon, start_col),
            '=' => if self.peek() == Some('=') {
                self.next(); self.make_token(TokenKind::EqEq, start_col)
            } else if self.peek() == Some('>') {
                self.next(); self.make_token(TokenKind::FatArrow, start_col)
            } else {
                self.make_token(TokenKind::Assign, start_col)
            },
            '-' => if self.peek() == Some('>') {
                self.next(); self.make_token(TokenKind::Arrow, start_col)
            } else {
                self.make_token(TokenKind::Minus, start_col)
            },
            '!' => if self.peek() == Some('=') { self.next(); self.make_token(TokenKind::Neq, start_col) }
                   else { self.make_token(TokenKind::Bang, start_col) },
            '<' => if self.peek() == Some('=') { self.next(); self.make_token(TokenKind::Leq, start_col) }
                   else { self.make_token(TokenKind::Lt, start_col) },
            '>' => if self.peek() == Some('=') { self.next(); self.make_token(TokenKind::Geq, start_col) }
                   else { self.make_token(TokenKind::Gt, start_col) },
            '+' => self.make_token(TokenKind::Plus, start_col),
            '*' => self.make_token(TokenKind::Star, start_col),
            '/' => self.comment(start_col),
            '%' => self.make_token(TokenKind::Percent, start_col),
            '|' => self.make_token(TokenKind::Pipe, start_col),
            _ => self.make_token(TokenKind::EOF, start_col),
        }
    }
}

// ==========================
// 型情報
// ==========================
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int, Float, Bool, String, Unit,
    Func(Vec<Type>, Box<Type>),
    Unknown,
}
impl Type {
    pub fn from_str(s: &str) -> Self {
        match s {
            "Int" => Type::Int,
            "Float" => Type::Float,
            "Bool" => Type::Bool,
            "String" => Type::String,
            "Unit" => Type::Unit,
            _ => Type::Unknown,
        }
    }
}

// ==========================
// AST構造
// ==========================
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64), Float(f64), Bool(bool), String(String), Ident(String),
    BinaryOp { left: Box<Expr>, op: String, right: Box<Expr> },
    Call { func: Box<Expr>, args: Vec<Expr> },
    Paren(Box<Expr>),
    If { cond: Box<Expr>, then_br: Box<Expr>, else_br: Option<Box<Expr>> },
    While { cond: Box<Expr>, body: Box<Expr> },
    For { var: String, iter: Box<Expr>, body: Box<Expr> },
    Match { expr: Box<Expr>, arms: Vec<(Expr, Expr)> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Expr(Expr),
    Let { name: String, ty: Option<Type>, expr: Expr },
    FnDef { name: String, params: Vec<(String, Type)>, ret_ty: Type, body: Vec<Stmt> },
    Return(Expr),
}

#[derive(Debug)]
pub struct ParseError {
    msg: String,
    pos: Position,
}

// ==========================
// Parser本体
// ==========================
pub struct Parser {
    lexer: Lexer,
    pub cur: Token,
    pub errors: Vec<ParseError>,
}

impl Parser {
    pub fn new(mut lexer: Lexer) -> Self {
        let cur = lexer.next_token();
        Parser { lexer, cur, errors: vec![] }
    }
    fn advance(&mut self) { self.cur = self.lexer.next_token(); }
    fn err(&mut self, msg: &str) {
        self.errors.push(ParseError { msg: msg.into(), pos: self.cur.pos.clone() });
    }
    pub fn parse_program(&mut self) -> Vec<Stmt> {
        let mut stmts = vec![];
        while self.cur.kind != TokenKind::EOF {
            match self.parse_stmt() {
                Ok(s) => stmts.push(s),
                Err(_) => self.advance(),
            }
        }
        stmts
    }
    fn expect(&mut self, kind: &TokenKind) -> bool {
        if &self.cur.kind == kind { self.advance(); true }
        else { self.err(&format!("expected {:?}, found {:?}", kind, self.cur.kind)); false }
    }
    fn parse_stmt(&mut self) -> Result<Stmt,()> {
        match &self.cur.kind {
            TokenKind::Let => self.parse_let(),
            TokenKind::Fn => self.parse_fn(),
            TokenKind::Return => { self.advance(); let e = self.parse_expr()?; Ok(Stmt::Return(e)) },
            _ => Ok(Stmt::Expr(self.parse_expr()?)),
        }
    }
    fn parse_let(&mut self) -> Result<Stmt,()> {
        self.advance();
        let name = if let TokenKind::Ident(id) = &self.cur.kind { id.clone() } else { self.err("expected identifier"); return Err(()); };
        self.advance();
        let mut ty = None;
        if self.cur.kind == TokenKind::Colon {
            self.advance();
            if let TokenKind::Ident(ref id) = self.cur.kind { ty = Some(Type::from_str(id)); self.advance(); }
            else { self.err("expected type after colon"); return Err(()); }
        }
        self.expect(&TokenKind::Assign);
        let expr = self.parse_expr()?;
        if self.cur.kind == TokenKind::Semicolon { self.advance(); }
        Ok(Stmt::Let { name, ty, expr })
    }
    fn parse_fn(&mut self) -> Result<Stmt,()> {
        self.advance();
        let name = if let TokenKind::Ident(id) = &self.cur.kind { id.clone() } else { self.err("expected identifier"); return Err(()); };
        self.advance();
        self.expect(&TokenKind::LParen);
        let mut params = vec![];
        while self.cur.kind != TokenKind::RParen && self.cur.kind != TokenKind::EOF {
            if let TokenKind::Ident(id) = &self.cur.kind {
                self.advance();
                self.expect(&TokenKind::Colon);
                if let TokenKind::Ident(ref tyname) = self.cur.kind { params.push((id.clone(), Type::from_str(tyname))); self.advance(); }
                else { self.err("expected type in param"); return Err(()); }
                if self.cur.kind == TokenKind::Comma { self.advance(); }
            } else { self.err("param name expected"); return Err(()); }
        }
        self.expect(&TokenKind::RParen);
        let ret_ty = if self.cur.kind == TokenKind::Arrow {
            self.advance();
            if let TokenKind::Ident(ref tyname) = self.cur.kind { let t = Type::from_str(tyname); self.advance(); t }
            else { self.err("expected type"); Type::Unknown }
        } else { Type::Unit };
        self.expect(&TokenKind::LBrace);
        let mut body = vec![];
        while self.cur.kind != TokenKind::RBrace && self.cur.kind != TokenKind::EOF {
            body.push(self.parse_stmt()?);
        }
        self.expect(&TokenKind::RBrace);
        Ok(Stmt::FnDef { name, params, ret_ty, body })
    }
    fn parse_expr(&mut self) -> Result<Expr,()> { self.parse_expr_bp(0) }

    // 演算子優先順位付きパーサ
    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr,()> {
        let mut lhs = match &self.cur.kind {
            TokenKind::Int(i) => { let v = *i; self.advance(); Expr::Int(v) },
            TokenKind::Float(f) => { let v = *f; self.advance(); Expr::Float(v) },
            TokenKind::Bool(b) => { let v = *b; self.advance(); Expr::Bool(v) },
            TokenKind::String(s) => { let v = s.clone(); self.advance(); Expr::String(v) },
            TokenKind::Ident(id) => {
                let v = id.clone();
                self.advance();
                if self.cur.kind == TokenKind::LParen {
                    self.advance();
                    let mut args = vec![];
                    while self.cur.kind != TokenKind::RParen && self.cur.kind != TokenKind::EOF {
                        args.push(self.parse_expr()?);
                        if self.cur.kind == TokenKind::Comma { self.advance(); }
                    }
                    self.expect(&TokenKind::RParen);
                    Expr::Call { func: Box::new(Expr::Ident(v)), args }
                } else { Expr::Ident(v) }
            }
            TokenKind::LParen => { self.advance(); let e = self.parse_expr()?; self.expect(&TokenKind::RParen); Expr::Paren(Box::new(e)) }
            TokenKind::If => self.parse_if()?,
            TokenKind::While => self.parse_while()?,
            TokenKind::For => self.parse_for()?,
            TokenKind::Match => self.parse_match()?,
            _ => { self.err("unexpected token"); return Err(()); }
        };

        loop {
            let op_info = match &self.cur.kind {
                TokenKind::Plus => Some((1, 2, "+")),
                TokenKind::Minus => Some((1, 2, "-")),
                TokenKind::Star => Some((3, 4, "*")),
                TokenKind::Slash => Some((3, 4, "/")),
                TokenKind::Percent => Some((3, 4, "%")),
                TokenKind::EqEq => Some((0, 1, "==")),
                TokenKind::Neq => Some((0, 1, "!=")),
                TokenKind::Lt => Some((0, 1, "<")),
                TokenKind::Gt => Some((0, 1, ">")),
                TokenKind::Leq => Some((0, 1, "<=")),
                TokenKind::Geq => Some((0, 1, ">=")),
                TokenKind::Assign => Some((0, 1, "=")),
                _ => None,
            };
            if let Some((left_bp, right_bp, op)) = op_info {
                if left_bp < min_bp { break; }
                let o = op.to_string();
                self.advance();
                let rhs = self.parse_expr_bp(right_bp)?;
                lhs = Expr::BinaryOp { left: Box::new(lhs), op: o, right: Box::new(rhs) };
                continue;
            }
            break;
        }
        Ok(lhs)
    }
    fn parse_if(&mut self) -> Result<Expr,()> {
        self.advance(); // if
        let cond = self.parse_expr()?;
        self.expect(&TokenKind::LBrace);
        let mut then_stmts = vec![];
        while self.cur.kind != TokenKind::RBrace && self.cur.kind != TokenKind::EOF {
            then_stmts.push(self.parse_stmt()?);
        }
        self.expect(&TokenKind::RBrace);
        let then_br = if then_stmts.len() == 1 { Box::new(match then_stmts.pop().unwrap() {
            Stmt::Expr(e) => e, _ => Expr::Ident("<stmt>".into())
        })} else { Box::new(Expr::Ident("<block>".into())) }; // 簡約例
        let else_br = if self.cur.kind == TokenKind::Else {
            self.advance();
            self.expect(&TokenKind::LBrace);
            let mut else_stmts = vec![];
            while self.cur.kind != TokenKind::RBrace && self.cur.kind != TokenKind::EOF {
                else_stmts.push(self.parse_stmt()?);
            }
            self.expect(&TokenKind::RBrace);
            Some(Box::new(Expr::Ident("<else>".into()))) // 本格化は省略
        } else { None };
        Ok(Expr::If { cond: Box::new(cond), then_br, else_br })
    }
    fn parse_while(&mut self) -> Result<Expr,()> {
        self.advance();
        let cond = self.parse_expr()?;
        self.expect(&TokenKind::LBrace);
        let mut body = vec![];
        while self.cur.kind != TokenKind::RBrace && self.cur.kind != TokenKind::EOF {
            body.push(self.parse_stmt()?);
        }
        self.expect(&TokenKind::RBrace);
        Ok(Expr::While { cond: Box::new(cond), body: Box::new(Expr::Ident("<block>".into())) })
    }
    fn parse_for(&mut self) -> Result<Expr,()> {
        self.advance();
        let var = if let TokenKind::Ident(id) = &self.cur.kind { id.clone() } else { self.err("for var"); return Err(()); };
        self.advance();
        self.expect(&TokenKind::In);
        let iter = self.parse_expr()?;
        self.expect(&TokenKind::LBrace);
        let mut body = vec![];
        while self.cur.kind != TokenKind::RBrace && self.cur.kind != TokenKind::EOF {
            body.push(self.parse_stmt()?);
        }
        self.expect(&TokenKind::RBrace);
        Ok(Expr::For { var, iter: Box::new(iter), body: Box::new(Expr::Ident("<block>".into())) })
    }
    fn parse_match(&mut self) -> Result<Expr,()> {
        self.advance();
        let expr = self.parse_expr()?;
        self.expect(&TokenKind::LBrace);
        let mut arms = vec![];
        while self.cur.kind != TokenKind::RBrace && self.cur.kind != TokenKind::EOF {
            let pat = self.parse_expr()?;
            self.expect(&TokenKind::FatArrow);
            let res = self.parse_expr()?;
            self.expect(&TokenKind::Comma);
            arms.push((pat, res));
        }
        self.expect(&TokenKind::RBrace);
        Ok(Expr::Match { expr: Box::new(expr), arms })
    }
}

// ==========================
// 型チェック（小規模）
// ==========================
pub fn type_check_expr(expr: &Expr) -> Result<Type, String> {
    match expr {
        Expr::Int(_) => Ok(Type::Int),
        Expr::Float(_) => Ok(Type::Float),
        Expr::Bool(_) => Ok(Type::Bool),
        Expr::String(_) => Ok(Type::String),
        Expr::BinaryOp { left, op, right } => {
            let t_l = type_check_expr(left)?;
            let t_r = type_check_expr(right)?;
            if t_l == t_r && (t_l == Type::Int || t_l == Type::Float) && ["+","-","*","/","%"].contains(&op.as_str()) {
                Ok(t_l)
            } else {
                Err(format!("type mismatch in binary op {:?}: {:?} vs {:?}", op, t_l, t_r))
            }
        },
        Expr::Paren(e) => type_check_expr(e),
        _ => Ok(Type::Unknown), // 省略
    }
}

// ==========================
// テスト
// ==========================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_priorities() {
        let mut lexer = Lexer::new("x * (y + 3) // test".into());
        assert_eq!(lexer.next_token().kind, TokenKind::Ident("x".into()));
        assert_eq!(lexer.next_token().kind, TokenKind::Star);
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Ident("y".into()));
        assert_eq!(lexer.next_token().kind, TokenKind::Plus);
        assert_eq!(lexer.next_token().kind, TokenKind::Int(3));
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert!(matches!(lexer.next_token().kind, TokenKind::Comment(_)));
    }

    #[test]
    fn test_parser_binop_and_paren_priority() {
        let mut parser = Parser::new(Lexer::new("1 + 2 * 3 + (4 + 5)".into()));
        let expr = parser.parse_expr().unwrap();
        // AST形など詳細な確認追加可能
    }

    #[test]
    fn test_if_while_for_match() {
        let src = "
            if 1 < 2 { let x: Int = 3; } else { let x: Int = 4; }
            while true { let y: Bool = false; }
            match x { 1 => 10, 2 => 20, }
        ";
        let mut parser = Parser::new(Lexer::new(src.into()));
        let _ast = parser.parse_program();
        assert!(parser.errors.is_empty());
    }

    #[test]
    fn test_error_report() {
        let mut parser = Parser::new(Lexer::new("let x: = 3;".into()));
        let ast = parser.parse_program();
        assert!(!parser.errors.is_empty());
        println!("error: {:?}", parser.errors);
    }

    #[test]
    fn test_type_check() {
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Int(1)),
            op: "+".to_string(),
            right: Box::new(Expr::Int(2)),
        };
        assert_eq!(type_check_expr(&expr).unwrap(), Type::Int);

        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Int(1)),
            op: "+".to_string(),
            right: Box::new(Expr::Float(2.0)),
        };
        assert!(type_check_expr(&expr).is_err());
    }
}
//❗ 現時点で足りない/改善できる点
//（必要なレベルで十分ですが、プロジェクトを拡張する場合）

//演算子優先度体系をさらに詳細化（例えば論理演算子&&/||/単項!も含めて将来的に拡張）

//match分岐やコードブロックの本格化（今は省略形ASTにしているので後ほど拡張）

//型推論アルゴリズムの拡張（現状は構造体のInt/Floatのチェックのみ、ジェネリクスや関数型拡張には追加が必要）

//関数呼び出しの型チェック本格化（現状は省略）

//エラー回復/同期/複数エラー報告の強化（今はadvanceでスキップ処理、企画によっては未完成部分をもっと詳しく表示可能）

//コメント・プリミティブ型類の拡張・Unicode識別子等の拡張（プロ仕様では拡張する）
