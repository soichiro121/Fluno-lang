// ==========================
// 既存の型システム（前回から継承）
// ==========================
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static TYPE_VAR_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int, Float, Bool, String, Unit,
    TypeVar(usize),
    Array(Box<Type>),
    Option(Box<Type>),
    Result(Box<Type>, Box<Type>),
    Func(Vec<Type>, Box<Type>),
    Gaussian, Uniform,
    Signal(Box<Type>),
    Event(Box<Type>),
    // 【新規】ユーザー定義型（struct/enum）
    Named(String),
    // 【新規】トレイトオブジェクト
    TraitObject(String),
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
            "Gaussian" => Type::Gaussian,
            "Uniform" => Type::Uniform,
            _ => Type::Named(s.to_string()),
        }
    }
    
    pub fn display(&self) -> String {
        match self {
            Type::Int => "Int".to_string(),
            Type::Float => "Float".to_string(),
            Type::Bool => "Bool".to_string(),
            Type::String => "String".to_string(),
            Type::Unit => "Unit".to_string(),
            Type::TypeVar(id) => format!("'t{}", id),
            Type::Array(t) => format!("Array<{}>", t.display()),
            Type::Option(t) => format!("Option<{}>", t.display()),
            Type::Result(t, e) => format!("Result<{}, {}>", t.display(), e.display()),
            Type::Func(params, ret) => {
                let params_str = params.iter()
                    .map(|t| t.display())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("Fn({}) -> {}", params_str, ret.display())
            }
            Type::Gaussian => "Gaussian".to_string(),
            Type::Uniform => "Uniform".to_string(),
            Type::Signal(t) => format!("Signal<{}>", t.display()),
            Type::Event(t) => format!("Event<{}>", t.display()),
            Type::Named(name) => name.clone(),
            Type::TraitObject(name) => format!("dyn {}", name),
            Type::Unknown => "?".to_string(),
        }
    }

    pub fn occurs_in(&self, var_id: usize, subst: &Substitution) -> bool {
        match self.apply_subst(subst) {
            Type::TypeVar(id) => id == var_id,
            Type::Array(t) | Type::Option(t) | Type::Signal(t) | Type::Event(t) => {
                t.occurs_in(var_id, subst)
            }
            Type::Result(t, e) => t.occurs_in(var_id, subst) || e.occurs_in(var_id, subst),
            Type::Func(params, ret) => {
                params.iter().any(|p| p.occurs_in(var_id, subst)) || ret.occurs_in(var_id, subst)
            }
            _ => false,
        }
    }

    pub fn apply_subst(&self, subst: &Substitution) -> Type {
        match self {
            Type::TypeVar(id) => subst.get(*id).unwrap_or_else(|| self.clone()),
            Type::Array(t) => Type::Array(Box::new(t.apply_subst(subst))),
            Type::Option(t) => Type::Option(Box::new(t.apply_subst(subst))),
            Type::Result(t, e) => Type::Result(
                Box::new(t.apply_subst(subst)),
                Box::new(e.apply_subst(subst)),
            ),
            Type::Func(params, ret) => Type::Func(
                params.iter().map(|p| p.apply_subst(subst)).collect(),
                Box::new(ret.apply_subst(subst)),
            ),
            Type::Signal(t) => Type::Signal(Box::new(t.apply_subst(subst))),
            Type::Event(t) => Type::Event(Box::new(t.apply_subst(subst))),
            _ => self.clone(),
        }
    }
}

pub type TypeEnv = HashMap<String, Type>;

#[derive(Debug, Clone)]
pub struct Substitution {
    map: HashMap<usize, Type>,
}

impl Substitution {
    pub fn new() -> Self {
        Substitution { map: HashMap::new() }
    }
    pub fn get(&self, var_id: usize) -> Option<Type> {
        self.map.get(&var_id).cloned()
    }
    pub fn insert(&mut self, var_id: usize, ty: Type) {
        self.map.insert(var_id, ty);
    }
    pub fn compose(&mut self, other: &Substitution) {
        for (k, v) in &self.map {
            self.map.insert(*k, v.apply_subst(other));
        }
        for (k, v) in &other.map {
            self.map.entry(*k).or_insert_with(|| v.clone());
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeError {
    pub msg: String,
    pub pos: Option<Position>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    pub line: usize,
    pub col: usize,
}

impl TypeError {
    pub fn new(msg: String) -> Self {
        TypeError { msg, pos: None }
    }
    pub fn with_pos(msg: String, pos: Position) -> Self {
        TypeError { msg, pos: Some(pos) }
    }
    pub fn format(&self) -> String {
        if let Some(ref pos) = self.pos {
            format!("Type error at line {}, col {}: {}", pos.line, pos.col, self.msg)
        } else {
            format!("Type error: {}", self.msg)
        }
    }
}

// ==========================
// トレイトシステムのAST定義
// ==========================

/// メソッドシグネチャ
#[derive(Debug, Clone, PartialEq)]
pub struct MethodSig {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub ret_ty: Type,
}

/// 関連型定義
#[derive(Debug, Clone, PartialEq)]
pub struct AssociatedType {
    pub name: String,
    pub bounds: Vec<String>, // トレイト境界
}

/// トレイト定義
#[derive(Debug, Clone, PartialEq)]
pub struct TraitDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub methods: Vec<MethodSig>,
    pub associated_types: Vec<AssociatedType>,
    pub default_methods: Vec<FnDef>,
}

/// トレイト境界（where句）
#[derive(Debug, Clone, PartialEq)]
pub struct TraitBound {
    pub type_param: String,
    pub traits: Vec<String>,
}

/// impl ブロック
#[derive(Debug, Clone, PartialEq)]
pub struct ImplBlock {
    pub trait_name: Option<String>,
    pub type_name: String,
    pub type_params: Vec<String>,
    pub where_clause: Vec<TraitBound>,
    pub methods: Vec<FnDef>,
    pub associated_type_impls: HashMap<String, Type>,
}

/// 関数定義（トレイトメソッド実装用）
#[derive(Debug, Clone, PartialEq)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub ret_ty: Type,
    pub body: Vec<Stmt>,
}

// ==========================
// 標準トレイトの定義
// ==========================

pub fn create_standard_traits() -> HashMap<String, TraitDef> {
    let mut traits = HashMap::new();
    
    // Clone トレイト
    traits.insert("Clone".to_string(), TraitDef {
        name: "Clone".to_string(),
        type_params: vec![],
        methods: vec![
            MethodSig {
                name: "clone".to_string(),
                params: vec![("self".to_string(), Type::Named("Self".to_string()))],
                ret_ty: Type::Named("Self".to_string()),
            }
        ],
        associated_types: vec![],
        default_methods: vec![],
    });
    
    // Eq トレイト
    traits.insert("Eq".to_string(), TraitDef {
        name: "Eq".to_string(),
        type_params: vec![],
        methods: vec![
            MethodSig {
                name: "eq".to_string(),
                params: vec![
                    ("self".to_string(), Type::Named("Self".to_string())),
                    ("other".to_string(), Type::Named("Self".to_string())),
                ],
                ret_ty: Type::Bool,
            },
            MethodSig {
                name: "ne".to_string(),
                params: vec![
                    ("self".to_string(), Type::Named("Self".to_string())),
                    ("other".to_string(), Type::Named("Self".to_string())),
                ],
                ret_ty: Type::Bool,
            }
        ],
        associated_types: vec![],
        default_methods: vec![],
    });
    
    // Ord トレイト
    traits.insert("Ord".to_string(), TraitDef {
        name: "Ord".to_string(),
        type_params: vec![],
        methods: vec![
            MethodSig {
                name: "cmp".to_string(),
                params: vec![
                    ("self".to_string(), Type::Named("Self".to_string())),
                    ("other".to_string(), Type::Named("Self".to_string())),
                ],
                ret_ty: Type::Named("Ordering".to_string()),
            },
            MethodSig {
                name: "lt".to_string(),
                params: vec![
                    ("self".to_string(), Type::Named("Self".to_string())),
                    ("other".to_string(), Type::Named("Self".to_string())),
                ],
                ret_ty: Type::Bool,
            },
            MethodSig {
                name: "le".to_string(),
                params: vec![
                    ("self".to_string(), Type::Named("Self".to_string())),
                    ("other".to_string(), Type::Named("Self".to_string())),
                ],
                ret_ty: Type::Bool,
            },
            MethodSig {
                name: "gt".to_string(),
                params: vec![
                    ("self".to_string(), Type::Named("Self".to_string())),
                    ("other".to_string(), Type::Named("Self".to_string())),
                ],
                ret_ty: Type::Bool,
            },
            MethodSig {
                name: "ge".to_string(),
                params: vec![
                    ("self".to_string(), Type::Named("Self".to_string())),
                    ("other".to_string(), Type::Named("Self".to_string())),
                ],
                ret_ty: Type::Bool,
            }
        ],
        associated_types: vec![],
        default_methods: vec![],
    });
    
    // Debug トレイト
    traits.insert("Debug".to_string(), TraitDef {
        name: "Debug".to_string(),
        type_params: vec![],
        methods: vec![
            MethodSig {
                name: "debug".to_string(),
                params: vec![("self".to_string(), Type::Named("Self".to_string()))],
                ret_ty: Type::String,
            }
        ],
        associated_types: vec![],
        default_methods: vec![],
    });
    
    // Display トレイト
    traits.insert("Display".to_string(), TraitDef {
        name: "Display".to_string(),
        type_params: vec![],
        methods: vec![
            MethodSig {
                name: "display".to_string(),
                params: vec![("self".to_string(), Type::Named("Self".to_string()))],
                ret_ty: Type::String,
            }
        ],
        associated_types: vec![],
        default_methods: vec![],
    });
    
    traits
}

// ==========================
// トレイトチェッカー
// ==========================

pub struct TraitChecker {
    pub traits: HashMap<String, TraitDef>,
    pub impls: Vec<ImplBlock>,
}

impl TraitChecker {
    pub fn new() -> Self {
        TraitChecker {
            traits: create_standard_traits(),
            impls: vec![],
        }
    }
    
    pub fn register_trait(&mut self, trait_def: TraitDef) {
        self.traits.insert(trait_def.name.clone(), trait_def);
    }
    
    pub fn register_impl(&mut self, impl_block: ImplBlock) {
        self.impls.push(impl_block);
    }
    
    /// impl ブロックがトレイトの全メソッドを実装しているか検証
    pub fn verify_impl(&self, impl_block: &ImplBlock) -> Result<(), TypeError> {
        if let Some(ref trait_name) = impl_block.trait_name {
            let trait_def = self.traits.get(trait_name)
                .ok_or_else(|| TypeError::new(format!("Undefined trait: {}", trait_name)))?;
            
            // 必須メソッドのチェック
            for method_sig in &trait_def.methods {
                let implemented = impl_block.methods.iter()
                    .any(|fn_def| fn_def.name == method_sig.name);
                
                if !implemented {
                    return Err(TypeError::new(format!(
                        "Missing method '{}' in impl {} for {}",
                        method_sig.name, trait_name, impl_block.type_name
                    )));
                }
            }
            
            // メソッドシグネチャの一致チェック
            for fn_def in &impl_block.methods {
                if let Some(method_sig) = trait_def.methods.iter()
                    .find(|m| m.name == fn_def.name) {
                    
                    // 引数の数チェック
                    if fn_def.params.len() != method_sig.params.len() {
                        return Err(TypeError::new(format!(
                            "Method '{}' has wrong number of parameters: expected {}, found {}",
                            fn_def.name, method_sig.params.len(), fn_def.params.len()
                        )));
                    }
                    
                    // 戻り値型チェック（Selfの置き換え考慮）
                    let expected_ret = self.substitute_self(&method_sig.ret_ty, &impl_block.type_name);
                    let actual_ret = &fn_def.ret_ty;
                    if !self.types_compatible(&expected_ret, actual_ret) {
                        return Err(TypeError::new(format!(
                            "Method '{}' has wrong return type: expected {}, found {}",
                            fn_def.name, expected_ret.display(), actual_ret.display()
                        )));
                    }
                }
            }
            
            // 関連型の実装チェック
            for assoc_ty in &trait_def.associated_types {
                if !impl_block.associated_type_impls.contains_key(&assoc_ty.name) {
                    return Err(TypeError::new(format!(
                        "Missing associated type '{}' in impl {} for {}",
                        assoc_ty.name, trait_name, impl_block.type_name
                    )));
                }
            }
        }
        
        Ok(())
    }
    
    /// where句の境界チェック
    pub fn verify_where_clause(&self, where_clause: &[TraitBound]) -> Result<(), TypeError> {
        for bound in where_clause {
            for trait_name in &bound.traits {
                if !self.traits.contains_key(trait_name) {
                    return Err(TypeError::new(format!(
                        "Undefined trait in where clause: {}",
                        trait_name
                    )));
                }
            }
        }
        Ok(())
    }
    
    /// 型がトレイトを実装しているかチェック
    pub fn type_implements_trait(&self, ty: &Type, trait_name: &str) -> bool {
        let type_name = match ty {
            Type::Named(name) => name.clone(),
            Type::Int => "Int".to_string(),
            Type::Float => "Float".to_string(),
            Type::Bool => "Bool".to_string(),
            Type::String => "String".to_string(),
            _ => return false,
        };
        
        self.impls.iter().any(|impl_block| {
            impl_block.type_name == type_name 
                && impl_block.trait_name.as_ref() == Some(&trait_name.to_string())
        })
    }
    
    fn substitute_self(&self, ty: &Type, type_name: &str) -> Type {
        match ty {
            Type::Named(name) if name == "Self" => Type::Named(type_name.to_string()),
            _ => ty.clone(),
        }
    }
    
    fn types_compatible(&self, t1: &Type, t2: &Type) -> bool {
        t1 == t2 || matches!((t1, t2), (Type::Named(_), Type::Named(_)))
    }
}

// ==========================
// AST定義（既存に追加）
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
    Array(Vec<Expr>),
    Lambda { params: Vec<String>, body: Box<Expr> },
    // 【新規】メソッド呼び出し
    MethodCall { receiver: Box<Expr>, method: String, args: Vec<Expr> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Expr(Expr),
    Let { name: String, ty: Option<Type>, expr: Expr },
    FnDef { name: String, params: Vec<(String, Type)>, ret_ty: Type, body: Vec<Stmt> },
    Return(Expr),
    // 【新規】トレイト定義
    TraitDef(TraitDef),
    // 【新規】impl ブロック
    ImplBlock(ImplBlock),
}

// ==========================
// トークン定義（拡張）
// ==========================

#[derive(Debug, PartialEq, Clone)]
pub enum TokenKind {
    Ident(String), Int(i64), Float(f64), Bool(bool), String(String),
    LParen, RParen, LBrace, RBrace, Comma, Semicolon, Colon, Arrow, Dot,
    Plus, Minus, Star, Slash, Percent, Bang, Eq, EqEq, Neq, Lt, Gt, Leq, Geq,
    Assign, If, Else, While, For, Match, Fn, Let, Return,
    // 【新規】トレイト関連キーワード
    Trait, Impl, Where, SelfType, SelfValue, Dyn, Pub,
    Pipe, FatArrow, Underscore, Comment(String),
    EOF,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub pos: Position,
}

// ==========================
// Lexer（トレイト対応）
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
            "fn" => TokenKind::Fn, "let" => TokenKind::Let, "if" => TokenKind::If,
            "else" => TokenKind::Else, "while" => TokenKind::While, "for" => TokenKind::For,
            "match" => TokenKind::Match, "return" => TokenKind::Return,
            "trait" => TokenKind::Trait, "impl" => TokenKind::Impl, "where" => TokenKind::Where,
            "Self" => TokenKind::SelfType, "self" => TokenKind::SelfValue, "dyn" => TokenKind::Dyn,
            "pub" => TokenKind::Pub,
            "true" => TokenKind::Bool(true), "false" => TokenKind::Bool(false),
            "_" => TokenKind::Underscore,
            _ => TokenKind::Ident(s),
        };
        self.make_token(kind, start_col)
    }
    fn number(&mut self, start_col: usize, _first: char) -> Token {
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
        self.next();
        while let Some(c) = self.peek() {
            if c == '"' { break; }
            s.push(self.next().unwrap());
        }
        self.next();
        self.make_token(TokenKind::String(s), start_col)
    }
    fn comment(&mut self, start_col: usize) -> Token {
        self.next();
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
            '.' => self.make_token(TokenKind::Dot, start_col),
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
// Parser（トレイト対応）
// ==========================

pub struct Parser {
    lexer: Lexer,
    pub cur: Token,
    pub errors: Vec<TypeError>,
}

impl Parser {
    pub fn new(mut lexer: Lexer) -> Self {
        let cur = lexer.next_token();
        Parser { lexer, cur, errors: vec![] }
    }
    fn advance(&mut self) { self.cur = self.lexer.next_token(); }
    fn err(&mut self, msg: &str) {
        self.errors.push(TypeError { msg: msg.into(), pos: Some(self.cur.pos.clone()) });
    }
    fn expect(&mut self, kind: &TokenKind) -> bool {
        if &self.cur.kind == kind { self.advance(); true }
        else { self.err(&format!("expected {:?}, found {:?}", kind, self.cur.kind)); false }
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
    
    fn parse_stmt(&mut self) -> Result<Stmt, ()> {
        match &self.cur.kind {
            TokenKind::Let => self.parse_let(),
            TokenKind::Fn => self.parse_fn(),
            TokenKind::Return => { self.advance(); let e = self.parse_expr()?; Ok(Stmt::Return(e)) },
            TokenKind::Trait => self.parse_trait(),
            TokenKind::Impl => self.parse_impl(),
            _ => Ok(Stmt::Expr(self.parse_expr()?)),
        }
    }
    
    /// トレイト定義のパース
    pub fn parse_trait(&mut self) -> Result<Stmt, ()> {
        self.advance(); // trait
        let name = if let TokenKind::Ident(id) = &self.cur.kind { id.clone() }
        else { self.err("expected trait name"); return Err(()); };
        self.advance();
        
        // ジェネリックパラメータ（簡易版）
        let type_params = vec![];
        
        self.expect(&TokenKind::LBrace);
        
        let mut methods = vec![];
        let mut associated_types = vec![];
        let mut default_methods = vec![];
        
        while self.cur.kind != TokenKind::RBrace && self.cur.kind != TokenKind::EOF {
            if self.cur.kind == TokenKind::Fn {
                self.advance();
                let method_name = if let TokenKind::Ident(id) = &self.cur.kind { id.clone() }
                else { self.err("method name"); return Err(()); };
                self.advance();
                
                self.expect(&TokenKind::LParen);
                let mut params = vec![];
                while self.cur.kind != TokenKind::RParen && self.cur.kind != TokenKind::EOF {
                    if let TokenKind::Ident(id) = &self.cur.kind {
                        let param_name = id.clone();
                        self.advance();
                        self.expect(&TokenKind::Colon);
                        let param_ty = self.parse_type();
                        params.push((param_name, param_ty));
                        if self.cur.kind == TokenKind::Comma { self.advance(); }
                    } else { break; }
                }
                self.expect(&TokenKind::RParen);
                
                let ret_ty = if self.cur.kind == TokenKind::Arrow {
                    self.advance();
                    self.parse_type()
                } else { Type::Unit };
                
                if self.cur.kind == TokenKind::Semicolon {
                    self.advance();
                    methods.push(MethodSig { name: method_name, params, ret_ty });
                } else if self.cur.kind == TokenKind::LBrace {
                    // デフォルト実装
                    self.advance();
                    let mut body = vec![];
                    while self.cur.kind != TokenKind::RBrace && self.cur.kind != TokenKind::EOF {
                        body.push(self.parse_stmt()?);
                    }
                    self.expect(&TokenKind::RBrace);
                    default_methods.push(FnDef { name: method_name, params, ret_ty, body });
                }
            } else {
                self.advance();
            }
        }
        
        self.expect(&TokenKind::RBrace);
        
        Ok(Stmt::TraitDef(TraitDef {
            name, type_params, methods, associated_types, default_methods
        }))
    }
    
    /// impl ブロックのパース
    pub fn parse_impl(&mut self) -> Result<Stmt, ()> {
        self.advance(); // impl
        
        // ジェネリックパラメータ（簡易版）
        let type_params = vec![];
        
        let trait_name = if let TokenKind::Ident(id) = &self.cur.kind {
            let name = id.clone();
            self.advance();
            if self.cur.kind == TokenKind::For {
                self.advance();
                Some(name)
            } else {
                None
            }
        } else { None };
        
        let type_name = if let TokenKind::Ident(id) = &self.cur.kind { id.clone() }
        else { self.err("expected type name"); return Err(()); };
        self.advance();
        
        // where句のパース
        let mut where_clause = vec![];
        if self.cur.kind == TokenKind::Where {
            self.advance();
            while self.cur.kind != TokenKind::LBrace && self.cur.kind != TokenKind::EOF {
                if let TokenKind::Ident(type_param) = &self.cur.kind {
                    let param = type_param.clone();
                    self.advance();
                    self.expect(&TokenKind::Colon);
                    let mut traits = vec![];
                    while let TokenKind::Ident(trait_name) = &self.cur.kind {
                        traits.push(trait_name.clone());
                        self.advance();
                        if self.cur.kind == TokenKind::Plus {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    where_clause.push(TraitBound { type_param: param, traits });
                    if self.cur.kind == TokenKind::Comma { self.advance(); }
                } else { break; }
            }
        }
        
        self.expect(&TokenKind::LBrace);
        
        let mut methods = vec![];
        while self.cur.kind != TokenKind::RBrace && self.cur.kind != TokenKind::EOF {
            if self.cur.kind == TokenKind::Fn {
                self.advance();
                let method_name = if let TokenKind::Ident(id) = &self.cur.kind { id.clone() }
                else { self.err("method name"); return Err(()); };
                self.advance();
                
                self.expect(&TokenKind::LParen);
                let mut params = vec![];
                while self.cur.kind != TokenKind::RParen && self.cur.kind != TokenKind::EOF {
                    if let TokenKind::Ident(id) = &self.cur.kind {
                        let param_name = id.clone();
                        self.advance();
                        self.expect(&TokenKind::Colon);
                        let param_ty = self.parse_type();
                        params.push((param_name, param_ty));
                        if self.cur.kind == TokenKind::Comma { self.advance(); }
                    } else if self.cur.kind == TokenKind::SelfValue {
                        params.push(("self".to_string(), Type::Named("Self".to_string())));
                        self.advance();
                        if self.cur.kind == TokenKind::Comma { self.advance(); }
                    } else { break; }
                }
                self.expect(&TokenKind::RParen);
                
                let ret_ty = if self.cur.kind == TokenKind::Arrow {
                    self.advance();
                    self.parse_type()
                } else { Type::Unit };
                
                self.expect(&TokenKind::LBrace);
                let mut body = vec![];
                while self.cur.kind != TokenKind::RBrace && self.cur.kind != TokenKind::EOF {
                    body.push(self.parse_stmt()?);
                }
                self.expect(&TokenKind::RBrace);
                
                methods.push(FnDef { name: method_name, params, ret_ty, body });
            } else {
                self.advance();
            }
        }
        
        self.expect(&TokenKind::RBrace);
        
        Ok(Stmt::ImplBlock(ImplBlock {
            trait_name, type_name, type_params, where_clause, methods,
            associated_type_impls: HashMap::new(),
        }))
    }
    
    fn parse_type(&mut self) -> ParseResult<Type> {
        match &self.current.kind {
            TokenKind::Identifier(name) => {
                let name_str = name.clone();
                self.advance()?;

                // 組み込み型の判定
                match name_str.as_str() {
                    "Int" => Ok(Type::Int),
                    "Float" => Ok(Type::Float),
                    "Bool" => Ok(Type::Bool),
                    "String" => Ok(Type::String),
                    "Unit" => Ok(Type::Unit),
                    "Gaussian" => Ok(Type::Gaussian),
                    "Uniform" => Ok(Type::Uniform),
                    
                    // ジェネリック型: Option<T>
                    "Option" => {
                        self.expect(TokenKind::Lt)?;
                        let inner = self.parse_type()?;
                        self.expect_gt_or_shr()?;
                        Ok(Type::Option(Box::new(inner)))
                    }
                    
                    // Result<T, E>
                    "Result" => {
                        self.expect(TokenKind::Lt)?;
                        let ok_type = self.parse_type()?;
                        self.expect(TokenKind::Comma)?;
                        let err_type = self.parse_type()?;
                        self.expect_gt_or_shr()?;
                        Ok(Type::Result {
                            ok_type: Box::new(ok_type),
                            err_type: Box::new(err_type),
                        })
                    }
                    
                    // ★ Rc<T>
                    "Rc" => {
                        self.expect(TokenKind::Lt)?;
                        let inner = self.parse_type()?;
                        self.expect_gt_or_shr()?;
                        Ok(Type::Rc(Box::new(inner)))
                    }
                    
                    // ★ Weak<T>
                    "Weak" => {
                        self.expect(TokenKind::Lt)?;
                        let inner = self.parse_type()?;
                        self.expect_gt_or_shr()?;
                        Ok(Type::Weak(Box::new(inner)))
                    }
                    
                    // Signal<T>
                    "Signal" => {
                        self.expect(TokenKind::Lt)?;
                        let inner = self.parse_type()?;
                        self.expect_gt_or_shr()?;
                        Ok(Type::Signal(Box::new(inner)))
                    }
                    
                    // Event<T>
                    "Event" => {
                        self.expect(TokenKind::Lt)?;
                        let inner = self.parse_type()?;
                        self.expect_gt_or_shr()?;
                        Ok(Type::Event(Box::new(inner)))
                    }
                    
                    // ユーザー定義型（構造体など）
                    _ => {
                        // 型引数があるかチェック
                        let type_args = if self.current.kind == TokenKind::Lt {
                            self.advance()?; // consume '<'
                            let mut args = Vec::new();
                            loop {
                                args.push(self.parse_type()?);
                                if self.current.kind == TokenKind::Comma {
                                    self.advance()?;
                                } else {
                                    break;
                                }
                            }
                            self.expect_gt_or_shr()?;
                            args
                        } else {
                            Vec::new()
                        };
                        
                        Ok(Type::Named {
                            name: Identifier::new(name_str, Span::initial()),
                            type_args,
                        })
                    }
                }
            }
            
            // Self 型
            TokenKind::SelfType => {
                self.advance()?;
                Ok(Type::Named {
                    name: Identifier::new("Self".to_string(), Span::initial()),
                    type_args: vec![],
                })
            }
            
            // 関数型: fn(T1, T2) -> T3
            TokenKind::Fn => {
                self.advance()?;
                self.expect(TokenKind::LParen)?;
                let params = self.parse_type_list()?;
                self.expect(TokenKind::RParen)?;
                self.expect(TokenKind::Arrow)?;
                let return_type = Box::new(self.parse_type()?);
                Ok(Type::Function { params, return_type })
            }
            
            // タプル型: (T1, T2, ...)
            TokenKind::LParen => {
                self.advance()?;
                if self.current.kind == TokenKind::RParen {
                    self.advance()?;
                    return Ok(Type::Unit); // 空のタプル = Unit
                }
                let types = self.parse_type_list()?;
                self.expect(TokenKind::RParen)?;
                if types.len() == 1 {
                    Ok(types.into_iter().next().unwrap()) // 単一要素は括弧付き型
                } else {
                    Ok(Type::Tuple(types))
                }
            }
            
            // 配列型: [T]
            TokenKind::LBracket => {
                self.advance()?;
                let inner = self.parse_type()?;
                self.expect(TokenKind::RBracket)?;
                Ok(Type::Array(Box::new(inner)))
            }
            
            _ => {
                Err(ParseError::ExpectedType {
                    found: self.current.kind.clone(),
                    line: self.current.line,
                    column: self.current.column,
                })
            }
        }
    }

    /// `>` または `>>` を適切に処理する（ネストした型引数用）
    fn expect_gt_or_shr(&mut self) -> ParseResult<()> {
        match self.current.kind {
            TokenKind::Gt => {
                self.advance()?;
                Ok(())
            }
            TokenKind::Shr => {
                // >> を > として1つ消費し、次のトークンは > として残す
                self.current.kind = TokenKind::Gt;
                Ok(())
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: TokenKind::Gt,
                found: self.current.kind.clone(),
                line: self.current.line,
                column: self.current.column,
            }),
        }
    }

    /// カンマ区切りの型リストをパース
    fn parse_type_list(&mut self) -> ParseResult<Vec<Type>> {
        let mut types = Vec::new();
        if self.current.kind == TokenKind::RParen {
            return Ok(types);
        }
        loop {
            types.push(self.parse_type()?);
            if self.current.kind == TokenKind::Comma {
                self.advance()?;
            } else {
                break;
            }
        }
        Ok(types)
    }

    
    fn parse_let(&mut self) -> Result<Stmt, ()> {
        self.advance();
        let name = if let TokenKind::Ident(id) = &self.cur.kind { id.clone() }
        else { self.err("expected identifier"); return Err(()); };
        self.advance();
        let mut ty = None;
        if self.cur.kind == TokenKind::Colon {
            self.advance();
            ty = Some(self.parse_type());
        }
        self.expect(&TokenKind::Assign);
        let expr = self.parse_expr()?;
        if self.cur.kind == TokenKind::Semicolon { self.advance(); }
        Ok(Stmt::Let { name, ty, expr })
    }
    
    fn parse_fn(&mut self) -> Result<Stmt, ()> {
        self.advance();
        let name = if let TokenKind::Ident(id) = &self.cur.kind { id.clone() }
        else { self.err("expected identifier"); return Err(()); };
        self.advance();
        self.expect(&TokenKind::LParen);
        let mut params = vec![];
        while self.cur.kind != TokenKind::RParen && self.cur.kind != TokenKind::EOF {
            if let TokenKind::Ident(id) = &self.cur.kind {
                self.advance();
                self.expect(&TokenKind::Colon);
                let param_ty = self.parse_type();
                params.push((id.clone(), param_ty));
                if self.cur.kind == TokenKind::Comma { self.advance(); }
            } else { self.err("param name expected"); return Err(()); }
        }
        self.expect(&TokenKind::RParen);
        let ret_ty = if self.cur.kind == TokenKind::Arrow {
            self.advance();
            self.parse_type()
        } else { Type::Unit };
        self.expect(&TokenKind::LBrace);
        let mut body = vec![];
        while self.cur.kind != TokenKind::RBrace && self.cur.kind != TokenKind::EOF {
            body.push(self.parse_stmt()?);
        }
        self.expect(&TokenKind::RBrace);
        Ok(Stmt::FnDef { name, params, ret_ty, body })
    }
    
    fn parse_expr(&mut self) -> Result<Expr, ()> {
        self.parse_expr_bp(0)
    }
    
    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, ()> {
        let mut lhs = match &self.cur.kind {
            TokenKind::Int(i) => { let v = *i; self.advance(); Expr::Int(v) }
            TokenKind::Float(f) => { let v = *f; self.advance(); Expr::Float(v) }
            TokenKind::Bool(b) => { let v = *b; self.advance(); Expr::Bool(v) }
            TokenKind::String(s) => { let v = s.clone(); self.advance(); Expr::String(v) }
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
                } else {
                    Expr::Ident(v)
                }
            }
            TokenKind::LParen => { self.advance(); let e = self.parse_expr()?; self.expect(&TokenKind::RParen); Expr::Paren(Box::new(e)) }
            _ => { self.err("unexpected token"); return Err(()); }
        };
        
        loop {
            // メソッド呼び出し
            if self.cur.kind == TokenKind::Dot {
                self.advance();
                if let TokenKind::Ident(method) = &self.cur.kind {
                    let method_name = method.clone();
                    self.advance();
                    if self.cur.kind == TokenKind::LParen {
                        self.advance();
                        let mut args = vec![];
                        while self.cur.kind != TokenKind::RParen && self.cur.kind != TokenKind::EOF {
                            args.push(self.parse_expr()?);
                            if self.cur.kind == TokenKind::Comma { self.advance(); }
                        }
                        self.expect(&TokenKind::RParen);
                        lhs = Expr::MethodCall { receiver: Box::new(lhs), method: method_name, args };
                        continue;
                    }
                }
            }
            
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
}

// ==========================
// テスト
// ==========================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_trait_definition() {
        let src = r#"
            trait Drawable {
                fn draw(self) -> Unit;
                fn area(self) -> Float;
            }
        "#;
        let mut parser = Parser::new(Lexer::new(src.to_string()));
        let ast = parser.parse_program();
        
        assert!(!ast.is_empty());
        if let Stmt::TraitDef(trait_def) = &ast[0] {
            assert_eq!(trait_def.name, "Drawable");
            assert_eq!(trait_def.methods.len(), 2);
            assert_eq!(trait_def.methods[0].name, "draw");
            assert_eq!(trait_def.methods[1].name, "area");
        } else {
            panic!("Expected trait definition");
        }
    }
    
    #[test]
    fn test_parse_impl_block() {
        let src = r#"
            impl Drawable for Point {
                fn draw(self) -> Unit {
                    return Unit;
                }
                fn area(self) -> Float {
                    return 0.0;
                }
            }
        "#;
        let mut parser = Parser::new(Lexer::new(src.to_string()));
        let ast = parser.parse_program();
        
        assert!(!ast.is_empty());
        if let Stmt::ImplBlock(impl_block) = &ast[0] {
            assert_eq!(impl_block.trait_name, Some("Drawable".to_string()));
            assert_eq!(impl_block.type_name, "Point");
            assert_eq!(impl_block.methods.len(), 2);
        } else {
            panic!("Expected impl block");
        }
    }
    
    #[test]
    fn test_verify_complete_impl() {
        let mut checker = TraitChecker::new();
        
        let trait_def = TraitDef {
            name: "Clone".to_string(),
            type_params: vec![],
            methods: vec![
                MethodSig {
                    name: "clone".to_string(),
                    params: vec![("self".to_string(), Type::Named("Self".to_string()))],
                    ret_ty: Type::Named("Self".to_string()),
                }
            ],
            associated_types: vec![],
            default_methods: vec![],
        };
        checker.register_trait(trait_def);
        
        let impl_block = ImplBlock {
            trait_name: Some("Clone".to_string()),
            type_name: "Point".to_string(),
            type_params: vec![],
            where_clause: vec![],
            methods: vec![
                FnDef {
                    name: "clone".to_string(),
                    params: vec![("self".to_string(), Type::Named("Self".to_string()))],
                    ret_ty: Type::Named("Point".to_string()),
                    body: vec![],
                }
            ],
            associated_type_impls: HashMap::new(),
        };
        
        assert!(checker.verify_impl(&impl_block).is_ok());
    }
    
    #[test]
    fn test_verify_incomplete_impl() {
        let mut checker = TraitChecker::new();
        
        let impl_block = ImplBlock {
            trait_name: Some("Clone".to_string()),
            type_name: "Point".to_string(),
            type_params: vec![],
            where_clause: vec![],
            methods: vec![], // メソッドなし
            associated_type_impls: HashMap::new(),
        };
        
        let result = checker.verify_impl(&impl_block);
        assert!(result.is_err());
        assert!(result.unwrap_err().msg.contains("Missing method"));
    }
    
    #[test]
    fn test_standard_traits() {
        let traits = create_standard_traits();
        assert!(traits.contains_key("Clone"));
        assert!(traits.contains_key("Eq"));
        assert!(traits.contains_key("Ord"));
        assert!(traits.contains_key("Debug"));
    }
    
    #[test]
    fn test_where_clause_parsing() {
        let src = r#"
            impl<T> Display for Array<T> where T: Display {
                fn display(self) -> String {
                    return "array";
                }
            }
        "#;
        let mut parser = Parser::new(Lexer::new(src.to_string()));
        let ast = parser.parse_program();
        
        if let Stmt::ImplBlock(impl_block) = &ast[0] {
            assert!(!impl_block.where_clause.is_empty());
        }
    }
    
    #[test]
    fn test_method_call_parsing() {
        let src = "x.clone()";
        let mut parser = Parser::new(Lexer::new(src.to_string()));
        let expr = parser.parse_expr().unwrap();
        
        if let Expr::MethodCall { receiver, method, args } = expr {
            assert_eq!(method, "clone");
            assert_eq!(args.len(), 0);
        } else {
            panic!("Expected method call");
        }
    }
    
    #[test]
    fn test_type_implements_trait() {
        let mut checker = TraitChecker::new();
        
        let impl_block = ImplBlock {
            trait_name: Some("Clone".to_string()),
            type_name: "Point".to_string(),
            type_params: vec![],
            where_clause: vec![],
            methods: vec![
                FnDef {
                    name: "clone".to_string(),
                    params: vec![("self".to_string(), Type::Named("Self".to_string()))],
                    ret_ty: Type::Named("Point".to_string()),
                    body: vec![],
                }
            ],
            associated_type_impls: HashMap::new(),
        };
        checker.register_impl(impl_block);
        
        assert!(checker.type_implements_trait(&Type::Named("Point".to_string()), "Clone"));
        assert!(!checker.type_implements_trait(&Type::Named("Point".to_string()), "Debug"));
    }
}
