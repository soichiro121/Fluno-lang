// ==========================
// 型システムと型変数
// ==========================
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static TYPE_VAR_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // プリミティブ型
    Int,
    Float,
    Bool,
    String,
    Unit,
    // 型変数（Hindley-Milner）
    TypeVar(usize),
    // ジェネリック型
    Array(Box<Type>),
    Option(Box<Type>),
    Result(Box<Type>, Box<Type>),
    // 関数型
    Func(Vec<Type>, Box<Type>),
    // 確率型（Flux拡張）
    Gaussian,
    Uniform,
    // リアクティブ型（Flux拡張）
    Signal(Box<Type>),
    Event(Box<Type>),
    // 未知型
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
            _ => Type::Unknown,
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

// ==========================
// 型環境と置換
// ==========================
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

// ==========================
// 型エラー
// ==========================
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
// 型推論エンジン
// ==========================
pub struct TypeInferer {
    subst: Substitution,
}

impl TypeInferer {
    pub fn new() -> Self {
        TypeInferer { subst: Substitution::new() }
    }

    /// 新しい型変数を生成
    pub fn fresh_type_var() -> Type {
        let id = TYPE_VAR_COUNTER.fetch_add(1, Ordering::SeqCst);
        Type::TypeVar(id)
    }

    /// 単一化アルゴリズム（Unification）
    pub fn unify(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        let t1 = t1.apply_subst(&self.subst);
        let t2 = t2.apply_subst(&self.subst);

        match (&t1, &t2) {
            // 同一型
            (Type::Int, Type::Int) | (Type::Float, Type::Float) 
            | (Type::Bool, Type::Bool) | (Type::String, Type::String)
            | (Type::Unit, Type::Unit) | (Type::Gaussian, Type::Gaussian)
            | (Type::Uniform, Type::Uniform) => Ok(()),

            // 型変数の単一化
            (Type::TypeVar(id1), Type::TypeVar(id2)) if id1 == id2 => Ok(()),
            (Type::TypeVar(id), ty) | (ty, Type::TypeVar(id)) => {
                if ty.occurs_in(*id, &self.subst) {
                    return Err(TypeError::new(format!(
                        "Occurs check failed: {} occurs in {}",
                        Type::TypeVar(*id).display(),
                        ty.display()
                    )));
                }
                self.subst.insert(*id, ty.clone());
                Ok(())
            }

            // ジェネリック型の単一化
            (Type::Array(t1), Type::Array(t2)) => self.unify(t1, t2),
            (Type::Option(t1), Type::Option(t2)) => self.unify(t1, t2),
            (Type::Result(t1, e1), Type::Result(t2, e2)) => {
                self.unify(t1, t2)?;
                self.unify(e1, e2)
            }
            (Type::Signal(t1), Type::Signal(t2)) => self.unify(t1, t2),
            (Type::Event(t1), Type::Event(t2)) => self.unify(t1, t2),

            // 関数型の単一化
            (Type::Func(params1, ret1), Type::Func(params2, ret2)) => {
                if params1.len() != params2.len() {
                    return Err(TypeError::new(format!(
                        "Function arity mismatch: expected {} parameters, found {}",
                        params1.len(), params2.len()
                    )));
                }
                for (p1, p2) in params1.iter().zip(params2.iter()) {
                    self.unify(p1, p2)?;
                }
                self.unify(ret1, ret2)
            }

            // 型不一致
            _ => Err(TypeError::new(format!(
                "Type mismatch: expected {}, found {}",
                t1.display(), t2.display()
            ))),
        }
    }

    /// 式の型推論
    pub fn infer_expr(&mut self, expr: &Expr, env: &TypeEnv) -> Result<Type, TypeError> {
        match expr {
            Expr::Int(_) => Ok(Type::Int),
            Expr::Float(_) => Ok(Type::Float),
            Expr::Bool(_) => Ok(Type::Bool),
            Expr::String(_) => Ok(Type::String),

            Expr::Ident(name) => {
                env.get(name)
                    .cloned()
                    .ok_or_else(|| TypeError::new(format!("Undefined variable: {}", name)))
            }

            Expr::BinaryOp { left, op, right } => {
                let t1 = self.infer_expr(left, env)?;
                let t2 = self.infer_expr(right, env)?;

                match op.as_str() {
                    "+" | "-" | "*" | "/" | "%" => {
                        self.unify(&t1, &Type::Int).or_else(|_| self.unify(&t1, &Type::Float))?;
                        self.unify(&t1, &t2)?;
                        Ok(t1.apply_subst(&self.subst))
                    }
                    "==" | "!=" | "<" | ">" | "<=" | ">=" => {
                        self.unify(&t1, &t2)?;
                        Ok(Type::Bool)
                    }
                    "&&" | "||" => {
                        self.unify(&t1, &Type::Bool)?;
                        self.unify(&t2, &Type::Bool)?;
                        Ok(Type::Bool)
                    }
                    _ => Err(TypeError::new(format!("Unknown operator: {}", op))),
                }
            }

            Expr::Call { func, args } => {
                let func_ty = self.infer_expr(func, env)?;
                let arg_types: Result<Vec<_>, _> = args.iter()
                    .map(|arg| self.infer_expr(arg, env))
                    .collect();
                let arg_types = arg_types?;

                let ret_ty = Self::fresh_type_var();
                let expected_func_ty = Type::Func(arg_types, Box::new(ret_ty.clone()));

                self.unify(&func_ty, &expected_func_ty)?;
                Ok(ret_ty.apply_subst(&self.subst))
            }

            Expr::If { cond, then_br, else_br } => {
                let cond_ty = self.infer_expr(cond, env)?;
                self.unify(&cond_ty, &Type::Bool)?;

                let then_ty = self.infer_expr(then_br, env)?;
                if let Some(else_expr) = else_br {
                    let else_ty = self.infer_expr(else_expr, env)?;
                    self.unify(&then_ty, &else_ty)?;
                }
                Ok(then_ty.apply_subst(&self.subst))
            }

            // 【改善】While式の型推論
            Expr::While { cond, body } => {
                let cond_ty = self.infer_expr(cond, env)?;
                self.unify(&cond_ty, &Type::Bool)?;
                self.infer_expr(body, env)?;
                Ok(Type::Unit)
            }

            // 【改善】For式の型推論
            Expr::For { var, iter, body } => {
                let iter_ty = self.infer_expr(iter, env)?;
                let elem_ty = Self::fresh_type_var();
                self.unify(&iter_ty, &Type::Array(Box::new(elem_ty.clone())))?;
                
                let mut new_env = env.clone();
                new_env.insert(var.clone(), elem_ty);
                self.infer_expr(body, &new_env)?;
                Ok(Type::Unit)
            }

            // 【改善】Match式の型推論
            Expr::Match { expr, arms } => {
                let expr_ty = self.infer_expr(expr, env)?;
                
                if arms.is_empty() {
                    return Err(TypeError::new("Match expression must have at least one arm".to_string()));
                }
                
                let first_result = self.infer_expr(&arms[0].1, env)?;
                for (pattern, result) in &arms[1..] {
                    // パターンの型チェック（簡易版）
                    let pattern_ty = self.infer_expr(pattern, env)?;
                    self.unify(&expr_ty, &pattern_ty)?;
                    
                    let result_ty = self.infer_expr(result, env)?;
                    self.unify(&first_result, &result_ty)?;
                }
                
                Ok(first_result.apply_subst(&self.subst))
            }

            Expr::Paren(e) => self.infer_expr(e, env),

            Expr::Array(elems) => {
                if elems.is_empty() {
                    Ok(Type::Array(Box::new(Self::fresh_type_var())))
                } else {
                    let elem_ty = self.infer_expr(&elems[0], env)?;
                    for elem in &elems[1..] {
                        let ty = self.infer_expr(elem, env)?;
                        self.unify(&elem_ty, &ty)?;
                    }
                    Ok(Type::Array(Box::new(elem_ty.apply_subst(&self.subst))))
                }
            }

            Expr::Lambda { params, body } => {
                let mut new_env = env.clone();
                let param_types: Vec<Type> = params.iter()
                    .map(|_| Self::fresh_type_var())
                    .collect();

                for (param, ty) in params.iter().zip(param_types.iter()) {
                    new_env.insert(param.clone(), ty.clone());
                }

                let body_ty = self.infer_expr(body, &new_env)?;
                Ok(Type::Func(param_types, Box::new(body_ty)))
            }

            _ => Ok(Type::Unknown),
        }
    }

    /// 【改善】文の型推論と型環境の更新（複数return文対応）
    pub fn infer_stmt(&mut self, stmt: &Stmt, env: &mut TypeEnv) -> Result<(), TypeError> {
        match stmt {
            Stmt::Let { name, ty, expr } => {
                let expr_ty = self.infer_expr(expr, env)?;
                if let Some(annot_ty) = ty {
                    self.unify(&expr_ty, annot_ty)?;
                }
                let final_ty = expr_ty.apply_subst(&self.subst);
                env.insert(name.clone(), final_ty);
                Ok(())
            }

            Stmt::FnDef { name, params, ret_ty, body } => {
                // 再帰関数サポート：関数名を先に環境に追加
                let param_types: Vec<Type> = params.iter()
                    .map(|(_, ty)| ty.clone())
                    .collect();
                let func_ty = Type::Func(param_types.clone(), Box::new(ret_ty.clone()));
                env.insert(name.clone(), func_ty.clone());

                let mut fn_env = env.clone();
                for (param_name, param_ty) in params {
                    fn_env.insert(param_name.clone(), param_ty.clone());
                }

                // 【改善】すべてのreturn文をチェック
                let mut return_types = Vec::new();
                for s in body {
                    if let Stmt::Return(expr) = s {
                        return_types.push(self.infer_expr(expr, &fn_env)?);
                    } else {
                        self.infer_stmt(s, &mut fn_env)?;
                    }
                }

                let body_ty = if return_types.is_empty() {
                    Type::Unit
                } else {
                    let first = &return_types[0];
                    for rt in &return_types[1..] {
                        self.unify(first, rt)?;
                    }
                    first.apply_subst(&self.subst)
                };

                self.unify(&body_ty, ret_ty)?;
                Ok(())
            }

            Stmt::Expr(expr) => {
                self.infer_expr(expr, env)?;
                Ok(())
            }

            Stmt::Return(_) => Ok(()),
        }
    }

    pub fn get_substitution(&self) -> &Substitution {
        &self.subst
    }
}

// ==========================
// AST定義
// ==========================
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Ident(String),
    BinaryOp { left: Box<Expr>, op: String, right: Box<Expr> },
    Call { func: Box<Expr>, args: Vec<Expr> },
    Paren(Box<Expr>),
    If { cond: Box<Expr>, then_br: Box<Expr>, else_br: Option<Box<Expr>> },
    While { cond: Box<Expr>, body: Box<Expr> },
    For { var: String, iter: Box<Expr>, body: Box<Expr> },
    Match { expr: Box<Expr>, arms: Vec<(Expr, Expr)> },
    Array(Vec<Expr>),
    Lambda { params: Vec<String>, body: Box<Expr> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Expr(Expr),
    Let { name: String, ty: Option<Type>, expr: Expr },
    FnDef { name: String, params: Vec<(String, Type)>, ret_ty: Type, body: Vec<Stmt> },
    Return(Expr),
}

// ==========================
// テスト
// ==========================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_type_var() {
        let t1 = TypeInferer::fresh_type_var();
        let t2 = TypeInferer::fresh_type_var();
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_unify_primitives() {
        let mut inferer = TypeInferer::new();
        assert!(inferer.unify(&Type::Int, &Type::Int).is_ok());
        assert!(inferer.unify(&Type::Int, &Type::Float).is_err());
    }

    #[test]
    fn test_unify_type_vars() {
        let mut inferer = TypeInferer::new();
        let t1 = TypeInferer::fresh_type_var();
        let t2 = TypeInferer::fresh_type_var();
        
        assert!(inferer.unify(&t1, &Type::Int).is_ok());
        assert!(inferer.unify(&t2, &t1).is_ok());
        
        let resolved = t2.apply_subst(&inferer.subst);
        assert_eq!(resolved, Type::Int);
    }

    #[test]
    fn test_occurs_check() {
        let mut inferer = TypeInferer::new();
        let t1 = TypeInferer::fresh_type_var();
        let t2 = Type::Array(Box::new(t1.clone()));
        
        let result = inferer.unify(&t1, &t2);
        assert!(result.is_err());
        assert!(result.unwrap_err().msg.contains("Occurs check failed"));
    }

    #[test]
    fn test_infer_simple_let() {
        let mut inferer = TypeInferer::new();
        let mut env = TypeEnv::new();
        
        let stmt = Stmt::Let {
            name: "x".to_string(),
            ty: None,
            expr: Expr::Int(42),
        };
        
        assert!(inferer.infer_stmt(&stmt, &mut env).is_ok());
        assert_eq!(env.get("x"), Some(&Type::Int));
    }

    #[test]
    fn test_infer_function() {
        let mut inferer = TypeInferer::new();
        let mut env = TypeEnv::new();
        
        let stmt = Stmt::FnDef {
            name: "double".to_string(),
            params: vec![("x".to_string(), Type::Int)],
            ret_ty: Type::Int,
            body: vec![Stmt::Return(Expr::BinaryOp {
                left: Box::new(Expr::Ident("x".to_string())),
                op: "*".to_string(),
                right: Box::new(Expr::Int(2)),
            })],
        };
        
        assert!(inferer.infer_stmt(&stmt, &mut env).is_ok());
        
        if let Some(Type::Func(params, ret)) = env.get("double") {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0], Type::Int);
            assert_eq!(**ret, Type::Int);
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_generic_array() {
        let mut inferer = TypeInferer::new();
        let env = TypeEnv::new();
        
        let expr = Expr::Array(vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)]);
        let ty = inferer.infer_expr(&expr, &env).unwrap();
        
        assert_eq!(ty, Type::Array(Box::new(Type::Int)));
    }

    #[test]
    fn test_type_error_undefined_variable() {
        let mut inferer = TypeInferer::new();
        let env = TypeEnv::new();
        
        let expr = Expr::Ident("undefined".to_string());
        let result = inferer.infer_expr(&expr, &env);
        
        assert!(result.is_err());
        assert!(result.unwrap_err().msg.contains("Undefined variable"));
    }

    #[test]
    fn test_type_error_mismatch() {
        let mut inferer = TypeInferer::new();
        let mut env = TypeEnv::new();
        
        let stmt = Stmt::Let {
            name: "x".to_string(),
            ty: Some(Type::Int),
            expr: Expr::String("string".to_string()),
        };
        
        let result = inferer.infer_stmt(&stmt, &mut env);
        assert!(result.is_err());
        assert!(result.unwrap_err().msg.contains("Type mismatch"));
        let stmt = Stmt::Let {
            name: "x".to_string(),
            ty: Some(Type::Int),
            expr: Expr::String("string".to_string()),
        };
        
        let result = inferer.infer_stmt(&stmt, &mut env);
        assert!(result.is_err());
        assert!(result.unwrap_err().msg.contains("Type mismatch"));
    }

    #[test]
    fn test_infer_lambda() {
        let mut inferer = TypeInferer::new();
        let env = TypeEnv::new();
        
        let expr = Expr::Lambda {
            params: vec!["x".to_string()],
            body: Box::new(Expr::BinaryOp {
                left: Box::new(Expr::Ident("x".to_string())),
                op: "*".to_string(),
                right: Box::new(Expr::Int(2)),
            }),
        };
        
        let ty = inferer.infer_expr(&expr, &env).unwrap();
        if let Type::Func(params, _) = ty {
            assert_eq!(params.len(), 1);
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_if_expression() {
        let mut inferer = TypeInferer::new();
        let env = TypeEnv::new();
        
        let expr = Expr::If {
            cond: Box::new(Expr::Bool(true)),
            then_br: Box::new(Expr::Int(1)),
            else_br: Some(Box::new(Expr::Int(2))),
        };
        
        let ty = inferer.infer_expr(&expr, &env).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_generic_result() {
        let mut inferer = TypeInferer::new();
        let t1 = Type::Result(Box::new(Type::Int), Box::new(Type::String));
        let t2 = Type::Result(Box::new(Type::Int), Box::new(Type::String));
        
        assert!(inferer.unify(&t1, &t2).is_ok());
    }

    #[test]
    fn test_complex_nested_generics() {
        let mut inferer = TypeInferer::new();
        
        let t1 = Type::Array(Box::new(Type::Option(Box::new(Type::Int))));
        let t2 = Type::Array(Box::new(Type::Option(Box::new(Type::Int))));
        
        assert!(inferer.unify(&t1, &t2).is_ok());
    }

    #[test]
    fn test_error_formatting() {
        let err = TypeError::with_pos(
            "Type mismatch: expected Int, found String".to_string(),
            Position { line: 5, col: 10 }
        );
        
        let formatted = err.format();
        assert!(formatted.contains("line 5"));
        assert!(formatted.contains("col 10"));
        assert!(formatted.contains("Type mismatch"));
    }

    // 【新規】複数return文のテスト
    #[test]
    fn test_multiple_returns() {
        let mut inferer = TypeInferer::new();
        let mut env = TypeEnv::new();
        
        // fn test(x: Int) -> Int {
        //     if x > 0 { return 1; }
        //     return -1;
        // }
        let stmt = Stmt::FnDef {
            name: "test".to_string(),
            params: vec![("x".to_string(), Type::Int)],
            ret_ty: Type::Int,
            body: vec![
                Stmt::Return(Expr::Int(1)),
                Stmt::Return(Expr::Int(-1)),
            ],
        };
        
        assert!(inferer.infer_stmt(&stmt, &mut env).is_ok());
    }

    // 【新規】While式のテスト
    #[test]
    fn test_while_loop() {
        let mut inferer = TypeInferer::new();
        let mut env = TypeEnv::new();
        env.insert("x".to_string(), Type::Int);
        
        let expr = Expr::While {
            cond: Box::new(Expr::BinaryOp {
                left: Box::new(Expr::Ident("x".to_string())),
                op: ">".to_string(),
                right: Box::new(Expr::Int(0)),
            }),
            body: Box::new(Expr::Ident("x".to_string())),
        };
        
        let ty = inferer.infer_expr(&expr, &env).unwrap();
        assert_eq!(ty, Type::Unit);
    }

    // 【新規】For式のテスト
    #[test]
    fn test_for_loop() {
        let mut inferer = TypeInferer::new();
        let env = TypeEnv::new();
        
        let expr = Expr::For {
            var: "i".to_string(),
            iter: Box::new(Expr::Array(vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)])),
            body: Box::new(Expr::Ident("i".to_string())),
        };
        
        let ty = inferer.infer_expr(&expr, &env).unwrap();
        assert_eq!(ty, Type::Unit);
    }

    // 【新規】Match式のテスト
    #[test]
    fn test_match_expression() {
        let mut inferer = TypeInferer::new();
        let env = TypeEnv::new();
        
        let expr = Expr::Match {
            expr: Box::new(Expr::Int(1)),
            arms: vec![
                (Expr::Int(1), Expr::String("one".to_string())),
                (Expr::Int(2), Expr::String("two".to_string())),
            ],
        };
        
        let ty = inferer.infer_expr(&expr, &env).unwrap();
        assert_eq!(ty, Type::String);
    }

    // 【新規】ネスト関数のテスト
    #[test]
    fn test_infer_nested_functions() {
        let mut inferer = TypeInferer::new();
        let mut env = TypeEnv::new();
        
        let inner_lambda = Expr::Lambda {
            params: vec!["x".to_string()],
            body: Box::new(Expr::BinaryOp {
                left: Box::new(Expr::Ident("x".to_string())),
                op: "+".to_string(),
                right: Box::new(Expr::Int(1)),
            }),
        };
        
        let stmt = Stmt::FnDef {
            name: "outer".to_string(),
            params: vec![],
            ret_ty: Type::Func(vec![Type::Int], Box::new(Type::Int)),
            body: vec![Stmt::Return(inner_lambda)],
        };
        
        assert!(inferer.infer_stmt(&stmt, &mut env).is_ok());
        
        if let Some(Type::Func(params, ret)) = env.get("outer") {
            assert_eq!(params.len(), 0);
            if let Type::Func(inner_params, inner_ret) = &**ret {
                assert_eq!(inner_params.len(), 1);
                assert_eq!(**inner_ret, Type::Int);
            } else {
                panic!("Expected nested function type");
            }
        } else {
            panic!("Expected function type");
        }
    }

    // 【新規】再帰関数のテスト
    #[test]
    fn test_infer_recursive_function() {
        let mut inferer = TypeInferer::new();
        let mut env = TypeEnv::new();
        
        // fn factorial(n: Int) -> Int {
        //     if n <= 1 { return 1; }
        //     return n * factorial(n - 1);
        // }
        let stmt = Stmt::FnDef {
            name: "factorial".to_string(),
            params: vec![("n".to_string(), Type::Int)],
            ret_ty: Type::Int,
            body: vec![
                Stmt::Return(Expr::Int(1)),
                Stmt::Return(Expr::BinaryOp {
                    left: Box::new(Expr::Ident("n".to_string())),
                    op: "*".to_string(),
                    right: Box::new(Expr::Call {
                        func: Box::new(Expr::Ident("factorial".to_string())),
                        args: vec![Expr::BinaryOp {
                            left: Box::new(Expr::Ident("n".to_string())),
                            op: "-".to_string(),
                            right: Box::new(Expr::Int(1)),
                        }],
                    }),
                }),
            ],
        };
        
        assert!(inferer.infer_stmt(&stmt, &mut env).is_ok());
    }

    // 【新規】部分適用のテスト
    #[test]
    fn test_partial_application() {
        let mut inferer = TypeInferer::new();
        let mut env = TypeEnv::new();
        
        // fn add(x: Int, y: Int) -> Int { return x + y; }
        let add_fn = Stmt::FnDef {
            name: "add".to_string(),
            params: vec![
                ("x".to_string(), Type::Int),
                ("y".to_string(), Type::Int),
            ],
            ret_ty: Type::Int,
            body: vec![Stmt::Return(Expr::BinaryOp {
                left: Box::new(Expr::Ident("x".to_string())),
                op: "+".to_string(),
                right: Box::new(Expr::Ident("y".to_string())),
            })],
        };
        
        assert!(inferer.infer_stmt(&add_fn, &mut env).is_ok());
        
        // カリー化されたadd関数の型
        // Fn(Int, Int) -> Int
        if let Some(Type::Func(params, ret)) = env.get("add") {
            assert_eq!(params.len(), 2);
            assert_eq!(params[0], Type::Int);
            assert_eq!(params[1], Type::Int);
            assert_eq!(**ret, Type::Int);
        } else {
            panic!("Expected function type");
        }
    }

    // 【新規】型不一致のreturn文検出
    #[test]
    fn test_inconsistent_return_types() {
        let mut inferer = TypeInferer::new();
        let mut env = TypeEnv::new();
        
        // fn bad() -> Int {
        //     return 1;
        //     return "string";
        // }
        let stmt = Stmt::FnDef {
            name: "bad".to_string(),
            params: vec![],
            ret_ty: Type::Int,
            body: vec![
                Stmt::Return(Expr::Int(1)),
                Stmt::Return(Expr::String("string".to_string())),
            ],
        };
        
        let result = inferer.infer_stmt(&stmt, &mut env);
        assert!(result.is_err());
    }

    // 【新規】Match式の空チェック
    #[test]
    fn test_empty_match_error() {
        let mut inferer = TypeInferer::new();
        let env = TypeEnv::new();
        
        let expr = Expr::Match {
            expr: Box::new(Expr::Int(1)),
            arms: vec![],
        };
        
        let result = inferer.infer_expr(&expr, &env);
        assert!(result.is_err());
        assert!(result.unwrap_err().msg.contains("at least one arm"));
    }
}
