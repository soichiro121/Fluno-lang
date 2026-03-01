// src/ast/node.rs

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModId(pub DefId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ItemId(pub DefId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

impl Span {
    pub fn new(line: usize, column: usize, length: usize) -> Self {
        Span {
            line,
            column,
            length,
        }
    }

    pub fn initial() -> Self {
        Span::new(1, 1, 0)
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Function(FunctionDef),
    Struct(StructDef),
    TypeAlias(TypeAlias),
    Enum(EnumDef),
    Trait(TraitDef),
    Impl(ImplBlock),
    Module(ModuleDef),
    Import(ImportStmt),
    Extern(ExternBlock),
}

#[derive(Debug, Clone)]
pub enum Attribute {
    Derive(Vec<String>),
    Simple(Identifier),
    Nested(Identifier, Vec<Attribute>),
    Value(Identifier, Literal),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeParam {
    pub name: String,
    pub bounds: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: Identifier,
    pub type_params: Vec<TypeParameter>,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub defid: Option<DefId>,
    pub is_async: bool,
    pub span: Span,
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: Identifier,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: Identifier,
    pub fields: Vec<StructField>,
    pub type_params: Vec<TypeParameter>,
    pub where_clause: Option<WhereClause>,
    pub span: Span,
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone)]
pub struct StructField {
    pub name: Identifier,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct EnumDef {
    pub name: Identifier,
    pub variants: Vec<EnumVariant>,
    pub type_params: Vec<TypeParameter>,
    pub span: Span,
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name: Identifier,
    pub data: VariantData,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum VariantData {
    Unit,
    Tuple(Vec<Type>),
    Struct(Vec<StructField>),
}

#[derive(Debug, Clone)]
pub struct TypeAlias {
    pub name: Identifier,
    pub type_params: Vec<TypeParameter>,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TraitDef {
    pub name: Identifier,
    pub type_params: Vec<TypeParameter>,
    pub assoc_types: Vec<Identifier>,
    pub methods: Vec<TraitMethod>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TraitMethod {
    pub name: Identifier,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub defid: Option<DefId>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ImplBlock {
    pub type_params: Vec<TypeParameter>,
    pub trait_ref: Option<Type>,
    pub self_ty: Type,
    pub items: Vec<ImplItem>,
    pub span: Span,
    pub where_preds: Vec<WherePredicate>,
}

#[derive(Debug, Clone)]
pub struct AssocTypeBinding {
    pub name: Identifier,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ImplItem {
    Method(FunctionDef),
    AssocType(AssocTypeBinding),
}

#[derive(Debug, Clone)]
pub struct ModuleDef {
    pub name: Identifier,
    pub items: Vec<Item>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ImportStmt {
    pub path: Path,
    pub alias: Option<Identifier>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExternBlock {
    pub abi: String,
    pub functions: Vec<ExternFn>,
    pub span: Span,
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone)]
pub struct ExternFn {
    pub name: Identifier,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub is_async: bool,
    pub span: Span,
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone)]
pub struct TypeParameter {
    pub name: Identifier,
    pub bounds: Vec<Type>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum WherePredicate {
    Bound {
        target_ty: Type,
        bound_ty: Type,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub struct WhereClause {
    pub predicates: Vec<WherePredicate>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    Bool,
    String,
    Unit,
    Any,

    Gaussian,
    Uniform,
    Bernoulli,
    Beta,
    VonMises,

    Signal(Box<Type>),
    Event(Box<Type>),

    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },

    Tuple(Vec<Type>),

    Array(Box<Type>),

    Option(Box<Type>),

    Result {
        ok_type: Box<Type>,
        err_type: Box<Type>,
    },

    Handle(Box<Type>),

    Named {
        name: Path,
        type_args: Vec<Type>,
    },

    Assoc {
        trait_def: Option<DefId>,
        self_ty: Box<Type>,
        name: String,
    },

    DynTrait {
        trait_path: Path,
    },

    TypeVar(Identifier),
    MetaVar(usize),

    Infer,
    Variadic(Box<Type>),
    Rc(Box<Type>),
    Weak(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Future(Box<Type>),
}

impl Type {
    pub fn is_probabilistic(&self) -> bool {
        matches!(
            self,
            Type::Gaussian | Type::Uniform | Type::Bernoulli | Type::Beta | Type::VonMises
        )
    }

    pub fn is_reactive(&self) -> bool {
        matches!(self, Type::Signal(_) | Type::Event(_))
    }

    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            Type::Int | Type::Float | Type::Bool | Type::String | Type::Unit
        )
    }

    pub fn matches(&self, other: &Type) -> bool {
        match (self, other) {
            (Type::Any, _) | (_, Type::Any) => true,

            (Type::Variadic(inner), Type::Variadic(other_inner)) => inner.matches(other_inner),
            (Type::Variadic(inner), o) => inner.matches(o),
            (o, Type::Variadic(inner)) => inner.matches(o),

            (Type::DynTrait { trait_path: a }, Type::DynTrait { trait_path: b }) => a == b,

            (Type::Array(a), Type::Array(b)) => a.matches(b),
            (Type::Option(a), Type::Option(b)) => a.matches(b),

            (
                Type::Result {
                    ok_type: ok1,
                    err_type: err1,
                },
                Type::Result {
                    ok_type: ok2,
                    err_type: err2,
                },
            ) => ok1.matches(ok2) && err1.matches(err2),

            (Type::Tuple(a), Type::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.matches(y))
            }

            (
                Type::Function {
                    params: p1,
                    return_type: r1,
                },
                Type::Function {
                    params: p2,
                    return_type: r2,
                },
            ) => {
                p1.len() == p2.len()
                    && p1.iter().zip(p2.iter()).all(|(x, y)| x.matches(y))
                    && r1.matches(r2)
            }

            _ => self == other,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "Int"),
            Type::Float => write!(f, "Float"),
            Type::Bool => write!(f, "Bool"),
            Type::String => write!(f, "String"),
            Type::Unit => write!(f, "Unit"),
            Type::Gaussian => write!(f, "Gaussian"),
            Type::Uniform => write!(f, "Uniform"),
            Type::Bernoulli => write!(f, "Bernoulli"),
            Type::Beta => write!(f, "Beta"),
            Type::VonMises => write!(f, "VonMises"),
            Type::Signal(t) => write!(f, "Signal<{}>", t),
            Type::Event(t) => write!(f, "Event<{}>", t),
            Type::Function {
                params,
                return_type,
            } => {
                write!(f, "Fn(")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, ") -> {}", return_type)
            }
            Type::Tuple(types) => {
                write!(f, "(")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            Type::Array(t) => write!(f, "Array<{}>", t),
            Type::Option(t) => write!(f, "Option<{}>", t),
            Type::Result { ok_type, err_type } => {
                write!(f, "Result<{}, {}>", ok_type, err_type)
            }
            Type::Named { name, type_args } => {
                if let Some(n) = name.last_name() {
                    write!(f, "{}", n)?;
                } else {
                    write!(f, "<path>")?;
                }
                if !type_args.is_empty() {
                    write!(f, "<")?;
                    for (i, arg) in type_args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            Type::TypeVar(name) => write!(f, "{}", name.name),
            Type::MetaVar(id) => write!(f, "_{}", id),
            Type::Infer => write!(f, "_"),
            Type::Any => write!(f, "Any"),
            Type::Variadic(inner) => write!(f, "Variadic<{}>", inner),
            Type::Rc(inner) => write!(f, "Rc<{}>", inner),
            Type::Weak(inner) => write!(f, "Weak<{}>", inner),
            Type::Assoc {
                trait_def: _,
                self_ty,
                name,
            } => write!(f, "{}::{}", self_ty, name),
            Type::DynTrait { trait_path } => write!(f, "dyn {}", trait_path),
            Type::Map(k, v) => write!(f, "Map<{}, {}>", k, v),
            Type::Handle(t) => write!(f, "Handle<{}>", t),
            Type::Future(t) => write!(f, "Future<{}>", t),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PathSeg {
    Ident(Identifier),
    Crate(Span),
    Self_(Span),
    Super(Span),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    pub segments: Vec<PathSeg>,
    pub span: Span,
    pub resolved: Option<DefId>,
}
impl Path {
    pub fn from_ident(ident: Identifier) -> Self {
        let span = ident.span;
        Path {
            segments: vec![PathSeg::Ident(ident)],
            span,
            resolved: None,
        }
    }

    pub fn as_single_ident(&self) -> Option<&Identifier> {
        match self.segments.as_slice() {
            [PathSeg::Ident(id)] => Some(id),
            _ => None,
        }
    }

    pub fn last_ident(&self) -> Option<&Identifier> {
        self.segments.iter().rev().find_map(|seg| {
            if let PathSeg::Ident(id) = seg {
                Some(id)
            } else {
                None
            }
        })
    }

    pub fn last_name(&self) -> Option<&str> {
        self.last_ident().map(|id| id.name.as_str())
    }

    pub fn iter_idents(&self) -> impl Iterator<Item = &Identifier> {
        self.segments.iter().filter_map(|seg| {
            if let PathSeg::Ident(id) = seg {
                Some(id)
            } else {
                None
            }
        })
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, seg) in self.segments.iter().enumerate() {
            if i > 0 {
                write!(f, "::")?;
            }
            match seg {
                PathSeg::Ident(id) => write!(f, "{}", id.name)?,
                PathSeg::Crate(_) => write!(f, "crate")?,
                PathSeg::Self_(_) => write!(f, "self")?,
                PathSeg::Super(_) => write!(f, "super")?,
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Let {
        pattern: Pattern,
        ty: Option<Type>,
        init: Option<Expression>,
        span: Span,
    },

    Expression(Expression),

    Return {
        value: Option<Expression>,
        span: Span,
    },

    While {
        condition: Expression,
        body: Block,
        span: Span,
    },

    For {
        pattern: Pattern,
        iterator: Expression,
        body: Block,
        span: Span,
    },

    Break {
        span: Span,
    },

    Continue {
        span: Span,
    },

    Empty,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Literal {
        value: Literal,
        span: Span,
    },
    Variable {
        name: Path,
        span: Span,
    },
    Binary {
        left: Box<Expression>,
        op: BinaryOp,
        right: Box<Expression>,
        span: Span,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Expression>,
        span: Span,
    },
    Call {
        callee: Box<Expression>,
        args: Vec<Expression>,
        span: Span,
    },
    MethodCall {
        receiver: Box<Expression>,
        method: Identifier,
        args: Vec<Expression>,
        span: Span,
        resolved: Option<DefId>,
    },
    FieldAccess {
        object: Box<Expression>,
        field: Identifier,
        span: Span,
    },
    Index {
        object: Box<Expression>,
        index: Box<Expression>,
        span: Span,
    },
    If {
        condition: Box<Expression>,
        then_branch: Block,
        else_branch: Option<Block>,
        span: Span,
    },
    Match {
        scrutinee: Box<Expression>,
        arms: Vec<MatchArm>,
        span: Span,
    },
    Block {
        block: Block,
        span: Span,
    },
    Lambda {
        params: Vec<Parameter>,
        body: Box<Expression>,
        span: Span,
    },
    Closure {
        params: Vec<Parameter>,
        body: Box<Expression>,
        span: Span,
    },
    Tuple {
        elements: Vec<Expression>,
        span: Span,
    },
    Array {
        elements: Vec<Expression>,
        span: Span,
    },
    Struct {
        name: Identifier,
        fields: Vec<FieldInit>,
        span: Span,
    },
    Enum {
        name: Identifier,
        variant: Identifier,
        args: Vec<Expression>,
        named_fields: Option<Vec<FieldInit>>,
        span: Span,
    },
    Some {
        expr: Box<Expression>,
        span: Span,
    },
    None {
        span: Span,
    },
    Ok {
        expr: Box<Expression>,
        span: Span,
    },
    Err {
        expr: Box<Expression>,
        span: Span,
    },
    Try {
        expr: Box<Expression>,
        span: Span,
    },
    Cast {
        expr: Box<Expression>,
        target_type: Type,
        span: Span,
    },
    Range {
        start: Option<Box<Expression>>,
        end: Option<Box<Expression>>,
        inclusive: bool,
        span: Span,
    },
    Paren {
        expr: Box<Expression>,
        span: Span,
    },
    Await {
        expr: Box<Expression>,
        span: Span,
    },
    Spawn {
        expr: Box<Expression>,
        span: Span,
    },
    With {
        name: Identifier,
        initializer: Box<Expression>,
        body: Block,
        span: Span,
    },
    UfcsMethod {
        trait_path: Path,
        method: Identifier,
        span: Span,
    },
}

impl Expression {
    pub fn span(&self) -> Span {
        use Expression::*;
        match self {
            Literal { span, .. }
            | Variable { span, .. }
            | Binary { span, .. }
            | Unary { span, .. }
            | Call { span, .. }
            | MethodCall { span, .. }
            | FieldAccess { span, .. }
            | Index { span, .. }
            | Lambda { span, .. }
            | If { span, .. }
            | Match { span, .. }
            | Block { span, .. }
            | Tuple { span, .. }
            | Array { span, .. }
            | Struct { span, .. }
            | Range { span, .. }
            | Cast { span, .. }
            | Paren { span, .. }
            | Await { span, .. }
            | Spawn { span, .. }
            | Some { span, .. }
            | None { span, .. }
            | Ok { span, .. }
            | Err { span, .. }
            | Try { span, .. }
            | With { span, .. }
            | Closure { span, .. }
            | Enum { span, .. }
            | UfcsMethod { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Precedence {
    Lowest,
    LogicalOr,
    LogicalAnd,
    Assignment,
    Or,
    And,
    Equals,
    LessGreater,
    Sum,
    Product,
    Prefix,
    Call,
    Index,
    Field,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expression>,
    pub body: Expression,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldInit {
    pub name: Identifier,
    pub value: Expression,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Unit,
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Int(n) => write!(f, "{}", n),
            Literal::Float(n) => write!(f, "{}", n),
            Literal::Bool(b) => write!(f, "{}", b),
            Literal::String(s) => write!(f, "\"{}\"", s),
            Literal::Unit => write!(f, "()"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    And,
    Or,

    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,

    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    ShlAssign,
    ShrAssign,
}

impl BinaryOp {
    pub fn assignment_op(&self) -> Option<BinaryOp> {
        match self {
            BinaryOp::AddAssign => Some(BinaryOp::Add),
            BinaryOp::SubAssign => Some(BinaryOp::Sub),
            BinaryOp::MulAssign => Some(BinaryOp::Mul),
            BinaryOp::DivAssign => Some(BinaryOp::Div),
            BinaryOp::ModAssign => Some(BinaryOp::Mod),
            BinaryOp::BitAndAssign => Some(BinaryOp::BitAnd),
            BinaryOp::BitOrAssign => Some(BinaryOp::BitOr),
            BinaryOp::BitXorAssign => Some(BinaryOp::BitXor),
            BinaryOp::ShlAssign => Some(BinaryOp::Shl),
            BinaryOp::ShrAssign => Some(BinaryOp::Shr),
            _ => None,
        }
    }

    pub fn desugared_op(&self) -> Option<BinaryOp> {
        self.assignment_op()
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::And => "&&",
            BinaryOp::Or => "||",
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
            BinaryOp::Assign => "=",
            BinaryOp::AddAssign => "+=",
            BinaryOp::SubAssign => "-=",
            BinaryOp::MulAssign => "*=",
            BinaryOp::DivAssign => "/=",
            BinaryOp::ModAssign => "%=",
            BinaryOp::BitAndAssign => "&=",
            BinaryOp::BitOrAssign => "|=",
            BinaryOp::BitXorAssign => "^=",
            BinaryOp::ShlAssign => "<<=",
            BinaryOp::ShrAssign => ">>=",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "!",
            UnaryOp::BitNot => "~",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard {
        span: Span,
    },

    Identifier {
        name: Identifier,
        span: Span,
    },

    Literal {
        value: Literal,
        span: Span,
    },

    Tuple {
        patterns: Vec<Pattern>,
        span: Span,
    },

    Struct {
        name: Identifier,
        fields: Vec<FieldPattern>,
        span: Span,
    },

    Enum {
        name: Identifier,
        variant: Identifier,
        args: Vec<Pattern>,
        named_fields: Option<Vec<FieldPattern>>,
        span: Span,
    },

    Or {
        patterns: Vec<Pattern>,
        span: Span,
    },

    Range {
        start: Literal,
        end: Literal,
        inclusive: bool,
        span: Span,
    },

    Some {
        pattern: Box<Pattern>,
        span: Span,
    },

    None {
        span: Span,
    },

    Ok {
        pattern: Box<Pattern>,
        span: Span,
    },

    Err {
        pattern: Box<Pattern>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldPattern {
    pub name: Identifier,
    pub pattern: Option<Pattern>,
    pub span: Span,
}

#[derive(Debug, Clone, Eq, Hash)]
pub struct Identifier {
    pub name: String,
    pub span: Span,
}

impl Identifier {
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Identifier {
            name: name.into(),
            span,
        }
    }
}

impl PartialEq for Identifier {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
}
