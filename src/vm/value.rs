// src/vm/value.rs

use crate::ast::node::{Block, Parameter};
use crate::vm::Environment;
use crate::gc::{Rc, Weak};
use std::fmt;
use std::collections::HashMap;
use crate::ad::types::ADFloat;
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq)]
pub enum Upvalue {
    Open(usize),
    Closed(Value),
}

#[derive(Debug, Clone, PartialEq)]
pub enum IteratorState {
    Range { current: i64, end: i64, inclusive: bool },
    Array { array: Rc<RefCell<Vec<Value>>>, index: usize },
}

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(ADFloat),
    Bool(bool),
    String(Rc<String>),
    Unit,
    Handle(usize),
    Tuple(Rc<Vec<Value>>),
    Array(Rc<RefCell<Vec<Value>>>),
    Struct {
        name: String,
        fields: Rc<RefCell<HashMap<String, Value>>>,
    },
    Enum { name: String, variant: String, fields: Rc<Vec<Value>> },
    Function {
        params: Vec<Parameter>,
        body: Block,
        closure: Rc<Environment>, 
        is_async: bool,
    },
    BytecodeFunction {
        name: String,
        chunk_index: usize,
        arity: usize,
        upvalue_count: usize,
    },
    BytecodeClosure {
        function: Box<Value>,
        upvalues: Vec<Rc<RefCell<Upvalue>>>,
    },
    Gaussian {
        mean: ADFloat,
        std: ADFloat,
    },
    Uniform {
        min: ADFloat,
        max: ADFloat,
    },
    Bernoulli {
        p: ADFloat,
    },
    Beta {
        alpha: ADFloat,
        beta: ADFloat,
    },
    Signal {
        id: usize,
        current_value: Rc<Value>,
    },
    Range {
        start: i64,
        end: i64,
        inclusive: bool,
    },
    Iterator(Rc<RefCell<IteratorState>>),
    Event {
        id: usize,
        latest_value: Option<Rc<Value>>,
    },
    Some(Box<Value>),
    None,
    Builtin(String),
    Map(Rc<std::cell::RefCell<HashMap<String, Value>>>),
    Rc(Rc<Value>),
    Weak(Weak<Value>),
    NativeFunction {
        name: String,
        library_index: usize,
        params: Vec<crate::ast::node::Type>,
        return_type: crate::ast::node::Type,
        is_async: bool,
    },
    Module { name: String },
    TraitObject {
        trait_name: String,
        vtable: Rc<HashMap<String, Value>>,
        data: Rc<Value>,
    },
    Future {
        task_id: usize,
        result: Arc<Mutex<Option<Result<Value, String>>>>,
    },
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Unit => false,
            Value::Int(0) => false,
            Value::Float(f) => {
                if f.value() == 0.0 { false } else { true }
            },
            _ => true,
        }
    }

    pub fn type_name(&self) -> &str {
        match self {
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Bool(_) => "Bool",
            Value::String(_) => "String",
            Value::Unit => "Unit",
            Value::Tuple(_) => "Tuple",
            Value::Array(_) => "Array",
            Value::Struct { .. } => "Struct",
            Value::Function { .. } => "Function",
            Value::BytecodeFunction { .. } => "Function",
            Value::BytecodeClosure { .. } => "Function",
            Value::Gaussian { .. } => "Gaussian",
            Value::Uniform { .. } => "Uniform",
            Value::Bernoulli { .. } => "Bernoulli",
            Value::Beta { .. } => "Beta",
            Value::Signal { .. } => "Signal",
            Value::Event { .. } => "Event",
            Value::Enum { name, .. } => name.as_str(),
            Value::Builtin(_) => "Builtin", 
            Value::Some(_) => "Some",
            Value::None => "None",
            Value::Rc(_) => "Rc",
            Value::Weak(_) => "Weak",
            Value::Module { .. } => "Module",
            Value::TraitObject { trait_name, .. } => trait_name,
            Value::Map(_) => "Map",
            Value::Handle(_) => "Handle",
            Value::NativeFunction { .. } => "NativeFunction",
            Value::Range { .. } => "Range",
            Value::Iterator(_) => "Iterator",
            Value::Future { .. } => "Future",
        }
    }
    
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(ad) => Some(ad.value()),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_ad_float(&self) -> Option<ADFloat> {
        match self {
            Value::Float(ad) => Some(ad.clone()),
            Value::Int(i) => Some(ADFloat::Concrete(*i as f64)),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(fl) => write!(f, "{}", fl.value()), 
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "{}", s),
            Value::Unit => write!(f, "()"),
            Value::Tuple(elements) => {
                write!(f, "(")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, ")")
            }
            Value::Array(elements) => {
                write!(f, "[")?;
                let arr = elements.borrow();
                for (i, elem) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            Value::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                let map = fields.borrow();
                for (i, (key, val)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", key, val)?;
                }
                write!(f, " }}")
            }
            Value::Enum { name, variant, fields } => {
                if fields.is_empty() {
                    write!(f, "{}::{}", name, variant)
                } else {
                    write!(f, "{}::{}(", name, variant)?;
                    for (i, v) in fields.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", v)?;
                    }
                    write!(f, ")")
                }
            }
            Value::Function { .. } => write!(f, "<function>"),
            Value::BytecodeFunction { name, arity, .. } => write!(f, "<fn {} (arity {})>", name, arity),
            Value::BytecodeClosure { function, .. } => write!(f, "{}", function),
            Value::Gaussian { mean, std } => write!(f, "Gaussian({}, {})", mean.value(), std.value()),
            Value::Uniform { min, max } => write!(f, "Uniform({}, {})", min.value(), max.value()),
            Value::Bernoulli { p } => write!(f, "Bernoulli({})", p.value()),
            Value::Beta { alpha, beta } => write!(f, "Beta({}, {})", alpha.value(), beta.value()),
            Value::Signal { id, .. } => write!(f, "<signal:{}>", id),
            Value::Event { id, .. } => write!(f, "<event:{}>", id),
            Value::Builtin(name) => write!(f, "<builtin:{}>", name),
            Value::Some(v) => write!(f, "Some({})", v),
            Value::None => write!(f, "None"), 
            Value::Rc(rc) => write!(f, "Rc({})", **rc),
            Value::Weak(_) => write!(f, "<weak-ref>"),
            Value::Module { name } => write!(f, "<module {}>", name),
            Value::TraitObject { trait_name, data, .. } => {
                write!(f, "<dyn {}: {}>", trait_name, data)
            }
            Value::Map(rc_map) => {
                write!(f, "{{")?;
                let map = rc_map.borrow();
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Handle(id) => write!(f, "Handle({})", id),
            Value::NativeFunction { name, .. } => write!(f, "<native fn {}>", name),
            Value::Range { start, end, inclusive } => {
                if *inclusive {
                    write!(f, "{}..={}", start, end)
                } else {
                    write!(f, "{}..{}", start, end)
                }
            }
            Value::Iterator(_) => write!(f, "<iterator>"),
            Value::Future { task_id, .. } => write!(f, "<future:{}>", task_id),
        }
    }
}



impl Add for Value {
    type Output = Result<Value, String>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(ADFloat::from(a as f64) + b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + ADFloat::from(b as f64))),
            (Value::String(a), Value::String(b)) => Ok(Value::String(Rc::new((**a).to_string() + &b))),
            (a, b) => Err(format!("Type mismatch in addition: {} + {}", a.type_name(), b.type_name())),
        }
    }
}

impl Sub for Value {
    type Output = Result<Value, String>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(ADFloat::from(a as f64) - b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - ADFloat::from(b as f64))),
            (a, b) => Err(format!("Type mismatch in subtraction: {} - {}", a.type_name(), b.type_name())),
        }
    }
}

impl Mul for Value {
    type Output = Result<Value, String>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(ADFloat::from(a as f64) * b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * ADFloat::from(b as f64))),
            (a, b) => Err(format!("Type mismatch in multiplication: {} * {}", a.type_name(), b.type_name())),
        }
    }
}

impl Div for Value {
    type Output = Result<Value, String>;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(ADFloat::from(a as f64) / b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / ADFloat::from(b as f64))),
            (a, b) => Err(format!("Type mismatch in division: {} / {}", a.type_name(), b.type_name())),
        }
    }
}

impl Neg for Value {
    type Output = Result<Value, String>;

    fn neg(self) -> Self::Output {
        match self {
            Value::Int(a) => Ok(Value::Int(-a)),
            Value::Float(a) => Ok(Value::Float(-a)),
            a => Err(format!("Type mismatch in negation: -{}", a.type_name())),
        }
    }
}

fn gaussian_sample(mean: f64, std: f64) -> f64 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let u1: f64 = rng.gen();
    let u2: f64 = rng.gen();
    mean + std * f64::sqrt(-2.0 * u1.ln()) * f64::cos(2.0 * std::f64::consts::PI * u2)
}

pub fn gaussian_to_float(mean: f64, std: f64) -> Value {
    let val = gaussian_sample(mean, std);
    Value::Float(ADFloat::Concrete(val))
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Unit, Value::Unit) => true,
            (Value::Handle(a), Value::Handle(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => *a.borrow() == *b.borrow(),
            (Value::Struct { name: n1, fields: f1 }, Value::Struct { name: n2, fields: f2 }) => {
                n1 == n2 && *f1.borrow() == *f2.borrow()
            }
            (Value::Enum { name: n1, variant: v1, fields: f1 }, Value::Enum { name: n2, variant: v2, fields: f2 }) => {
                n1 == n2 && v1 == v2 && f1 == f2
            }
            (
                Value::Function {
                    params: p1,
                    body: b1,
                    closure: c1,
                    is_async: a1,
                },
                Value::Function {
                    params: p2,
                    body: b2,
                    closure: c2,
                    is_async: a2,
                },
            ) => p1 == p2 && b1 == b2 && Rc::ptr_eq(c1, c2) && a1 == a2,
            (Value::BytecodeFunction { name: n1, chunk_index: c1, arity: a1, upvalue_count: u1 }, 
             Value::BytecodeFunction { name: n2, chunk_index: c2, arity: a2, upvalue_count: u2 }) => {
                n1 == n2 && c1 == c2 && a1 == a2 && u1 == u2
            }
            (Value::BytecodeClosure { function: f1, upvalues: u1 }, 
             Value::BytecodeClosure { function: f2, upvalues: u2 }) => {
                f1 == f2 && u1 == u2
            }
            (Value::Gaussian { mean: m1, std: s1 }, Value::Gaussian { mean: m2, std: s2 }) => {
                m1 == m2 && s1 == s2
            }
            (Value::Uniform { min: mn1, max: mx1 }, Value::Uniform { min: mn2, max: mx2 }) => {
                mn1 == mn2 && mx1 == mx2
            }
            (Value::Bernoulli { p: p1 }, Value::Bernoulli { p: p2 }) => p1 == p2,
            (Value::Beta { alpha: a1, beta: b1 }, Value::Beta { alpha: a2, beta: b2 }) => {
                a1 == a2 && b1 == b2
            }
            (Value::Signal { id: i1, current_value: v1 }, Value::Signal { id: i2, current_value: v2 }) => {
                i1 == i2 && v1 == v2
            }
            (Value::Range { start: s1, end: e1, inclusive: i1 }, 
             Value::Range { start: s2, end: e2, inclusive: i2 }) => {
                s1 == s2 && e1 == e2 && i1 == i2
            }
            (Value::Iterator(a), Value::Iterator(b)) => *a.borrow() == *b.borrow(),
            (Value::Event { id: i1, latest_value: v1 }, Value::Event { id: i2, latest_value: v2 }) => {
                i1 == i2 && v1 == v2
            }
            (Value::Some(a), Value::Some(b)) => a == b,
            (Value::None, Value::None) => true,
            (Value::Builtin(a), Value::Builtin(b)) => a == b,
            (Value::Map(a), Value::Map(b)) => *a.borrow() == *b.borrow(),
            (Value::Rc(a), Value::Rc(b)) => a == b,
            (Value::Weak(_), Value::Weak(_)) => false,
            (Value::NativeFunction { name: n1, library_index: l1, params: p1, return_type: r1, is_async: a1 },
             Value::NativeFunction { name: n2, library_index: l2, params: p2, return_type: r2, is_async: a2 }) => {
                n1 == n2 && l1 == l2 && p1 == p2 && r1 == r2 && a1 == a2
            }
            (Value::Module { name: n1 }, Value::Module { name: n2 }) => n1 == n2,
            (Value::TraitObject { trait_name: t1, vtable: v1, data: d1 }, 
             Value::TraitObject { trait_name: t2, vtable: v2, data: d2 }) => {
                t1 == t2 && v1 == v2 && d1 == d2
            }
            (Value::Future { task_id: t1, .. }, Value::Future { task_id: t2, .. }) => t1 == t2,
            _ => false,
        }
    }
}