// Runtime value representation for Flux with Reference Counting GC.

use crate::ast::node::{Block, Parameter};
use crate::vm::Environment;
use crate::gc::{Rc, Weak}; // 自作Rcを使用
use std::fmt;
use std::collections::HashMap;
use crate::ad::types::ADFloat;
use std::ops::{Add, Sub, Mul, Div, Neg};

// Runtime value in the Flux interpreter.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    // Integer value
    Int(i64),
    // Floating-point value
    Float(ADFloat),
    // Boolean value
    Bool(bool),
    // String value (Heap allocated)
    String(Rc<String>),
    // Unit value (void)
    Unit,
    // Opaque handle to an external resource
    Handle(usize),
    
    // Tuple value (Heap allocated)
    Tuple(Rc<Vec<Value>>),
    // Array value (Heap allocated)
    Array(Rc<Vec<Value>>),
    
    // Struct instance (Heap allocated)
    Struct {
        name: String,
        fields: Rc<HashMap<String, Value>>,
    },
    
    // Enum variant (Heap allocated if data is present)
    /*Enum {
        name: String,
        variant: String,
        data: Rc<Value>, // BoxではなくRcに変更
    },*/
    Enum { name: String, variant: String, fields: Rc<Vec<Value>> },
    
    // Function closure (Heap allocated)
    Function {
        params: Vec<Parameter>,
        body: Block,
        closure: Rc<Environment>, 
    },
    
    // Probabilistic value (Gaussian)
    Gaussian {
        mean: ADFloat,
        std: ADFloat,
    },
    
    // Probabilistic value (Uniform)
    Uniform {
        min: ADFloat,
        max: ADFloat,
    },
    
    // Probabilistic value (Bernoulli)
    Bernoulli {
        p: ADFloat, // probability of success (0 <= p <= 1)
    },
    
    // Probabilistic value (Beta)
    Beta {
        alpha: ADFloat,
        beta: ADFloat,
    },
    
    // Signal value (reactive)
    Signal {
        id: usize,
        current_value: Rc<Value>, // Box -> Rc
    },
    
    // Event value (reactive)
    Event {
        id: usize,
        latest_value: Option<Rc<Value>>, // Box -> Rc
    },
    
    Some(Box<Value>),
    None,
    
    // Built-in function
    Builtin(String),
    //Option(Option<Rc<Value>>), // Box -> Rc

    // Map value (Heap allocated, Mutable)
    Map(Rc<std::cell::RefCell<HashMap<String, Value>>>),
    
    // Internal use for Rc/Weak
    Rc(Rc<Value>),
    Weak(Weak<Value>),

    // Foreign Function (Dynamic Library Call)
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
}

impl Value {
    // Check if this value is "truthy" (for conditionals).
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

    // Get the type name of this value.
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
            Value::Gaussian { .. } => "Gaussian",
            Value::Uniform { .. } => "Uniform",
            Value::Bernoulli { .. } => "Bernoulli",
            Value::Beta { .. } => "Beta",
            Value::Signal { .. } => "Signal",
            Value::Event { .. } => "Event",
            //Value::Option(_) => "Option",
            //Value::Result(_) => "Result",
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
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            Value::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                for (i, (key, val)) in fields.iter().enumerate() {
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
            Value::Gaussian { mean, std } => write!(f, "Gaussian({}, {})", mean.value(), std.value()),
            Value::Uniform { min, max } => write!(f, "Uniform({}, {})", min.value(), max.value()),
            Value::Bernoulli { p } => write!(f, "Bernoulli({})", p.value()),
            Value::Beta { alpha, beta } => write!(f, "Beta({}, {})", alpha.value(), beta.value()),
            Value::Signal { id, .. } => write!(f, "<signal:{}>", id),
            Value::Event { id, .. } => write!(f, "<event:{}>", id),
            //Value::Option(Some(val)) => write!(f, "Some({})", val),
            //Value::Option(None) => write!(f, "None"),
            //Value::Result(Ok(val)) => write!(f, "Ok({})", val),
            //Value::Result(Err(err)) => write!(f, "Err({})", err),
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
        }
    }
}



impl Add for Value {
    type Output = Result<Value, String>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            
            // ADFloat同士
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            
            // IntとFloatの混合 -> Float (ADFloat)
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(ADFloat::from(a as f64) + b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + ADFloat::from(b as f64))),
            
            // 文字列連結
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
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)), // 整数除算
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




// Gaussianサンプリング: Box-Muller
// Gaussianの.sample()的動作で呼ばれる関数
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