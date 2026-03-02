use crate::ad::types::ADFloat;
use crate::vm::region::ContainerStore;
use crate::vm::value::Value;
use crate::vm::Environment;
use std::collections::HashMap;

/// Send-safe representation of Value for cross-thread transfer.
#[derive(Debug, Clone)]
pub enum SendableValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Unit,
    Tuple(Vec<SendableValue>),
    Array(Vec<SendableValue>),
    Struct {
        name: String,
        fields: HashMap<String, SendableValue>,
    },
    Enum {
        name: String,
        variant: String,
        fields: Vec<SendableValue>,
    },
    Function {
        params: Vec<crate::ast::node::Parameter>,
        body: crate::ast::node::Block,
        closure: SendableEnvironment,
        is_async: bool,
    },
    Gaussian {
        mean: f64,
        std: f64,
    },
    Uniform {
        min: f64,
        max: f64,
    },
    Bernoulli {
        p: f64,
    },
    Beta {
        alpha: f64,
        beta: f64,
    },
    VonMises {
        mu: f64,
        kappa: f64,
    },
    Range {
        start: i64,
        end: i64,
        inclusive: bool,
    },
    Some(Box<SendableValue>),
    None,
    Builtin(String),
    Map(HashMap<String, SendableValue>),
    BytecodeFunction {
        name: String,
        chunk_index: usize,
        arity: usize,
        upvalue_count: usize,
    },
    BytecodeClosure {
        function: Box<SendableValue>,
        upvalues: Vec<SendableUpvalue>,
    },
}

#[derive(Debug, Clone)]
pub enum SendableUpvalue {
    Open(usize),
    Closed(SendableValue),
}

// Compile-time assertion that SendableValue is Send + 'static
const _: fn() = || {
    fn assert_send<T: Send + 'static>() {}
    assert_send::<SendableValue>();
};

#[derive(Debug, Clone)]
pub struct SendableEnvironment {
    pub scopes: Vec<HashMap<String, SendableValue>>,
}

pub fn value_to_sendable(val: &Value, store: &ContainerStore) -> Result<SendableValue, String> {
    match val {
        Value::Int(i) => Ok(SendableValue::Int(*i)),
        Value::Float(f) => Ok(SendableValue::Float(f.value())),
        Value::Bool(b) => Ok(SendableValue::Bool(*b)),
        Value::String(s) => Ok(SendableValue::String((**s).clone())),
        Value::Unit => Ok(SendableValue::Unit),
        Value::Builtin(name) => Ok(SendableValue::Builtin(name.clone())),
        Value::None => Ok(SendableValue::None),

        Value::Tuple(elems) => {
            let sendable: Result<Vec<_>, _> =
                elems.iter().map(|e| value_to_sendable(e, store)).collect();
            Ok(SendableValue::Tuple(sendable?))
        }

        Value::Array(handle) => {
            let arr = store
                .clone_array(*handle)
                .ok_or_else(|| "Invalid array handle during spawn".to_string())?;
            let sendable: Result<Vec<_>, _> =
                arr.iter().map(|e| value_to_sendable(e, store)).collect();
            Ok(SendableValue::Array(sendable?))
        }

        Value::Map(handle) => {
            let dict = store
                .clone_dict(*handle)
                .ok_or_else(|| "Invalid map handle during spawn".to_string())?;
            let mut sendable = HashMap::new();
            for (k, v) in &dict {
                sendable.insert(k.clone(), value_to_sendable(v, store)?);
            }
            Ok(SendableValue::Map(sendable))
        }

        Value::Struct { name, fields } => {
            let dict = store
                .clone_dict(*fields)
                .ok_or_else(|| "Invalid struct handle during spawn".to_string())?;
            let mut sendable = HashMap::new();
            for (k, v) in &dict {
                sendable.insert(k.clone(), value_to_sendable(v, store)?);
            }
            Ok(SendableValue::Struct {
                name: name.clone(),
                fields: sendable,
            })
        }

        Value::Enum {
            name,
            variant,
            fields,
        } => {
            let sendable: Result<Vec<_>, _> =
                fields.iter().map(|f| value_to_sendable(f, store)).collect();
            Ok(SendableValue::Enum {
                name: name.clone(),
                variant: variant.clone(),
                fields: sendable?,
            })
        }

        Value::Function {
            params,
            body,
            closure,
            is_async,
        } => {
            let sendable_env = env_to_sendable(closure, store)?;
            Ok(SendableValue::Function {
                params: params.clone(),
                body: body.clone(),
                closure: sendable_env,
                is_async: *is_async,
            })
        }

        Value::Some(inner) => Ok(SendableValue::Some(Box::new(value_to_sendable(
            inner, store,
        )?))),

        Value::Gaussian { mean, std } => Ok(SendableValue::Gaussian {
            mean: mean.value(),
            std: std.value(),
        }),
        Value::Uniform { min, max } => Ok(SendableValue::Uniform {
            min: min.value(),
            max: max.value(),
        }),
        Value::Bernoulli { p } => Ok(SendableValue::Bernoulli { p: p.value() }),
        Value::Beta { alpha, beta } => Ok(SendableValue::Beta {
            alpha: alpha.value(),
            beta: beta.value(),
        }),
        Value::VonMises { mu, kappa } => Ok(SendableValue::VonMises {
            mu: mu.value(),
            kappa: kappa.value(),
        }),
        Value::Range {
            start,
            end,
            inclusive,
        } => Ok(SendableValue::Range {
            start: *start,
            end: *end,
            inclusive: *inclusive,
        }),

        Value::Rc(inner) => value_to_sendable(inner, store),

        Value::BytecodeFunction {
            name,
            chunk_index,
            arity,
            upvalue_count,
        } => Ok(SendableValue::BytecodeFunction {
            name: name.clone(),
            chunk_index: *chunk_index,
            arity: *arity,
            upvalue_count: *upvalue_count,
        }),

        Value::BytecodeClosure { function, upvalues } => {
            let sendable_fn = value_to_sendable(function, store)?;
            let mut sendable_upvals = Vec::new();
            for uv in upvalues {
                let uv_ref = uv.borrow();
                match &*uv_ref {
                    crate::vm::value::Upvalue::Open(idx) => {
                        sendable_upvals.push(SendableUpvalue::Open(*idx));
                    }
                    crate::vm::value::Upvalue::Closed(val) => {
                        sendable_upvals
                            .push(SendableUpvalue::Closed(value_to_sendable(val, store)?));
                    }
                }
            }
            Ok(SendableValue::BytecodeClosure {
                function: Box::new(sendable_fn),
                upvalues: sendable_upvals,
            })
        }

        Value::Handle(_) => Err("Cannot send Handle across threads".into()),
        Value::Iterator(_) => Err("Cannot send Iterator across threads".into()),
        Value::Signal { .. } => Err("Cannot send Signal across threads".into()),
        Value::Event { .. } => Err("Cannot send Event across threads".into()),
        Value::Weak(_) => Err("Cannot send Weak reference across threads".into()),
        Value::NativeFunction { .. } => Err("Cannot send NativeFunction across threads".into()),
        Value::TraitObject { .. } => Err("Cannot send TraitObject across threads".into()),
        Value::Module { .. } => Err("Cannot send Module across threads".into()),
        Value::Future { .. } => Err("Cannot send Future across threads".into()),
    }
}

pub fn sendable_to_value(sval: SendableValue, store: &mut ContainerStore) -> Value {
    match sval {
        SendableValue::Int(i) => Value::Int(i),
        SendableValue::Float(f) => Value::Float(ADFloat::Concrete(f)),
        SendableValue::Bool(b) => Value::Bool(b),
        SendableValue::String(s) => Value::String(crate::gc::Rc::new(s)),
        SendableValue::Unit => Value::Unit,
        SendableValue::Builtin(name) => Value::Builtin(name),
        SendableValue::None => Value::None,

        SendableValue::Tuple(elems) => {
            let values: Vec<Value> = elems
                .into_iter()
                .map(|e| sendable_to_value(e, store))
                .collect();
            Value::Tuple(crate::gc::Rc::new(values))
        }

        SendableValue::Array(elems) => {
            let values: Vec<Value> = elems
                .into_iter()
                .map(|e| sendable_to_value(e, store))
                .collect();
            let handle = store.alloc_array(values);
            Value::Array(handle)
        }

        SendableValue::Map(dict) => {
            let values: HashMap<String, Value> = dict
                .into_iter()
                .map(|(k, v)| (k, sendable_to_value(v, store)))
                .collect();
            let handle = store.alloc_dict(values);
            Value::Map(handle)
        }

        SendableValue::Struct { name, fields } => {
            let values: HashMap<String, Value> = fields
                .into_iter()
                .map(|(k, v)| (k, sendable_to_value(v, store)))
                .collect();
            let handle = store.alloc_dict(values);
            Value::Struct {
                name,
                fields: handle,
            }
        }

        SendableValue::Enum {
            name,
            variant,
            fields,
        } => {
            let values: Vec<Value> = fields
                .into_iter()
                .map(|f| sendable_to_value(f, store))
                .collect();
            Value::Enum {
                name,
                variant,
                fields: crate::gc::Rc::new(values),
            }
        }

        SendableValue::Function {
            params,
            body,
            closure,
            is_async,
        } => {
            let env = sendable_to_env(closure, store);
            Value::Function {
                params,
                body,
                closure: crate::gc::Rc::new(env),
                is_async,
            }
        }

        SendableValue::Some(inner) => Value::Some(Box::new(sendable_to_value(*inner, store))),

        SendableValue::Gaussian { mean, std } => Value::Gaussian {
            mean: ADFloat::Concrete(mean),
            std: ADFloat::Concrete(std),
        },
        SendableValue::Uniform { min, max } => Value::Uniform {
            min: ADFloat::Concrete(min),
            max: ADFloat::Concrete(max),
        },
        SendableValue::Bernoulli { p } => Value::Bernoulli {
            p: ADFloat::Concrete(p),
        },
        SendableValue::Beta { alpha, beta } => Value::Beta {
            alpha: ADFloat::Concrete(alpha),
            beta: ADFloat::Concrete(beta),
        },
        SendableValue::VonMises { mu, kappa } => Value::VonMises {
            mu: ADFloat::Concrete(mu),
            kappa: ADFloat::Concrete(kappa),
        },
        SendableValue::Range {
            start,
            end,
            inclusive,
        } => Value::Range {
            start,
            end,
            inclusive,
        },

        SendableValue::BytecodeFunction {
            name,
            chunk_index,
            arity,
            upvalue_count,
        } => Value::BytecodeFunction {
            name,
            chunk_index,
            arity,
            upvalue_count,
        },

        SendableValue::BytecodeClosure { function, upvalues } => {
            let func_val = sendable_to_value(*function, store);
            let mut uv_vec = Vec::new();
            for suv in upvalues {
                let upvalue = match suv {
                    SendableUpvalue::Open(idx) => crate::vm::value::Upvalue::Open(idx),
                    SendableUpvalue::Closed(sval) => {
                        crate::vm::value::Upvalue::Closed(sendable_to_value(sval, store))
                    }
                };
                uv_vec.push(crate::gc::Rc::new(std::cell::RefCell::new(upvalue)));
            }
            Value::BytecodeClosure {
                function: Box::new(func_val),
                upvalues: uv_vec,
            }
        }
    }
}

pub fn env_to_sendable(
    env: &Environment,
    store: &ContainerStore,
) -> Result<SendableEnvironment, String> {
    let mut sendable_scopes = Vec::new();
    for scope in env.scopes() {
        let mut sendable_scope = HashMap::new();
        for (name, val) in scope {
            sendable_scope.insert(name.clone(), value_to_sendable(val, store)?);
        }
        sendable_scopes.push(sendable_scope);
    }
    Ok(SendableEnvironment {
        scopes: sendable_scopes,
    })
}

pub fn sendable_to_env(senv: SendableEnvironment, store: &mut ContainerStore) -> Environment {
    let mut scopes = Vec::new();
    for sendable_scope in senv.scopes {
        let mut scope = HashMap::new();
        for (name, sval) in sendable_scope {
            scope.insert(name, sendable_to_value(sval, store));
        }
        scopes.push(scope);
    }
    Environment::from_scopes(scopes)
}
