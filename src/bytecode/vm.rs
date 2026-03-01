use super::chunk::Chunk;
use super::opcode::Opcode;
use crate::ad::types::ADFloat;
use crate::gc::Rc;
use crate::vm::Upvalue;
use crate::Value;
use rand::Rng;
use std::cell::RefCell;
use std::collections::HashMap;

const STACK_MAX: usize = 65536;
const FRAMES_MAX: usize = 256;

#[derive(Debug)]
pub struct CallFrame {
    pub chunk_index: usize,
    pub ip: usize,
    pub slot_base: usize,
    pub closure: Value,
}

pub struct BytecodeVM {
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    globals: HashMap<String, Value>,
    chunks: Vec<Chunk>,
    current_frame: usize,
    open_upvalues: Vec<Rc<RefCell<Upvalue>>>,
    methods: HashMap<String, HashMap<String, Value>>,

    log_weight: f64,
    rng: rand::rngs::ThreadRng,
    prob_context: Option<crate::vm::prob_runtime::ProbContext>,
    reactive_runtime: crate::vm::reactive::ReactiveRuntime,
    pub store: crate::vm::region::ContainerStore,
    task_id_counter: usize,
}

#[derive(Debug)]
pub enum VMError {
    StackOverflow,
    StackUnderflow,
    TypeError(String),
    UndefinedVariable(String),
    UndefinedField(String),
    DivisionByZero,
    IndexOutOfBounds,
    InvalidOperation(String),
    ArityMismatch { expected: usize, found: usize },
}

impl From<crate::vm::RuntimeError> for VMError {
    fn from(err: crate::vm::RuntimeError) -> Self {
        use crate::vm::RuntimeError;
        match err {
            RuntimeError::TypeMismatch { message, .. } => VMError::TypeError(message),
            RuntimeError::UndefinedVariable { name, .. } => VMError::UndefinedVariable(name),
            RuntimeError::InvalidOperation { message, .. } => VMError::InvalidOperation(message),
            RuntimeError::ArityMismatch {
                expected, found, ..
            } => VMError::ArityMismatch { expected, found },
            RuntimeError::IndexOutOfBounds { .. } => VMError::IndexOutOfBounds,
            _ => VMError::InvalidOperation(format!("Runtime error: {:?}", err)),
        }
    }
}

pub type VMResult<T> = Result<T, VMError>;

use crate::ad::types::ADGradient;
use crate::vm::prob_runtime::{Adam, InferenceMode, Optimizer, ProbContext, SGD};
use crate::vm::reactive::ReactiveRuntime;

impl BytecodeVM {
    pub fn new() -> Self {
        let mut vm = BytecodeVM {
            stack: Vec::with_capacity(256),
            frames: Vec::with_capacity(16),
            globals: HashMap::new(),
            chunks: Vec::new(),
            current_frame: 0,
            open_upvalues: Vec::new(),
            methods: HashMap::new(),
            log_weight: 0.0,
            rng: rand::thread_rng(),
            prob_context: None,
            reactive_runtime: ReactiveRuntime::new(),
            store: crate::vm::region::ContainerStore::new(),
            task_id_counter: 0,
        };

        vm.globals
            .insert("infer".to_string(), Value::Builtin("infer".to_string()));
        vm.globals.insert(
            "infer_hmc".to_string(),
            Value::Builtin("infer_hmc".to_string()),
        ); // Added infer_hmc
        vm.globals
            .insert("len".to_string(), Value::Builtin("len".to_string()));
        vm.globals.insert(
            "gaussian".to_string(),
            Value::Builtin("gaussian".to_string()),
        );
        vm.globals.insert(
            "Gaussian".to_string(),
            Value::Builtin("gaussian".to_string()),
        ); // Alias
        vm.globals
            .insert("uniform".to_string(), Value::Builtin("uniform".to_string()));
        vm.globals.insert(
            "bernoulli".to_string(),
            Value::Builtin("bernoulli".to_string()),
        );
        vm.globals
            .insert("beta".to_string(), Value::Builtin("beta".to_string()));
        vm.globals
            .insert("sample".to_string(), Value::Builtin("sample".to_string())); // Added sample (commonly used as wrapper)
        vm.globals
            .insert("observe".to_string(), Value::Builtin("observe".to_string())); // Added observe
        vm.globals
            .insert("Map".to_string(), Value::Builtin("Map".to_string())); // Added Map constructor
        vm.globals.insert(
            "infer_vi".to_string(),
            Value::Builtin("infer_vi".to_string()),
        );
        vm.globals
            .insert("param".to_string(), Value::Builtin("param".to_string()));
        vm.globals
            .insert("println".to_string(), Value::Builtin("println".to_string()));
        vm.globals
            .insert("sin".to_string(), Value::Builtin("sin".to_string()));
        vm.globals
            .insert("cos".to_string(), Value::Builtin("cos".to_string()));
        vm.globals
            .insert("tan".to_string(), Value::Builtin("tan".to_string()));
        vm.globals
            .insert("exp".to_string(), Value::Builtin("exp".to_string()));
        vm.globals
            .insert("ln".to_string(), Value::Builtin("ln".to_string()));
        vm.globals
            .insert("sqrt".to_string(), Value::Builtin("sqrt".to_string()));
        vm.globals
            .insert("pow".to_string(), Value::Builtin("pow".to_string()));

        vm.globals.insert(
            "Map::insert".to_string(),
            Value::Builtin("Map::insert".to_string()),
        );
        vm.globals.insert(
            "Map::get".to_string(),
            Value::Builtin("Map::get".to_string()),
        );

        vm.globals.insert(
            "Array::push".to_string(),
            Value::Builtin("Array::push".to_string()),
        );
        vm.globals.insert(
            "Signal_new".to_string(),
            Value::Builtin("Signal_new".to_string()),
        );
        vm.globals.insert(
            "Signal_get".to_string(),
            Value::Builtin("Signal_get".to_string()),
        );
        vm.globals
            .insert("grad".to_string(), Value::Builtin("grad".to_string()));
        vm.globals.insert(
            "backward".to_string(),
            Value::Builtin("backward".to_string()),
        );

        vm.globals.insert(
            "Signal_set".to_string(),
            Value::Builtin("Signal_set".to_string()),
        );

        vm
    }

    pub fn load_chunk(&mut self, chunk: Chunk) -> usize {
        let index = self.chunks.len();
        self.chunks.push(chunk);
        index
    }

    pub fn execute(&mut self, chunk_index: usize) -> VMResult<Value> {
        self.frames.push(CallFrame {
            chunk_index,
            ip: 0,
            slot_base: 0,
            closure: Value::Unit,
        });
        self.current_frame = 0;

        self.run(0)
    }

    fn frame(&self) -> &CallFrame {
        if self.frames.is_empty() {
            panic!("BytecodeVM: No frames available");
        }
        &self.frames[self.current_frame]
    }

    fn frame_mut(&mut self) -> &mut CallFrame {
        if self.frames.is_empty() {
            panic!("BytecodeVM: No frames available");
        }
        &mut self.frames[self.current_frame]
    }

    fn chunk(&self) -> &Chunk {
        &self.chunks[self.frame().chunk_index]
    }

    fn push(&mut self, value: Value) -> VMResult<()> {
        if self.stack.len() >= STACK_MAX {
            return Err(VMError::StackOverflow);
        }
        self.stack.push(value);
        Ok(())
    }

    fn pop(&mut self) -> VMResult<Value> {
        self.stack.pop().ok_or(VMError::StackUnderflow)
    }

    fn peek(&self, distance: usize) -> VMResult<&Value> {
        if distance >= self.stack.len() {
            return Err(VMError::StackUnderflow);
        }
        Ok(&self.stack[self.stack.len() - 1 - distance])
    }

    fn get_constant(&self, index: u16) -> &Value {
        &self.chunk().constants[index as usize]
    }

    fn run(&mut self, target_depth: usize) -> VMResult<Value> {
        loop {
            let chunk_index = self.frame().chunk_index;
            let ip = self.frame().ip;

            if ip >= self.chunks[chunk_index].code.len() {
                break;
            }

            let instr = self.chunks[chunk_index].code[ip].clone();
            self.frame_mut().ip += 1;

            match instr.opcode {
                Opcode::Const => {
                    let idx = instr.read_u16();
                    let value = self.chunks[chunk_index].constants[idx as usize].clone();
                    self.push(value)?;
                }

                Opcode::Pop => {
                    self.pop()?;
                }

                Opcode::Dup => {
                    let value = self.peek(0)?.clone();
                    self.push(value)?;
                }

                Opcode::Nil => {
                    self.push(Value::Unit)?;
                }

                Opcode::True => {
                    self.push(Value::Bool(true))?;
                }

                Opcode::False => {
                    self.push(Value::Bool(false))?;
                }

                Opcode::Add => self.binary_add()?,
                Opcode::Sub => self.binary_sub()?,
                Opcode::Mul => self.binary_mul()?,
                Opcode::Div => self.binary_div()?,
                Opcode::Mod => self.binary_mod()?,
                Opcode::Neg => self.unary_neg()?,

                Opcode::Eq => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(values_equal(&a, &b)))?;
                }
                Opcode::Ne => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(!values_equal(&a, &b)))?;
                }
                Opcode::Lt => self.comparison_op(|a, b| a < b)?,
                Opcode::Le => self.comparison_op(|a, b| a <= b)?,
                Opcode::Gt => self.comparison_op(|a, b| a > b)?,
                Opcode::Ge => self.comparison_op(|a, b| a >= b)?,

                Opcode::And => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(is_truthy(&a) && is_truthy(&b)))?;
                }
                Opcode::Or => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(is_truthy(&a) || is_truthy(&b)))?;
                }
                Opcode::Not => {
                    let a = self.pop()?;
                    self.push(Value::Bool(!is_truthy(&a)))?;
                }

                Opcode::GetLocal => {
                    let slot = instr.read_u16() as usize;
                    let base = self.frame().slot_base;
                    if base + slot >= self.stack.len() {
                        return Err(VMError::StackUnderflow);
                    }
                    let value = self.stack[base + slot].clone();
                    self.push(value)?;
                }

                Opcode::SetLocal => {
                    let slot = instr.read_u16() as usize;
                    let base = self.frame().slot_base;
                    if base + slot >= self.stack.len() {
                        return Err(VMError::StackUnderflow);
                    }
                    let value = self.peek(0)?.clone();
                    self.stack[base + slot] = value;
                }

                Opcode::GetGlobal => {
                    let idx = instr.read_u16();
                    let name = match &self.chunks[chunk_index].constants[idx as usize] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid global name".into())),
                    };
                    let value = self
                        .globals
                        .get(&name)
                        .ok_or_else(|| VMError::UndefinedVariable(name))?
                        .clone();
                    self.push(value)?;
                }

                Opcode::SetGlobal => {
                    let idx = instr.read_u16();
                    let name = match &self.chunks[chunk_index].constants[idx as usize] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid global name".into())),
                    };
                    if !self.globals.contains_key(&name) {
                        return Err(VMError::UndefinedVariable(name));
                    }
                    let value = self.peek(0)?.clone();
                    self.globals.insert(name, value);
                }

                Opcode::DefineGlobal => {
                    let idx = instr.read_u16();
                    let name = match &self.chunks[chunk_index].constants[idx as usize] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid global name".into())),
                    };
                    let value = self.pop()?;
                    self.globals.insert(name, value);
                }

                Opcode::Jump => {
                    let offset = instr.read_i16();
                    let new_ip = (self.frame().ip as i32 + offset as i32) as usize;
                    self.frame_mut().ip = new_ip;
                }

                Opcode::JumpIfFalse => {
                    let offset = instr.read_i16();
                    if !is_truthy(&self.pop()?) {
                        let new_ip = (self.frame().ip as i32 + offset as i32) as usize;
                        self.frame_mut().ip = new_ip;
                    }
                }

                Opcode::JumpIfTrue => {
                    let offset = instr.read_i16();
                    if is_truthy(&self.pop()?) {
                        let new_ip = (self.frame().ip as i32 + offset as i32) as usize;
                        self.frame_mut().ip = new_ip;
                    }
                }

                Opcode::Loop => {
                    let offset = instr.read_u16() as usize;
                    self.frame_mut().ip -= offset;
                }

                Opcode::MakeArray => {
                    let count = instr.read_u16() as usize;
                    let mut elements = Vec::with_capacity(count);
                    for _ in 0..count {
                        elements.push(self.pop()?);
                    }
                    elements.reverse();
                    let handle = self.store.alloc_array(elements);
                    self.push(Value::Array(handle))?;
                }

                Opcode::Index => {
                    let index = self.pop()?;
                    let obj = self.pop()?;
                    match (&obj, &index) {
                        (Value::Array(handle), Value::Int(i)) => {
                            let idx = *i as usize;
                            if let Some(val) = self
                                .store
                                .with_array(*handle, |a| a.get(idx).cloned())
                                .flatten()
                            {
                                self.push(val)?;
                            } else {
                                return Err(VMError::IndexOutOfBounds);
                            }
                        }
                        (Value::String(s), Value::Int(i)) => {
                            let idx = *i as usize;
                            if idx >= s.len() {
                                return Err(VMError::IndexOutOfBounds);
                            }
                            let ch = s.chars().nth(idx).ok_or(VMError::IndexOutOfBounds)?;
                            self.push(Value::String(Rc::new(ch.to_string())))?;
                        }
                        _ => {
                            return Err(VMError::TypeError(format!(
                                "Cannot index {} with {}",
                                obj.type_name(),
                                index.type_name()
                            )))
                        }
                    }
                }

                Opcode::IsType => {
                    let idx = instr.read_u16();
                    let type_name = match &self.chunks[chunk_index].constants[idx as usize] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid type name".into())),
                    };
                    let value = self.peek(0)?;
                    let is_match = match value {
                        Value::Int(_) => type_name == "Int",
                        Value::Float(_) => type_name == "Float",
                        Value::Bool(_) => type_name == "Bool",
                        Value::String(_) => type_name == "String",
                        Value::Struct { name, .. } => *name == type_name,
                        Value::Enum { name, .. } => *name == type_name,
                        _ => false,
                    };
                    self.push(Value::Bool(is_match))?;
                }

                Opcode::SetField => {
                    let val = self.pop()?;
                    let obj = self.pop()?;
                    let name_idx = instr.read_u16();

                    let name = match &self.chunks[chunk_index].constants[name_idx as usize] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid field name".into())),
                    };

                    if let Value::Struct { fields, .. } = obj {
                        self.store.with_dict_mut(fields, |d| {
                            d.insert(name, val.clone());
                        });
                        self.push(val)?;
                    } else {
                        return Err(VMError::TypeError("SetField on non-struct".into()));
                    }
                }

                Opcode::SetIndex => {
                    let val = self.pop()?;
                    let idx_val = self.pop()?;
                    let obj = self.pop()?;

                    match (obj, idx_val) {
                        (Value::Array(handle), Value::Int(i)) => {
                            let idx = i as usize;
                            let mut success = false;
                            self.store.with_array_mut(handle, |a| {
                                if idx < a.len() {
                                    a[idx] = val.clone();
                                    success = true;
                                }
                            });
                            if !success {
                                return Err(VMError::IndexOutOfBounds);
                            }
                            self.push(val)?;
                        }
                        (Value::Map(handle), key_val) => {
                            let key = match key_val {
                                Value::String(s) => s.to_string(),
                                _ => {
                                    return Err(VMError::TypeError("Map key must be string".into()))
                                }
                            };
                            self.store.with_dict_mut(handle, |m| {
                                m.insert(key, val.clone());
                            });
                            self.push(val)?;
                        }
                        _ => return Err(VMError::TypeError("Invalid item assignment".into())),
                    }
                }

                Opcode::MakeRange => {
                    let inclusive = instr.read_u8() != 0;
                    let end_val = self.pop()?;
                    let start_val = self.pop()?;

                    let start = match start_val {
                        Value::Int(n) => n,
                        _ => return Err(VMError::TypeError("Range start must be Int".into())),
                    };
                    let end = match end_val {
                        Value::Int(n) => n,
                        _ => return Err(VMError::TypeError("Range end must be Int".into())),
                    };

                    self.push(Value::Range {
                        start,
                        end,
                        inclusive,
                    })?;
                }

                Opcode::Iterator => {
                    let val = self.pop()?;
                    let iter_state = match val {
                        Value::Range {
                            start,
                            end,
                            inclusive,
                        } => {
                            use crate::vm::value::IteratorState;
                            IteratorState::Range {
                                current: start,
                                end,
                                inclusive,
                            }
                        }
                        Value::Array(handle) => {
                            use crate::vm::value::IteratorState;
                            IteratorState::Array {
                                array_handle: handle,
                                index: 0,
                            }
                        }
                        _ => {
                            return Err(VMError::TypeError(format!(
                                "Cannot iterate over {:?}",
                                val
                            )))
                        }
                    };
                    self.push(Value::Iterator(Rc::new(RefCell::new(iter_state))))?;
                }

                Opcode::Next => {
                    let offset = instr.read_u16();
                    let iter_val = self.peek(0)?;

                    let next_val = match iter_val {
                        Value::Iterator(state_rc) => {
                            use crate::vm::value::IteratorState;
                            let mut state = state_rc.borrow_mut();
                            match *state {
                                IteratorState::Range {
                                    ref mut current,
                                    end,
                                    inclusive,
                                } => {
                                    if (inclusive && *current <= end)
                                        || (!inclusive && *current < end)
                                    {
                                        let val = *current;
                                        *current += 1;
                                        Some(Value::Int(val))
                                    } else {
                                        None
                                    }
                                }
                                IteratorState::Array {
                                    array_handle,
                                    ref mut index,
                                } => {
                                    if let Some(val) = self
                                        .store
                                        .with_array(array_handle, |a| a.get(*index).cloned())
                                        .flatten()
                                    {
                                        *index += 1;
                                        Some(val)
                                    } else {
                                        None
                                    }
                                }
                            }
                        }
                        _ => {
                            return Err(VMError::InvalidOperation(
                                "Next called on non-iterator".into(),
                            ))
                        }
                    };

                    if let Some(val) = next_val {
                        self.push(val)?;
                    } else {
                        self.frame_mut().ip += offset as usize;
                    }
                }

                Opcode::MakeGaussian => {
                    let std = self.pop()?;
                    let mean = self.pop()?;
                    let mean_f = value_to_adfloat(&mean)?;
                    let std_f = value_to_adfloat(&std)?;
                    self.push(Value::Gaussian {
                        mean: mean_f,
                        std: std_f,
                    })?;
                }

                Opcode::MakeUniform => {
                    let max = self.pop()?;
                    let min = self.pop()?;
                    let min_f = value_to_adfloat(&min)?;
                    let max_f = value_to_adfloat(&max)?;
                    self.push(Value::Uniform {
                        min: min_f,
                        max: max_f,
                    })?;
                }

                Opcode::Closure => {
                    let const_idx = instr.read_u16();
                    let prototype = self.get_constant(const_idx).clone();

                    match prototype {
                        Value::BytecodeFunction { upvalue_count, .. } => {
                            let mut upvalues = Vec::with_capacity(upvalue_count);
                            let mut offset = 2;

                            for _ in 0..upvalue_count {
                                if offset + 1 >= instr.operands.len() {
                                    return Err(VMError::InvalidOperation(
                                        "Malformed closure instruction".into(),
                                    ));
                                }
                                let is_local = instr.operands[offset] != 0;
                                let index = instr.operands[offset + 1] as usize;
                                offset += 2;

                                if is_local {
                                    let slot = self.frame().slot_base + index;
                                    upvalues.push(self.capture_upvalue(slot));
                                } else {
                                    if let Value::BytecodeClosure {
                                        upvalues: ref current_upvalues,
                                        ..
                                    } = self.frame().closure
                                    {
                                        if index >= current_upvalues.len() {
                                            return Err(VMError::IndexOutOfBounds);
                                        }
                                        upvalues.push(current_upvalues[index].clone());
                                    } else {
                                        return Err(VMError::InvalidOperation(
                                            "Closure instruction in non-closure frame".into(),
                                        ));
                                    }
                                }
                            }

                            let closure = Value::BytecodeClosure {
                                function: Box::new(prototype),
                                upvalues,
                            };
                            self.push(closure)?;
                        }
                        _ => {
                            return Err(VMError::InvalidOperation(
                                "Closure operand must be a function".into(),
                            ))
                        }
                    }
                }

                Opcode::GetUpvalue => {
                    let slot = instr.read_u16() as usize;

                    let upvalue =
                        if let Value::BytecodeClosure { ref upvalues, .. } = self.frame().closure {
                            upvalues[slot].clone()
                        } else {
                            return Err(VMError::InvalidOperation(
                                "GetUpvalue in non-closure frame".into(),
                            ));
                        };

                    let value = match &*upvalue.borrow() {
                        Upvalue::Open(idx) => self.stack[*idx].clone(),
                        Upvalue::Closed(val) => val.clone(),
                    };
                    self.push(value)?;
                }

                Opcode::SetUpvalue => {
                    let slot = instr.read_u16() as usize;
                    let value = self.peek(0)?.clone();

                    let upvalue =
                        if let Value::BytecodeClosure { ref upvalues, .. } = self.frame().closure {
                            upvalues[slot].clone()
                        } else {
                            return Err(VMError::InvalidOperation(
                                "SetUpvalue in non-closure frame".into(),
                            ));
                        };

                    let mut upvalue_guard = upvalue.borrow_mut();
                    match &mut *upvalue_guard {
                        Upvalue::Open(idx) => self.stack[*idx] = value,
                        Upvalue::Closed(val) => *val = value,
                    }
                }

                Opcode::CloseUpvalue => {
                    self.close_upvalues(self.stack.len() - 1);
                    self.pop()?;
                }

                Opcode::MakeBernoulli => {
                    let p = self.pop()?;
                    let p_f = value_to_adfloat(&p)?;
                    self.push(Value::Bernoulli { p: p_f })?;
                }

                Opcode::MakeBeta => {
                    let beta = self.pop()?;
                    let alpha = self.pop()?;
                    let alpha_f = value_to_adfloat(&alpha)?;
                    let beta_f = value_to_adfloat(&beta)?;
                    self.push(Value::Beta {
                        alpha: alpha_f,
                        beta: beta_f,
                    })?;
                }

                Opcode::Sample => {
                    let dist = self.pop()?;

                    let val = if let Some(ctx) = &mut self.prob_context {
                        ctx.sample_counter += 1;
                        let key = format!("sample_{}", ctx.sample_counter);
                        let mode = ctx.mode;
                        let _tape_id = ctx.tape_id;

                        match mode {
                            InferenceMode::Sampling => {
                                if let Some(trace_val) = ctx.trace.get(&key) {
                                    use crate::vm::prob_runtime::get_distribution_log_pdf_ad;
                                    if let Value::Float(f) = trace_val {
                                        let log_prob = get_distribution_log_pdf_ad(&dist, f)?;
                                        ctx.accumulated_log_prob =
                                            ctx.accumulated_log_prob.clone() + log_prob;
                                    }
                                    trace_val.clone()
                                } else {
                                    let sampled = sample_distribution(&dist, &mut self.rng)?;
                                    ctx.trace.insert(key.clone(), sampled.clone());

                                    use crate::vm::prob_runtime::get_distribution_log_pdf_ad;
                                    if let Value::Float(f) = &sampled {
                                        let log_prob = get_distribution_log_pdf_ad(&dist, f)?;
                                        ctx.accumulated_log_prob =
                                            ctx.accumulated_log_prob.clone() + log_prob;
                                    }
                                    sampled
                                }
                            }
                            InferenceMode::Gradient { tape_id } => {
                                let trace_val = ctx.trace.get(&key).cloned().ok_or(
                                    VMError::InvalidOperation(
                                        "Missing trace in Gradient mode".into(),
                                    ),
                                )?;

                                let val_node = match trace_val {
                                    Value::Float(ad) => match ad {
                                        ADFloat::Concrete(v) => {
                                            let new_node = ADFloat::new_input(v, tape_id);
                                            if let Some(id) = new_node.node_id() {
                                                ctx.param_nodes.insert(key.clone(), id);
                                            }
                                            new_node
                                        }
                                        _ => ad,
                                    },
                                    _ => {
                                        return Err(VMError::TypeError(
                                            "Sample must be float in gradient mode".into(),
                                        ))
                                    }
                                };

                                use crate::vm::prob_runtime::get_distribution_log_pdf_ad;
                                let score = get_distribution_log_pdf_ad(&dist, &val_node)?;
                                ctx.accumulated_log_prob = ctx.accumulated_log_prob.clone() + score;

                                Value::Float(val_node)
                            }
                            InferenceMode::Guide { tape_id: _ } => {
                                if let Some(trace_val) = ctx.trace.get(&key).cloned() {
                                    let val_ad = match &trace_val {
                                        Value::Float(ad) => ad.clone(),
                                        Value::Int(i) => ADFloat::Concrete(*i as f64),
                                        _ => {
                                            return Err(VMError::TypeError(
                                                "Replay value must be number".into(),
                                            ))
                                        }
                                    };

                                    use crate::vm::prob_runtime::get_distribution_log_pdf_ad;
                                    let score = get_distribution_log_pdf_ad(&dist, &val_ad)?;
                                    ctx.accumulated_log_prob =
                                        ctx.accumulated_log_prob.clone() + score;
                                    trace_val
                                } else {
                                    use crate::vm::prob_runtime::{
                                        get_distribution_log_pdf_ad, get_distribution_sample,
                                    };

                                    let val_ad =
                                        get_distribution_sample(&dist, &mut self.rng, mode)?;
                                    let log_q = get_distribution_log_pdf_ad(&dist, &val_ad)?;
                                    let val_result = Value::Float(val_ad);

                                    ctx.accumulated_log_prob =
                                        ctx.accumulated_log_prob.clone() + log_q;
                                    ctx.trace.insert(key, val_result.clone());

                                    val_result
                                }
                            }
                        }
                    } else {
                        sample_distribution(&dist, &mut self.rng)?
                    };
                    self.push(val)?;
                }

                Opcode::Observe => {
                    let value = self.pop()?;
                    let dist = self.pop()?;

                    if let Some(ctx) = &mut self.prob_context {
                        use crate::vm::prob_runtime::get_distribution_log_pdf_ad;
                        let val_ad = value_to_adfloat(&value)?;
                        let log_prob = get_distribution_log_pdf_ad(&dist, &val_ad)?;
                        ctx.accumulated_log_prob = ctx.accumulated_log_prob.clone() + log_prob;
                    } else {
                        let log_prob = self.log_prob(&dist, &value)?;
                        self.log_weight += log_prob;
                    }
                    self.push(Value::Unit)?;
                }

                Opcode::Call => {
                    let arg_count = instr.read_u8() as usize;
                    self.call_value(arg_count)?;
                }

                Opcode::Return => {
                    let result = self.pop()?;

                    let base = self.frame().slot_base;

                    self.close_upvalues(base);

                    self.frames.pop();

                    if self.frames.len() == target_depth {
                        return Ok(result);
                    }

                    if self.frames.is_empty() {
                        return Ok(result);
                    }

                    self.current_frame = self.frames.len() - 1;

                    if base > 0 {
                        if base <= self.stack.len() {
                            if base >= 1 {
                                let return_slot = base - 1;
                                self.stack[return_slot] = result;
                                self.stack.truncate(return_slot + 1);
                            } else {
                                return Err(VMError::StackUnderflow);
                            }
                        } else {
                            return Err(VMError::StackUnderflow);
                        }
                    } else {
                        self.push(result)?;
                    }
                }

                Opcode::CloseScope => {
                    let count = instr.read_u16() as usize;
                    let result = self.pop()?;

                    let target_len = self.stack.len().saturating_sub(count);

                    for i in (target_len..self.stack.len()).rev() {
                        let value = self.stack[i].clone();
                        self.perform_cleanup(&value)?;
                    }

                    self.close_upvalues(target_len);
                    self.stack.truncate(target_len);
                    self.push(result)?;
                }

                Opcode::DefineMethod => {
                    let type_name_idx = u16::from_le_bytes([instr.operands[0], instr.operands[1]]);
                    let method_name_idx =
                        u16::from_le_bytes([instr.operands[2], instr.operands[3]]);

                    let type_name =
                        match &self.chunks[chunk_index].constants[type_name_idx as usize] {
                            Value::String(s) => s.to_string(),
                            _ => return Err(VMError::InvalidOperation("Invalid type name".into())),
                        };
                    let method_name = match &self.chunks[chunk_index].constants
                        [method_name_idx as usize]
                    {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid method name".into())),
                    };

                    let func = self.pop()?;
                    self.methods
                        .entry(type_name)
                        .or_default()
                        .insert(method_name, func);
                }

                Opcode::MakeStruct => {
                    let name_idx = u16::from_le_bytes([instr.operands[0], instr.operands[1]]);
                    let field_count = instr.operands[2] as usize;

                    let name = match &self.chunks[chunk_index].constants[name_idx as usize] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid struct name".into())),
                    };

                    let mut fields = HashMap::new();
                    for _ in 0..field_count {
                        let val = self.pop()?;
                        let key_val = self.pop()?;
                        let key = match key_val {
                            Value::String(s) => s.to_string(),
                            _ => {
                                return Err(VMError::TypeError("Struct key must be string".into()))
                            }
                        };
                        fields.insert(key, val);
                    }

                    let handle = self.store.alloc_dict(fields);
                    self.push(Value::Struct {
                        name,
                        fields: handle,
                    })?;
                }

                Opcode::GetField => {
                    let name_idx = instr.read_u16();
                    let field_name = match &self.chunks[chunk_index].constants[name_idx as usize] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid field name".into())),
                    };

                    let obj = self.pop()?;
                    match obj {
                        Value::Struct { fields, .. } => {
                            if let Some(val) = self
                                .store
                                .with_dict(fields, |d| d.get(&field_name).cloned())
                                .flatten()
                            {
                                self.push(val)?;
                            } else {
                                return Err(VMError::UndefinedField(format!(
                                    "Field '{}' not found",
                                    field_name
                                )));
                            }
                        }
                        _ => {
                            return Err(VMError::TypeError(
                                "Expected struct for field access".into(),
                            ))
                        }
                    }
                }

                Opcode::MakeEnum => {
                    let name_idx = u16::from_le_bytes([instr.operands[0], instr.operands[1]]);
                    let var_idx = u16::from_le_bytes([instr.operands[2], instr.operands[3]]);
                    let count = instr.operands[4] as usize;

                    let name = match &self.chunks[chunk_index].constants[name_idx as usize] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid enum name".into())),
                    };
                    let variant = match &self.chunks[chunk_index].constants[var_idx as usize] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid variant name".into())),
                    };

                    let mut fields = Vec::with_capacity(count);
                    for _ in 0..count {
                        fields.push(self.pop()?);
                    }
                    fields.reverse();

                    self.push(Value::Enum {
                        name,
                        variant,
                        fields: Rc::new(fields),
                    })?;
                }

                Opcode::IsVariant => {
                    let var_idx = instr.read_u16();
                    let variant = match &self.chunks[chunk_index].constants[var_idx as usize] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid variant name".into())),
                    };

                    let obj = self.peek(0)?;
                    let is_match = match obj {
                        Value::Enum { variant: v, .. } => v == &variant,
                        _ => false,
                    };
                    self.push(Value::Bool(is_match))?;
                }

                Opcode::UnpackEnum => {
                    let obj = self.pop()?;
                    match obj {
                        Value::Enum { fields, .. } => {
                            for f in fields.iter() {
                                self.push(f.clone())?;
                            }
                        }
                        _ => {
                            return Err(VMError::TypeError(
                                format!("UnpackEnum on non-enum: {}", obj.type_name()).into(),
                            ))
                        }
                    }
                }

                Opcode::MethodCall => {
                    let method_name_idx = instr.read_u16();
                    let arg_count = instr.operands[2] as usize;

                    let method_name = match &self.chunks[chunk_index].constants
                        [method_name_idx as usize]
                    {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::InvalidOperation("Invalid method name".into())),
                    };

                    let receiver = self.peek(arg_count)?.clone();

                    match (&receiver, method_name.as_str()) {
                        (Value::Array(arr), "len") => {
                            let len = self.store.with_array(*arr, |a| a.len() as i64).unwrap_or(0);
                            for _ in 0..arg_count {
                                self.pop()?;
                            }
                            self.pop()?;
                            self.push(Value::Int(len))?;
                            continue;
                        }
                        (Value::Gaussian { .. }, "sample")
                        | (Value::Uniform { .. }, "sample")
                        | (Value::Bernoulli { .. }, "sample")
                        | (Value::Beta { .. }, "sample") => {
                            for _ in 0..arg_count {
                                self.pop()?;
                            }
                            let dist = self.pop()?;
                            let sampled = sample_distribution(&dist, &mut self.rng)?;
                            self.push(sampled)?;
                            continue;
                        }
                        (Value::Gaussian { .. }, "clone")
                        | (Value::Uniform { .. }, "clone")
                        | (Value::Bernoulli { .. }, "clone")
                        | (Value::Beta { .. }, "clone") => {
                            for _ in 0..arg_count {
                                self.pop()?;
                            }
                            let obj = self.pop()?;
                            self.push(obj)?;
                            continue;
                        }
                        (Value::Map(map_rc), "insert") => {
                            if arg_count != 2 {
                                return Err(VMError::ArityMismatch {
                                    expected: 2,
                                    found: arg_count,
                                });
                            }
                            let value = self.pop()?;
                            let key_val = self.pop()?;
                            let key = match key_val {
                                Value::String(s) => s.to_string(),
                                _ => {
                                    return Err(VMError::TypeError("Map key must be string".into()))
                                }
                            };
                            self.pop()?;

                            self.store.with_dict_mut(*map_rc, |map| {
                                map.insert(key, value);
                            });
                            self.push(Value::Unit)?;
                            continue;
                        }
                        (Value::Map(map_rc), "get") => {
                            if arg_count != 1 {
                                return Err(VMError::ArityMismatch {
                                    expected: 1,
                                    found: arg_count,
                                });
                            }
                            let key_val = self.pop()?;
                            let key = match key_val {
                                Value::String(s) => s.to_string(),
                                _ => {
                                    return Err(VMError::TypeError("Map key must be string".into()))
                                }
                            };
                            self.pop()?;

                            let val = self
                                .store
                                .with_dict(*map_rc, |map| map.get(&key).cloned())
                                .flatten()
                                .unwrap_or(Value::Unit);
                            self.push(val)?;
                            continue;
                        }
                        _ => {}
                    }

                    let type_name = match &receiver {
                        Value::Int(_) => "Int".to_string(),
                        Value::Float(_) => "Float".to_string(),
                        Value::Bool(_) => "Bool".to_string(),
                        Value::String(_) => "String".to_string(),
                        Value::Struct { name, .. } => name.clone(),
                        Value::Enum { name, .. } => name.clone(),
                        Value::Array(_) => "Array".to_string(),
                        Value::Unit => "Unit".to_string(),
                        Value::Gaussian { .. } => "Gaussian".to_string(),
                        Value::Uniform { .. } => "Uniform".to_string(),
                        Value::Bernoulli { .. } => "Bernoulli".to_string(),
                        Value::Beta { .. } => "Beta".to_string(),
                        _ => {
                            return Err(VMError::TypeError(format!(
                                "Method call not supported on {:?}",
                                receiver
                            )))
                        }
                    };

                    let method = if let Some(methods) = self.methods.get(&type_name) {
                        methods.get(&method_name).cloned()
                    } else {
                        None
                    };

                    if let Some(m) = method {
                        let receiver_idx = self.stack.len() - 1 - arg_count;
                        self.stack.insert(receiver_idx, m);

                        self.call_value(arg_count + 1)?;
                    } else {
                        return Err(VMError::InvalidOperation(format!(
                            "Method '{}' not found for type '{}'",
                            method_name, type_name
                        )));
                    }
                }

                Opcode::Print => {
                    let value = self.pop()?;
                    println!("{}", value);
                }

                Opcode::Spawn => {
                    use crate::vm::deep_clone::{
                        sendable_to_value, value_to_sendable, SendableValue,
                    };
                    use std::sync::{Arc, Mutex};

                    let arg_count = instr.operands[0] as usize;
                    let mut args = Vec::with_capacity(arg_count);
                    for _ in 0..arg_count {
                        args.push(self.pop()?);
                    }
                    args.reverse();

                    let callee = self.pop()?;
                    self.task_id_counter += 1;
                    let task_id = self.task_id_counter;

                    let is_callable = matches!(
                        callee,
                        Value::BytecodeFunction { .. }
                            | Value::BytecodeClosure { .. }
                            | Value::NativeFunction { .. }
                            | Value::Builtin(_)
                            | Value::Function { .. }
                    );

                    if is_callable {
                        let sendable_func = value_to_sendable(&callee, &self.store)
                            .map_err(|e| VMError::InvalidOperation(format!("spawn: {}", e)))?;

                        let sendable_args: Vec<SendableValue> = args
                            .iter()
                            .map(|v| value_to_sendable(v, &self.store))
                            .collect::<Result<_, _>>()
                            .map_err(|e| VMError::InvalidOperation(format!("spawn: {}", e)))?;

                        let mut sendable_chunks = Vec::new();
                        for chunk in &self.chunks {
                            let mut sendable_constants = Vec::new();
                            for constant in &chunk.constants {
                                // Some constants may be closure values or complex ones, convert safely
                                sendable_constants.push(
                                    value_to_sendable(constant, &self.store).map_err(|e| {
                                        VMError::InvalidOperation(format!(
                                            "spawn chunk error: {}",
                                            e
                                        ))
                                    })?,
                                );
                            }
                            sendable_chunks.push((
                                chunk.name.clone(),
                                chunk.code.clone(),
                                sendable_constants,
                            ));
                        }

                        let join_handle =
                            std::thread::spawn(move || -> Box<dyn std::any::Any + Send> {
                                let mut child_vm = BytecodeVM::new();
                                for (name, code, s_consts) in sendable_chunks {
                                    let mut constants = Vec::new();
                                    for sc in s_consts {
                                        constants.push(sendable_to_value(sc, &mut child_vm.store));
                                    }
                                    child_vm.chunks.push(crate::bytecode::chunk::Chunk {
                                        name,
                                        code,
                                        constants,
                                    });
                                }

                                let func = sendable_to_value(sendable_func, &mut child_vm.store);
                                let re_args: Vec<Value> = sendable_args
                                    .into_iter()
                                    .map(|s| sendable_to_value(s, &mut child_vm.store))
                                    .collect();

                                let result = child_vm.call_function(&func, re_args);

                                let sendable_result: Result<SendableValue, String> = match result {
                                    Ok(val) => value_to_sendable(&val, &child_vm.store)
                                        .map_err(|e| format!("spawn result: {}", e)),
                                    Err(e) => Err(format!("{:?}", e)),
                                };
                                Box::new(sendable_result)
                            });

                        let handle = Arc::new(Mutex::new(Some(join_handle)));
                        let result_slot = Arc::new(Mutex::new(None));
                        self.push(Value::Future {
                            task_id,
                            handle,
                            result: result_slot,
                        })?;
                    } else {
                        let result_slot = Arc::new(Mutex::new(Some(Ok(callee))));
                        let handle = Arc::new(Mutex::new(None));
                        self.push(Value::Future {
                            task_id,
                            handle,
                            result: result_slot,
                        })?;
                    }
                }

                Opcode::Await => {
                    use crate::vm::deep_clone::{sendable_to_value, SendableValue};

                    let future_val = self.pop()?;
                    match future_val {
                        Value::Future { handle, result, .. } => {
                            {
                                let lock = result.lock().unwrap();
                                if let Some(ref r) = *lock {
                                    match r {
                                        Ok(v) => {
                                            self.push(v.clone())?;
                                            continue;
                                        }
                                        Err(e) => {
                                            return Err(VMError::InvalidOperation(format!(
                                                "Async task failed: {}",
                                                e
                                            )))
                                        }
                                    }
                                }
                            }

                            let join_handle = {
                                let mut lock = handle.lock().unwrap();
                                lock.take()
                            };

                            if let Some(jh) = join_handle {
                                let thread_result = jh.join().map_err(|_| {
                                    VMError::InvalidOperation("Spawned task panicked".into())
                                })?;
                                let sendable_result = thread_result
                                    .downcast::<Result<SendableValue, String>>()
                                    .map_err(|_| {
                                        VMError::InvalidOperation(
                                            "Internal error: unexpected task result type".into(),
                                        )
                                    })?;
                                match *sendable_result {
                                    Ok(sval) => {
                                        let val = sendable_to_value(sval, &mut self.store);
                                        let mut lock = result.lock().unwrap();
                                        *lock = Some(Ok(val.clone()));
                                        self.push(val)?;
                                    }
                                    Err(e) => {
                                        return Err(VMError::InvalidOperation(format!(
                                            "Async task failed: {}",
                                            e
                                        )))
                                    }
                                }
                            } else {
                                let lock = result.lock().unwrap();
                                match lock.as_ref() {
                                    Some(Ok(v)) => {
                                        self.push(v.clone())?;
                                    }
                                    Some(Err(e)) => {
                                        return Err(VMError::InvalidOperation(format!(
                                            "Async task failed: {}",
                                            e
                                        )))
                                    }
                                    None => {
                                        return Err(VMError::InvalidOperation(
                                            "Future has no result".into(),
                                        ))
                                    }
                                }
                            }
                        }
                        _ => {
                            return Err(VMError::TypeError(format!(
                                "'await' requires a Future, got {}",
                                future_val.type_name()
                            )))
                        }
                    }
                }

                Opcode::Cast => {
                    let type_idx = u16::from_le_bytes([instr.operands[0], instr.operands[1]]);
                    let type_name = match &self.chunks[chunk_index].constants[type_idx as usize] {
                        Value::String(s) => s.to_string(),
                        _ => {
                            return Err(VMError::InvalidOperation(
                                "Invalid cast type string".into(),
                            ))
                        }
                    };

                    let val = self.pop()?;

                    let casted = match (val.clone(), type_name.as_str()) {
                        (Value::Int(i), "Float") => Value::Float(ADFloat::Concrete(i as f64)),
                        (Value::Float(f), "Int") => Value::Int(f.value() as i64),
                        (Value::Int(i), "String") => Value::String(Rc::new(i.to_string())),
                        (Value::Float(f), "String") => {
                            Value::String(Rc::new(f.value().to_string()))
                        }
                        (Value::Bool(b), "String") => Value::String(Rc::new(b.to_string())),
                        _ => val,
                    };
                    self.push(casted)?;
                }

                Opcode::TryMethodCall | Opcode::MakeTraitObject => {
                    return Err(VMError::InvalidOperation(format!(
                        "Opcode {:?} not yet implemented",
                        instr.opcode
                    )));
                }
            }
        }

        if self.stack.is_empty() {
            Ok(Value::Unit)
        } else {
            self.pop()
        }
    }

    fn binary_mod(&mut self) -> VMResult<()> {
        let b = self.pop()?;
        let a = self.pop()?;
        match (&a, &b) {
            (Value::Int(x), Value::Int(y)) => {
                if *y == 0 {
                    return Err(VMError::DivisionByZero);
                }
                self.push(Value::Int(x % y))
            }
            _ => {
                let x = value_to_adfloat(&a)?.value();
                let y = value_to_adfloat(&b)?.value();
                self.push(Value::Float(ADFloat::Concrete(x % y)))
            }
        }
    }

    fn binary_add(&mut self) -> VMResult<()> {
        let b = self.pop()?;
        let a = self.pop()?;

        match (&a, &b) {
            (Value::Gaussian { mean: m1, std: s1 }, Value::Gaussian { mean: m2, std: s2 }) => {
                let new_mean = ADFloat::Concrete(m1.value() + m2.value());
                let new_std = ADFloat::Concrete((s1.value().powi(2) + s2.value().powi(2)).sqrt());
                self.push(Value::Gaussian {
                    mean: new_mean,
                    std: new_std,
                })
            }
            (Value::Int(x), Value::Int(y)) => self.push(Value::Int(x + y)),
            (Value::Float(x), Value::Float(y)) => {
                self.push(Value::Float(ADFloat::Concrete(x.value() + y.value())))
            }
            (Value::Int(x), Value::Float(y)) => {
                self.push(Value::Float(ADFloat::Concrete(*x as f64 + y.value())))
            }
            (Value::Float(x), Value::Int(y)) => {
                self.push(Value::Float(ADFloat::Concrete(x.value() + *y as f64)))
            }
            (Value::String(x), Value::String(y)) => {
                let mut s = x.to_string();
                s.push_str(y);
                self.push(Value::String(Rc::new(s)))
            }
            (Value::Float(x), Value::Bool(y)) => {
                let yf = if *y { 1.0 } else { 0.0 };
                self.push(Value::Float(ADFloat::Concrete(x.value() + yf)))
            }
            (Value::Bool(x), Value::Float(y)) => {
                let xf = if *x { 1.0 } else { 0.0 };
                self.push(Value::Float(ADFloat::Concrete(xf + y.value())))
            }
            (Value::Int(x), Value::Bool(y)) => {
                let yi = if *y { 1 } else { 0 };
                self.push(Value::Int(x + yi))
            }
            (Value::Bool(x), Value::Int(y)) => {
                let xi = if *x { 1 } else { 0 };
                self.push(Value::Int(xi + y))
            }
            (Value::String(x), other) => {
                let mut s = x.to_string();
                s.push_str(&format!("{}", other));
                self.push(Value::String(Rc::new(s)))
            }
            (other, Value::String(x)) => {
                let mut s = format!("{}", other);
                s.push_str(x);
                self.push(Value::String(Rc::new(s)))
            }
            _ => Err(VMError::TypeError(format!(
                "Cannot add {} and {}",
                a.type_name(),
                b.type_name()
            ))),
        }
    }

    fn binary_sub(&mut self) -> VMResult<()> {
        let b = self.pop()?;
        let a = self.pop()?;
        match (&a, &b) {
            (Value::Int(x), Value::Int(y)) => self.push(Value::Int(x - y)),
            (Value::Float(x), Value::Float(y)) => {
                self.push(Value::Float(ADFloat::Concrete(x.value() - y.value())))
            }
            (Value::Int(x), Value::Float(y)) => {
                self.push(Value::Float(ADFloat::Concrete(*x as f64 - y.value())))
            }
            (Value::Float(x), Value::Int(y)) => {
                self.push(Value::Float(ADFloat::Concrete(x.value() - *y as f64)))
            }
            _ => Err(VMError::TypeError(format!(
                "Cannot subtract {} by {}",
                a.type_name(),
                b.type_name()
            ))),
        }
    }

    fn binary_mul(&mut self) -> VMResult<()> {
        let b = self.pop()?;
        let a = self.pop()?;
        match (&a, &b) {
            (Value::Int(x), Value::Int(y)) => self.push(Value::Int(x * y)),
            (Value::Float(x), Value::Float(y)) => {
                self.push(Value::Float(ADFloat::Concrete(x.value() * y.value())))
            }
            (Value::Int(x), Value::Float(y)) => {
                self.push(Value::Float(ADFloat::Concrete(*x as f64 * y.value())))
            }
            (Value::Float(x), Value::Int(y)) => {
                self.push(Value::Float(ADFloat::Concrete(x.value() * *y as f64)))
            }
            _ => Err(VMError::TypeError(format!(
                "Cannot multiply {} and {}",
                a.type_name(),
                b.type_name()
            ))),
        }
    }

    fn binary_div(&mut self) -> VMResult<()> {
        let b = self.pop()?;
        let a = self.pop()?;
        match (&a, &b) {
            (Value::Int(x), Value::Int(y)) => {
                if *y == 0 {
                    return Err(VMError::DivisionByZero);
                }
                self.push(Value::Int(x / y))
            }
            (Value::Float(x), Value::Float(y)) => {
                self.push(Value::Float(ADFloat::Concrete(x.value() / y.value())))
            }
            (Value::Int(x), Value::Float(y)) => {
                self.push(Value::Float(ADFloat::Concrete(*x as f64 / y.value())))
            }
            (Value::Float(x), Value::Int(y)) => {
                self.push(Value::Float(ADFloat::Concrete(x.value() / *y as f64)))
            }
            _ => Err(VMError::TypeError(format!(
                "Cannot divide {} by {}",
                a.type_name(),
                b.type_name()
            ))),
        }
    }

    fn unary_neg(&mut self) -> VMResult<()> {
        let a = self.pop()?;
        match a {
            Value::Int(x) => self.push(Value::Int(-x)),
            Value::Float(x) => self.push(Value::Float(ADFloat::Concrete(-x.value()))),
            _ => Err(VMError::TypeError(format!(
                "Cannot negate {}",
                a.type_name()
            ))),
        }
    }

    fn comparison_op<F>(&mut self, op: F) -> VMResult<()>
    where
        F: Fn(f64, f64) -> bool,
    {
        let b = self.pop()?;
        let a = self.pop()?;

        let result = match (&a, &b) {
            (Value::Int(x), Value::Int(y)) => op(*x as f64, *y as f64),
            (Value::Float(x), Value::Float(y)) => op(x.value(), y.value()),
            (Value::Int(x), Value::Float(y)) => op(*x as f64, y.value()),
            (Value::Float(x), Value::Int(y)) => op(x.value(), *y as f64),
            _ => {
                return Err(VMError::TypeError(format!(
                    "Cannot compare {} and {}",
                    a.type_name(),
                    b.type_name()
                )))
            }
        };

        self.push(Value::Bool(result))
    }

    fn call_value(&mut self, arg_count: usize) -> VMResult<()> {
        let callee_idx = self.stack.len() - 1 - arg_count;
        let callee = self.stack[callee_idx].clone();

        match callee {
            Value::BytecodeFunction {
                chunk_index, arity, ..
            } => {
                if arity != arg_count {
                    return Err(VMError::ArityMismatch {
                        expected: arity,
                        found: arg_count,
                    });
                }

                if self.frames.len() >= FRAMES_MAX {
                    return Err(VMError::StackOverflow);
                }

                let slot_base = self.stack.len() - arg_count;

                let frame = CallFrame {
                    chunk_index,
                    ip: 0,
                    slot_base,
                    closure: self.stack[callee_idx].clone(),
                };

                self.frames.push(frame);
                self.current_frame = self.frames.len() - 1;

                Ok(())
            }
            Value::BytecodeClosure {
                ref function,
                upvalues: _,
            } => {
                if let Value::BytecodeFunction {
                    chunk_index, arity, ..
                } = **function
                {
                    if arity != arg_count {
                        return Err(VMError::ArityMismatch {
                            expected: arity,
                            found: arg_count,
                        });
                    }

                    if self.frames.len() >= FRAMES_MAX {
                        return Err(VMError::StackOverflow);
                    }

                    let slot_base = self.stack.len() - arg_count;

                    let frame = CallFrame {
                        chunk_index,
                        ip: 0,
                        slot_base,
                        closure: callee.clone(),
                    };

                    self.frames.push(frame);
                    self.current_frame = self.frames.len() - 1;
                    Ok(())
                } else {
                    Err(VMError::InvalidOperation(
                        "Closure does not contain a function".into(),
                    ))
                }
            }
            Value::Function { .. } => Err(VMError::InvalidOperation(
                "AST function calls not supported in bytecode VM without compilation".into(),
            )),
            Value::NativeFunction { .. } => Err(VMError::InvalidOperation(
                "Native function calls not yet supported in bytecode VM".into(),
            )),
            Value::Builtin(ref name) => {
                let name = name.clone();
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count {
                    args.push(self.pop()?);
                }
                args.reverse();

                self.pop()?;

                let result = self.call_builtin(&name, args)?;
                self.push(result)?;
                Ok(())
            }
            _ => Err(VMError::TypeError(format!(
                "Type {} is not callable",
                callee.type_name()
            ))),
        }
    }

    fn call_builtin(&mut self, name: &str, args: Vec<Value>) -> VMResult<Value> {
        match name {
            "infer" => {
                if args.len() != 2 {
                    return Err(VMError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                    });
                }

                let num_samples = match args[0] {
                    Value::Int(n) => n as usize,
                    _ => {
                        return Err(VMError::TypeError(
                            "infer arg 1 must be Int (num_samples)".into(),
                        ))
                    }
                };

                let model_fn = args[1].clone();
                self.run_inference(num_samples, model_fn)
            }
            "len" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                match &args[0] {
                    Value::Array(arr) => Ok(Value::Int(
                        self.store.with_array(*arr, |a| a.len() as i64).unwrap_or(0),
                    )),
                    Value::String(s) => Ok(Value::Int(s.len() as i64)),
                    _ => Err(VMError::TypeError("len requires Array or String".into())),
                }
            }
            "gaussian" => {
                if args.len() != 2 {
                    return Err(VMError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                    });
                }
                let mean = value_to_adfloat(&args[0])?;
                let std = value_to_adfloat(&args[1])?;
                Ok(Value::Gaussian { mean, std })
            }
            "param" => {
                if args.len() != 2 {
                    return Err(VMError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                    });
                }

                let name = match &args[0] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(VMError::TypeError("param name must be string".into())),
                };
                let init_val = match &args[1] {
                    Value::Float(f) => f.value(),
                    Value::Int(i) => *i as f64,
                    _ => return Err(VMError::TypeError("param init value must be number".into())),
                };

                let val = if let Some(ctx) = self.prob_context.as_mut() {
                    use crate::vm::prob_runtime::register_param;
                    register_param(ctx, &name, init_val)
                } else {
                    ADFloat::Concrete(init_val)
                };

                Ok(Value::Float(val))
            }
            "uniform" => {
                if args.len() != 2 {
                    return Err(VMError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                    });
                }
                let min = value_to_adfloat(&args[0])?;
                let max = value_to_adfloat(&args[1])?;
                Ok(Value::Uniform { min, max })
            }
            "bernoulli" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                let p = value_to_adfloat(&args[0])?;
                Ok(Value::Bernoulli { p })
            }
            "beta" => {
                if args.len() != 2 {
                    return Err(VMError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                    });
                }
                let alpha = value_to_adfloat(&args[0])?;
                let beta = value_to_adfloat(&args[1])?;
                Ok(Value::Beta { alpha, beta })
            }
            "Map" => {
                if args.len() != 0 {
                    return Err(VMError::ArityMismatch {
                        expected: 0,
                        found: args.len(),
                    });
                }
                Ok(Value::Map(self.store.alloc_dict(HashMap::new())))
            }
            "sample" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                sample_distribution(&args[0], &mut self.rng)
            }
            "observe" => {
                if args.len() != 2 {
                    return Err(VMError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                    });
                }
                let dist = &args[0];
                let value = &args[1];
                let log_prob = self.log_prob(dist, value)?;
                self.log_weight += log_prob;
                Ok(Value::Unit)
            }
            "println" => {
                for arg in &args {
                    print!("{}", arg);
                }
                println!();
                Ok(Value::Unit)
            }
            "sin" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                let val = value_to_adfloat(&args[0])?;
                Ok(Value::Float(val.sin()))
            }
            "cos" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                let val = value_to_adfloat(&args[0])?;
                Ok(Value::Float(val.cos()))
            }
            "tan" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                let val = value_to_adfloat(&args[0])?;
                Ok(Value::Float(val.tan()))
            }
            "exp" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                let val = value_to_adfloat(&args[0])?;
                Ok(Value::Float(val.exp()))
            }
            "ln" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                let val = value_to_adfloat(&args[0])?;
                Ok(Value::Float(val.ln()))
            }
            "sqrt" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                let val = value_to_adfloat(&args[0])?;
                Ok(Value::Float(val.sqrt()))
            }
            "pow" => {
                if args.len() != 2 {
                    return Err(VMError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                    });
                }
                let base = value_to_adfloat(&args[0])?;
                let exp = match &args[1] {
                    Value::Float(f) => f.value(),
                    Value::Int(i) => *i as f64,
                    _ => return Err(VMError::TypeError("pow exponent must be number".into())),
                };
                Ok(Value::Float(base.powf(exp)))
            }
            "Array::push" => {
                if args.len() != 2 {
                    return Err(VMError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                    });
                }

                let new_arr = if let Value::Array(handle) = &args[0] {
                    let mut new_vec = self.store.clone_array(*handle).unwrap_or_default();
                    new_vec.push(args[1].clone());
                    Value::Array(self.store.alloc_array(new_vec))
                } else {
                    return Err(VMError::TypeError("First argument must be Array".into()));
                };
                Ok(new_arr)
            }
            "Signal_new" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                let initial_value = args[0].clone();
                let id = self
                    .reactive_runtime
                    .create_root(initial_value.clone(), crate::vm::reactive::NodeKind::Signal);
                Ok(Value::Signal {
                    id,
                    current_value: Rc::new(initial_value),
                })
            }
            "Signal_get" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                if let Value::Signal { id, .. } = &args[0] {
                    let val = self
                        .reactive_runtime
                        .get_value(*id)
                        .ok_or_else(|| VMError::InvalidOperation("Signal not found".into()))?;
                    Ok(val)
                } else {
                    Err(VMError::TypeError("Argument must be Signal".into()))
                }
            }
            "Signal_set" => {
                if args.len() != 2 {
                    return Err(VMError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                    });
                }
                if let Value::Signal { id, .. } = &args[0] {
                    self.reactive_runtime.set_value(*id, args[1].clone());
                    Ok(Value::Unit)
                } else {
                    Err(VMError::TypeError("First argument must be Signal".into()))
                }
            }
            "grad" => {
                if args.len() < 2 {
                    return Err(VMError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                    });
                }
                let func = args[0].clone();
                let x_val = args[1].clone();

                let tape_id = crate::ad::create_tape();

                let x_ad = value_to_adfloat(&x_val)?;
                let x_node = match x_ad {
                    ADFloat::Concrete(v) => ADFloat::new_input(v, tape_id),
                    _ => x_ad,
                };
                let x_node_id = x_node.node_id();

                let result = self.call_function(&func, vec![Value::Float(x_node)])?;

                let mut grad_val = 0.0;
                if let Value::Float(res_ad) = result {
                    if let Some(res_id) = res_ad.node_id() {
                        let grads = crate::ad::with_tape(tape_id, |tape| {
                            crate::ad::backward::backward(tape, res_id)
                        });
                        if let Some(nid) = x_node_id {
                            if let Some(ADGradient::Scalar(g)) = grads.get(&nid) {
                                grad_val = *g;
                            }
                        }
                    }
                }

                crate::ad::remove_tape(tape_id);
                Ok(Value::Float(ADFloat::Concrete(grad_val)))
            }
            "backward" => {
                if args.len() != 1 {
                    return Err(VMError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                    });
                }
                Ok(Value::Unit)
            }

            "infer_hmc" => {
                if args.len() != 5 {
                    return Err(VMError::ArityMismatch {
                        expected: 5,
                        found: args.len(),
                    });
                }

                let num_samples = match &args[1] {
                    Value::Int(i) => *i as usize,
                    _ => {
                        return Err(VMError::TypeError(
                            "Argument 1 (num_samples) must be Int".into(),
                        ))
                    }
                };
                let burn_in = match &args[2] {
                    Value::Int(i) => *i as usize,
                    _ => {
                        return Err(VMError::TypeError(
                            "Argument 2 (burn_in) must be Int".into(),
                        ))
                    }
                };
                let epsilon = match &args[3] {
                    Value::Float(f) => f.value(),
                    _ => {
                        return Err(VMError::TypeError(
                            "Argument 3 (epsilon) must be Float".into(),
                        ))
                    }
                };
                let l_steps = match &args[4] {
                    Value::Int(i) => *i as usize,
                    _ => {
                        return Err(VMError::TypeError(
                            "Argument 4 (l_steps) must be Int".into(),
                        ))
                    }
                };

                let samples = self.run_hmc(&args[0], num_samples, burn_in, epsilon, l_steps)?;

                let mut samples_val = Vec::new();
                for sample in samples {
                    let mut map = std::collections::HashMap::new();
                    for (k, v) in sample {
                        map.insert(k, Value::Float(ADFloat::Concrete(v)));
                    }
                    samples_val.push(Value::Map(self.store.alloc_dict(map)));
                }
                Ok(Value::Array(self.store.alloc_array(samples_val)))
            }
            "infer_vi" => {
                if args.len() != 4 {
                    return Err(VMError::ArityMismatch {
                        expected: 4,
                        found: args.len(),
                    });
                }
                let config = args[0].clone();
                let model_fn = args[1].clone();
                let guide_fn = args[2].clone();
                let _data = args[3].clone();

                self.run_vi(config, model_fn, guide_fn)
            }
            "Map::insert" => {
                if args.len() != 3 {
                    return Err(VMError::ArityMismatch {
                        expected: 3,
                        found: args.len(),
                    });
                }
                if let Value::Map(handle) = &args[0] {
                    let key = match &args[1] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::TypeError("Map key must be string".into())),
                    };
                    self.store.with_dict_mut(*handle, |map| {
                        map.insert(key, args[2].clone());
                    });
                    Ok(Value::Unit)
                } else {
                    Err(VMError::TypeError(
                        "First argument to Map::insert must be Map".into(),
                    ))
                }
            }
            "Map::get" => {
                if args.len() != 2 {
                    return Err(VMError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                    });
                }
                if let Value::Map(handle) = &args[0] {
                    let key = match &args[1] {
                        Value::String(s) => s.to_string(),
                        _ => return Err(VMError::TypeError("Map key must be string".into())),
                    };
                    let val = self
                        .store
                        .with_dict(*handle, |map| map.get(&key).cloned())
                        .flatten()
                        .unwrap_or(Value::None);
                    Ok(val)
                } else {
                    Err(VMError::TypeError(
                        "First argument to Map::get must be Map".into(),
                    ))
                }
            }
            _ => Err(VMError::InvalidOperation(format!(
                "Unknown builtin: {}",
                name
            ))),
        }
    }

    fn run_inference(&mut self, num_samples: usize, model_fn: Value) -> VMResult<Value> {
        let (chunk_index, upvalues_opt) = match &model_fn {
            Value::BytecodeFunction { chunk_index, .. } => (*chunk_index, None),
            Value::BytecodeClosure { function, upvalues } => {
                if let Value::BytecodeFunction { chunk_index, .. } = &**function {
                    (*chunk_index, Some(upvalues.clone()))
                } else {
                    return Err(VMError::TypeError(
                        "infer requires a bytecode function".into(),
                    ));
                }
            }
            _ => return Err(VMError::TypeError("infer requires a function".into())),
        };

        let mut samples = Vec::with_capacity(num_samples);
        let mut weights = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            self.log_weight = 0.0;

            let saved_stack_len = self.stack.len();
            let saved_frames_len = self.frames.len();

            let slot_base = self.stack.len();

            let closure = if let Some(ref upvals) = upvalues_opt {
                Value::BytecodeClosure {
                    function: Box::new(model_fn.clone()),
                    upvalues: upvals.clone(),
                }
            } else {
                model_fn.clone()
            };

            self.frames.push(CallFrame {
                chunk_index,
                ip: 0,
                slot_base,
                closure,
            });
            self.current_frame = self.frames.len() - 1;

            let result = self.run(saved_frames_len);

            while self.frames.len() > saved_frames_len {
                self.frames.pop();
            }
            if !self.frames.is_empty() {
                self.current_frame = self.frames.len() - 1;
            } else {
                self.current_frame = 0;
            }

            self.stack.truncate(saved_stack_len);

            match result {
                Ok(value) => {
                    let sample_val = match &value {
                        Value::Float(f) => f.value(),
                        Value::Int(i) => *i as f64,
                        _ => 0.0,
                    };
                    samples.push(sample_val);
                    weights.push(self.log_weight.exp());
                }
                Err(_) => {
                    continue;
                }
            }
        }

        if samples.is_empty() {
            return Err(VMError::InvalidOperation(
                "No valid samples in inference".into(),
            ));
        }

        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            let result: Vec<Value> = samples
                .iter()
                .map(|s| Value::Float(ADFloat::Concrete(*s)))
                .collect();
            return Ok(Value::Array(self.store.alloc_array(result)));
        }

        let normalized: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();

        let mut resampled = Vec::with_capacity(num_samples);
        for i in 0..samples.len().min(num_samples) {
            if normalized[i] > 0.001 || resampled.len() < num_samples / 2 {
                resampled.push(Value::Float(ADFloat::Concrete(samples[i])));
            }
        }

        while resampled.len() < num_samples {
            let idx = (self.rng.gen::<f64>() * samples.len() as f64) as usize;
            let idx = idx.min(samples.len() - 1);
            resampled.push(Value::Float(ADFloat::Concrete(samples[idx])));
        }

        Ok(Value::Array(self.store.alloc_array(resampled)))
    }

    fn run_vi(&mut self, config: Value, model_fn: Value, guide_fn: Value) -> VMResult<Value> {
        let (num_iters, optimizer_name, lr) = if let Value::Map(handle) = config {
            let n = self
                .store
                .with_dict(handle, |map| match map.get("num_iters") {
                    Some(Value::Int(i)) => *i as usize,
                    _ => 1000,
                })
                .unwrap_or(1000);
            let opt = self
                .store
                .with_dict(handle, |map| match map.get("optimizer") {
                    Some(Value::String(s)) => s.as_str().to_string(),
                    _ => "adam".to_string(),
                })
                .unwrap_or_else(|| "adam".to_string());
            let r = self
                .store
                .with_dict(handle, |map| match map.get("learning_rate") {
                    Some(Value::Float(f)) => f.value(),
                    Some(Value::Int(i)) => *i as f64,
                    _ => 0.01,
                })
                .unwrap_or(0.01);
            (n, opt, r)
        } else {
            (1000, "adam".to_string(), 0.01)
        };

        let mut optimizer: Box<dyn Optimizer> = if optimizer_name == "sgd" {
            Box::new(SGD { lr })
        } else {
            Box::new(Adam::new(lr))
        };

        let mut vi_params: HashMap<String, f64> = HashMap::new();

        self.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
        if let Some(ctx) = self.prob_context.as_mut() {
            ctx.mode = InferenceMode::Guide {
                tape_id: ctx.tape_id,
            };
        }
        let _ = self.call_function(&guide_fn, vec![])?;

        if let Some(ctx) = self.prob_context.as_ref() {
            for (name, val) in &ctx.vi_params {
                vi_params.insert(name.clone(), val.value());
            }
        }
        self.prob_context = None;
        crate::ad::remove_tape(crate::ad::create_tape());

        for _ in 0..num_iters {
            let (grads, param_nodes) =
                self.run_guide_and_model_vi(&model_fn, &guide_fn, &mut vi_params)?;
            optimizer.step(&mut vi_params, &grads, &param_nodes);
        }

        let mut result_map = HashMap::new();
        for (k, v) in vi_params {
            result_map.insert(k, Value::Float(ADFloat::Concrete(v)));
        }

        Ok(Value::Map(self.store.alloc_dict(result_map)))
    }

    fn run_guide_and_model_vi(
        &mut self,
        model_fn: &Value,
        guide_fn: &Value,
        vi_params: &mut HashMap<String, f64>,
    ) -> VMResult<(HashMap<usize, ADGradient>, HashMap<String, usize>)> {
        self.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
        let tape_id = self.prob_context.as_ref().unwrap().tape_id;

        if let Some(ctx) = self.prob_context.as_mut() {
            ctx.mode = InferenceMode::Guide { tape_id };
            for (name, val) in vi_params.iter() {
                let ad_val = ADFloat::new_input(*val, tape_id);
                ctx.vi_params.insert(name.clone(), ad_val.clone());
                if let ADFloat::Dual { node_id, .. } = ad_val {
                    ctx.param_nodes.insert(name.clone(), node_id);
                }
            }
        }

        let _ = self.call_function(guide_fn, vec![])?;

        let (_trace_z, log_q) = if let Some(ctx) = self.prob_context.as_ref() {
            (ctx.trace.clone(), ctx.accumulated_log_prob.clone())
        } else {
            (HashMap::new(), ADFloat::Concrete(0.0))
        };

        if let Some(ctx) = self.prob_context.as_mut() {
            ctx.mode = InferenceMode::Gradient { tape_id };
            ctx.accumulated_log_prob = ADFloat::Concrete(0.0);
            ctx.sample_counter = 0;
        }

        let _ = self.call_function(model_fn, vec![])?;

        let log_p = if let Some(ctx) = self.prob_context.as_ref() {
            ctx.accumulated_log_prob.clone()
        } else {
            ADFloat::Concrete(0.0)
        };

        let elbo = log_p - log_q;

        let mut grads = HashMap::new();
        let mut param_nodes = HashMap::new();

        if let Some(loss_node_id) = elbo.node_id() {
            let all_grads = crate::ad::with_tape(tape_id, |tape| {
                crate::ad::backward::backward(tape, loss_node_id)
            });

            if let Some(ctx) = self.prob_context.as_ref() {
                param_nodes = ctx.param_nodes.clone();
            }

            grads = all_grads;
        }

        crate::ad::remove_tape(tape_id);
        self.prob_context = None;

        Ok((grads, param_nodes))
    }

    fn call_function(&mut self, function: &Value, args: Vec<Value>) -> VMResult<Value> {
        let saved_frames_len = self.frames.len();
        let saved_stack_len = self.stack.len();

        for arg in args {
            self.push(arg)?;
        }
        match function {
            Value::BytecodeFunction {
                chunk_index, arity, ..
            } => {
                self.frames.push(CallFrame {
                    chunk_index: *chunk_index,
                    ip: 0,
                    slot_base: self.stack.len() - arity,
                    closure: Value::BytecodeClosure {
                        function: Box::new(function.clone()),
                        upvalues: vec![],
                    },
                });
            }
            Value::BytecodeClosure {
                function: func,
                upvalues: _,
            } => {
                if let Value::BytecodeFunction {
                    chunk_index, arity, ..
                } = &**func
                {
                    self.frames.push(CallFrame {
                        chunk_index: *chunk_index,
                        ip: 0,
                        slot_base: self.stack.len() - arity,
                        closure: function.clone(),
                    });
                }
            }
            _ => return Err(VMError::TypeError("Expected function".into())),
        }
        self.current_frame = self.frames.len() - 1;

        let result = self.run(saved_frames_len)?;

        while self.frames.len() > saved_frames_len {
            self.frames.pop();
        }
        if !self.frames.is_empty() {
            self.current_frame = self.frames.len() - 1;
        } else {
            self.current_frame = 0;
        }
        self.stack.truncate(saved_stack_len);
        Ok(result)
    }

    fn run_model_and_get_grads(
        &mut self,
        model_fn: &Value,
        q_values: &HashMap<String, f64>,
    ) -> VMResult<(f64, HashMap<String, f64>, HashMap<String, f64>)> {
        self.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
        let current_tape_id = self.prob_context.as_ref().unwrap().tape_id;

        if let Some(ctx) = self.prob_context.as_mut() {
            for (k, v) in q_values {
                ctx.trace
                    .insert(k.clone(), Value::Float(ADFloat::Concrete(*v)));
            }
            if q_values.is_empty() {
                ctx.mode = InferenceMode::Sampling;
            } else {
                ctx.mode = InferenceMode::Gradient {
                    tape_id: ctx.tape_id,
                };
            }
        }

        let saved_frames_len = self.frames.len();
        let saved_stack_len = self.stack.len();

        let (chunk_index, upvalues_opt) = match model_fn {
            Value::BytecodeFunction { chunk_index, .. } => (*chunk_index, None),
            Value::BytecodeClosure { function, upvalues } => {
                if let Value::BytecodeFunction { chunk_index, .. } = &**function {
                    (*chunk_index, Some(upvalues.clone()))
                } else {
                    return Err(VMError::TypeError(
                        "infer requires a bytecode function".into(),
                    ));
                }
            }
            _ => return Err(VMError::TypeError("infer requires a function".into())),
        };

        let slot_base = self.stack.len();
        let closure = if let Some(ref upvals) = upvalues_opt {
            Value::BytecodeClosure {
                function: Box::new(model_fn.clone()),
                upvalues: upvals.clone(),
            }
        } else {
            Value::Unit
        };

        self.frames.push(CallFrame {
            chunk_index,
            ip: 0,
            slot_base,
            closure,
        });
        self.current_frame = self.frames.len() - 1;

        if let Some(ctx) = self.prob_context.as_mut() {
            ctx.sample_counter = 0;
            ctx.accumulated_log_prob = ADFloat::Concrete(0.0);
        }

        let _result = self.run(saved_frames_len).unwrap_or(Value::Unit);

        while self.frames.len() > saved_frames_len {
            self.frames.pop();
        }
        if !self.frames.is_empty() {
            self.current_frame = self.frames.len() - 1;
        } else {
            self.current_frame = 0;
        }
        self.stack.truncate(saved_stack_len);

        let res = if let Some(ctx) = self.prob_context.as_ref() {
            let log_prob = ctx.accumulated_log_prob.value();

            let mut new_q = HashMap::new();
            for (k, v) in &ctx.trace {
                if let Value::Float(ad) = v {
                    new_q.insert(k.clone(), ad.value());
                }
            }

            let mut grads = HashMap::new();
            if let Some(loss_node_id) = ctx.accumulated_log_prob.node_id() {
                let all_grads = crate::ad::with_tape(ctx.tape_id, |tape| {
                    crate::ad::backward::backward(tape, loss_node_id)
                });

                for (name, id) in &ctx.param_nodes {
                    if let Some(grad_enum) = all_grads.get(id) {
                        if let ADGradient::Scalar(g) = grad_enum {
                            grads.insert(name.clone(), *g);
                        }
                    }
                }
            }

            Ok((log_prob, grads, new_q))
        } else {
            Ok((0.0, HashMap::new(), HashMap::new()))
        };

        crate::ad::remove_tape(current_tape_id);
        self.prob_context = None;

        res
    }

    pub fn run_hmc(
        &mut self,
        model: &Value,
        num_samples: usize,
        burn_in: usize,
        epsilon: f64,
        l_steps: usize,
    ) -> VMResult<Vec<HashMap<String, f64>>> {
        let mut samples = Vec::new();
        let mut rng = rand::thread_rng();
        use rand::Rng;
        use rand_distr::StandardNormal;

        let (mut current_log_prob, mut current_grads, mut current_q) =
            self.run_model_and_get_grads(model, &HashMap::new())?;

        if current_q.is_empty() {
            return Ok(samples);
        }

        for i in 0..(num_samples + burn_in) {
            let mut current_p: HashMap<String, f64> = HashMap::new();
            for (key, _) in &current_q {
                let p_val: f64 = rng.sample(StandardNormal);
                current_p.insert(key.clone(), p_val);
            }

            let current_k: f64 = current_p.values().map(|p| 0.5 * p * p).sum();
            let current_h = -current_log_prob + current_k;

            let mut q_new = current_q.clone();
            let mut p_new = current_p.clone();
            let mut grads_new = current_grads.clone();
            let mut log_prob_new = current_log_prob;

            for (key, p) in p_new.iter_mut() {
                if let Some(grad) = grads_new.get(key) {
                    *p -= (epsilon / 2.0) * (-grad);
                }
            }

            for step in 0..l_steps {
                for (key, q) in q_new.iter_mut() {
                    if let Some(p) = p_new.get(key) {
                        *q += epsilon * (*p);
                    }
                }

                let (lp, grads, _) = self.run_model_and_get_grads(model, &q_new)?;
                log_prob_new = lp;
                grads_new = grads;

                if step != l_steps - 1 {
                    for (key, p) in p_new.iter_mut() {
                        if let Some(grad) = grads_new.get(key) {
                            *p -= epsilon * (-grad);
                        }
                    }
                }
            }

            for (key, p) in p_new.iter_mut() {
                if let Some(grad) = grads_new.get(key) {
                    *p -= (epsilon / 2.0) * (-grad);
                }
            }

            let new_k: f64 = p_new.values().map(|p| 0.5 * p * p).sum();
            let new_h = -log_prob_new + new_k;

            let acceptance_prob = (current_h - new_h).exp();

            if rng.gen::<f64>() < acceptance_prob {
                current_q = q_new;
                current_log_prob = log_prob_new;
                current_grads = grads_new;
            }

            if i >= burn_in {
                samples.push(current_q.clone());
            }
        }

        Ok(samples)
    }
}

fn sample_distribution(dist: &Value, rng: &mut rand::rngs::ThreadRng) -> VMResult<Value> {
    match dist {
        Value::Gaussian { mean, std } => {
            use rand_distr::{Distribution, Normal};
            let normal = Normal::new(mean.value(), std.value())
                .map_err(|e| VMError::InvalidOperation(e.to_string()))?;
            let sample = normal.sample(rng);
            Ok(Value::Float(ADFloat::Concrete(sample)))
        }
        Value::Uniform { min, max } => {
            let sample = rng.gen_range(min.value()..max.value());
            Ok(Value::Float(ADFloat::Concrete(sample)))
        }
        Value::Bernoulli { p } => {
            let sample = rng.gen::<f64>() < p.value();
            Ok(Value::Bool(sample))
        }
        Value::Beta { alpha, beta } => {
            use rand_distr::{Beta, Distribution};
            let beta_dist = Beta::new(alpha.value(), beta.value())
                .map_err(|e| VMError::InvalidOperation(e.to_string()))?;
            let sample = beta_dist.sample(rng);
            Ok(Value::Float(ADFloat::Concrete(sample)))
        }
        _ => Err(VMError::TypeError(format!(
            "Cannot sample from {}",
            dist.type_name()
        ))),
    }
}

impl BytecodeVM {
    fn log_prob(&self, dist: &Value, value: &Value) -> VMResult<f64> {
        let x = match value {
            Value::Float(f) => f.value(),
            Value::Int(i) => *i as f64,
            Value::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            _ => {
                return Err(VMError::TypeError(format!(
                    "Cannot compute log probability for {}",
                    value.type_name()
                )))
            }
        };

        match dist {
            Value::Gaussian { mean, std } => {
                let z = (x - mean.value()) / std.value();
                let log_prob =
                    -0.5 * z * z - std.value().ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
                Ok(log_prob)
            }
            Value::Uniform { min, max } => {
                if x >= min.value() && x <= max.value() {
                    Ok(-((max.value() - min.value()).ln()))
                } else {
                    Ok(f64::NEG_INFINITY)
                }
            }
            Value::Bernoulli { p } => {
                let prob = if x > 0.5 { p.value() } else { 1.0 - p.value() };
                Ok(prob.ln())
            }
            Value::Beta { alpha, beta } => {
                use statrs::function::gamma::ln_gamma;
                if x <= 0.0 || x >= 1.0 {
                    return Ok(f64::NEG_INFINITY);
                }
                let a = alpha.value();
                let b = beta.value();
                let log_prob = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() + ln_gamma(a + b)
                    - ln_gamma(a)
                    - ln_gamma(b);
                Ok(log_prob)
            }
            _ => Err(VMError::TypeError(format!(
                "Cannot compute log probability for distribution {}",
                dist.type_name()
            ))),
        }
    }

    pub fn get_log_weight(&self) -> f64 {
        self.log_weight
    }

    fn capture_upvalue(&mut self, local_idx: usize) -> Rc<RefCell<Upvalue>> {
        for upvalue in &self.open_upvalues {
            let borrow = upvalue.borrow();
            if let Upvalue::Open(idx) = *borrow {
                if idx == local_idx {
                    return upvalue.clone();
                }
            }
        }

        if local_idx >= self.stack.len() {}

        let created_upvalue = Rc::new(RefCell::new(Upvalue::Open(local_idx)));
        self.open_upvalues.push(created_upvalue.clone());
        created_upvalue
    }

    fn close_upvalues(&mut self, last_idx: usize) {
        let mut i = 0;
        while i < self.open_upvalues.len() {
            let should_close = {
                let upvalue = self.open_upvalues[i].borrow();
                if let Upvalue::Open(idx) = *upvalue {
                    idx >= last_idx
                } else {
                    false
                }
            };

            if should_close {
                let upvalue_rc = self.open_upvalues.remove(i);
                let mut upvalue = upvalue_rc.borrow_mut();
                if let Upvalue::Open(idx) = *upvalue {
                    if idx < self.stack.len() {
                        let value = self.stack[idx].clone();
                        *upvalue = Upvalue::Closed(value);
                    } else {
                        *upvalue = Upvalue::Closed(Value::Unit);
                    }
                }
            } else {
                i += 1;
            }
        }
    }

    pub fn reset_log_weight(&mut self) {
        self.log_weight = 0.0;
    }

    fn perform_cleanup(&mut self, resource: &Value) -> VMResult<()> {
        let type_name = match resource {
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Bool(_) => "Bool",
            Value::String(_) => "String",
            Value::Struct { name, .. } => name,
            Value::Enum { name, .. } => name,
            Value::Array(_) => "Array",
            Value::Unit => "Unit",
            Value::Gaussian { .. } => "Gaussian",
            Value::Uniform { .. } => "Uniform",
            Value::Bernoulli { .. } => "Bernoulli",
            Value::Beta { .. } => "Beta",
            _ => return Ok(()),
        };

        let method = if let Some(methods) = self.methods.get(type_name) {
            methods.get("close").cloned()
        } else {
            None
        };

        if let Some(m) = method {
            self.call_function(&m, vec![resource.clone()])?;
        }
        Ok(())
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => (x.value() - y.value()).abs() < f64::EPSILON,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Unit, Value::Unit) => true,
        _ => false,
    }
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Bool(b) => *b,
        Value::Unit => false,
        Value::Int(0) => false,
        _ => true,
    }
}

fn value_to_adfloat(value: &Value) -> VMResult<ADFloat> {
    match value {
        Value::Float(f) => Ok(f.clone()),
        Value::Int(i) => Ok(ADFloat::Concrete(*i as f64)),
        _ => Err(VMError::TypeError(format!(
            "Expected number, got {}",
            value.type_name()
        ))),
    }
}

impl Default for BytecodeVM {
    fn default() -> Self {
        Self::new()
    }
}
