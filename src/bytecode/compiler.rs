use super::chunk::Chunk;
use super::opcode::{Instruction, Opcode};
use crate::ad::types::ADFloat;
use crate::ast::node::*;
use crate::gc::Rc;
use crate::Value;

#[derive(Debug)]
pub struct Local {
    name: String,
    depth: usize,
    is_captured: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Upvalue {
    index: u8,
    is_local: bool,
}

struct CompilerState {
    chunk: Chunk,
    locals: Vec<Local>,
    upvalues: Vec<Upvalue>,
    scope_depth: usize,
    loop_start: Option<usize>,
    loop_end_jumps: Vec<usize>,
    function_name: String,
}

impl CompilerState {
    fn new(name: impl Into<String>) -> Self {
        CompilerState {
            chunk: Chunk::new(name),
            locals: Vec::new(),
            upvalues: Vec::new(),
            scope_depth: 0,
            loop_start: None,
            loop_end_jumps: Vec::new(),
            function_name: String::new(),
        }
    }

    fn resolve_local(&self, name: &str) -> Option<u16> {
        for (i, local) in self.locals.iter().enumerate().rev() {
            if local.name == name {
                return Some(i as u16);
            }
        }
        None
    }

    fn add_upvalue(&mut self, index: u8, is_local: bool) -> u16 {
        for (i, upvalue) in self.upvalues.iter().enumerate() {
            if upvalue.index == index && upvalue.is_local == is_local {
                return i as u16;
            }
        }

        self.upvalues.push(Upvalue { index, is_local });
        (self.upvalues.len() - 1) as u16
    }
}

pub struct BytecodeCompiler {
    states: Vec<CompilerState>,
    finished_functions: Vec<Chunk>,
}

#[derive(Debug)]
pub enum CompileError {
    TooManyConstants,
    TooManyLocals,
    TooManyUpvalues,
    InvalidAssignmentTarget,
    UndefinedVariable(String),
    NotYetImplemented(String),
}

pub type CompileResult<T> = Result<T, CompileError>;

impl BytecodeCompiler {
    pub fn new(name: impl Into<String>) -> Self {
        let mut compiler = BytecodeCompiler {
            states: Vec::new(),
            finished_functions: Vec::new(),
        };
        let mut global_state = CompilerState::new(name);
        global_state.function_name = "main".to_string();
        compiler.states.push(global_state);
        compiler
    }

    fn current_state(&mut self) -> &mut CompilerState {
        self.states.last_mut().expect("Compiler stack empty")
    }

    fn current_chunk(&mut self) -> &mut Chunk {
        &mut self.current_state().chunk
    }

    pub fn compile_program(mut self, program: &Program) -> CompileResult<Vec<Chunk>> {
        for item in &program.items {
            self.compile_item(item)?;
        }

        let mut has_main = false;
        for item in &program.items {
            if let Item::Function(func) = item {
                if func.name.name == "main" {
                    has_main = true;
                    break;
                }
            }
        }

        if has_main {
            let name_idx = self.make_constant(Value::String(Rc::new("main".to_string())))?;
            self.emit(Instruction::with_u16(Opcode::GetGlobal, name_idx, 0));
            self.emit(Instruction::with_u8(Opcode::Call, 0, 0));
            self.emit(Instruction::simple(Opcode::Pop, 0));
        }

        self.emit(Instruction::simple(Opcode::Nil, 0));
        self.emit(Instruction::simple(Opcode::Return, 0));

        if let Some(global_state) = self.states.pop() {
            let global_chunk = global_state.chunk;

            let mut all_chunks = Vec::new();
            all_chunks.push(global_chunk);
            all_chunks.extend(self.finished_functions);

            for (i, c) in all_chunks.iter().enumerate() {
                println!("DEBUG BYTECODE Chunk {}:\n{}", i, c);
            }

            Ok(all_chunks)
        } else {
            Ok(Vec::new())
        }
    }

    fn compile_item(&mut self, item: &Item) -> CompileResult<()> {
        match item {
            Item::Function(func) => self.compile_function_def(func),
            Item::Struct(struct_def) => {
                let name = struct_def.name.name.clone();

                let arity = struct_def.fields.len();
                let mut state = CompilerState::new(&name);
                state.function_name = name.clone();
                state.scope_depth = 1;
                for field in &struct_def.fields {
                    state.locals.push(Local {
                        name: field.name.name.clone(),
                        depth: 1,
                        is_captured: false,
                    });
                }

                for (i, field) in struct_def.fields.iter().enumerate() {
                    let n_idx = state
                        .chunk
                        .add_constant(Value::String(Rc::new(field.name.name.clone())));

                    state.chunk.write(Instruction::with_u16(
                        Opcode::Const,
                        n_idx,
                        struct_def.span.line,
                    ));

                    state.chunk.write(Instruction::with_u16(
                        Opcode::GetLocal,
                        i as u16,
                        struct_def.span.line,
                    ));
                }

                let struct_name_idx = state
                    .chunk
                    .add_constant(Value::String(Rc::new(name.clone())));

                let mut mk = Instruction::with_u16(
                    Opcode::MakeStruct,
                    struct_name_idx,
                    struct_def.span.line,
                );
                mk.operands.push(arity as u8);
                state.chunk.write(mk);

                state
                    .chunk
                    .write(Instruction::simple(Opcode::Return, struct_def.span.line));

                let chunk = state.chunk;
                let chunk_idx = self.finished_functions.len() + 1;
                self.finished_functions.push(chunk);
                let func_val = Value::BytecodeFunction {
                    name: name.clone(),
                    chunk_index: chunk_idx,
                    arity,
                    upvalue_count: 0,
                };
                let func_const_idx = self.make_constant(func_val)?;
                let global_name_idx = self.make_constant(Value::String(Rc::new(name.clone())))?;

                self.emit(Instruction::with_u16(
                    Opcode::Closure,
                    func_const_idx,
                    struct_def.span.line,
                ));
                self.emit(Instruction::with_u16(
                    Opcode::DefineGlobal,
                    global_name_idx,
                    struct_def.span.line,
                ));

                Ok(())
            }
            Item::Enum(enum_def) => {
                let name = enum_def.name.name.clone();
                let name_idx = self.make_constant(Value::String(Rc::new(name.clone())))?;
                let val_idx =
                    self.make_constant(Value::String(Rc::new(format!("Enum:{}", name))))?;
                self.emit(Instruction::with_u16(
                    Opcode::Const,
                    val_idx,
                    enum_def.span.line,
                ));
                self.emit(Instruction::with_u16(
                    Opcode::DefineGlobal,
                    name_idx,
                    enum_def.span.line,
                ));
                Ok(())
            }
            Item::Impl(impl_def) => {
                let type_name = match &impl_def.self_ty {
                    Type::Named { name, .. } => {
                        name.last_ident().expect("Invalid impl path").name.clone()
                    }
                    _ => return Err(CompileError::NotYetImplemented("Complex impl types".into())),
                };
                let type_name_idx =
                    self.make_constant(Value::String(Rc::new(type_name.clone())))?;

                for item in &impl_def.items {
                    if let ImplItem::Method(method) = item {
                        let mut state = CompilerState::new(&method.name.name);
                        state.function_name = method.name.name.clone();
                        state.scope_depth = 1;
                        self.states.push(state);

                        for param in &method.params {
                            self.add_local(param.name.name.clone())?;
                        }

                        self.compile_block(&method.body)?;
                        self.emit(Instruction::simple(Opcode::Return, method.span.line));

                        let state = self.states.pop().expect("State mismatch");

                        let chunk_index = self.finished_functions.len() + 1;
                        self.finished_functions.push(state.chunk);

                        let func_val = Value::BytecodeFunction {
                            name: method.name.name.clone(),
                            chunk_index,
                            arity: method.params.len(),
                            upvalue_count: state.upvalues.len(),
                        };
                        let func_idx = self.make_constant(func_val)?;

                        self.emit(Instruction::with_u16(
                            Opcode::Closure,
                            func_idx,
                            method.span.line,
                        ));

                        let is_instance = method
                            .params
                            .first()
                            .map(|p| p.name.name == "self")
                            .unwrap_or(false);

                        if !is_instance {
                            let full_name = format!("{}::{}", type_name, method.name.name);
                            let name_idx = self.make_constant(Value::String(Rc::new(full_name)))?;
                            self.emit(Instruction::with_u16(
                                Opcode::DefineGlobal,
                                name_idx,
                                method.span.line,
                            ));
                        } else {
                            let method_name_idx = self
                                .make_constant(Value::String(Rc::new(method.name.name.clone())))?;

                            let mut instr =
                                Instruction::simple(Opcode::DefineMethod, method.span.line);
                            instr.operands.push((type_name_idx & 0xFF) as u8);
                            instr.operands.push((type_name_idx >> 8) as u8);
                            instr.operands.push((method_name_idx & 0xFF) as u8);
                            instr.operands.push((method_name_idx >> 8) as u8);
                            self.emit(instr);
                        }
                    }
                }
                Ok(())
            }
            Item::Module(m) => {
                for item in &m.items {
                    self.compile_item(item)?;
                }
                Ok(())
            }
            Item::Import(_) => Ok(()),
            Item::TypeAlias(_) => Ok(()),
            Item::Trait(_) => Ok(()),
            Item::Extern(_) => Ok(()),
        }
    }

    fn compile_function_def(&mut self, func: &FunctionDef) -> CompileResult<()> {
        let mut state = CompilerState::new(&func.name.name);
        state.function_name = func.name.name.clone();
        state.scope_depth = 1;

        self.states.push(state);

        for param in &func.params {
            self.add_local(param.name.name.clone())?;
        }

        self.compile_block(&func.body)?;

        self.emit(Instruction::simple(Opcode::Return, func.span.line));

        let state = self.states.pop().expect("State mismatch");

        let chunk_index = self.finished_functions.len() + 1;

        let upvalue_count = state.upvalues.len();
        let upvalues = state.upvalues.clone();

        let func_val = Value::BytecodeFunction {
            name: func.name.name.clone(),
            chunk_index,
            arity: func.params.len(),
            upvalue_count,
        };

        self.finished_functions.push(state.chunk);

        let const_idx = self.make_constant(func_val)?;

        if upvalue_count > 0 {
            let mut instr = Instruction::with_u16(Opcode::Closure, const_idx, func.span.line);

            for up in upvalues {
                instr.operands.push(if up.is_local { 1 } else { 0 });
                instr.operands.push(up.index);
            }
            self.emit(instr);
        } else {
            let instr = Instruction::with_u16(Opcode::Closure, const_idx, func.span.line);
            self.emit(instr);
        }

        if self.states.len() == 1 {
            let name_idx = self.make_constant(Value::String(Rc::new(func.name.name.clone())))?;
            self.emit(Instruction::with_u16(
                Opcode::DefineGlobal,
                name_idx,
                func.span.line,
            ));
        } else {
            self.add_local(func.name.name.clone())?;
        }

        Ok(())
    }

    fn compile_block(&mut self, block: &Block) -> CompileResult<()> {
        self.begin_scope();

        let mut emitted_value = false;

        for (i, stmt) in block.statements.iter().enumerate() {
            let is_last = i == block.statements.len() - 1;

            if is_last {
                if let Statement::Expression(expr) = stmt {
                    self.compile_expression(expr)?;
                    emitted_value = true;
                    continue;
                }
            }

            self.compile_statement(stmt)?;
        }

        if !emitted_value {
            self.emit(Instruction::simple(Opcode::Nil, block.span.line));
        }

        self.end_scope_with_value(block.span.line);
        Ok(())
    }

    fn compile_statement(&mut self, stmt: &Statement) -> CompileResult<()> {
        match stmt {
            Statement::Let { pattern, init, .. } => {
                if let Some(expr) = init {
                    self.compile_expression(expr)?;
                } else {
                    self.emit(Instruction::simple(Opcode::Nil, 0));
                }
                self.compile_pattern_binding(pattern)?;
                Ok(())
            }
            Statement::Expression(expr) => {
                self.compile_expression(expr)?;
                self.emit(Instruction::simple(Opcode::Pop, expr.span().line));
                Ok(())
            }
            Statement::Return { value, span } => {
                if let Some(expr) = value {
                    self.compile_expression(expr)?;
                } else {
                    self.emit(Instruction::simple(Opcode::Nil, span.line));
                }
                self.emit(Instruction::simple(Opcode::Return, span.line));
                Ok(())
            }
            Statement::While {
                condition,
                body,
                span,
            } => {
                let loop_start = self.current_chunk().current_offset();

                let old_loop_start = self.current_state().loop_start;
                let old_loop_end_jumps = std::mem::take(&mut self.current_state().loop_end_jumps);
                self.current_state().loop_start = Some(loop_start);

                self.compile_expression(condition)?;
                let exit_jump = self.emit_jump(Opcode::JumpIfFalse, span.line);

                self.compile_block(body)?;
                self.emit(Instruction::simple(Opcode::Pop, span.line));
                self.emit_loop(loop_start, span.line)?;

                self.patch_jump(exit_jump)?;

                let end_jumps = std::mem::take(&mut self.current_state().loop_end_jumps);
                for jump in end_jumps {
                    self.patch_jump(jump)?;
                }

                self.current_state().loop_start = old_loop_start;
                self.current_state().loop_end_jumps = old_loop_end_jumps;
                Ok(())
            }
            Statement::For {
                pattern,
                iterator,
                body,
                span,
            } => {
                self.begin_scope();

                self.compile_expression(iterator)?;

                self.emit(Instruction::simple(Opcode::Iterator, span.line));

                self.add_local("(iterator)".to_string())?;

                let loop_start = self.current_chunk().current_offset();

                let old_loop_start = self.current_state().loop_start;
                let old_loop_end_jumps = std::mem::take(&mut self.current_state().loop_end_jumps);
                self.current_state().loop_start = Some(loop_start);

                let exit_jump = self.emit_jump(Opcode::Next, span.line);

                self.begin_scope();
                self.compile_pattern_binding(pattern)?;
                self.compile_block(body)?;
                self.emit(Instruction::simple(Opcode::Pop, span.line));

                self.end_scope();

                self.emit_loop(loop_start, span.line)?;

                self.patch_jump(exit_jump)?;

                self.end_scope();

                let end_jumps = std::mem::take(&mut self.current_state().loop_end_jumps);
                for jump in end_jumps {
                    self.patch_jump(jump)?;
                }

                self.current_state().loop_start = old_loop_start;
                self.current_state().loop_end_jumps = old_loop_end_jumps;

                Ok(())
            }
            Statement::Break { span } => {
                let jump = self.emit_jump(Opcode::Jump, span.line);
                self.current_state().loop_end_jumps.push(jump);
                Ok(())
            }
            Statement::Continue { span } => {
                if let Some(loop_start) = self.current_state().loop_start {
                    self.emit_loop(loop_start, span.line)?;
                }
                Ok(())
            }
            Statement::Empty => Ok(()),
        }
    }

    fn compile_expression(&mut self, expr: &Expression) -> CompileResult<()> {
        let line = expr.span().line;
        match expr {
            Expression::Literal { value, .. } => self.compile_literal(value, line),
            Expression::Variable { name, .. } => {
                let full_name = name
                    .iter_idents()
                    .map(|id| id.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::");

                if full_name.is_empty() {
                    return Err(CompileError::UndefinedVariable("unknown".into()));
                }
                self.resolve_and_load_variable(&full_name, line)
            }
            Expression::Binary {
                left, op, right, ..
            } => {
                if matches!(op, BinaryOp::Assign) {
                    return self.compile_assignment(left, right);
                }
                if let Some(op_type) = op.assignment_op() {
                    return self.compile_compound_assignment(left, right, op_type);
                }
                match op {
                    BinaryOp::And => {
                        self.compile_expression(left)?;
                        self.emit(Instruction::simple(Opcode::Dup, line));
                        let end_jump = self.emit_jump(Opcode::JumpIfFalse, line);
                        self.emit(Instruction::simple(Opcode::Pop, line));
                        self.compile_expression(right)?;
                        self.patch_jump(end_jump)?;
                        return Ok(());
                    }
                    BinaryOp::Or => {
                        self.compile_expression(left)?;
                        self.emit(Instruction::simple(Opcode::Dup, line));
                        let end_jump = self.emit_jump(Opcode::JumpIfTrue, line);
                        self.emit(Instruction::simple(Opcode::Pop, line));
                        self.compile_expression(right)?;
                        self.patch_jump(end_jump)?;
                        return Ok(());
                    }
                    _ => {}
                }
                self.compile_expression(left)?;
                self.compile_expression(right)?;
                let opcode = match op {
                    BinaryOp::Add => Opcode::Add,
                    BinaryOp::Sub => Opcode::Sub,
                    BinaryOp::Mul => Opcode::Mul,
                    BinaryOp::Div => Opcode::Div,
                    BinaryOp::Mod => Opcode::Mod,
                    BinaryOp::Eq => Opcode::Eq,
                    BinaryOp::Ne => Opcode::Ne,
                    BinaryOp::Lt => Opcode::Lt,
                    BinaryOp::Le => Opcode::Le,
                    BinaryOp::Gt => Opcode::Gt,
                    BinaryOp::Ge => Opcode::Ge,
                    _ => return Err(CompileError::NotYetImplemented(format!("{:?}", op))),
                };
                self.emit(Instruction::simple(opcode, line));
                Ok(())
            }
            Expression::Unary { op, operand, .. } => {
                self.compile_expression(operand)?;
                let opcode = match op {
                    UnaryOp::Neg => Opcode::Neg,
                    UnaryOp::Not => Opcode::Not,
                    _ => return Err(CompileError::NotYetImplemented(format!("{:?}", op))),
                };
                self.emit(Instruction::simple(opcode, line));
                Ok(())
            }
            Expression::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.compile_expression(condition)?;
                let then_jump = self.emit_jump(Opcode::JumpIfFalse, line);
                self.compile_block(then_branch)?;
                let else_jump = self.emit_jump(Opcode::Jump, line);
                self.patch_jump(then_jump)?;
                if let Some(else_block) = else_branch {
                    self.compile_block(else_block)?;
                } else {
                    self.emit(Instruction::simple(Opcode::Nil, line));
                }
                self.patch_jump(else_jump)?;
                Ok(())
            }
            Expression::Block { block, .. } => self.compile_block(block),
            Expression::Array { elements, .. } => {
                for elem in elements {
                    self.compile_expression(elem)?;
                }
                self.emit(Instruction::with_u16(
                    Opcode::MakeArray,
                    elements.len() as u16,
                    line,
                ));
                Ok(())
            }
            Expression::Index { object, index, .. } => {
                self.compile_expression(object)?;
                self.compile_expression(index)?;
                self.emit(Instruction::simple(Opcode::Index, line));
                Ok(())
            }
            Expression::FieldAccess { object, field, .. } => {
                self.compile_expression(object)?;
                let idx = self.make_constant(Value::String(Rc::new(field.name.clone())))?;
                self.emit(Instruction::with_u16(Opcode::GetField, idx, line));
                Ok(())
            }
            Expression::Struct { name, fields, .. } => {
                let name_idx = self.make_constant(Value::String(Rc::new(name.name.clone())))?;
                for field in fields {
                    let field_name_idx =
                        self.make_constant(Value::String(Rc::new(field.name.name.clone())))?;
                    self.emit(Instruction::with_u16(Opcode::Const, field_name_idx, line));
                    self.compile_expression(&field.value)?;
                }
                let mut instr = Instruction::with_u16(Opcode::MakeStruct, name_idx, line);
                instr.operands.push(fields.len() as u8);
                self.emit(instr);
                Ok(())
            }
            Expression::Call { callee, args, .. } => {
                if let Expression::Variable { name, .. } = callee.as_ref() {
                    if let Some(ident) = name.last_ident() {
                        match ident.name.as_str() {
                            "print" | "println" => {
                                for arg in args {
                                    self.compile_expression(arg)?;
                                    self.emit(Instruction::simple(Opcode::Print, line));
                                }
                                self.emit(Instruction::simple(Opcode::Nil, line));
                                return Ok(());
                            }
                            "Gaussian" if args.len() == 2 => {
                                self.compile_expression(&args[0])?;
                                self.compile_expression(&args[1])?;
                                self.emit(Instruction::simple(Opcode::MakeGaussian, line));
                                return Ok(());
                            }
                            "Uniform" if args.len() == 2 => {
                                self.compile_expression(&args[0])?;
                                self.compile_expression(&args[1])?;
                                self.emit(Instruction::simple(Opcode::MakeUniform, line));
                                return Ok(());
                            }
                            "Bernoulli" if args.len() == 1 => {
                                self.compile_expression(&args[0])?;
                                self.emit(Instruction::simple(Opcode::MakeBernoulli, line));
                                return Ok(());
                            }
                            "Beta" if args.len() == 2 => {
                                self.compile_expression(&args[0])?;
                                self.compile_expression(&args[1])?;
                                self.emit(Instruction::simple(Opcode::MakeBeta, line));
                                return Ok(());
                            }
                            "sample" if args.len() == 1 => {
                                self.compile_expression(&args[0])?;
                                self.emit(Instruction::simple(Opcode::Sample, line));
                                return Ok(());
                            }
                            "observe" if args.len() == 2 => {
                                self.compile_expression(&args[0])?;
                                self.compile_expression(&args[1])?;
                                self.emit(Instruction::simple(Opcode::Observe, line));
                                return Ok(());
                            }
                            _ => {}
                        }
                    }
                }
                self.compile_expression(callee)?;
                for arg in args {
                    self.compile_expression(arg)?;
                }
                self.emit(Instruction::with_u8(Opcode::Call, args.len() as u8, line));
                Ok(())
            }
            Expression::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                self.compile_expression(receiver)?;
                for arg in args {
                    self.compile_expression(arg)?;
                }
                let method_idx = self.make_constant(Value::String(Rc::new(method.name.clone())))?;
                let mut instr = Instruction::with_u16(Opcode::MethodCall, method_idx, line);
                instr.operands.push(args.len() as u8);
                self.emit(instr);
                Ok(())
            }
            Expression::Match {
                scrutinee,
                arms,
                span,
            } => self.compile_match(scrutinee, arms, span.line),
            Expression::Lambda { params, body, span } => {
                self.compile_lambda(params, body, span.clone())
            }
            Expression::Closure { params, body, span } => {
                self.compile_lambda(params, body, span.clone())
            }
            Expression::Tuple { elements, .. } => {
                for elem in elements {
                    self.compile_expression(elem)?;
                }
                self.emit(Instruction::with_u16(
                    Opcode::MakeArray,
                    elements.len() as u16,
                    line,
                ));
                Ok(())
            }
            Expression::Enum {
                name,
                variant,
                args,
                named_fields: _,
                span: _,
            } => {
                let name_idx = self.make_constant(Value::String(Rc::new(name.name.clone())))?;
                let variant_idx =
                    self.make_constant(Value::String(Rc::new(variant.name.clone())))?;

                for arg in args {
                    self.compile_expression(arg)?;
                }

                let mut instr = Instruction::with_u16(Opcode::MakeEnum, name_idx, line);
                let variant_bytes = variant_idx.to_le_bytes();
                instr.operands.push(variant_bytes[0]);
                instr.operands.push(variant_bytes[1]);
                instr.operands.push(args.len() as u8);
                self.emit(instr);
                Ok(())
            }
            Expression::Some { expr, .. } => {
                self.compile_expression(expr)?;
                let name_idx = self.make_constant(Value::String(Rc::new("Option".to_string())))?;
                let variant_idx = self.make_constant(Value::String(Rc::new("Some".to_string())))?;
                let mut instr = Instruction::with_u16(Opcode::MakeEnum, name_idx, line);
                let variant_bytes = variant_idx.to_le_bytes();
                instr.operands.push(variant_bytes[0]);
                instr.operands.push(variant_bytes[1]);
                instr.operands.push(1);
                self.emit(instr);
                Ok(())
            }
            Expression::None { .. } => {
                let name_idx = self.make_constant(Value::String(Rc::new("Option".to_string())))?;
                let variant_idx = self.make_constant(Value::String(Rc::new("None".to_string())))?;
                let mut instr = Instruction::with_u16(Opcode::MakeEnum, name_idx, line);
                let variant_bytes = variant_idx.to_le_bytes();
                instr.operands.push(variant_bytes[0]);
                instr.operands.push(variant_bytes[1]);
                instr.operands.push(0);
                self.emit(instr);
                Ok(())
            }
            Expression::Paren { expr, .. } => self.compile_expression(expr),
            Expression::UfcsMethod {
                trait_path, method, ..
            } => {
                let path_str = trait_path
                    .iter_idents()
                    .map(|id| id.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::");
                let full_name = format!("{}::{}", path_str, method.name);
                self.resolve_and_load_variable(&full_name, line)
            }
            Expression::Range {
                start,
                end,
                inclusive,
                ..
            } => {
                if let Some(s) = start {
                    self.compile_expression(s)?;
                } else {
                    self.compile_literal(&Literal::Int(0), line)?;
                }

                if let Some(e) = end {
                    self.compile_expression(e)?;
                } else {
                    self.compile_literal(&Literal::Int(100_000), line)?;
                }

                let mut instr = Instruction::simple(Opcode::MakeRange, line);
                instr.operands.push(if *inclusive { 1 } else { 0 });
                self.emit(instr);
                Ok(())
            }
            Expression::Ok { expr, .. } => {
                self.compile_expression(expr)?;
                let name_idx = self.make_constant(Value::String(Rc::new("Result".to_string())))?;
                let variant_idx = self.make_constant(Value::String(Rc::new("Ok".to_string())))?;
                let mut instr = Instruction::with_u16(Opcode::MakeEnum, name_idx, line);
                let variant_bytes = variant_idx.to_le_bytes();
                instr.operands.push(variant_bytes[0]);
                instr.operands.push(variant_bytes[1]);
                instr.operands.push(1);
                self.emit(instr);
                Ok(())
            }
            Expression::Err { expr, .. } => {
                self.compile_expression(expr)?;
                let name_idx = self.make_constant(Value::String(Rc::new("Result".to_string())))?;
                let variant_idx = self.make_constant(Value::String(Rc::new("Err".to_string())))?;
                let mut instr = Instruction::with_u16(Opcode::MakeEnum, name_idx, line);
                let variant_bytes = variant_idx.to_le_bytes();
                instr.operands.push(variant_bytes[0]);
                instr.operands.push(variant_bytes[1]);
                instr.operands.push(1);
                self.emit(instr);
                Ok(())
            }
            Expression::Try { expr, .. } => {
                self.compile_expression(expr)?;

                let _name_idx = self.make_constant(Value::String(Rc::new("Result".to_string())))?;
                let err_variant_idx =
                    self.make_constant(Value::String(Rc::new("Err".to_string())))?;

                let is_variant_instr =
                    Instruction::with_u16(Opcode::IsVariant, err_variant_idx, line);
                self.emit(is_variant_instr);

                let jump_if_ok = self.emit_jump(Opcode::JumpIfFalse, line);

                self.emit(Instruction::simple(Opcode::Return, line));

                self.patch_jump(jump_if_ok)?;
                self.emit(Instruction::simple(Opcode::UnpackEnum, line));
                Ok(())
            }
            Expression::Spawn { expr, .. } => {
                match expr.as_ref() {
                    Expression::Call { callee, args, .. } => {
                        self.compile_expression(callee)?;
                        for arg in args {
                            self.compile_expression(arg)?;
                        }
                        self.emit(Instruction::with_u8(Opcode::Spawn, args.len() as u8, line));
                    }
                    _ => {
                        self.compile_expression(expr)?;
                        self.emit(Instruction::with_u8(Opcode::Spawn, 0, line));
                    }
                }
                Ok(())
            }
            Expression::Await { expr, .. } => {
                self.compile_expression(expr)?;
                self.emit(Instruction::simple(Opcode::Await, line));
                Ok(())
            }
            Expression::Cast {
                expr, target_type, ..
            } => {
                self.compile_expression(expr)?;
                let type_name = format!("{:?}", target_type);
                let type_idx = self.make_constant(Value::String(Rc::new(type_name)))?;
                self.emit(Instruction::with_u16(Opcode::Cast, type_idx, line));
                Ok(())
            }
            Expression::With {
                name,
                initializer,
                body,
                ..
            } => {
                self.compile_expression(initializer)?;
                self.begin_scope();
                self.add_local(name.name.clone())?;
                self.compile_block(body)?;
                self.end_scope_with_value(line);
                Ok(())
            }
        }
    }

    fn compile_lambda(
        &mut self,
        params: &[Parameter],
        body: &Expression,
        span: Span,
    ) -> CompileResult<()> {
        let mut state = CompilerState::new("lambda");
        state.function_name = "lambda".to_string();
        state.scope_depth = 1;

        self.states.push(state);

        for param in params {
            self.add_local(param.name.name.clone())?;
        }

        self.compile_expression(body)?;
        self.emit(Instruction::simple(Opcode::Return, span.line));

        let state = self.states.pop().expect("State mismatch");

        let chunk_index = 1 + self.finished_functions.len();
        let upvalue_count = state.upvalues.len();
        let upvalues = state.upvalues.clone();

        let func_val = Value::BytecodeFunction {
            name: "lambda".to_string(),
            chunk_index,
            arity: params.len(),
            upvalue_count,
        };

        self.finished_functions.push(state.chunk);

        let const_idx = self.make_constant(func_val)?;

        let mut instr = Instruction::with_u16(Opcode::Closure, const_idx, span.line);
        for up in upvalues {
            instr.operands.push(if up.is_local { 1 } else { 0 });
            instr.operands.push(up.index);
        }
        self.emit(instr);
        Ok(())
    }

    fn compile_match(
        &mut self,
        scrutinee: &Expression,
        arms: &[MatchArm],
        line: usize,
    ) -> CompileResult<()> {
        self.compile_expression(scrutinee)?;

        let mut match_end_jumps = Vec::new();

        for arm in arms {
            let mut fail_jumps = Vec::new();

            self.emit(Instruction::simple(Opcode::Dup, line));

            let needs_pop_on_fail = match &arm.pattern {
                Pattern::Literal { value, .. } => {
                    self.compile_literal(value, line)?;
                    self.emit(Instruction::simple(Opcode::Eq, line));
                    let jump = self.emit_jump(Opcode::JumpIfFalse, line);
                    fail_jumps.push(jump);
                    false
                }
                Pattern::Some { .. } => {
                    let idx = self.make_constant(Value::String(Rc::new("Some".to_string())))?;
                    self.emit(Instruction::with_u16(Opcode::IsVariant, idx, line));
                    let jump = self.emit_jump(Opcode::JumpIfFalse, line);
                    fail_jumps.push(jump);
                    true
                }
                Pattern::None { .. } => {
                    let idx = self.make_constant(Value::String(Rc::new("None".to_string())))?;
                    self.emit(Instruction::with_u16(Opcode::IsVariant, idx, line));
                    let jump = self.emit_jump(Opcode::JumpIfFalse, line);
                    fail_jumps.push(jump);
                    true
                }
                Pattern::Enum { variant, .. } => {
                    let idx = self.make_constant(Value::String(Rc::new(variant.name.clone())))?;
                    self.emit(Instruction::with_u16(Opcode::IsVariant, idx, line));
                    let jump = self.emit_jump(Opcode::JumpIfFalse, line);
                    fail_jumps.push(jump);
                    true
                }
                Pattern::Ok { .. } => {
                    let idx = self.make_constant(Value::String(Rc::new("Ok".to_string())))?;
                    self.emit(Instruction::with_u16(Opcode::IsVariant, idx, line));
                    let jump = self.emit_jump(Opcode::JumpIfFalse, line);
                    fail_jumps.push(jump);
                    true
                }
                Pattern::Err { .. } => {
                    let idx = self.make_constant(Value::String(Rc::new("Err".to_string())))?;
                    self.emit(Instruction::with_u16(Opcode::IsVariant, idx, line));
                    let jump = self.emit_jump(Opcode::JumpIfFalse, line);
                    fail_jumps.push(jump);
                    true
                }
                Pattern::Identifier { .. } | Pattern::Wildcard { .. } => {
                    self.emit(Instruction::simple(Opcode::Pop, line));
                    false
                }
                _ => return Err(CompileError::NotYetImplemented("Complex patterns".into())),
            };

            if needs_pop_on_fail {
                self.emit(Instruction::simple(Opcode::Pop, line));
            }

            self.begin_scope();
            self.compile_pattern_bind(&arm.pattern, line)?;

            self.compile_expression(&arm.body)?;

            self.end_scope_with_value(line);

            let end_jump = self.emit_jump(Opcode::Jump, line);
            match_end_jumps.push(end_jump);

            for jump in fail_jumps {
                self.patch_jump(jump)?;
            }
            if needs_pop_on_fail {
                self.emit(Instruction::simple(Opcode::Pop, line));
            }
        }

        self.emit(Instruction::simple(Opcode::Pop, line));
        self.emit(Instruction::simple(Opcode::Nil, line));

        for jump in match_end_jumps {
            self.patch_jump(jump)?;
        }
        Ok(())
    }

    fn compile_pattern_bind(&mut self, pattern: &Pattern, line: usize) -> CompileResult<()> {
        match pattern {
            Pattern::Identifier { name, .. } => {
                self.add_local(name.name.clone())?;
            }
            Pattern::Wildcard { .. } => {
                self.emit(Instruction::simple(Opcode::Pop, line));
            }
            Pattern::Some { pattern: inner, .. } => {
                self.emit(Instruction::with_u8(Opcode::UnpackEnum, 1, line));
                self.compile_pattern_bind(inner, line)?;
            }
            Pattern::Enum { args, .. } => {
                self.emit(Instruction::with_u8(
                    Opcode::UnpackEnum,
                    args.len() as u8,
                    line,
                ));
                for arg in args.iter().rev() {
                    self.compile_pattern_bind(arg, line)?;
                }
            }
            Pattern::Ok { pattern: inner, .. } => {
                self.emit(Instruction::with_u8(Opcode::UnpackEnum, 1, line));
                self.compile_pattern_bind(inner, line)?;
            }
            Pattern::Err { pattern: inner, .. } => {
                self.emit(Instruction::with_u8(Opcode::UnpackEnum, 1, line));
                self.compile_pattern_bind(inner, line)?;
            }
            Pattern::None { .. } => {
                self.emit(Instruction::simple(Opcode::Pop, line));
            }
            Pattern::Literal { .. } => {
                self.emit(Instruction::simple(Opcode::Pop, line));
            }
            _ => {}
        }
        Ok(())
    }

    fn compile_literal(&mut self, lit: &Literal, line: usize) -> CompileResult<()> {
        match lit {
            Literal::Int(n) => {
                let idx = self.make_constant(Value::Int(*n))?;
                self.emit(Instruction::with_u16(Opcode::Const, idx, line));
            }
            Literal::Float(n) => {
                let idx = self.make_constant(Value::Float(ADFloat::Concrete(*n)))?;
                self.emit(Instruction::with_u16(Opcode::Const, idx, line));
            }
            Literal::Bool(true) => {
                self.emit(Instruction::simple(Opcode::True, line));
            }
            Literal::Bool(false) => {
                self.emit(Instruction::simple(Opcode::False, line));
            }
            Literal::String(s) => {
                let idx = self.make_constant(Value::String(Rc::new(s.clone())))?;
                self.emit(Instruction::with_u16(Opcode::Const, idx, line));
            }
            Literal::Unit => {
                self.emit(Instruction::simple(Opcode::Nil, line));
            }
        }
        Ok(())
    }

    fn resolve_and_load_variable(&mut self, name: &str, line: usize) -> CompileResult<()> {
        let state_len = self.states.len();
        if state_len == 0 {
            return Err(CompileError::UndefinedVariable(name.to_string()));
        }

        if let Some(idx) = self.current_state().resolve_local(name) {
            self.emit(Instruction::with_u16(Opcode::GetLocal, idx, line));
            return Ok(());
        }

        if let Some(idx) = self.resolve_upvalue(state_len - 1, name) {
            self.emit(Instruction::with_u16(Opcode::GetUpvalue, idx, line));
            return Ok(());
        }

        let idx = self.make_constant(Value::String(Rc::new(name.to_string())))?;
        self.emit(Instruction::with_u16(Opcode::GetGlobal, idx, line));
        Ok(())
    }

    fn resolve_upvalue(&mut self, state_index: usize, name: &str) -> Option<u16> {
        if state_index == 0 {
            return None;
        }

        let parent_index = state_index - 1;

        if let Some(local_idx) = self.states[parent_index].resolve_local(name) {
            self.states[parent_index].locals[local_idx as usize].is_captured = true;
            return Some(self.states[state_index].add_upvalue(local_idx as u8, true));
        }

        if let Some(upvalue_idx) = self.resolve_upvalue(parent_index, name) {
            if upvalue_idx > 255 {
                return None;
            }
            return Some(self.states[state_index].add_upvalue(upvalue_idx as u8, false));
        }

        None
    }

    fn compile_assignment(&mut self, target: &Expression, value: &Expression) -> CompileResult<()> {
        let line = target.span().line;
        match target {
            Expression::Variable { name, .. } => {
                self.compile_expression(value)?;
                if let Some(ident) = name.last_ident() {
                    let state_len = self.states.len();

                    if let Some(idx) = self.current_state().resolve_local(&ident.name) {
                        self.emit(Instruction::with_u16(Opcode::SetLocal, idx, line));
                    } else if let Some(idx) = self.resolve_upvalue(state_len - 1, &ident.name) {
                        self.emit(Instruction::with_u16(Opcode::SetUpvalue, idx, line));
                    } else {
                        let idx = self.make_constant(Value::String(Rc::new(ident.name.clone())))?;
                        self.emit(Instruction::with_u16(Opcode::SetGlobal, idx, line));
                    }
                }
                Ok(())
            }
            Expression::FieldAccess { object, field, .. } => {
                self.compile_expression(object)?;
                self.compile_expression(value)?;
                let idx = self.make_constant(Value::String(Rc::new(field.name.clone())))?;
                self.emit(Instruction::with_u16(Opcode::SetField, idx, line));
                Ok(())
            }
            Expression::Index { object, index, .. } => {
                self.compile_expression(object)?;
                self.compile_expression(index)?;
                self.compile_expression(value)?;
                self.emit(Instruction::simple(Opcode::SetIndex, line));
                Ok(())
            }
            _ => Err(CompileError::InvalidAssignmentTarget),
        }
    }

    fn compile_compound_assignment(
        &mut self,
        target: &Expression,
        value: &Expression,
        op: BinaryOp,
    ) -> CompileResult<()> {
        let line = target.span().line;
        match target {
            Expression::Variable { name, .. } => {
                self.compile_expression(target)?;
                self.compile_expression(value)?;

                let opcode = match op {
                    BinaryOp::Add => Opcode::Add,
                    BinaryOp::Sub => Opcode::Sub,
                    BinaryOp::Mul => Opcode::Mul,
                    BinaryOp::Div => Opcode::Div,
                    BinaryOp::Mod => Opcode::Mod,
                    _ => {
                        return Err(CompileError::NotYetImplemented(format!(
                            "Compound assignment for {:?}",
                            op
                        )))
                    }
                };
                self.emit(Instruction::simple(opcode, line));

                if let Some(ident) = name.last_ident() {
                    let state_len = self.states.len();
                    if let Some(idx) = self.current_state().resolve_local(&ident.name) {
                        self.emit(Instruction::with_u16(Opcode::SetLocal, idx, line));
                    } else if let Some(idx) = self.resolve_upvalue(state_len - 1, &ident.name) {
                        self.emit(Instruction::with_u16(Opcode::SetUpvalue, idx, line));
                    } else {
                        let idx = self.make_constant(Value::String(Rc::new(ident.name.clone())))?;
                        self.emit(Instruction::with_u16(Opcode::SetGlobal, idx, line));
                    }
                }
                Ok(())
            }
            _ => Err(CompileError::NotYetImplemented(
                "Compound assignment only supported for variables".into(),
            )),
        }
    }

    fn compile_pattern_binding(&mut self, pattern: &Pattern) -> CompileResult<()> {
        match pattern {
            Pattern::Identifier { name, .. } => {
                if self.current_state().scope_depth > 0 {
                    self.add_local(name.name.clone())?;
                } else {
                    let idx = self.make_constant(Value::String(Rc::new(name.name.clone())))?;
                    self.emit(Instruction::with_u16(Opcode::DefineGlobal, idx, 0));
                }
                Ok(())
            }
            Pattern::Wildcard { .. } => {
                self.emit(Instruction::simple(Opcode::Pop, 0));
                Ok(())
            }
            Pattern::Tuple { patterns, .. } => {
                for (i, elem) in patterns.iter().enumerate() {
                    self.emit(Instruction::simple(Opcode::Dup, 0));
                    let idx = self.make_constant(Value::Int(i as i64))?;
                    self.emit(Instruction::with_u16(Opcode::Const, idx, 0));
                    self.emit(Instruction::simple(Opcode::Index, 0));
                    self.compile_pattern_binding(elem)?;
                }
                self.emit(Instruction::simple(Opcode::Pop, 0));
                Ok(())
            }
            _ => Err(CompileError::NotYetImplemented(format!(
                "pattern {:?}",
                pattern
            ))),
        }
    }

    fn begin_scope(&mut self) {
        self.current_state().scope_depth += 1;
    }

    fn end_scope(&mut self) {
        let state = self.current_state();
        state.scope_depth -= 1;
        let depth = state.scope_depth;

        let mut ops = Vec::new();

        while !state.locals.is_empty() && state.locals.last().unwrap().depth > depth {
            let local = state.locals.pop().unwrap();
            if local.is_captured {
                ops.push(Instruction::simple(Opcode::CloseUpvalue, 0));
            } else {
                ops.push(Instruction::simple(Opcode::Pop, 0));
            }
        }

        for op in ops {
            self.emit(op);
        }
    }

    fn end_scope_with_value(&mut self, line: usize) {
        let state = self.current_state();
        state.scope_depth -= 1;
        let depth = state.scope_depth;

        let mut count = 0;
        while !state.locals.is_empty() && state.locals.last().unwrap().depth > depth {
            state.locals.pop();
            count += 1;
        }

        if count > 0 {
            self.emit(Instruction::with_u16(
                Opcode::CloseScope,
                count as u16,
                line,
            ));
        }
    }

    fn add_local(&mut self, name: String) -> CompileResult<()> {
        let state = self.current_state();
        if state.locals.len() >= u16::MAX as usize {
            return Err(CompileError::TooManyLocals);
        }
        state.locals.push(Local {
            name,
            depth: state.scope_depth,
            is_captured: false,
        });
        Ok(())
    }

    fn emit(&mut self, instruction: Instruction) -> usize {
        self.current_chunk().write(instruction)
    }

    fn emit_jump(&mut self, opcode: Opcode, line: usize) -> usize {
        self.emit(Instruction::with_i16(opcode, 0, line))
    }

    fn emit_loop(&mut self, loop_start: usize, line: usize) -> CompileResult<()> {
        let offset = self.current_chunk().current_offset() - loop_start + 1;
        if offset > u16::MAX as usize {
            return Err(CompileError::NotYetImplemented("loop too large".into()));
        }
        self.emit(Instruction::with_u16(Opcode::Loop, offset as u16, line));
        Ok(())
    }

    fn patch_jump(&mut self, offset: usize) -> CompileResult<()> {
        let jump = self.current_chunk().current_offset() as i32 - offset as i32 - 1;
        if jump > i16::MAX as i32 || jump < i16::MIN as i32 {
            return Err(CompileError::NotYetImplemented("jump too large".into()));
        }
        self.current_chunk().patch_jump(offset, jump as i16);
        Ok(())
    }

    fn make_constant(&mut self, value: Value) -> CompileResult<u16> {
        let idx = self.current_chunk().add_constant(value);
        if idx > u16::MAX {
            return Err(CompileError::TooManyConstants);
        }
        Ok(idx)
    }
}
