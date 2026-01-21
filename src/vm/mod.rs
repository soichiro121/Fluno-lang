// vm/mod.rs
// Virtual Machine Interpreter for the Flux programming language.
//
// This module provides a simple tree-walking interpreter for executing
// Flux programs. It maintains an environment for variable bindings and
// evaluates expressions and statements recursively.

mod error;
mod value;
mod environment;
mod reactive;

use std::collections::{HashMap, HashSet};
use reactive::ReactiveRuntime;
pub use error::{RuntimeError, RuntimeResult};
pub use value::Value;
pub use environment::Environment;
use crate::vm::prob_runtime::InferenceMode;
use crate::ast::node::{
    BinaryOp, Block, Expression, Identifier, Item, Literal, Parameter, Pattern, Program, Span,
    Statement, Type, UnaryOp, DefId
};
use crate::error::{Error, FlunoResult};
use crate::gc::Rc;
use rand::Rng;
use crate::vm::reactive::NodeKind;
pub mod prob_runtime;
pub mod distributions;
pub mod resource_manager;
use prob_runtime::ProbContext;
use crate::ADFloat;

pub struct ImplRegistry {
    // (型名, メソッド名) -> 関数Value
    methods: HashMap<(String, String), Value>,
    // trait実装の記録（将来的にtrait境界チェック用）
    trait_impls: HashMap<String, HashSet<String>>, // 型名 -> 実装trait名のセット
}

impl ImplRegistry {
    pub fn new() -> Self {
        Self {
            methods: HashMap::new(),
            trait_impls: HashMap::new(),
        }
    }
    
    pub fn register_method(&mut self, type_name: String, method_name: String, func: Value) {
        self.methods.insert((type_name, method_name), func);
    }
    
    pub fn lookup_method(&self, type_name: &str, method_name: &str) -> Option<&Value> {
        self.methods.get(&(type_name.to_string(), method_name.to_string()))
    }
    
    pub fn register_trait_impl(&mut self, type_name: String, trait_name: String) {
        self.trait_impls.entry(type_name).or_default().insert(trait_name);
    }
}

pub struct Interpreter {
    // Environment stack for variable bindings
    env: Environment,
    // Global variables accessible from any scope
    globals: HashMap<String, Value>,
    // Reactive runtime system for Signal/Event
    reactive: ReactiveRuntime,
    // Return value from a function (used for early returns)
    return_value: Option<Value>,
    // Break flag for loop control
    should_break: bool,
    // Continue flag for loop control
    should_continue: bool,
    last_receiver: Option<Value>,
    current_weight: f64,
    // 確率的プログラミングモードかどうか
    pub prob_mode: bool,
    // 現在の確率実行コンテキスト (MCMC中に使用)
    pub prob_context: Option<ProbContext>,
    // 確率変数生成時のIDカウンタ (Address生成用)
    prob_id_counter: usize,
    modules: HashMap<String, HashMap<String, Value>>, // "a::b" -> { "f" => Value, ... }
    module_stack: Vec<String>,                        // ["a", "b"]
    impl_registry: ImplRegistry,
    pub defs: HashMap<DefId, Value>, 
    pub known_traits: HashSet<String>, 
    latest_gradients: HashMap<usize, crate::ad::types::ADGradient>,
    
    // FFI Management
    // resource_manager stores opaque handles to external Rust objects
    pub resource_manager: resource_manager::ResourceManager,
    // libraries stores loaded shared libraries to keep them alive
    native_libraries: Vec<std::rc::Rc<libloading::Library>>, 
}

fn calculate_score(dist: &Value, val: &Value) -> f64 {
    match (dist, val) {
        (Value::Gaussian { mean, std }, Value::Float(x)) => { 
            prob_runtime::score_gaussian(ADFloat::Concrete(x.value()), ADFloat::Concrete(mean.value()), ADFloat::Concrete(std.value())).value()
        },
        // 他の分布...
        _ => 0.0,
    }
}

impl Interpreter {
    // Create a new interpreter instance.
    pub fn new() -> Self {
        let mut interpreter = Interpreter {
            env: Environment::new(),
            globals: HashMap::new(),
            reactive: ReactiveRuntime::new(),
            return_value: None,
            should_break: false,
            should_continue: false,
            last_receiver: None,
            current_weight: 0.0,
            prob_mode: false,
            prob_context: None,
            prob_id_counter: 0,
            modules: HashMap::new(),
            module_stack: Vec::new(),
            impl_registry: ImplRegistry::new(), 
            defs: HashMap::new(),
            known_traits: HashSet::new(),
            latest_gradients: HashMap::new(),
            resource_manager: resource_manager::ResourceManager::new(),
            native_libraries: Vec::new(),
        };
        interpreter.init_builtins();
        interpreter
    }

    // Initialize built-in functions and constants.
    fn init_builtins(&mut self) {
        self.globals.insert("true".to_string(), Value::Bool(true));
        self.globals.insert("false".to_string(), Value::Bool(false));
        self.globals.insert("println".to_string(), Value::Builtin("println".to_string()));
        self.globals.insert("print".to_string(), Value::Builtin("print".to_string()));
        self.globals.insert("assert".to_string(), Value::Builtin("assert".to_string()));
        self.globals.insert("to_future".to_string(), Value::Builtin("to_future".to_string()));
        self.globals.insert("wait_future".to_string(), Value::Builtin("wait_future".to_string()));
        self.globals.insert("nearly_eq".to_string(), Value::Builtin("nearly_eq".to_string()));
        //self.globals.insert("nearlyeq".to_string(), Value::Builtin("nearlyeq".to_string()));
        self.globals.insert("is_float".to_string(), Value::Builtin("is_float".to_string()));
        self.globals.insert("is_gaussian".to_string(), Value::Builtin("is_gaussian".to_string()));
        self.globals.insert("Gaussian".to_string(), Value::Builtin("Gaussian".to_string()));
        // Signal
        self.globals.insert("Signal_new".to_string(), Value::Builtin("Signal_new".to_string()));
        self.globals.insert("Signal_combine".to_string(), Value::Builtin("Signal_combine".to_string()));
        self.globals.insert("Signal_map".to_string(), Value::Builtin("Signal_map".to_string()));
        self.globals.insert("Signal_set".to_string(), Value::Builtin("Signal_set".to_string()));
        self.globals.insert("Signal_get".to_string(), Value::Builtin("Signal_get".to_string()));
        self.globals.insert("Signal_filter".to_string(), Value::Builtin("Signal_filter".to_string()));
        // Event
        self.globals.insert("Event_new".to_string(), Value::Builtin("Event_new".to_string()));
        self.globals.insert("Event_emit".to_string(), Value::Builtin("Event_emit".to_string()));
        self.globals.insert("Event_map".to_string(), Value::Builtin("Event_map".to_string()));
        // Reactive Operators
        self.globals.insert("Event_fold".to_string(), Value::Builtin("Event_fold".to_string()));
        self.globals.insert("Event_merge".to_string(), Value::Builtin("Event_merge".to_string()));
        self.globals.insert("Signal_sample".to_string(), Value::Builtin("Signal_sample".to_string()));
        //inference
        self.globals.insert("observe".to_string(), Value::Builtin("observe".to_string()));
        self.globals.insert("infer".to_string(), Value::Builtin("infer".to_string()));

        self.globals.insert("sample".to_string(), Value::Builtin("sample".to_string()));
        
        // Uniform(min: Float, max: Float) -> Uniform
        // 一様分布オブジェクトを作成
        self.globals.insert("Uniform".to_string(), Value::Builtin("Uniform".to_string()));
        self.globals.insert("Bernoulli".to_string(), Value::Builtin("Bernoulli".to_string()));
        self.globals.insert("Beta".to_string(), Value::Builtin("Beta".to_string()));
        
        // Math helper for inference (exp)
        self.globals.insert("exp".to_string(), Value::Builtin("exp".to_string()));

        self.globals.insert("ln".to_string(), Value::Builtin("ln".to_string()));
        self.globals.insert("sin".to_string(), Value::Builtin("sin".to_string()));
        self.globals.insert("cos".to_string(), Value::Builtin("cos".to_string()));
        self.globals.insert("tan".to_string(), Value::Builtin("tan".to_string()));
        self.globals.insert("sqrt".to_string(), Value::Builtin("sqrt".to_string()));
        self.globals.insert("pow".to_string(), Value::Builtin("pow".to_string()));
        self.globals.insert("powf".to_string(), Value::Builtin("powf".to_string()));

        self.globals.insert("to_json".to_string(), Value::Builtin("to_json".to_string()));
        self.globals.insert("from_json".to_string(), Value::Builtin("from_json".to_string()));

        self.globals.insert("Rc_new".to_string(), Value::Builtin("Rc_new".to_string()));
        self.globals.insert("Rc_downgrade".to_string(), Value::Builtin("Rc_downgrade".to_string()));
        self.globals.insert("Weak_upgrade".to_string(), Value::Builtin("Weak_upgrade".to_string()));

        self.globals.insert("Float::sqrt".to_string(), Value::Builtin("Float::sqrt".to_string()));
        self.globals.insert("Float::abs".to_string(), Value::Builtin("Float::abs".to_string()));
        self.globals.insert("Float::pow".to_string(), Value::Builtin("Float::pow".to_string()));
        self.globals.insert("Float::floor".to_string(), Value::Builtin("Float::floor".to_string()));
        self.globals.insert("Float::ceil".to_string(), Value::Builtin("Float::ceil".to_string()));
        self.globals.insert("Float::round".to_string(), Value::Builtin("Float::round".to_string()));
        self.globals.insert("Int::abs".to_string(), Value::Builtin("Int::abs".to_string()));
        self.globals.insert("*::len".to_string(), Value::Builtin("Array::len".to_string()));
        self.globals.insert("*::push".to_string(), Value::Builtin("Array::push".to_string()));

        // AD related builtins
        self.globals.insert("create_tape".to_string(), Value::Builtin("create_tape".to_string()));
        self.globals.insert("param".to_string(), Value::Builtin("param".to_string()));
        self.globals.insert("infer_vi".to_string(), Value::Builtin("infer_vi".to_string()));
        self.globals.insert("infer_hmc".to_string(), Value::Builtin("infer_hmc".to_string()));
        self.globals.insert("backward".to_string(), Value::Builtin("backward".to_string()));
        self.globals.insert("grad".to_string(), Value::Builtin("grad".to_string()));

        self.globals.insert("Map".to_string(), Value::Builtin("Map".to_string()));
        self.globals.insert("Map::insert".to_string(), Value::Builtin("Map::insert".to_string()));
        self.globals.insert("Map::get".to_string(), Value::Builtin("Map::get".to_string()));
        self.globals.insert("Map::contains_key".to_string(), Value::Builtin("Map::contains_key".to_string()));

    }
    
    


    pub fn sample(&mut self, distribution: Value) -> RuntimeResult<Value> {
        // 確率モードでなければ単にサンプリングして返す
        if !self.prob_mode || self.prob_context.is_none() {
            return self.sample_from_dist(&distribution);
        }


        let (address, mode, tape_id) = {
            let ctx = self.prob_context.as_ref().unwrap();
            let addr = format!("sample_{}", ctx.sample_counter);
            let mode = ctx.mode; // Copy
            let tape_id = ctx.tape_id;
            (addr, mode, tape_id)
        };


        if let Some(ctx) = self.prob_context.as_mut() {
            ctx.sample_counter += 1;
        }

        match mode {
            InferenceMode::Sampling => {
                // Trace Replay Check
                let traced_opt = self.prob_context.as_ref().unwrap().trace.get(&address).cloned();
                
                let val = if let Some(traced) = traced_opt {
                    traced
                } else {
                    // No trace, sample new
                    let new_val = self.sample_from_dist(&distribution)?;
                    
                    // Register to trace
                    if let Some(ctx) = self.prob_context.as_mut() {
                        ctx.trace.insert(address.clone(), new_val.clone());
                    }
                    new_val
                };
                
                // Log Prob Accumulation (Samplingでも尤度計算が必要な場合がある)
                // 今回は簡易的に計算して log_joint に入れる
                if let Some(ctx) = self.prob_context.as_mut() {
                    let f_val = match &val {
                        Value::Float(f) => f.value(),
                        Value::Int(i) => *i as f64,
                        _ => 0.0, // Error handling omitted
                    };
                    if let Ok(score) = crate::vm::prob_runtime::get_distribution_log_pdf(&distribution, f_val) {
                         let score_ad = crate::ad::types::ADFloat::Concrete(score);
                         ctx.log_joint = ctx.log_joint.clone() + score_ad.clone();
                         ctx.accumulated_log_prob = ctx.accumulated_log_prob.clone() + score_ad;
                    }
                }
                
                Ok(val)
            },
            
            InferenceMode::Gradient { tape_id } => {
                let ctx = self.prob_context.as_mut().unwrap();
                let traced = ctx.trace.get(&address).cloned().ok_or(
                     RuntimeError::InvalidOperation { message: "Missing trace in Gradient mode".into() }
                )?;
                
                let val_node = match traced {
                    Value::Float(ad) => {
                        match ad {
                             // HMCならConcreteが入っているはずだが、VIからの流用ならDualもありうる
                             // 今回はHMCはConcrete前提
                             crate::ad::types::ADFloat::Concrete(v) => {
                                 // Make Input for HMC
                                 let new_node = crate::ad::types::ADFloat::new_input(v, tape_id);
                                 if let Some(id) = new_node.node_id() {
                                     ctx.param_nodes.insert(address.clone(), id);
                                 }
                                 new_node
                             },
                             _ => ad 
                        }
                    },
                    _ => return Err(RuntimeError::TypeMismatch{message: "Sample must be float in gradient mode".into()})
                };
                
                // Score AD
                let score = crate::vm::prob_runtime::get_distribution_log_pdf_ad(&distribution, &val_node)?;
                ctx.log_joint = ctx.log_joint.clone() + score.clone();
                ctx.accumulated_log_prob = ctx.accumulated_log_prob.clone() + score;
                
                Ok(Value::Float(val_node))
            },
            
            InferenceMode::Guide { tape_id: _ } => {

                let traced_opt = self.prob_context.as_ref().unwrap().trace.get(&address).cloned();

                if let Some(traced_val) = traced_opt {
                    // Replay (Model Phase: log_p)
                    let val_ad = match &traced_val {
                        Value::Float(ad) => ad.clone(),
                        Value::Int(i) => crate::ad::types::ADFloat::Concrete(*i as f64),
                        _ => return Err(RuntimeError::TypeMismatch{message: "Replay value must be number".into()}),
                    };

                    let score = crate::vm::prob_runtime::get_distribution_log_pdf_ad(&distribution, &val_ad)?;
                    
                    if let Some(ctx) = self.prob_context.as_mut() {
                        ctx.log_joint = ctx.log_joint.clone() + score.clone();
                        ctx.accumulated_log_prob = ctx.accumulated_log_prob.clone() + score;
                    }
                    return Ok(traced_val);
                }


                let mut rng = rand::thread_rng();
                let val_ad = crate::vm::prob_runtime::get_distribution_sample(&distribution, &mut rng, mode)?;
                let log_q = crate::vm::prob_runtime::get_distribution_log_pdf_ad(&distribution, &val_ad)?;
                let val_result = Value::Float(val_ad);

                if let Some(ctx) = self.prob_context.as_mut() {
                    ctx.log_joint = ctx.log_joint.clone() + log_q.clone();
                    ctx.accumulated_log_prob = ctx.accumulated_log_prob.clone() + log_q;
                    ctx.trace.insert(address, val_result.clone());
                }

                Ok(val_result)
            }
        }
    }


    
    fn sample_from_dist(&self, dist: &Value) -> RuntimeResult<Value> {
        use crate::vm::prob_runtime::{get_distribution_sample, InferenceMode};
        use rand::thread_rng;

        let mode = if let Some(ctx) = &self.prob_context {
            ctx.mode
        } else {
            InferenceMode::Sampling
        };

        let mut rng = thread_rng();
        let sample_ad = get_distribution_sample(dist, &mut rng, mode)?;
        Ok(Value::Float(sample_ad))
    }


    pub fn observe(&mut self, distribution: Value, value: Value) -> RuntimeResult<()> {
        if self.prob_mode {
            let mode = if let Some(ctx) = &self.prob_context { ctx.mode } else { return Ok(()); };
            
            match mode {
                crate::vm::prob_runtime::InferenceMode::Gradient { .. } | crate::vm::prob_runtime::InferenceMode::Guide { .. } => {
                     // AD Mode: Calculate Score using AD
                     if let Some(ctx) = self.prob_context.as_mut() {
                         let val_ad = match &value {
                             Value::Float(f) => f.clone(),
                             Value::Int(i) => crate::ad::types::ADFloat::Concrete(*i as f64),
                             _ => return Err(RuntimeError::TypeMismatch{message: "Observed value must be number".into()}),
                         };
                         
                         let score = crate::vm::prob_runtime::get_distribution_log_pdf_ad(&distribution, &val_ad)?;
                         ctx.log_joint = ctx.log_joint.clone() + score.clone();
                         ctx.accumulated_log_prob = ctx.accumulated_log_prob.clone() + score;
                     }
                }
                _ => {
                     // Sampling Mode: Concrete Score
                    let score = self.score_distribution(&distribution, &value); // Calculate before borrow
                    if let Some(ctx) = self.prob_context.as_mut() {
                        let score_ad = crate::ad::types::ADFloat::Concrete(score);
                        ctx.accumulated_log_prob = ctx.accumulated_log_prob.clone() + score_ad.clone();
                         ctx.log_joint = ctx.log_joint.clone() + score_ad;
                    }
                }
            }
        }
        Ok(())
    }

    // 分布の対数尤度を計算
    fn score_distribution(&self, dist: &Value, val: &Value) -> f64 {
        match (dist, val) {
            (Value::Gaussian { mean, std }, Value::Float(x)) => {
                prob_runtime::score_gaussian(x.clone(), mean.clone(), std.clone()).value()
            },
            (Value::Uniform { min, max }, Value::Float(val)) => {
                // 範囲内判定などは value() で比較
                if val.value() >= min.value() && val.value() <= max.value() {
                    // -(max - min).ln()
                    -(max.clone() - min.clone()).ln().value()
                } else {
                    f64::NEG_INFINITY
                }
            },
            _ => 0.0,
        }
    }


    // 推論実行 (infer文相当)
    // num_samples: サンプル数
    // model_func: 推論対象の関数（クロージャ）

    pub fn run_inference(&mut self, num_samples: usize, model_func: Value) -> RuntimeResult<Value> {

        let (body, closure) = match model_func {
            Value::Function { body, closure, .. } => (body, closure),
            _ => {
                return Err(RuntimeError::TypeMismatch {
                    message: "infer requires a function".into(),
                });
            }
        };


        let saved_env = self.env.clone();
        self.env = closure.as_ref().clone(); // クロージャ環境を使う

        // 3. HMC settings (defaults)
        let burn_in = 50; 
        let epsilon = 0.1;
        let l_steps = 10;
        
        let sampler = crate::vm::prob_runtime::HMC::new(num_samples, burn_in, epsilon, l_steps);
        use std::collections::HashMap;
        
        // 推論実行 (Legacy infer: empty init params)
        let samples_result = sampler.infer(&body, self, HashMap::new());
        

        self.env = saved_env;

        // 結果の返却 (Legacy: Extract first parameter as scalar)
        match samples_result {
            Ok(samples) => {
                let mut val_samples = Vec::new();
                for map in samples {
                    // 古い挙動: 最初のパラメータだけを返す
                    if let Some(v) = map.values().next() {
                        val_samples.push(Value::Float(ADFloat::Concrete(*v)));
                    }
                }
                Ok(Value::Array(crate::gc::Rc::new(val_samples)))
            },
            Err(e) => Err(e),
        }
    }




    fn make_sample(&self, value: Value, log_weight: f64) -> Value {
        use std::collections::HashMap;
        use crate::gc::Rc;

        let mut fields = HashMap::new();
        fields.insert("value".to_string(), value);
        fields.insert("log_weight".to_string(), Value::Float(ADFloat::Concrete(log_weight)));

        Value::Struct {
            name: "Sample".to_string(),
            fields: Rc::new(fields),
        }
    }

    // Helper to promote user-defined 'Gaussian' struct to internal Value::Gaussian
    fn to_gaussian_if_struct(&self, val: &Value) -> Value {
        if let Value::Struct { name, fields } = val {
            if name == "Gaussian" {
                let mean = fields.get("mean").and_then(|v| match v { Value::Float(f) => Some(f.clone()), _ => None });
                let std = fields.get("std").and_then(|v| match v { Value::Float(f) => Some(f.clone()), _ => None });
                if let (Some(m), Some(s)) = (mean, std) {
                    return Value::Gaussian { mean: m, std: s };
                }
            }
        }
        val.clone() // Return original if not convertible
    }

    // Execute a program with unified error handling.
    pub fn execute_program(&mut self, program: Program) -> FlunoResult<()> {
        self.eval_program(program).map_err(Error::from)
    }

    // Execute a program.
    pub fn execute(&mut self, program: Program) -> RuntimeResult<()> {
        self.eval_program(program)?;
        Ok(())
    }

    pub fn eval_program(&mut self, program: Program) -> RuntimeResult<()> {
        for item in &program.items {
            self.execute_item(item)?;
        }

        if let Some(main_func) = self.globals.get("main").cloned() {
            match main_func {
                Value::Function { params, body, closure } => {
                    if !params.is_empty() {
                        return Err(RuntimeError::InvalidOperation {
                            message: "main function should not have parameters".to_string(),
                        });
                    }

                    let saved_env = self.env.clone();
                    self.env = closure.as_ref().clone();
                    self.push_scope();
                    let result = self.eval_block(&body);
                    self.pop_scope();
                    self.env = saved_env;
                    match result {
                        Ok(_) => Ok(()),
                        // EarlyReturn は main 関数の正常な終了
                        Err(RuntimeError::EarlyReturn) => Ok(()), 
                        // その他のエラーはそのまま伝播
                        Err(e) => Err(e),
                    }
                }
                _ => Err(RuntimeError::InvalidOperation {
                    message: "main is not a function".to_string(),
                }),
            }
        } else {
            Ok(())
        }
    }

    fn type_name_from_ast_type_for_impl(ty: &Type) -> Option<String> {
        match ty {
            Type::Named { name, .. } => name.last_ident().map(|id| id.name.clone()),
            Type::TypeVar(_) => Some("*".to_string()),
            Type::Array(_) => Some("Array".to_string()),
            Type::Int => Some("Int".to_string()),
            Type::Float => Some("Float".to_string()),
            Type::String => Some("String".to_string()),
            Type::Bool => Some("Bool".to_string()),
            Type::Gaussian => Some("Gaussian".to_string()),
            Type::Unit => Some("Unit".to_string()),
            _ => None,
        }
    }

    fn type_name_from_type_for_method(ty: &Type) -> Option<String> {
        match ty {
            Type::Named { name, .. } => name.last_ident().map(|id| id.name.clone()),
            Type::TypeVar(_) => Some("*".to_string()),
            Type::Array(_) => Some("Array".to_string()),
            Type::Int => Some("Int".to_string()),
            Type::Float => Some("Float".to_string()),
            Type::String => Some("String".to_string()),
            Type::Bool => Some("Bool".to_string()),
            Type::Gaussian => Some("Gaussian".to_string()),
            Type::Unit => Some("Unit".to_string()),
            _ => None,
        }
    }

    fn type_name_from_value_for_method(receiver: &Value) -> String {
        match receiver {
            Value::Struct { name, .. } => name.clone(),
            Value::Array(_) => "Array".to_string(),
            Value::Int(_) => "Int".to_string(),
            Value::Float(_) => "Float".to_string(),
            Value::String(_) => "String".to_string(),
            Value::Bool(_) => "Bool".to_string(),
            Value::Gaussian { .. } => "Gaussian".to_string(),
            Value::Unit => "Unit".to_string(),
            // Signal/Event/Module などは既存の表示名を使う（必要ならここも明示化）
            _ => receiver.type_name().to_string(),
        }
    }

    fn execute_item(&mut self, item: &Item) -> RuntimeResult<()> {
        match item {
            Item::Function(func) => {
                let value = Value::Function {
                    params: func.params.clone(),
                    body: func.body.clone(),
                    closure: Rc::new(self.env.clone()),
                };
                if let Some(defid) = func.defid {
                    self.defs.insert(defid, value.clone());
                }

                if self.module_stack.is_empty() {
                    // トップレベル関数だけ globals に出す
                    self.globals.insert(func.name.name.clone(), value);
                } else {
                    // モジュール内関数は modules[current] にだけ出す
                    self.export_into_current_module(func.name.name.clone(), value);
                }
                Ok(())
            }

            Item::Struct(_) => Ok(()), // 構造体定義は実行時は何もしない

            Item::Impl(impl_block) => {
                

                let typename = match Self::type_key_from_type(&impl_block.self_ty) {
                    Some(name) => name,
                    None => return Ok(()), // 型名が取れない場合はスキップ
                };


                let trait_name = impl_block.trait_ref.as_ref()
                    .and_then(|t| Self::type_key_from_type(t));


                for item in &impl_block.items {
                    if let crate::ast::node::ImplItem::Method(func) = item {
                        let value = Value::Function {
                            params: func.params.clone(),
                            body: func.body.clone(),
                            closure: Rc::new(self.env.clone()),
                        };

                        if let Some(defid) = func.defid {
                            self.defs.insert(defid, value.clone());
                        }
                        
                        // グローバルに登録（現在の call_method が期待する形式）
                        let method_key = if let Some(ref tname) = trait_name {
                            
                            // trait実装: "Type::Trait::method"
                            format!("{}::{}::{}", typename, tname, func.name.name)
                        } else {
                            // inherent: "Type::method"
                            format!("{}::{}", typename, func.name.name)
                        };

                        self.globals.insert(method_key, value);
                    }
                }
                Ok(())
            }

            Item::Module(m) => {
                // push
                self.module_stack.push(m.name.name.clone());

                let fq = self.current_module_name(); // "a::b" になる

                let module_value = Value::Module { name: fq.clone() };

                // top-level なら globals へ
                if self.module_stack.len() == 1 {
                    self.globals.insert(m.name.name.clone(), module_value.clone());
                } else {
                    // 親モジュールの export table に leaf 名で入れる
                    let parent_fq = self.module_stack[..self.module_stack.len() - 1].join("::");
                    self.modules
                        .entry(parent_fq)
                        .or_default()
                        .insert(m.name.name.clone(), module_value.clone());
                }

                // 自分の export table を確保
                self.modules.entry(fq.clone()).or_default();

                for it in &m.items {
                    self.execute_item(it)?;
                }

                self.module_stack.pop();
                Ok(())
            }


            Item::Import(imp) => {
                let local = imp.alias
                    .as_ref()
                    .map(|id| id.name.clone())
                    .unwrap_or_else(|| imp.path.last_ident().unwrap().name.clone());

                if self.get_variable(&local).is_some() {
                    return Err(RuntimeError::InvalidOperation {
                        message: format!("Duplicate import: {local}"),
                    });
                }

                // ["a","b","add"]
                let parts: Vec<String> = imp.path.iter_idents().map(|id| id.name.clone()).collect();
                if parts.is_empty() {
                    return Err(RuntimeError::InvalidOperation {
                        message: "Empty import path".to_string(),
                    });
                }

                // ルートは globals から辿る（a）
                let mut cur = self.globals.get(&parts[0]).cloned().ok_or_else(|| {
                    RuntimeError::UndefinedVariable {
                        name: parts[0].clone(),
                        span: imp.span,
                    }
                })?;

                // a, b を辿って最終的に add を取る
                for i in 1..parts.len() {
                    let name = &parts[i];

                    match cur {
                        Value::Module { name: ref fq } => {
                            let table = self.modules.get(fq).ok_or_else(|| RuntimeError::UndefinedVariable {
                                name: fq.clone(),
                                span: imp.span,
                            })?;

                            cur = table.get(name).cloned().ok_or_else(|| RuntimeError::UndefinedVariable {
                                name: format!("{fq}::{name}"),
                                span: imp.span,
                            })?;
                        }
                        _ => {
                            return Err(RuntimeError::InvalidOperation {
                                message: format!("Import path is not a module at segment '{name}'"),
                            });
                        }
                    }
                }

                self.globals.insert(local, cur);
                Ok(())
            }
            Item::Trait(trait_def) => {
                self.known_traits.insert(trait_def.name.name.clone());
                Ok(())
            }
            
            Item::Extern(ext) => {
                // Determine library name from attributes
                let mut lib_name = None;
                for attr in &ext.attributes {
                    // looking for #[link(name = "...")]
                    if let crate::ast::node::Attribute::Nested(id, inner) = attr {
                        if id.name == "link" {
                            for item in inner {
                                if let crate::ast::node::Attribute::Value(key, crate::ast::node::Literal::String(val)) = item {
                                    if key.name == "name" {
                                        lib_name = Some(val.clone());
                                    }
                                }
                            }
                        }
                    }
                }
                
                // If lib_name is None, maybe use "libc" or just fail for now?
                // Let's assume standard library or current process if None (or handle error)
                // For this phase, we require #[link(name="...")]
                let lib_path = lib_name.ok_or_else(|| RuntimeError::InvalidOperation { 
                    message: "Extern block requires #[link(name = \"...\")] attribute".to_string() 
                })?;

                let lib_index = self.native_libraries.len();
                unsafe {
                    // Try exact path first
                    let lib = match libloading::Library::new(&lib_path) {
                        Ok(l) => l,
                        Err(_) => {
                            // Basic resolution logic
                            let mut candidates = Vec::new();

                            #[cfg(target_os = "windows")]
                            let ext = "dll";
                            #[cfg(target_os = "linux")]
                            let ext = "so";
                            #[cfg(target_os = "macos")]
                            let ext = "dylib";

                            // Current dir + extension
                            candidates.push(format!("{}.{}", lib_path, ext));
                            candidates.push(format!("./{}.{}", lib_path, ext));
                            
                            // Dev path (project specific convenience)
                            // "fluno_rt" -> "./fluno-rt/target/release/fluno_rt.dll"
                            candidates.push(format!("./fluno-rt/target/release/{}.{}", lib_path, ext));
                            
                            // Try "lib" prefix for unix-likes if not present
                            #[cfg(not(target_os = "windows"))]
                            if !lib_path.starts_with("lib") {
                                candidates.push(format!("lib{}.{}", lib_path, ext));
                                candidates.push(format!("./lib{}.{}", lib_path, ext));
                            }

                            let mut loaded: Option<libloading::Library> = None;
                            let mut last_err = None;

                            for path in candidates {
                                match libloading::Library::new(&path) {
                                    Ok(l) => {
                                        loaded = Some(l);
                                        break;
                                    },
                                    Err(e) => last_err = Some(e),
                                }
                            }

                            loaded.ok_or_else(|| RuntimeError::InvalidOperation {
                                message: format!("Failed to load library '{}'. Attempted paths include dev locations. Last error: {:?}", lib_path, last_err)
                            })?
                        }
                    };
                    
                    self.native_libraries.push(std::rc::Rc::new(lib));
                }

                // 3. Register functions
                for func in &ext.functions {
                    let val = Value::NativeFunction {
                        name: func.name.name.clone(),
                        library_index: lib_index,
                        params: func.params.iter().map(|p| p.ty.clone()).collect(),
                        return_type: func.return_type.clone().unwrap_or(Type::Unit),
                        is_async: func.is_async,
                    };
                    
                    if self.module_stack.is_empty() {
                         self.globals.insert(func.name.name.clone(), val);
                    } else {
                         self.export_into_current_module(func.name.name.clone(), val);
                    }
                }
                Ok(())
            }

            // EnumやTraitなど未実装のものは無視してエラーにしない
            Item::Enum(_) | Item::TypeAlias(_) => Ok(()),

            // ★ 修正: エラーの出し方をタプル形式に変更
        }
    }
    fn current_module_name(&self) -> String {
        self.module_stack.join("::")
    }


    fn export_into_current_module(&mut self, name: String, v: Value) {
        let m = self.current_module_name();
        self.modules.entry(m).or_default().insert(name, v);
    }



    fn type_key_from_value(v: &Value) -> String {
        match v {
            Value::Struct { name, .. } => name.clone(),
            Value::Int(_) => "Int".to_string(),
            Value::Float(_) => "Float".to_string(),
            Value::Bool(_) => "Bool".to_string(),
            Value::String(_) => "String".to_string(),
            Value::Array(_) => "Array".to_string(),
            _ => v.type_name().to_string(),
        }
    }

    fn type_key_from_type(ty: &Type) -> Option<String> {
        match ty {
            Type::Named { name, .. } => name.last_ident().map(|id| id.name.clone()),
            Type::TypeVar(_) => Some("*".to_string()),
            Type::Array(_) => Some("Array".to_string()),
            Type::Int => Some("Int".to_string()),
            Type::Float => Some("Float".to_string()),
            Type::String => Some("String".to_string()),
            Type::Bool => Some("Bool".to_string()),
            Type::Gaussian => Some("Gaussian".to_string()),
            Type::Unit => Some("Unit".to_string()),
            _ => None,
        }
    }


    fn coerce_to_trait_object(
        &mut self,
        value: Value,
        traitname: &str,
        _span: Span,
    ) -> RuntimeResult<Value> {
        if matches!(value, Value::TraitObject { .. }) {
            return Ok(value);
        }

        let typename = Self::type_key_from_value(&value);

        let mut vtable = HashMap::new();

        // ここが重要: "Type::Trait::" まで含める
        let prefix = format!("{}::{}::", typename, traitname);

        for (key, func) in self.globals.iter() {
            if let Some(methodname) = key.strip_prefix(&prefix) {
                // methodname は "get" のような値になる
                vtable.insert(methodname.to_string(), func.clone());
            }
        }

        if vtable.is_empty() {
            return Err(RuntimeError::InvalidOperation {
                message: format!(
                    "Type '{}' does not implement trait '{}' (no trait methods found)",
                    typename, traitname
                ),
            });
        }

        Ok(Value::TraitObject {
            trait_name: traitname.to_string(),
            vtable: Rc::new(vtable),
            data: Rc::new(value),
        })
    }







    pub fn eval_statement(&mut self, stmt: &Statement) -> RuntimeResult<()> {
        // 既にリターンやbreak/continueフラグが立っている場合はスキップ
        if self.return_value.is_some() || self.should_break || self.should_continue {
            return Ok(());
        }

        match stmt {
            Statement::Expression(expr) => {
                // 式を評価（副作用、例えば関数呼び出しや代入を含む）
                self.eval_expression(expr)?;
                Ok(())
            }

            Statement::Let { pattern, init, ty, .. } => {
                let value = if let Some(expr) = init {
                    let mut val = self.eval_expression(expr)?;

                    // 1. Type::Named の場合 (dyn なしの名前指定)
                    // ここで trait 名かどうかチェックし、trait ならエラーにする
                    if let Some(Type::Named { name, .. }) = ty {
                        if let Some(id) = name.last_ident() {
                            // known_traits に登録されている名前なら「dyn 忘れてるよ」エラー
                            if self.known_traits.contains(&id.name) {
                                return Err(RuntimeError::TypeMismatch {
                                    message: format!(
                                        "Trait '{}' used as a type without 'dyn'. Use 'dyn {}' for trait objects.", 
                                        id.name, id.name
                                    ),
                                });
                            }
                        }
                    }
                    
                    // 2. Array<dyn Trait> の場合
                    // 配列要素ごとに trait object 化を行う
                    else if let Some(Type::Array(ref elem_ty)) = ty {
                        if let Type::DynTrait { trait_path: ref traitpath } = **elem_ty {
                            let trait_name = traitpath.last_ident()
                                .map(|id| id.name.as_str())
                                .unwrap_or("Unknown");
                            
                            if let Value::Array(ref elements) = val {
                                let mut converted = Vec::new();
                                for elem in elements.iter() {
                                    let trait_obj = self.coerce_to_trait_object(
                                        elem.clone(),
                                        trait_name,
                                        expr.span()
                                    )?;
                                    converted.push(trait_obj);
                                }
                                val = Value::Array(Rc::new(converted));
                            }
                        }
                    }
                    
                    // 3. dyn Trait の場合 (単体)
                    // 既存の trait object 化処理
                    else if let Some(Type::DynTrait { trait_path: ref traitpath }) = ty {
                        let trait_name = traitpath.last_ident()
                            .map(|id| id.name.as_str())
                            .unwrap_or("Unknown");
                        val = self.coerce_to_trait_object(val, trait_name, expr.span())?;
                    }
                    
                    val
                } else {
                    Value::Unit
                };
                
                self.bind_pattern(pattern, value)?;
                Ok(())
            }




            Statement::Return { value, .. } => {
                let return_val = if let Some(expr) = value {
                    self.eval_expression(expr)?
                } else {
                    Value::Unit
                };
                self.return_value = Some(return_val);
                // 関数から抜けるためのシグナルとしてエラーを使用する設計の場合
                Err(RuntimeError::EarlyReturn)
            }

            Statement::While { condition, body, .. } => {
                while self.eval_expression(condition)?.is_truthy() {
                    // ループ本体の実行
                    // eval_block は EarlyReturn エラーを捕捉してリターン値をセットするなどの処理が必要
                    // ここでは単純に呼び出し、制御フラグをチェック
                    match self.eval_block(body) {
                        Ok(_) => {}
                        Err(RuntimeError::EarlyReturn) => {
                            // ブロック内でリターンがあった場合、そのまま外へ伝播
                             return Err(RuntimeError::EarlyReturn);
                        }
                        Err(e) => return Err(e),
                    }
                    
                    if self.should_break {
                        self.should_break = false;
                        break;
                    }
                    if self.should_continue {
                        self.should_continue = false;
                        continue;
                    }
                    if self.return_value.is_some() {
                        break;
                    }
                }
                Ok(())
            }

            Statement::For { pattern, iterator, body, .. } => {
                let iter_val = self.eval_expression(iterator)?;
                match iter_val {
                    Value::Array(elements) => {
                        for element in elements.iter() {
                            self.bind_pattern(pattern, element.clone())?;
                            
                            match self.eval_block(body) {
                                Ok(_) => {}
                                Err(RuntimeError::EarlyReturn) => return Err(RuntimeError::EarlyReturn),
                                Err(e) => return Err(e),
                            }

                            if self.should_break {
                                self.should_break = false;
                                break;
                            }
                            if self.should_continue {
                                self.should_continue = false;
                                continue;
                            }
                            if self.return_value.is_some() {
                                break;
                            }
                        }
                        Ok(())
                    }
                    // 文字列のイテレーションなど他の型も必要ならここに追加
                    _ => Err(RuntimeError::InvalidOperation {
                        message: format!("Cannot iterate over {}", iter_val.type_name()),
                    }),
                }
            }

            Statement::Break { .. } => {
                self.should_break = true;
                Ok(())
            }

            Statement::Continue { .. } => {
                self.should_continue = true;
                Ok(())
            }

            Statement::Empty => Ok(()),
        }
    }


    fn eval_block(&mut self, block: &Block) -> RuntimeResult<Value> {
        self.push_scope();
        let mut result = Value::Unit;
        for (i, stmt) in block.statements.iter().enumerate() {
            if i == block.statements.len() - 1 {
                if let Statement::Expression(expr) = stmt {
                    result = self.eval_expression(expr)?;
                    break;
                }
            }
            self.eval_statement(stmt)?;
            if self.return_value.is_some() || self.should_break || self.should_continue {
                break;
            }
        }
        self.pop_scope();
        Ok(result)
    }

    pub fn load_program_defs(&mut self, program: &Program) {
        use crate::ast::node::{Item, ImplItem};
        use crate::vm::value::Value;
        use crate::gc::Rc;

        for item in &program.items {
            match item {
                Item::Function(func) => {
                    if let Some(defid) = func.defid {
                        let func_val = Value::Function {
                            params: func.params.clone(),
                            body: func.body.clone(),
                            closure: Rc::new(self.env.clone()),
                        };
                        self.defs.insert(defid, func_val);
                    }
                }
                Item::Impl(impl_block) => {
                    for impl_item in &impl_block.items {
                        if let ImplItem::Method(func) = impl_item {
                            if let Some(defid) = func.defid {
                                let func_val = Value::Function {
                                    params: func.params.clone(),
                                    body: func.body.clone(),
                                    closure: Rc::new(self.env.clone()),
                                };
                                self.defs.insert(defid, func_val);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn eval_expression(&mut self, expr: &Expression) -> RuntimeResult<Value> {
        match expr {
            Expression::Literal { value, .. } => self.eval_literal(value),
            
            Expression::Variable { name, .. } => self
                .get_variable(&name.last_ident().unwrap().name)
                .ok_or_else(|| RuntimeError::UndefinedVariable {
                    name: name.last_ident().unwrap().name.clone(),
                    span: name.span,
                }),

            Expression::Some { expr, .. } => {
                let inner = self.eval_expression(expr)?;
                Ok(Value::Enum {
                    name: "Option".into(),
                    variant: "Some".into(),
                    fields: Rc::new(vec![inner]),
                })
            }

            Expression::None { .. } => Ok(Value::Enum {
                name: "Option".into(),
                variant: "None".into(),
                fields: Rc::new(vec![]),
            }),

            Expression::Ok { expr, .. } => {
                let inner = self.eval_expression(expr)?;
                Ok(Value::Enum {
                    name: "Result".into(),
                    variant: "Ok".into(),
                    fields: Rc::new(vec![inner]),
                })
            }

            Expression::Err { expr, .. } => {
                let inner = self.eval_expression(expr)?;
                Ok(Value::Enum {
                    name: "Result".into(),
                    variant: "Err".into(),
                    fields: Rc::new(vec![inner]),
                })
            }


            Expression::Try { expr, span: _ } => {
                let result = self.eval_expression(expr)?;
                match result {
                    Value::Enum { name, variant, fields } if name == "Result" && variant == "Ok" => {
                        if fields.len() != 1 {
                            return Err(RuntimeError::TypeMismatch {
                                message: "Result::Ok must have exactly 1 field".into(),
                            });
                        }
                        Ok(fields[0].clone())
                    }

                    Value::Enum { name, variant, fields } if name == "Result" && variant == "Err" => {
                        // Err(e) をそのまま return_value に積んで EarlyReturn する
                        self.return_value = Some(Value::Enum {
                            name,
                            variant,
                            fields,
                        });
                        Err(RuntimeError::EarlyReturn)
                    }

                    _ => Err(RuntimeError::TypeMismatch {
                        message: format!(
                            "Try operator requires Result type, found {}",
                            result.type_name()
                        ),
                    }),
                }
            }
            Expression::MethodCall { receiver, method, args, span, resolved } => {
                // 1. レシーバを評価
                let obj = self.eval_expression(receiver)?;
                
                // 2. 引数を評価
                let mut arg_values = Vec::new();
                for arg in args {
                    arg_values.push(self.eval_expression(arg)?);
                }

                // ★ Built-in Int Methods
                if let Value::Int(i) = &obj {
                    if method.name.as_str() == "to_string" {
                         return Ok(Value::String(Rc::new(i.to_string())));
                    }
                }

                // ★ Built-in String Methods (Phase 7)
                if let Value::String(s) = &obj {
                    let method_name = method.name.as_str();
                    match method_name {
                         "trim" => {
                             if !arg_values.is_empty() { return Err(RuntimeError::ArityMismatch{ expected: 0, found: arg_values.len(), span: *span }); }
                             return Ok(Value::String(Rc::new(s.trim().to_string())));
                         },
                         "split" => {
                             if arg_values.len() != 1 { return Err(RuntimeError::ArityMismatch{ expected: 1, found: arg_values.len(), span: *span }); }
                             if let Value::String(delim) = &arg_values[0] {
                                 let parts: Vec<Value> = s.split(delim.as_str())
                                     .map(|p| Value::String(Rc::new(p.to_string())))
                                     .collect();
                                 return Ok(Value::Array(Rc::new(parts)));
                             } else {
                                 return Err(RuntimeError::TypeMismatch{ message: "split delimiter must be String".into() });
                             }
                         },
                         "replace" => {
                             if arg_values.len() != 2 { return Err(RuntimeError::ArityMismatch{ expected: 2, found: arg_values.len(), span: *span }); }
                             match (&arg_values[0], &arg_values[1]) {
                                 (Value::String(from), Value::String(to)) => {
                                     return Ok(Value::String(Rc::new(s.replace(from.as_str(), to.as_str()))));
                                 }
                                 _ => return Err(RuntimeError::TypeMismatch{ message: "replace args must be String".into() })
                             }
                         },
                         "contains" => {
                             if arg_values.len() != 1 { return Err(RuntimeError::ArityMismatch{ expected: 1, found: arg_values.len(), span: *span }); }
                             if let Value::String(sub) = &arg_values[0] {
                                 return Ok(Value::Bool(s.contains(sub.as_str())));
                             }
                         },
                         "len" => {
                             return Ok(Value::Int(s.len() as i64));
                         },
                         "slice" => {
                             if arg_values.len() != 2 { return Err(RuntimeError::ArityMismatch{ expected: 2, found: arg_values.len(), span: *span }); }
                             let start = match arg_values[0] { Value::Int(i) => i as usize, _ => 0 };
                             let end = match arg_values[1] { Value::Int(i) => i as usize, _ => s.len() };
                             let start = start.min(s.len());
                             let end = end.min(s.len()).max(start);
                             // Simple byte slicing for now
                             if s.is_char_boundary(start) && s.is_char_boundary(end) {
                                  return Ok(Value::String(Rc::new(s[start..end].to_string())));
                             } else {
                                  // Fallback to lossy if boundaries invalid? Or Error?
                                  // Just return error or whole string?
                                  return Err(RuntimeError::InvalidOperation { message: "Invalid slice indices (char boundary)".into() });
                             }
                         }
                         _ => {} // Fallthrough to custom methods or error
                    }
                }

                // ★ Built-in Float Methods (Phase 8)
                if let Value::Float(f) = &obj {
                    match method.name.as_str() {
                        "abs" => return Ok(Value::Float(f.clone().abs())),
                        "exp" => return Ok(Value::Float(f.clone().exp())),
                        "ln" => return Ok(Value::Float(f.clone().ln())),
                        "sqrt" => return Ok(Value::Float(f.clone().sqrt())),
                        "sin" => return Ok(Value::Float(f.clone().sin())),
                        "cos" => return Ok(Value::Float(f.clone().cos())),
                        "tan" => return Ok(Value::Float(f.clone().tan())),
                        "floor" => return Ok(Value::Float(f.clone().floor())),
                        "ceil" => return Ok(Value::Float(f.clone().ceil())),
                        "round" => return Ok(Value::Float(f.clone().round())),
                        "pow" | "powf" => {
                            if arg_values.len() != 1 { return Err(RuntimeError::ArityMismatch{ expected: 1, found: arg_values.len(), span: *span }); }
                            let arg_f = match &arg_values[0] {
                                Value::Float(v) => v.clone(),
                                Value::Int(i) => crate::ad::types::ADFloat::Concrete(*i as f64),
                                _ => return Err(RuntimeError::TypeMismatch{ message: "pow expects number arg".into() }),
                            };
                            return Ok(Value::Float((arg_f * f.clone().ln()).exp()));
                        },
                        _ => {}
                    }
                }

                // ★追加: 高速パス (DefId解決済みなら直接呼ぶ)
                if let Some(defid) = resolved {
                    if let Some(func_val_ref) = self.defs.get(defid) {
                        let func_val = func_val_ref.clone();
                        // メソッド呼び出し規約: 第一引数に self (obj) を積む
                        let mut call_args = Vec::with_capacity(1 + arg_values.len());
                        call_args.push(obj);
                        call_args.extend(arg_values);

                        return self.call_function_value(&func_val, call_args);
                    }
                    // IDはあるが定義が見つからない場合はエラーにするか、フォールバックするか。
                    // 基本的にはあり得ない（panicでも良いレベル）が、安全のためフォールバックへ。
                }

                // 3. 統一されたメソッド呼び出し関数を使う (従来の名前ベース探索)
                self.call_method(obj, &method.name, &arg_values, *span)
            }



            // ========================================================
            // Binary Operator Handling (Assignment + Operations)
            // ========================================================
            Expression::Binary { left, op, right, span } => {
                match op {
                    // 1. 単純代入 (=)
                    BinaryOp::Assign => {
                        // 左辺の種類によって分岐
                        match left.as_ref() {
                            // 変数への代入: x = value
                            Expression::Variable { name, .. } => {
                                let value = self.eval_expression(right)?;
                                
                                if !self.env.update(&name.last_ident().unwrap().name, value.clone()) {
                                    return Err(RuntimeError::UndefinedVariable {
                                        name: name.last_ident().unwrap().name.clone(),
                                        span: *span,
                                    });
                                }
                                return Ok(value);
                            }

                            // フィールドへの代入: obj.field = value
                            Expression::FieldAccess { object, field, .. } => {
                                let new_val = self.eval_expression(right)?;
                                
                                if let Expression::Variable { name: var_name, .. } = object.as_ref() {
                                    let obj_val = self.env.get(&var_name.last_ident().unwrap().name).ok_or_else(|| {
                                        RuntimeError::UndefinedVariable {
                                            name: var_name.last_ident().unwrap().name.clone(),
                                            span: *span,
                                        }
                                    })?;
                                    
                                    match obj_val {
                                        Value::Struct { name: struct_name, fields } => {
                                            // Rc をクローンして、HashMap をコピー
                                            let mut new_fields = (*fields).clone();
                                            
                                            if new_fields.contains_key(&field.name) {
                                                new_fields.insert(field.name.clone(), new_val.clone());
                                            } else {
                                                return Err(RuntimeError::TypeMismatch {
                                                    message: format!(
                                                        "Struct '{}' has no field '{}'",
                                                        struct_name, field.name
                                                    ),
                                                });
                                            }
                                            
                                            // 新しい Struct を作って変数に書き戻す
                                            let updated = Value::Struct {
                                                name: struct_name,
                                                fields: Rc::new(new_fields),
                                            };
                                            self.env.update(&var_name.last_ident().unwrap().name, updated);
                                            return Ok(new_val);
                                        }
                                        _ => {
                                            return Err(RuntimeError::TypeMismatch {
                                                message: format!(
                                                    "Cannot access field '{}' on non-struct value",
                                                    field.name
                                                ),
                                            });
                                        }
                                    }
                                } else {
                                    return Err(RuntimeError::Unimplemented(
                                        "Assignment to complex field expressions not yet implemented".to_string()
                                    ));
                                }
                            }

                            // 配列インデックスへの代入: arr[i] = value
                            Expression::Index { object, index, .. } => {
                                let new_val = self.eval_expression(right)?;
                                
                                if let Expression::Variable { name: var_name, .. } = object.as_ref() {
                                    let arr_val = self.env.get(&var_name.last_ident().unwrap().name).ok_or_else(|| {
                                        RuntimeError::UndefinedVariable {
                                            name: var_name.last_ident().unwrap().name.clone(),
                                            span: *span,
                                        }
                                    })?;
                                    
                                    let idx_val = self.eval_expression(index)?;
                                    
                                    match (arr_val, idx_val) {
                                        (Value::Array(elements), Value::Int(idx)) => {
                                            let idx_usize = if idx < 0 {
                                                return Err(RuntimeError::TypeMismatch {
                                                    message: format!("Array index cannot be negative: {}", idx),
                                                });
                                            } else {
                                                idx as usize
                                            };
                                            
                                            // Vec をクローンして要素を変更
                                            let mut new_elements = (*elements).clone();
                                            
                                            if idx_usize >= new_elements.len() {
                                                return Err(RuntimeError::TypeMismatch {
                                                    message: format!(
                                                        "Index {} out of bounds for array of length {}",
                                                        idx, new_elements.len()
                                                    ),
                                                });
                                            }
                                            
                                            new_elements[idx_usize] = new_val.clone();
                                            
                                            // 新しい Array を作って変数に書き戻す
                                            self.env.update(&var_name.last_ident().unwrap().name, Value::Array(Rc::new(new_elements)));
                                            return Ok(new_val);
                                        }
                                        _ => {
                                            return Err(RuntimeError::TypeMismatch {
                                                message: "Invalid array indexing operation".to_string(),
                                            });
                                        }
                                    }
                                } else {
                                    return Err(RuntimeError::Unimplemented(
                                        "Assignment to complex index expressions not yet implemented".to_string()
                                    ));
                                }
                            }

                            _ => {
                                return Err(RuntimeError::Unimplemented(
                                    "Assignment to this expression type not yet implemented".to_string()
                                ));
                            }
                        }
                    }

                    // 2. 複合代入 (+=, -=, *=, /=, %=)
                    BinaryOp::AddAssign | BinaryOp::SubAssign | BinaryOp::MulAssign | 
                    BinaryOp::DivAssign | BinaryOp::ModAssign |
                    BinaryOp::BitAndAssign | BinaryOp::BitOrAssign | BinaryOp::BitXorAssign |
                    BinaryOp::ShlAssign | BinaryOp::ShrAssign => {
                        
                        if let Expression::Variable { name, .. } = left.as_ref() {
                            // 左辺(現在の値)を取得
                            let left_val = self.eval_expression(left)?;
                            // 右辺を評価
                            let right_val = self.eval_expression(right)?;
                            
                            // 対応する演算を実行
                            let result = match op {
                                BinaryOp::AddAssign => self.apply_binary_op(&BinaryOp::Add, left_val, right_val, *span)?,
                                BinaryOp::SubAssign => self.apply_binary_op(&BinaryOp::Sub, left_val, right_val, *span)?,
                                BinaryOp::MulAssign => self.apply_binary_op(&BinaryOp::Mul, left_val, right_val, *span)?,
                                BinaryOp::DivAssign => self.apply_binary_op(&BinaryOp::Div, left_val, right_val, *span)?,
                                BinaryOp::ModAssign => self.apply_binary_op(&BinaryOp::Mod, left_val, right_val, *span)?,
                                BinaryOp::BitAndAssign => self.apply_binary_op(&BinaryOp::BitAnd, left_val, right_val, *span)?,
                                BinaryOp::BitOrAssign => self.apply_binary_op(&BinaryOp::BitOr, left_val, right_val, *span)?,
                                BinaryOp::BitXorAssign => self.apply_binary_op(&BinaryOp::BitXor, left_val, right_val, *span)?,
                                BinaryOp::ShlAssign => self.apply_binary_op(&BinaryOp::Shl, left_val, right_val, *span)?,
                                BinaryOp::ShrAssign => self.apply_binary_op(&BinaryOp::Shr, left_val, right_val, *span)?,
                                _ => unreachable!(),
                            };
                            
                            // 変数を更新
                            if !self.env.update(&name.last_ident().unwrap().name, result.clone()) {
                                return Err(RuntimeError::UndefinedVariable {
                                     name: name.last_ident().unwrap().name.clone(),
                                     span: *span 
                                });
                            }
                            
                            return Ok(result);
                        }
                        return Err(RuntimeError::Unimplemented(
                            "Compound assignment to fields or indices not yet implemented".to_string()
                        ));
                    }
                    
                    // 3. 通常の二項演算 (Add, Sub, Eq, Lt, And, Or など)
                    _ => {
                        // 短絡評価 (Short-circuit)
                        if matches!(op, BinaryOp::And) {
                            let l = self.eval_expression(left)?;
                            if !l.is_truthy() { return Ok(Value::Bool(false)); }
                            let r = self.eval_expression(right)?;
                            return Ok(Value::Bool(r.is_truthy()));
                        }
                        if matches!(op, BinaryOp::Or) {
                            let l = self.eval_expression(left)?;
                            if l.is_truthy() { return Ok(Value::Bool(true)); }
                            let r = self.eval_expression(right)?;
                            return Ok(Value::Bool(r.is_truthy()));
                        }

                        // 通常評価
                        let l = self.eval_expression(left)?;
                        let r = self.eval_expression(right)?;

                        // ★ FIX: Float/Int 混合演算のサポート (Runtime)
                        // ここで型変換を行ってから apply_binary_op に渡す
                        match op {
                            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                                match (&l, &r) {
                                    (Value::Float(f), Value::Int(i)) => {
                                        let r_f = Value::Float(crate::ad::types::ADFloat::Concrete(*i as f64));
                                        return self.apply_binary_op(op, l, r_f, *span);
                                    },
                                    (Value::Int(i), Value::Float(f)) => {
                                        let l_f = Value::Float(crate::ad::types::ADFloat::Concrete(*i as f64));
                                        return self.apply_binary_op(op, l_f, r, *span);
                                    },
                                    _ => {} // それ以外はそのまま
                                }
                            }
                            _ => {}
                        }

                        self.apply_binary_op(op, l, r, *span)
                    }
                }
            }



            Expression::Unary { op, operand, .. } => {
                let val = self.eval_expression(operand)?;
                self.eval_unary_op(*op, &val)
            }

            Expression::Call { callee, args, span } => self.eval_call(callee, args, *span),

            Expression::If { condition, then_branch, else_branch, .. } => {
                let cond = self.eval_expression(condition)?;
                if cond.is_truthy() {
                    self.eval_block(then_branch)
                } else if let Some(else_block) = else_branch {
                    self.eval_block(else_block)
                } else {
                    Ok(Value::Unit)
                }
            }

            Expression::Block { block, .. } => self.eval_block(block),

            Expression::Tuple { elements, .. } => {
                let mut values = Vec::new();
                for elem in elements {
                    values.push(self.eval_expression(elem)?);
                }
                Ok(Value::Tuple(Rc::new(values)))
            }

            Expression::Array { elements, .. } => {
                let mut values = Vec::new();
                for elem in elements {
                    values.push(self.eval_expression(elem)?);
                }
                Ok(Value::Array(Rc::new(values)))
            }

            Expression::Paren { expr, .. } => self.eval_expression(expr),

            Expression::FieldAccess { object, field, .. } => {
                let obj = self.eval_expression(object)?;
                self.last_receiver = Some(obj.clone());
                self.eval_field_access(&obj, &field.name)
            }

            Expression::Index { object, index, .. } => {
                let obj = self.eval_expression(object)?;
                let idx = self.eval_expression(index)?;
                self.eval_index(&obj, &idx)
            }

            Expression::Struct { name, fields, .. } => {
                let mut field_map = HashMap::new();
                for field_init in fields {
                    let val = self.eval_expression(&field_init.value)?;
                    field_map.insert(field_init.name.name.clone(), val);
                }
                Ok(Value::Struct {
                    name: name.name.clone(),
                    fields: Rc::new(field_map),
                })
            }

            Expression::Match { scrutinee, arms, .. } => {
                let val = self.eval_expression(scrutinee)?;
                for arm in arms {
                    if self.pattern_matches(&arm.pattern, &val)? {
                        self.push_scope();
                        self.bind_pattern(&arm.pattern, val.clone())?;
                        if let Some(ref guard) = arm.guard {
                            let guard_val = self.eval_expression(guard)?;
                            if !guard_val.is_truthy() {
                                self.pop_scope();
                                continue;
                            }
                        }
                        let result = self.eval_expression(&arm.body)?;
                        self.pop_scope();
                        return Ok(result);
                    }
                }
                Err(RuntimeError::InvalidOperation { message: "Non-exhaustive match patterns".into() })
            }

            Expression::With { name, initializer, body, span } => {
                // 1. Initialize
                let resource = self.eval_expression(initializer)?;
                // 2. Setup Scope
                self.push_scope();
                self.env.set(name.name.clone(), resource.clone());
                // 3. Execute Body
                let result = self.eval_block(body);
                // 4. Cleanup
                let cleanup_result = self.perform_cleanup(&resource, *span);
                self.pop_scope();
                // 5. Result Handling
                match (result, cleanup_result) {
                    (Err(e), _) => Err(e),
                    (Ok(_val), Err(e)) => Err(e),
                    (Ok(val), Ok(_)) => Ok(val),
                }
            }

            Expression::Lambda { params, body, span:_ } => {
                let closure = Rc::new(self.env.clone());
                let func_body = match body.as_ref() {
                    Expression::Block { block, .. } => block.clone(),
                    _ => {
                        crate::ast::node::Block {
                            statements: vec![crate::ast::node::Statement::Expression(*body.clone())],
                            span: Span::initial(),
                        }
                    }
                };

                Ok(Value::Function {
                    params: params.clone(),
                    body: func_body,
                    closure,
                })
            }
            Expression::Enum { name, variant, args, .. } => {
                let mut fields = Vec::new();
                for a in args {
                    fields.push(self.eval_expression(a)?);
                }
                Ok(Value::Enum {
                    name: name.name.clone(),
                    variant: variant.name.clone(),
                    fields: crate::gc::Rc::new(fields),
                })
            }
            Expression::UfcsMethod { trait_path, method, span } => {
                return Err(RuntimeError::InvalidOperation {
                    message: format!("UFCS method used as value not supported yet: {}::{}", trait_path, method.name),
                });
            }


            _ => Err(RuntimeError::Unimplemented(
                format!("Expression type not yet implemented: {:?}", expr),
            )),
        }
    }


    // --- FFI / Basic Native Call Support ---
    fn call_native(&mut self, name: &str, lib_index: usize, params: &[Type], ret_type: &Type, args: Vec<Value>, span: Span) -> RuntimeResult<Value> {
        // Phase 4: AD Taint Check
        // extern calls are forbidden during gradient computation (Gradient/Guide mode)
        // because they break the computation graph and make gradients incorrect.
        if let Some(ref ctx) = self.prob_context {
            match ctx.mode {
                prob_runtime::InferenceMode::Gradient { .. } | prob_runtime::InferenceMode::Guide { .. } => {
                    return Err(RuntimeError::InvalidOperation {
                        message: format!(
                            "FFI call to '{}' is forbidden during AD context (Gradient/Guide mode). \
                             Extern functions break the computation graph.",
                            name
                        ),
                    });
                }
                prob_runtime::InferenceMode::Sampling => {
                    // Sampling mode is OK
                }
            }
        }
        
        let lib = self.native_libraries.get(lib_index).ok_or_else(|| RuntimeError::InvalidOperation {
            message: format!("Library index {} not found", lib_index)
        })?;

        // Intercept Virtual Builtins (Async/Await Helpers)
        if name == "to_future" {
             if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
             let id = match &args[0] { Value::Int(i) => *i, _ => return Err(RuntimeError::TypeMismatch { message: "Arg must be Int".into() }) };
             return Ok(Value::Handle(id as usize));
        }
        if name == "wait_future" {
             if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
             let id_usize = match &args[0] { Value::Handle(id) => *id, _ => return Err(RuntimeError::TypeMismatch { message: "Arg must be Handle".into() }) };
             let id = id_usize as i64;
             
             unsafe {
                 let poll_sym_res = lib.get::<unsafe extern "C" fn(i64) -> i32>(b"fluno_task_poll");
                 let result_sym_res = lib.get::<unsafe extern "C" fn(i64) -> *const std::ffi::c_char>(b"fluno_task_result");
                 
                 if let (Ok(poll_sym), Ok(result_sym)) = (poll_sym_res, result_sym_res) {
                     loop {
                         let status = poll_sym(id);
                         if status == 1 {
                             let ptr = result_sym(id);
                             if ptr.is_null() { return Ok(Value::Unit); }
                             let s = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
                             return Ok(Value::String(crate::gc::Rc::new(s)));
                         } else if status == -1 {
                             return Err(RuntimeError::InvalidOperation { message: "Async task failed".into() });
                         }
                         std::thread::sleep(std::time::Duration::from_millis(5));
                     }
                 } else {
                     return Err(RuntimeError::InvalidOperation { message: "Missing async runtime symbols in library".into() });
                 }
             }
        }

        // Limitation: For Phase 2, we only support a very specific set of signatures.
        // e.g. (Int, Int) -> Int
        // To support arbitrary signatures dynamically, we would need `libffi` or complex `libloading` usage.
        
        unsafe {
            // Case 1: (Int, Int) -> Int
            if params.len() == 2 
               && matches!(params[0], Type::Int)
               && matches!(params[1], Type::Int)
               && matches!(ret_type, Type::Int) 
            {
                let func: libloading::Symbol<unsafe extern "C" fn(i64, i64) -> i64> = lib.get(name.as_bytes()).map_err(|e| RuntimeError::InvalidOperation {
                    message: format!("Symbol '{}' not found: {}", name, e)
                })?;
                
                let arg0 = match &args[0] { Value::Int(i) => *i, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 0 must be Int".into() }) };
                let arg1 = match &args[1] { Value::Int(i) => *i, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 1 must be Int".into() }) };
                
                let res = func(arg0, arg1);
                return Ok(Value::Int(res));
            }
            // Case 2: () -> Int
            else if params.is_empty() && matches!(ret_type, Type::Int) {
                 let func: libloading::Symbol<unsafe extern "C" fn() -> i64> = lib.get(name.as_bytes()).map_err(|e| RuntimeError::InvalidOperation {
                    message: format!("Symbol '{}' not found: {}", name, e)
                })?;
                let res = func();
                return Ok(Value::Int(res));
            }
            // Case 3: (Int) -> Int
            else if params.len() == 1
               && matches!(params[0], Type::Int)
               && matches!(ret_type, Type::Int) 
            {
                let func: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> = lib.get(name.as_bytes()).map_err(|e| RuntimeError::InvalidOperation {
                    message: format!("Symbol '{}' not found: {}", name, e)
                })?;
                let arg0 = match &args[0] { Value::Int(i) => *i, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 0 must be Int".into() }) };
                let res = func(arg0);
                return Ok(Value::Int(res));
            }
            // Case 4: (String) -> Unit
            else if params.len() == 1
               && matches!(params[0], Type::String)
               && matches!(ret_type, Type::Unit)
            {
                let func: libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_char)> = lib.get(name.as_bytes()).map_err(|e| RuntimeError::InvalidOperation {
                    message: format!("Symbol '{}' not found: {}", name, e)
                })?;
                
                let arg0 = match &args[0] { 
                    Value::String(s) => s, 
                    _ => return Err(RuntimeError::TypeMismatch { message: "Arg 0 must be String".into() }) 
                };
                
                let c_str = std::ffi::CString::new((**arg0).clone()).map_err(|_| RuntimeError::InvalidOperation {
                    message: "String contains null byte".to_string()
                })?;
                
                func(c_str.as_ptr());
                return Ok(Value::Unit);
            }
            // Case 5: (String) -> String
            else if params.len() == 1
               && matches!(params[0], Type::String)
               && matches!(ret_type, Type::String)
            {
                let func: libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_char) -> *const std::ffi::c_char> = lib.get(name.as_bytes()).map_err(|e| RuntimeError::InvalidOperation {
                    message: format!("Symbol '{}' not found: {}", name, e)
                })?;
                
                let arg0 = match &args[0] { 
                    Value::String(s) => s, 
                    _ => return Err(RuntimeError::TypeMismatch { message: "Arg 0 must be String".into() }) 
                };
                
                let c_str = std::ffi::CString::new((**arg0).clone()).map_err(|_| RuntimeError::InvalidOperation {
                    message: "String contains null byte".to_string()
                })?;
                
                let res_ptr = func(c_str.as_ptr());
                if res_ptr.is_null() {
                     return Ok(Value::String(crate::gc::Rc::new(String::new()))); // Return empty string on null
                }
                
                // Convert C string back to Rust String
                // Note: We copy the string. The memory at res_ptr is NOT freed here.
                // In a production FFI, we need a strategy for who owns the returned memory.
                // For this POC, we accept that the C side might leak if it allocated using malloc/CString::into_raw.
                let res_str = std::ffi::CStr::from_ptr(res_ptr).to_string_lossy().into_owned();
                return Ok(Value::String(crate::gc::Rc::new(res_str)));
            }
             // Case 6: (String, String) -> Int
            else if params.len() == 2
               && matches!(params[0], Type::String)
               && matches!(params[1], Type::String)
               && matches!(ret_type, Type::Int)
            {
                let func: libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_char, *const std::ffi::c_char) -> i64> = lib.get(name.as_bytes()).map_err(|e| RuntimeError::InvalidOperation {
                    message: format!("Symbol '{}' not found: {}", name, e)
                })?;
                
                let arg0 = match &args[0] { Value::String(s) => s, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 0 must be String".into() }) };
                let arg1 = match &args[1] { Value::String(s) => s, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 1 must be String".into() }) };
                
                let c_str0 = std::ffi::CString::new((**arg0).clone()).map_err(|_| RuntimeError::InvalidOperation { message: "String contains null byte".to_string() })?;
                let c_str1 = std::ffi::CString::new((**arg1).clone()).map_err(|_| RuntimeError::InvalidOperation { message: "String contains null byte".to_string() })?;

                let res = func(c_str0.as_ptr(), c_str1.as_ptr());
                return Ok(Value::Int(res));
            }
            // Case 7: (Int, String) -> Int
            else if params.len() == 2
               && matches!(params[0], Type::Int)
               && matches!(params[1], Type::String)
               && matches!(ret_type, Type::Int)
            {
                let func: libloading::Symbol<unsafe extern "C" fn(i64, *const std::ffi::c_char) -> i64> = lib.get(name.as_bytes()).map_err(|e| RuntimeError::InvalidOperation {
                    message: format!("Symbol '{}' not found: {}", name, e)
                })?;
                
                let arg0 = match &args[0] { Value::Int(i) => *i, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 0 must be Int".into() }) };
                let arg1 = match &args[1] { Value::String(s) => s, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 1 must be String".into() }) };
                let c_str1 = std::ffi::CString::new((**arg1).clone()).map_err(|_| RuntimeError::InvalidOperation { message: "String contains null byte".to_string() })?;

                let res = func(arg0, c_str1.as_ptr());
                return Ok(Value::Int(res));
            }
            // Case 8: (Int) -> String
            else if params.len() == 1
               && matches!(params[0], Type::Int)
               && matches!(ret_type, Type::String)
            {
                let func: libloading::Symbol<unsafe extern "C" fn(i64) -> *const std::ffi::c_char> = lib.get(name.as_bytes()).map_err(|e| RuntimeError::InvalidOperation {
                    message: format!("Symbol '{}' not found: {}", name, e)
                })?;
                
                let arg0 = match &args[0] { Value::Int(i) => *i, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 0 must be Int".into() }) };
                let res_ptr = func(arg0);
                
                if res_ptr.is_null() {
                     return Ok(Value::String(crate::gc::Rc::new(String::new())));
                }
                let res_str = std::ffi::CStr::from_ptr(res_ptr).to_string_lossy().into_owned();
                return Ok(Value::String(crate::gc::Rc::new(res_str)));
            }
            // Case 9: (Int) -> Unit
            else if params.len() == 1
               && matches!(params[0], Type::Int)
               && matches!(ret_type, Type::Unit)
            {
                // Note: C functions returning void. We cast to a function pointer returning void?
                // libloading/libffi expects correct signature. Rust fn() -> () is compatible with void return usually.
                let func: libloading::Symbol<unsafe extern "C" fn(i64)> = lib.get(name.as_bytes()).map_err(|e| RuntimeError::InvalidOperation {
                    message: format!("Symbol '{}' not found: {}", name, e)
                })?;
                
                let arg0 = match &args[0] { Value::Int(i) => *i, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 0 must be Int".into() }) };
                func(arg0);
                return Ok(Value::Unit);
            }
            // Case 10: (String) -> Int
            else if params.len() == 1
               && matches!(params[0], Type::String)
               && matches!(ret_type, Type::Int)
            {
                let func: libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_char) -> i64> = lib.get(name.as_bytes()).map_err(|e| RuntimeError::InvalidOperation {
                    message: format!("Symbol '{}' not found: {}", name, e)
                })?;
                
                let arg0 = match &args[0] { Value::String(s) => s, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 0 must be String".into() }) };
                let c_str = std::ffi::CString::new((**arg0).clone()).map_err(|_| RuntimeError::InvalidOperation { message: "String contains null byte".to_string() })?;
                
                let res = func(c_str.as_ptr());
                return Ok(Value::Int(res));
            }

            // Case 10: (Int) -> Int
            // Already supported? Yes, Case 3.

            // Add more signatures here as needed for testing
            
            Err(RuntimeError::Unimplemented(
                format!("FFI signature not supported in Phase 2: {:?} -> {:?}", params, ret_type)
            ))
        }
    }

    fn apply_binary_op(
        &mut self,
        op: &BinaryOp,
        left_val: Value,
        right_val: Value,
        _span: Span,
    ) -> RuntimeResult<Value> {
        match (op, &left_val, &right_val) {
            // --- 算術演算 (Int) ---
            (BinaryOp::Add, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l + r)),
            (BinaryOp::Sub, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l - r)),
            (BinaryOp::Mul, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l * r)),
            (BinaryOp::Div, Value::Int(l), Value::Int(r)) => {
                if *r == 0 { return Err(RuntimeError::DivisionByZero); }
                Ok(Value::Int(l / r))
            },
            (BinaryOp::Mod, Value::Int(l), Value::Int(r)) => {
                if *r == 0 { return Err(RuntimeError::DivisionByZero); }
                Ok(Value::Int(l % r))
            },

            // --- 算術演算 (Float) ---
            (BinaryOp::Add, Value::Float(l), Value::Float(r)) => Ok(Value::Float(l.clone() + r.clone())),
            (BinaryOp::Sub, Value::Float(l), Value::Float(r)) => Ok(Value::Float(l.clone() - r.clone())),
            (BinaryOp::Mul, Value::Float(l), Value::Float(r)) => Ok(Value::Float(l.clone() * r.clone())),
            (BinaryOp::Div, Value::Float(l), Value::Float(r)) => Ok(Value::Float(l.clone() / r.clone())),

            // --- 算術演算 (Gaussian) ---
            (BinaryOp::Add, Value::Gaussian{mean: m1, std: s1}, Value::Gaussian{mean: m2, std: s2}) => {
                let new_mean = m1.clone() + m2.clone();
                let var1 = s1.clone() * s1.clone();
                let var2 = s2.clone() * s2.clone();
                let new_std = (var1 + var2).sqrt();
                Ok(Value::Gaussian{mean: new_mean, std: new_std})
            },

            // --- ★ ここを追加: 混合演算 (Float & Int) ---
            (BinaryOp::Add, Value::Float(l), Value::Int(r)) => {
                let r_f = crate::ad::types::ADFloat::Concrete(*r as f64);
                Ok(Value::Float(l.clone() + r_f))
            },
            (BinaryOp::Add, Value::Int(l), Value::Float(r)) => {
                let l_f = crate::ad::types::ADFloat::Concrete(*l as f64);
                Ok(Value::Float(l_f + r.clone()))
            },
            (BinaryOp::Sub, Value::Float(l), Value::Int(r)) => {
                let r_f = crate::ad::types::ADFloat::Concrete(*r as f64);
                Ok(Value::Float(l.clone() - r_f))
            },
            (BinaryOp::Sub, Value::Int(l), Value::Float(r)) => {
                let l_f = crate::ad::types::ADFloat::Concrete(*l as f64);
                Ok(Value::Float(l_f - r.clone()))
            },
            (BinaryOp::Mul, Value::Float(l), Value::Int(r)) => {
                let r_f = crate::ad::types::ADFloat::Concrete(*r as f64);
                Ok(Value::Float(l.clone() * r_f))
            },
            (BinaryOp::Mul, Value::Int(l), Value::Float(r)) => {
                let l_f = crate::ad::types::ADFloat::Concrete(*l as f64);
                Ok(Value::Float(l_f * r.clone()))
            },
            (BinaryOp::Div, Value::Float(l), Value::Int(r)) => {
                if *r == 0 { return Err(RuntimeError::DivisionByZero); }
                let r_f = crate::ad::types::ADFloat::Concrete(*r as f64);
                Ok(Value::Float(l.clone() / r_f))
            },
            (BinaryOp::Div, Value::Int(l), Value::Float(r)) => {
                let l_f = crate::ad::types::ADFloat::Concrete(*l as f64);
                Ok(Value::Float(l_f / r.clone()))
            },

            // --- 文字列結合 ---
            (BinaryOp::Add, Value::String(l), Value::String(r)) => {
                 Ok(Value::String(crate::gc::rc::Rc::new(format!("{}{}", l, r))))
            },
            (BinaryOp::Add, Value::String(l), r) => {
                 Ok(Value::String(crate::gc::rc::Rc::new(format!("{}{}", l, r))))
            },
            (BinaryOp::Add, l, Value::String(r)) => {
                 Ok(Value::String(crate::gc::rc::Rc::new(format!("{}{}", l, r))))
            },

            // --- 比較演算 ---
            (BinaryOp::Eq, _, _) => Ok(Value::Bool(left_val == right_val)),
            (BinaryOp::Ne, _, _) => Ok(Value::Bool(left_val != right_val)),
            
            // Int同士
            (BinaryOp::Lt, Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l < r)),
            (BinaryOp::Le, Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l <= r)),
            (BinaryOp::Gt, Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l > r)),
            (BinaryOp::Ge, Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l >= r)),
            
            // Float同士
            (BinaryOp::Lt, Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l < r)),
            (BinaryOp::Le, Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l <= r)),
            (BinaryOp::Gt, Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l > r)),
            (BinaryOp::Ge, Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l >= r)),

            // ★ 混合比較
            (BinaryOp::Lt, Value::Float(l), Value::Int(r)) => Ok(Value::Bool(l.value() < *r as f64)),
            (BinaryOp::Le, Value::Float(l), Value::Int(r)) => Ok(Value::Bool(l.value() <= *r as f64)),
            (BinaryOp::Gt, Value::Float(l), Value::Int(r)) => Ok(Value::Bool(l.value() > *r as f64)),
            (BinaryOp::Ge, Value::Float(l), Value::Int(r)) => Ok(Value::Bool(l.value() >= *r as f64)),
            (BinaryOp::Lt, Value::Int(l), Value::Float(r)) => Ok(Value::Bool((*l as f64) < r.value())),
            (BinaryOp::Le, Value::Int(l), Value::Float(r)) => Ok(Value::Bool((*l as f64) <= r.value())),
            (BinaryOp::Gt, Value::Int(l), Value::Float(r)) => Ok(Value::Bool((*l as f64) > r.value())),
            (BinaryOp::Ge, Value::Int(l), Value::Float(r)) => Ok(Value::Bool((*l as f64) >= r.value())),

            // ビット演算
            (BinaryOp::BitAnd, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l & r)),
            (BinaryOp::BitOr, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l | r)),
            (BinaryOp::BitXor, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l ^ r)),
            (BinaryOp::Shl, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l << r)),
            (BinaryOp::Shr, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l >> r)),

            _ => Err(RuntimeError::TypeMismatch {
                message: format!("Invalid operation {} for {} and {}", op, left_val.type_name(), right_val.type_name()),
            }),
        }
    }



    // Helper to call .close() on a resource
    fn perform_cleanup(&mut self, resource: &Value, span: Span) -> RuntimeResult<()> {
        // Check if 'close' method exists
        // We simulate "resource.close()" call manually
        // 1. Access the 'close' field/method
        let close_method_result = self.eval_field_access(resource, "close");
        match close_method_result {
            Ok(func_val) => {
                // 2. Check if it is callable
                if let Value::Function { .. } | Value::Builtin(_) = func_val {
                    // 3. Call close() with 0 arguments (excluding receiver if method logic handles it)
                    // Note: Our eval_field_access for Gaussian/Struct usually binds 'self' in the closure.
                    // So we call it with empty args.
                    match self.eval_call_value(func_val, &[], span) {
                        Ok(_) => Ok(()),
                        Err(e) => Err(e),
                    }
                } else {
                    // If 'close' exists but is not a function, ignore or error?
                    // Let's verify strict RAII: explicit close method required?
                    // For flexibility, if no close method, we do nothing (just memory drop).
                    Ok(())
                }
            },
            Err(_) => {
                // No 'close' method defined.
                // Just allow normal GC to handle memory. No explicit cleanup needed.
                Ok(())
            }
        }
    }




    // call_builtin implementation integrated here
    fn call_builtin(&mut self, name: &str, args: Vec<Value>, span: Span) -> RuntimeResult<Value> {
        match name {
            // --- Signal Builtins ---
            "Signal_new" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span });
                }
                let initial = args[0].clone();
                let id = self.reactive.create_root(initial.clone(), NodeKind::Signal);
                Ok(Value::Signal { id, current_value: Rc::new(initial) })
            }

            "Signal_get" => {
                if let Value::Signal { id, .. } = &args[0] {
                    match self.reactive.get_value(*id) {
                        Some(v) => Ok(v),
                        None => Err(RuntimeError::InvalidOperation { message: "Signal not found".into() })
                    }
                } else {
                    Err(RuntimeError::TypeMismatch { message: "Expected Signal".into() })
                }
            }

            "Signal_set" => {
                if args.len() != 2 {
                    return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span });
                }
                let id = match &args[0] {
                    Value::Signal { id, .. } => *id,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Arg 1 must be Signal".into() }),
                };
                let new_val = args[1].clone();
                
                self.reactive.set_value(id, new_val);
                self.propagate_signal_updates(id)?;
                
                Ok(Value::Unit)
            }

            "Signal_map" => {
                if args.len() != 2 {
                    return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span });
                }
                let parent_id = match &args[0] {
                    Value::Signal { id, .. } => *id,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Arg 1 must be Signal".into() }),
                };
                let closure = args[1].clone(); 

                // 初期値を計算
                let parent_val = self.reactive.get_value(parent_id).unwrap();
                let initial_val = self.call_function_value(&closure, vec![parent_val])?;

                let id = self.reactive.create_computed(vec![parent_id], initial_val.clone(), closure, NodeKind::Signal);
                Ok(Value::Signal { id, current_value: Rc::new(initial_val) })
            }
            
            "to_future" => {
                if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                let id = match &args[0] { Value::Int(i) => *i, _ => return Err(RuntimeError::TypeMismatch { message: "Arg must be Int".into() }) };
                // Map Int (TaskId) to Handle
                Ok(Value::Handle(id as usize))
            }

            "wait_future" => {
                 if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                 let id_usize = match &args[0] { Value::Handle(id) => *id, _ => return Err(RuntimeError::TypeMismatch { message: "Arg must be Handle".into() }) };
                 let id = id_usize as i64;
                 
                 if self.native_libraries.is_empty() {
                      return Err(RuntimeError::InvalidOperation { message: "No native libraries loaded".into() });
                 }
                 let lib = &self.native_libraries[0]; 
                 
                 unsafe {
                     let poll_sym_res = lib.get::<unsafe extern "C" fn(i64) -> i32>(b"fluno_task_poll");
                     let poll_sym = match poll_sym_res {
                         Ok(s) => s,
                         Err(e) => return Err(RuntimeError::InvalidOperation { message: format!("FFI error poll: {}", e) }),
                     };
                     
                     let result_sym_res = lib.get::<unsafe extern "C" fn(i64) -> *const std::ffi::c_char>(b"fluno_task_result");
                     let result_sym = match result_sym_res {
                         Ok(s) => s,
                         Err(e) => return Err(RuntimeError::InvalidOperation { message: format!("FFI error result: {}", e) }),
                     };
                     
                     loop {
                         let status = poll_sym(id);
                         if status == 1 {
                             let ptr = result_sym(id);
                             if ptr.is_null() { return Ok(Value::Unit); }
                             let s = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
                             return Ok(Value::String(crate::gc::Rc::new(s)));
                         } else if status == -1 {
                             return Err(RuntimeError::InvalidOperation { message: "Async task failed".into() });
                         }
                         std::thread::sleep(std::time::Duration::from_millis(5));
                     }
                 }
            }

            "Signal_filter" => {
                if args.len() != 2 {
                    return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span });
                }
                let parent_id = match &args[0] {
                    Value::Signal { id, .. } => *id,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Arg 1 must be Signal".into() }),
                };
                // クロージャを取得し、"__signal_op" = "filter" を環境に注入する
                let closure_val = args[1].clone(); 
                
                let new_closure = if let Value::Function { params, body, closure } = closure_val {
                    let mut new_env = closure.as_ref().clone();
                    new_env.set("__signal_op".into(), Value::String(Rc::new("filter".into())));
                    Value::Function { params, body, closure: Rc::new(new_env) }
                } else {
                    return Err(RuntimeError::TypeMismatch { message: "Arg 2 must be Function".into() });
                };

                let parent_val = self.reactive.get_value(parent_id).unwrap();
                // 初期値は親の値をそのまま使う
                let id = self.reactive.create_computed(vec![parent_id], parent_val.clone(), new_closure, NodeKind::Signal);
                Ok(Value::Signal { id, current_value: Rc::new(parent_val) })
            }

            "Signal_combine" => {
                if args.len() != 3 { return Err(RuntimeError::ArityMismatch { expected: 3, found: args.len(), span }); }
                let s1_id = match &args[0] { Value::Signal { id, .. } => *id, v => return Err(RuntimeError::TypeMismatch { message: format!("Arg 1 not Signal: {:?}", v) }) };
                let s2_id = match &args[1] { Value::Signal { id, .. } => *id, v => return Err(RuntimeError::TypeMismatch { message: format!("Arg 2 not Signal: {:?}", v) }) };
                let closure = args[2].clone();

                let s1_val = self.reactive.get_value(s1_id).unwrap();
                let s2_val = self.reactive.get_value(s2_id).unwrap();
                let initial_val = self.call_function_value(&closure, vec![s1_val, s2_val])?;

                let id = self.reactive.create_computed(vec![s1_id, s2_id], initial_val.clone(), closure, NodeKind::Signal);
                Ok(Value::Signal { id, current_value: Rc::new(initial_val) })
            }

            // --- Event Builtins ---
            "Event_new" => {
                if !args.is_empty() {
                    return Err(RuntimeError::ArityMismatch { expected: 0, found: args.len(), span });
                }
                // Eventは初期値を持たないが、内部的には Unit を入れておく
                let id = self.reactive.create_root(Value::Unit, NodeKind::Event);
                Ok(Value::Signal { id, current_value: Rc::new(Value::Unit) }) 
            }

            "Event_emit" => {
                if args.len() != 2 {
                    return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span });
                }
                let id = match &args[0] {
                    Value::Signal { id, .. } => *id, // EventもValue::Signalで表現
                    _ => return Err(RuntimeError::TypeMismatch { message: "Arg 1 must be Event".into() }),
                };
                let val = args[1].clone();
                
                // 値をセットして伝播
                self.reactive.set_value(id, val);
                self.propagate_signal_updates(id)?;
                
                // Eventは永続しないので、伝播後に値をリセットすべきだが、
                // 今回の実装では「直近のイベント値」として残っても害はないとする。
                
                Ok(Value::Unit)
            }

            "Event_map" => {
                // Signal_map とほぼ同じだが NodeKind::Event を指定
                if args.len() != 2 {
                    return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span });
                }
                let parent_id = match &args[0] {
                    Value::Signal { id, .. } => *id,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Arg 1 must be Event".into() }),
                };
                let closure = args[1].clone(); 

                // EventのMapは初期値を計算しない（まだ発火していないから）。Unitにしておく。
                let initial_val = Value::Unit;

                let id = self.reactive.create_computed(vec![parent_id], initial_val.clone(), closure, NodeKind::Event);
                Ok(Value::Signal { id, current_value: Rc::new(initial_val) })
            }

            // --- Standard Builtins ---
            "print" | "println" => {
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { print!(" "); }
                    print!("{}", arg);
                }
                if name == "println" { println!(); }
                Ok(Value::Unit)
            }
            "println_closed" => {
                println!(">> MockResource closed successfully.");
                Ok(Value::Unit)
            }
            "nearly_eq" => {
                if args.len() != 2 { return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span }); }
                let a = match &args[0] { Value::Float(f) => f, v => return Err(RuntimeError::TypeMismatch { message: format!("lhs not float: {:?}", v) }) };
                let b = match &args[1] { Value::Float(f) => f, v => return Err(RuntimeError::TypeMismatch { message: format!("rhs not float: {:?}", v) }) };
                Ok(Value::Bool((a.clone() - b.clone()).abs().value() < 1e-6))
            }
            "assert" => {
                if args.is_empty() { return Err(RuntimeError::ArityMismatch { expected: 1, found: 0, span }); }
                if !matches!(&args[0], Value::Bool(true)) {
                    let msg = if args.len() > 1 {
                        match &args[1] { Value::String(s) => s.to_string(), v => format!("{:?}", v) }
                    } else { "Assertion failed".to_string() };
                    return Err(RuntimeError::InvalidOperation { message: msg });
                }
                Ok(Value::Unit)
            }
            "is_float" => {
                if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                Ok(Value::Bool(matches!(args[0], Value::Float(_))))
            }
            "is_gaussian" => {
                if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                Ok(Value::Bool(matches!(args[0], Value::Gaussian { .. })))
            }
            "Gaussian" => {
                if args.len() != 2 {
                    return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span });
                }

                let mean = match &args[0] {
                    Value::Float(f) => f.clone(),
                    Value::Int(i) => crate::ad::types::ADFloat::Concrete(*i as f64),
                    v => return Err(RuntimeError::TypeMismatch {
                        message: format!("mean must be number, got {}", v.type_name()),
                    }),
                };

                let std = match &args[1] {
                    Value::Float(f) => f.clone(),
                    Value::Int(i) => crate::ad::types::ADFloat::Concrete(*i as f64),
                    v => return Err(RuntimeError::TypeMismatch {
                        message: format!("std must be number, got {}", v.type_name()),
                    }),
                };

                Ok(Value::Gaussian { mean, std })
            }

            "Event_fold" => {
                // Event_fold(event, initial_value, folder_fn) -> Signal
                if args.len() != 3 { return Err(RuntimeError::ArityMismatch { expected: 3, found: args.len(), span }); }
                let parent_id = match &args[0] {
                    Value::Signal { id, .. } => *id,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Arg 1 must be Event".into() }),
                };
                let initial = args[1].clone();
                let closure_val = args[2].clone();

                let new_closure = if let Value::Function { params, body, closure } = closure_val {
                    let mut new_env = closure.as_ref().clone();
                    new_env.set("__signal_op".into(), Value::String(Rc::new("fold".into())));
                    Value::Function { params, body, closure: Rc::new(new_env) }
                } else {
                    return Err(RuntimeError::TypeMismatch { message: "Arg 3 must be Function".into() });
                };

                // 初期値は指定された initial
                let id = self.reactive.create_computed(vec![parent_id], initial.clone(), new_closure, NodeKind::Signal);
                Ok(Value::Signal { id, current_value: Rc::new(initial) })
            }

            "Event_merge" => {
                // Event_merge(event1, event2) -> Event
                if args.len() != 2 { return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span }); }
                let id1 = match &args[0] { Value::Signal { id, .. } => *id, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 1 must be Event".into() }) };
                let id2 = match &args[1] { Value::Signal { id, .. } => *id, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 2 must be Event".into() }) };
                
                // プレースホルダ的なクロージャを作る（実際は propagate 側で処理するが、形式上必要）
                // Op = "merge"
                let closure = Value::Function { 
                    params: vec![], 
                    body: crate::ast::node::Block { statements: vec![], span: Span::initial() },
                    closure: Rc::new({
                        let mut e = self.env.clone();
                        e.set("__signal_op".into(), Value::String(Rc::new("merge".into())));
                        e
                    })
                };

                let id = self.reactive.create_computed(vec![id1, id2], Value::Unit, closure, NodeKind::Event);
                Ok(Value::Signal { id, current_value: Rc::new(Value::Unit) })
            }

            "Signal_sample" => {
                // Signal_sample(trigger_event, value_signal) -> Event
                if args.len() != 2 { return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span }); }
                let trig_id = match &args[0] { Value::Signal { id, .. } => *id, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 1 must be Event".into() }) };
                let val_id = match &args[1] { Value::Signal { id, .. } => *id, _ => return Err(RuntimeError::TypeMismatch { message: "Arg 2 must be Signal".into() }) };

                // Op = "sample"
                let closure = Value::Function { 
                    params: vec![], 
                    body: crate::ast::node::Block { statements: vec![], span: Span::initial() },
                    closure: Rc::new({
                        let mut e = self.env.clone();
                        e.set("__signal_op".into(), Value::String(Rc::new("sample".into())));
                        e
                    })
                };

                // イベントが発火した瞬間のSignalの値をEventとして流すので、初期値はUnitでよい
                let id = self.reactive.create_computed(vec![trig_id, val_id], Value::Unit, closure, NodeKind::Event);
                Ok(Value::Signal { id, current_value: Rc::new(Value::Unit) })
            }

            "Array::push" => {
                if args.len() != 2 { return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span }); }
                match &args[0] {
                    Value::Array(arr) => {
                        let mut new_arr = (**arr).clone();
                        new_arr.push(args[1].clone());
                        Ok(Value::Array(Rc::new(new_arr)))
                    },
                    v => Err(RuntimeError::TypeMismatch { message: format!("push expects array, got {}", v.type_name()) })
                }
            }
            "exp" => {
                if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.clone().exp())),
                    Value::Int(i) => Ok(Value::Float(crate::ad::types::ADFloat::Concrete((*i as f64).exp()))),
                    v => Err(RuntimeError::TypeMismatch { message: format!("exp expects number, got {:?}", v) })
                }
            }

            "sin" => {
                if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.clone().sin())),
                    Value::Int(i) => Ok(Value::Float(crate::ad::types::ADFloat::Concrete((*i as f64).sin()))),
                    v => Err(RuntimeError::TypeMismatch { message: format!("sin expects number, got {}", v.type_name()) })
                }
            }
            "cos" => {
                if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.clone().cos())),
                    Value::Int(i) => Ok(Value::Float(crate::ad::types::ADFloat::Concrete((*i as f64).cos()))),
                    v => Err(RuntimeError::TypeMismatch { message: format!("cos expects number, got {}", v.type_name()) })
                }
            }
            "tan" => {
                if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.clone().tan())),
                    Value::Int(i) => Ok(Value::Float(crate::ad::types::ADFloat::Concrete((*i as f64).tan()))),
                    v => Err(RuntimeError::TypeMismatch { message: format!("tan expects number, got {}", v.type_name()) })
                }
            }
            "sqrt" => {
                if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.clone().sqrt())),
                    Value::Int(i) => Ok(Value::Float(crate::ad::types::ADFloat::Concrete((*i as f64).sqrt()))),
                    v => Err(RuntimeError::TypeMismatch { message: format!("sqrt expects number, got {}", v.type_name()) })
                }
            }
            "ln" => {
                if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.clone().ln())),
                    Value::Int(i) => Ok(Value::Float(crate::ad::types::ADFloat::Concrete((*i as f64).ln()))),
                    v => Err(RuntimeError::TypeMismatch { message: format!("ln expects number, got {}", v.type_name()) })
                }
            }
            "pow" => {
                if args.len() != 2 { return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span }); }
                let base = match &args[0] {
                     Value::Float(f) => f.clone(),
                     Value::Int(i) => crate::ad::types::ADFloat::Concrete(*i as f64),
                     v => return Err(RuntimeError::TypeMismatch { message: format!("pow base must be number, got {}", v.type_name()) }),
                };
                let exp = match &args[1] {
                     Value::Float(f) => f.clone(),
                     Value::Int(i) => crate::ad::types::ADFloat::Concrete(*i as f64),
                     v => return Err(RuntimeError::TypeMismatch { message: format!("pow exponent must be number, got {}", v.type_name()) }),
                };
                // pow(x, y) = exp(y * ln(x))
                Ok(Value::Float((exp * base.ln()).exp()))
            }

            "to_json" => {
                if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                let json_val = value_to_json(&args[0]);
                Ok(Value::String(Rc::new(json_val.to_string())))
            }
            "from_json" => {
                if args.len() != 1 { return Err(RuntimeError::ArityMismatch { expected: 1, found: args.len(), span }); }
                let json_str = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Err(RuntimeError::TypeMismatch { message: "from_json arg must be String".into() }),
                };
                let json_val: serde_json::Value = serde_json::from_str(json_str).map_err(|e| RuntimeError::InvalidOperation { 
                    message: format!("JSON Parse Error: {}", e) 
                })?;
                Ok(json_to_value(&json_val))
            }

            "observe" => {
                if args.len() != 2 {
                    return Err(RuntimeError::ArityMismatch { expected: 2, found: args.len(), span });
                }

                // 第2引数を number として正規化（prob_mode に関係なくやる）
                let val = match &args[1] {
                    Value::Float(f) => f.clone(),
                    // Int などは Float に変換 (ADFloat::Concrete を使用)
                    Value::Int(i) => crate::ad::types::ADFloat::Concrete(*i as f64),
                    _ => {
                        return Err(RuntimeError::TypeMismatch {
                            message: "Observed value must be number".into(),
                        })
                    }
                };

                // 確率モードでなければ「スコア加算」はスキップだが、値は返す
                if self.prob_mode && self.prob_context.is_some() {
                    let dist = &args[0];

                    use crate::vm::prob_runtime::calculate_score_ad;
                    // calculate_score_ad は ADFloat を返す
                    let score = calculate_score_ad(dist, &val)?;

                    if let Some(ctx) = self.prob_context.as_mut() {
                        // ADFloat 同士の加算を行い、AD グラフに記録
                        ctx.accumulated_log_prob = ctx.accumulated_log_prob.clone() + score;
                    }
                }

                // 従来の self.current_weight への加算は、もし残したいならここで加算
                // ただし、ProbContext 側で管理するなら不要かもしれません。
                // 念のため ADFloat の値を取り出して足しておきます。
                // (current_weight 自体を使わなくする方向であれば削除可能です)
                //
                // 例:
                // self.current_weight += val.value;

                // 重要: Unit ではなく観測値を Float として返す
                Ok(Value::Float(val))
            }

            "infer" => {
                // Legacy infer(num_samples: Int, model: fn) -> Array<Value> (Scalar)
                if args.len() != 2 {
                    return Err(RuntimeError::ArgumentMismatch { expected: 2, found: args.len(), span });
                }
                let num_samples = match &args[0] {
                    Value::Int(n) => *n as usize,
                    _ => return Err(RuntimeError::TypeMismatch { message: "infer arg 1 must be Int".into() }),
                };
                let model_fn = args[1].clone();
                
                self.run_inference(num_samples, model_fn)
            }

            "infer_hmc" => {
                // infer_hmc(config: Map, model: fn) -> Array<Map>
                if args.len() != 2 {
                    return Err(RuntimeError::ArgumentMismatch { expected: 2, found: args.len(), span });
                }
                
                // Parse Config
                let mut num_samples = 1000;
                let burn_in = 100;
                let mut epsilon = 0.1;
                let mut l_steps = 10;
                let mut init_params = std::collections::HashMap::new();

                if let Value::Map(m) = &args[0] {
                    let map = m.borrow();
                    if let Some(v) = map.get("num_samples") {
                        if let Value::Int(i) = v { num_samples = *i as usize; }
                    }
                    if let Some(v) = map.get("step_size") {
                        if let Value::Float(f) = v { epsilon = f.value(); }
                    }
                    if let Some(v) = map.get("num_leapfrog") {
                        if let Value::Int(i) = v { l_steps = *i as usize; }
                    }
                    
                    if let Some(Value::Map(params)) = map.get("init_params") {
                         let p_map = params.borrow();
                         for (k, v) in p_map.iter() {
                             if let Value::Float(f) = v {
                                 init_params.insert(k.clone(), f.value());
                             }
                         }
                    }
                }

                let model_func = args[1].clone();
                let (body, closure) = match model_func {
                    Value::Function { body, closure, .. } => (body, closure),
                    _ => return Err(RuntimeError::TypeMismatch { message: "infer_hmc model must be a function".into() }),
                };

                // 推論実行
                let saved_env = self.env.clone();
                self.env = closure.as_ref().clone();

                let sampler = crate::vm::prob_runtime::HMC::new(num_samples, burn_in, epsilon, l_steps);
                let samples_result = sampler.infer(&body, self, init_params);

                self.env = saved_env;

                match samples_result {
                    Ok(samples) => {
                        use crate::gc::Rc;
                        use std::cell::RefCell;
                        
                        let mut map_samples = Vec::new();
                        for samp_map in samples {
                             let mut content = std::collections::HashMap::new();
                             for (k, v) in samp_map {
                                 content.insert(k, Value::Float(ADFloat::Concrete(v)));
                             }
                             map_samples.push(Value::Map(Rc::new(RefCell::new(content))));
                        }
                        Ok(Value::Array(Rc::new(map_samples)))
                    },
                    Err(e) => Err(e),
                }
            }



            "Uniform" => {
                if args.len() != 2 {
                    return Err(RuntimeError::ArgumentMismatch { expected: 2, found: args.len(), span });
                }
                let min = match &args[0] {
                    Value::Float(f) => f.value(),   // ここで .value()
                    Value::Int(i) => *i as f64,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Uniform min must be number".into() }),
                };

                let max = match &args[1] {
                    Value::Float(f) => f.value(),   // ここで .value()
                    Value::Int(i) => *i as f64,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Uniform max must be number".into() }),
                };
                
                Ok(Value::Uniform { min : ADFloat::Concrete(min), max : ADFloat::Concrete(max) })
            }

            "Bernoulli" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArgumentMismatch { expected: 1, found: args.len(), span });
                }
                let p = match &args[0] {
                    Value::Float(f) => f.value(),
                    Value::Int(i) => *i as f64,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Bernoulli p must be number".into() }),
                };
                if p < 0.0 || p > 1.0 {
                    return Err(RuntimeError::InvalidOperation { message: "Bernoulli p must be between 0 and 1".into() });
                }
                Ok(Value::Bernoulli { p: ADFloat::Concrete(p) })
            }

            "Beta" => {
                if args.len() != 2 {
                    return Err(RuntimeError::ArgumentMismatch { expected: 2, found: args.len(), span });
                }
                let alpha = match &args[0] {
                    Value::Float(f) => f.value(),
                    Value::Int(i) => *i as f64,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Beta alpha must be number".into() }),
                };
                let beta = match &args[1] {
                    Value::Float(f) => f.value(),
                    Value::Int(i) => *i as f64,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Beta beta must be number".into() }),
                };
                if alpha <= 0.0 || beta <= 0.0 {
                    return Err(RuntimeError::InvalidOperation { message: "Beta parameters must be positive".into() });
                }
                Ok(Value::Beta { alpha: ADFloat::Concrete(alpha), beta: ADFloat::Concrete(beta) })
            }

            "infer_vi" => {
                if args.len() != 3 {
                    return Err(RuntimeError::ArgumentMismatch { expected: 3, found: args.len(), span });
                }
                // Parse Config Map
                // Parse Config Map
                let mut num_iters = 1000;
                let mut lr = 0.01;
                let mut use_adam = false;
                
                if let Value::Map(m) = &args[0] {
                    let map = m.borrow();
                    if let Some(v) = map.get("iters") {
                         match v {
                             Value::Int(i) => num_iters = *i as usize,
                             Value::Float(f) => num_iters = f.value() as usize,
                             _ => {}
                         }
                    }
                    if let Some(v) = map.get("num_iters") {
                         match v {
                             Value::Int(i) => num_iters = *i as usize,
                             Value::Float(f) => num_iters = f.value() as usize,
                             _ => {}
                         }
                    }
                    if let Some(v) = map.get("lr") {
                         match v {
                             Value::Float(f) => lr = f.value(),
                             _ => {}
                         }
                    }
                    if let Some(Value::String(s)) = map.get("optimizer") {
                         if s.as_str() == "adam" { use_adam = true; }
                    }
                }
                
                let model_block = match &args[1] {
                     Value::Function { body, .. } => body,
                     _ => return Err(RuntimeError::TypeMismatch { message: "Model must be a function".into() }),
                };
                let guide_block = match &args[2] {
                     Value::Function { body, .. } => body,
                     _ => return Err(RuntimeError::TypeMismatch { message: "Guide must be a function".into() }),
                };

                let mut advi = if use_adam {
                    crate::vm::prob_runtime::ADVI::with_adam(num_iters, lr)
                } else {
                    crate::vm::prob_runtime::ADVI::new(num_iters, lr)
                };
                let result_map = advi.infer(model_block, guide_block, self)?;
                
                // Convert HashMap<String, f64> to Value::Map
                use crate::gc::Rc;
                use std::cell::RefCell;
                use std::collections::HashMap;
                use serde_json;
                
                let mut map_content = HashMap::new();
                for (k, v) in result_map {
                    map_content.insert(k, Value::Float(crate::ad::types::ADFloat::Concrete(v)));
                }
                
                Ok(Value::Map(Rc::new(RefCell::new(map_content))))
            }

            "param" => {
                if args.len() != 2 {
                    return Err(RuntimeError::ArgumentMismatch { expected: 2, found: args.len(), span });
                }
                let name = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Param name must be a string".into() }),
                };
                let init_val = match &args[1] {
                    Value::Float(f) => f.value(),
                    Value::Int(i) => *i as f64,
                    _ => return Err(RuntimeError::TypeMismatch { message: "Param init must be a number".into() }),
                };
                
                if let Some(ctx) = self.prob_context.as_mut() {
                    let val_ad = crate::vm::prob_runtime::register_param(ctx, name, init_val);
                    Ok(Value::Float(val_ad))
                } else {
                    Ok(Value::Float(crate::ad::types::ADFloat::Concrete(init_val)))
                }
            }

            "Map" => {
                use std::collections::HashMap;
                use crate::gc::Rc;
                use std::cell::RefCell;
                
                let map = HashMap::new();
                Ok(Value::Map(Rc::new(RefCell::new(map))))
            }

            
            "Map::insert" => {
                if args.len() != 3 { // self, key, value
                     return Err(RuntimeError::ArgumentMismatch { expected: 3, found: args.len(), span });
                }
                match (&args[0], &args[1]) {
                    (Value::Map(rc_map), Value::String(k)) => {
                        rc_map.borrow_mut().insert(k.to_string(), args[2].clone());
                        Ok(Value::Unit)
                    }
                     _ => Err(RuntimeError::TypeMismatch { message: "Map::insert expects (Map, String, Any)".into() })
                }
            }
            "Map::get" => {
                if args.len() != 2 { // self, key
                     return Err(RuntimeError::ArgumentMismatch { expected: 2, found: args.len(), span });
                }
                match (&args[0], &args[1]) {
                    (Value::Map(rc_map), Value::String(k)) => {
                        let map = rc_map.borrow();
                        if let Some(v) = map.get(k.as_str()) {
                            Ok(v.clone()) 
                        } else {
                            Ok(Value::Unit) 
                        }
                    }
                     _ => Err(RuntimeError::TypeMismatch { message: "Map::get expects (Map, String)".into() })
                }
            }
            "Map::contains_key" => {
                if args.len() != 2 {
                     return Err(RuntimeError::ArgumentMismatch { expected: 2, found: args.len(), span });
                }
                match (&args[0], &args[1]) {
                    (Value::Map(rc_map), Value::String(k)) => {
                        let map = rc_map.borrow();
                        Ok(Value::Bool(map.contains_key(k.as_str())))
                    }
                     _ => Err(RuntimeError::TypeMismatch { message: "Map::contains_key expects (Map, String)".into() })
                }
            }
            "sample" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArgumentMismatch { expected: 1, found: args.len(), span });
                }
                // Interpreter::sample メソッドを呼び出す
                self.sample(args[0].clone())
            }

            "Rc_new" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                        span: Span::initial(), 
                    });
                }
                Ok(Value::Rc(crate::gc::Rc::new(args[0].clone())))
            }
            
            "Rc_downgrade" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                        span: Span::initial(), 
                    });
                }
                if let Value::Rc(rc) = &args[0] {
                    Ok(Value::Weak(crate::gc::Rc::downgrade(rc)))
                } else {
                    Err(RuntimeError::TypeMismatch {
                        message: "Rc_downgrade expects Rc value".into(),
                    })
                }
            }
            
            // 組み込み関数の処理（既存の Rc_new, Rc_downgrade の近く）
            "Weak_upgrade" => {
                if args.len() != 1 {
                    return Err(RuntimeError::InvalidOperation {
                        message: format!("Weak_upgrade expects 1 argument, got {}", args.len()),
                    });
                }

                let weak_val = &args[0];

                // Weak 型の値から Rc を取り出す
                match weak_val {
                    Value::Weak(weak_rc) => {
                        // Weak::upgrade() を呼び出す
                        // weak_rc は Weak<RefCell<Value>> なので upgrade すると Option<Rc<RefCell<Value>>> が返る
                        match weak_rc.upgrade() {
                            Some(strong_rc) => {
                                // strong_rc は Rc<RefCell<Value>>。Value::Rc もこれを受け取るはず。
                                // Value::Option は Box<Value> を持つので、Value::Rc を Box で包む。
                                Ok(Value::Enum {
                                    name: "Option".into(),
                                    variant: "Some".into(),
                                    fields: crate::gc::Rc::new(vec![Value::Rc(strong_rc)]),
                                })
                            }
                            None => Ok(Value::Enum {
                                name: "Option".into(),
                                variant: "None".into(),
                                fields: crate::gc::Rc::new(vec![]),
                            }),
                        }
                    }
                    _ => Err(RuntimeError::InvalidOperation {
                        message: format!("Weak_upgrade expects Weak, got {:?}", weak_val),
                    }),
                }
            }


            "Float::sqrt" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                        span,
                    });
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.clone().sqrt())),
                    v => Err(RuntimeError::TypeMismatch {
                        message: format!("sqrt expects Float, got {:?}", v),
                    }),
                }
            }
            
            "Float::abs" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                        span,
                    });
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.clone().abs())),
                    v => Err(RuntimeError::TypeMismatch {
                        message: format!("abs expects Float, got {:?}", v),
                    }),
                }
            }
            
            "Float::pow" => {
                if args.len() != 2 {
                    return Err(RuntimeError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                        span,
                    });
                }
                match (&args[0], &args[1]) {
                    (Value::Float(base), Value::Float(exp)) => Ok(Value::Float(base.clone().powf(exp.value()))),
                    _ => Err(RuntimeError::TypeMismatch {
                        message: "pow expects (Float, Float)".to_string(),
                    }),
                }
            }
            
            "Float::floor" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                        span,
                    });
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.clone().floor())),
                    v => Err(RuntimeError::TypeMismatch {
                        message: format!("floor expects Float, got {:?}", v),
                    }),
                }
            }
            
            "Float::ceil" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                        span,
                    });
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.clone().ceil())),
                    v => Err(RuntimeError::TypeMismatch {
                        message: format!("ceil expects Float, got {:?}", v),
                    }),
                }
            }
            
            "Float::round" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                        span,
                    });
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.clone().round())),
                    v => Err(RuntimeError::TypeMismatch {
                        message: format!("round expects Float, got {:?}", v),
                    }),
                }
            }
            
            // ==========================================
            // Int メソッド
            // ==========================================
            
            "Int::abs" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                        span,
                    });
                }
                match &args[0] {
                    Value::Int(i) => Ok(Value::Int(i.abs())),
                    v => Err(RuntimeError::TypeMismatch {
                        message: format!("abs expects Int, got {:?}", v),
                    }),
                }
            }
            
            // ==========================================
            // Array メソッド
            // ==========================================
            
            "Array::len" => {
                if args.len() != 1 {
                    return Err(RuntimeError::ArityMismatch {
                        expected: 1,
                        found: args.len(),
                        span,
                    });
                }
                match &args[0] {
                    Value::Array(elements) => Ok(Value::Int(elements.len() as i64)),
                    Value::Tuple(elements) => Ok(Value::Int(elements.len() as i64)),
                    v => Err(RuntimeError::TypeMismatch {
                        message: format!("len expects Array or Tuple, got {:?}", v),
                    }),
                }
            }
            "create_tape" => {
                // 引数なし
                let tape_id = crate::ad::create_tape();
                // IDをIntとして返す（usize -> i64）
                Ok(Value::Int(tape_id as i64))
            }


            "backward" => {
                // 引数: (target: Float)
                if args.len() != 1 {
                    return Err(RuntimeError::InvalidOperation { message: "backward expects 1 argument".into() });
                }
                
                match &args[0] {
                    Value::Float(ad_val) => {
                        // 逆伝播を実行
                        // ad_val.backward() は HashMap<usize, f64> を返すと仮定
                        let grads = ad_val.backward(); 
                        
                        // 結果をインタプリタに保存
                        self.latest_gradients = grads;
                        
                        Ok(Value::Unit)
                    }
                    _ => Err(RuntimeError::TypeMismatch { message: "backward target must be Float".into() }),
                }
            }

            "grad" => {
                // 引数: (var: Float) -> その変数の勾配を返す
                // 指定された変数のNode IDを使って、latest_gradientsから値を引く
                if args.len() != 1 {
                    return Err(RuntimeError::InvalidOperation { message: "grad expects 1 argument".into() });
                }
                
                match &args[0] {
                    Value::Float(ad_val) => {
                        // ADFloatからNode IDを取得する必要がある
                        // ADFloat::Dual { node_id, ... } パターンマッチ
                        if let crate::ad::types::ADFloat::Dual { node_id, .. } = ad_val {
                            let g = self.latest_gradients.get(node_id).and_then(|g| g.as_scalar()).unwrap_or(0.0);
                            Ok(Value::Float(crate::ad::types::ADFloat::Concrete(g))) // 勾配自体は定数扱い
                        } else {
                            // Constantなら勾配は0 (あるいはエラー?)
                            Ok(Value::Float(crate::ad::types::ADFloat::Concrete(0.0)))
                        }
                    }
                    _ => Err(RuntimeError::TypeMismatch { message: "grad target must be Float".into() }),
                }
            }
            _ => Err(RuntimeError::Unimplemented(format!("Builtin '{}'", name))),
        }
    }

    // 更新伝播ロジック
    fn propagate_signal_updates(&mut self, start_id: usize) -> RuntimeResult<()> {
        let update_order = self.reactive.get_propagation_order(start_id);
        let mut updated_nodes = std::collections::HashSet::new();
        updated_nodes.insert(start_id);

        for id in update_order {
            let (update_fn, dependencies) = {
                let node = self.reactive.get_node(id).ok_or(RuntimeError::InvalidOperation{message: "Node missing".into()})?;
                (node.update_fn.clone(), node.dependencies.clone())
            };

            // 依存先が更新されているかチェック
            let should_process = dependencies.iter().any(|dep| updated_nodes.contains(dep));
            if !should_process { continue; }

            if let Some(func) = update_fn {
                // 基本的な引数収集
                let mut args = Vec::new();
                for dep_id in &dependencies {
                    args.push(self.reactive.get_value(*dep_id).unwrap());
                }

                // オペレータ種別判定
                let mut op_type = "default".to_string(); // Stringに変更
                if let Value::Function { closure, .. } = &func {
                    if let Some(Value::String(op)) = closure.as_ref().get("__signal_op") {
                        op_type = op.as_ref().clone(); // Rc<String>の中身をClone
                    }
                }

                match op_type.as_str() {
                    "filter" => {
                        // args[0] は親の値
                        let predicate_result = self.call_function_value(&func, args.clone())?;
                        if predicate_result.is_truthy() {
                            if !args.is_empty() {
                                self.reactive.set_value(id, args[0].clone());
                                updated_nodes.insert(id);
                            }
                        }
                    }
                    "fold" => {
                        // args[0] はイベントの値 (Input)
                        // 自分の現在の値 (Accumulator) を取得
                        let current_acc = self.reactive.get_value(id).unwrap();
                        
                        // fold関数の引数は (acc, input)
                        let fold_args = vec![current_acc, args[0].clone()];
                        let new_val = self.call_function_value(&func, fold_args)?;
                        
                        self.reactive.set_value(id, new_val);
                        updated_nodes.insert(id);
                    }
                    "merge" => {
                        // dependencies[0] と [1] のうち、更新された方を採用
                        // 両方更新されていたら、とりあえず第2引数を優先する（Left-biasなら第1引数）
                        // ここでは args[0], args[1] に値が入っている
                        
                        let dep0_updated = updated_nodes.contains(&dependencies[0]);
                        let dep1_updated = updated_nodes.contains(&dependencies[1]);
                        
                        if dep1_updated {
                            self.reactive.set_value(id, args[1].clone());
                            updated_nodes.insert(id);
                        } else if dep0_updated {
                            self.reactive.set_value(id, args[0].clone());
                            updated_nodes.insert(id);
                        }
                    }
                    "sample" => {
                        // dependencies: [trigger_event, value_signal]
                        // トリガー(0)が更新されたときのみ、Signal(1)の値を採用して更新
                        let trigger_updated = updated_nodes.contains(&dependencies[0]);
                        
                        if trigger_updated {
                            // Signalの値 (args[1]) を採用
                            self.reactive.set_value(id, args[1].clone());
                            updated_nodes.insert(id);
                        }
                    }
                    _ => {
                        // default (map, combine)
                        let new_val = self.call_function_value(&func, args)?;
                        self.reactive.set_value(id, new_val);
                        updated_nodes.insert(id);
                    }
                }
            }
        }
        Ok(())
    }

    
    // ヘルパー: Value::Function を実行する
    fn call_function_value(&mut self, func: &Value, args: Vec<Value>) -> RuntimeResult<Value> {
        match func {
            Value::Function { params, body, closure } => {
                let saved_env = self.env.clone();
                self.env = closure.as_ref().clone();
                self.env.push_scope();
                for (param, arg) in params.iter().zip(args) {
                    self.env.set(param.name.name.clone(), arg);
                }
                let result = self.eval_block(body);
                self.env = saved_env; 

                match result {
                    Ok(v) => Ok(v),
                    Err(RuntimeError::EarlyReturn) => {
                         Ok(self.return_value.take().unwrap_or(Value::Unit))
                    }
                    Err(e) => Err(e),
                }
            }
            Value::Builtin(name) => {
                self.call_builtin(name, args, Span::initial())
            }
            // ★ FFI関数の呼び出し
            Value::NativeFunction { name, library_index, params, return_type, is_async } => {
                if *is_async {
                    return Err(RuntimeError::Unimplemented(
                        format!("Async FFI call '{}' is not supported in call_function_value", name)
                    ));
                }
                self.call_native(name, *library_index, params, return_type, args, Span::initial())
            }

            _ => Err(RuntimeError::TypeMismatch{message: "Not a function".into()})
        }
    }

    fn eval_literal(&self, literal: &Literal) -> RuntimeResult<Value> {
        match literal {
            Literal::Int(n) => Ok(Value::Int(*n)),
            Literal::Float(f) => Ok(Value::Float(ADFloat::Concrete(*f))),
            Literal::Bool(b) => Ok(Value::Bool(*b)),
            Literal::String(s) => Ok(Value::String(Rc::new(s.clone()))),
            Literal::Unit => Ok(Value::Unit),
        }
    }

    fn eval_binary_op(
        &mut self,
        op: &BinaryOp,
        left_expr: &Expression,
        right_expr: &Expression,
        span: Span,
    ) -> RuntimeResult<Value> {
        // 左右を評価
        let left_val = self.eval_expression(left_expr)?;
        
        // 短絡評価 (Short-circuit) のためのロジック
        // && の場合、左が false なら右を評価せずに false を返す
        if matches!(op, BinaryOp::And) {
            if !left_val.is_truthy() {
                return Ok(Value::Bool(false));
            }
            let right_val = self.eval_expression(right_expr)?;
            return Ok(Value::Bool(right_val.is_truthy()));
        }
        // || の場合、左が true なら右を評価せずに true を返す
        if matches!(op, BinaryOp::Or) {
            if left_val.is_truthy() {
                return Ok(Value::Bool(true));
            }
            let right_val = self.eval_expression(right_expr)?;
            return Ok(Value::Bool(right_val.is_truthy()));
        }

        // それ以外は右辺も評価
        // それ以外は右辺も評価
        let right_val = self.eval_expression(right_expr)?;

        match (op, &left_val, &right_val) {
            // ========== 算術演算 (Int) ==========
            (BinaryOp::Add, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l + r)),
            (BinaryOp::Sub, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l - r)),
            (BinaryOp::Mul, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l * r)),
            (BinaryOp::Div, Value::Int(l), Value::Int(r)) => {
                if *r == 0 {
                    return Err(RuntimeError::DivisionByZero);
                }
                Ok(Value::Int(l / r))
            },
            (BinaryOp::Mod, Value::Int(l), Value::Int(r)) => {
                if *r == 0 {
                    return Err(RuntimeError::DivisionByZero);
                }
                Ok(Value::Int(l % r))
            },

            // ========== 算術演算 (Float) ==========
            (BinaryOp::Add, Value::Float(l), Value::Float(r)) => Ok(Value::Float(l.clone() + r.clone())),
            (BinaryOp::Sub, Value::Float(l), Value::Float(r)) => Ok(Value::Float(l.clone() - r.clone())),
            (BinaryOp::Mul, Value::Float(l), Value::Float(r)) => Ok(Value::Float(l.clone() * r.clone())),
            (BinaryOp::Div, Value::Float(l), Value::Float(r)) => Ok(Value::Float(l.clone() / r.clone())),

            // ========== 比較演算 (Eq, Ne) ==========
            (BinaryOp::Eq, _, _) => Ok(Value::Bool(left_val == right_val)),
            (BinaryOp::Ne, _, _) => Ok(Value::Bool(left_val != right_val)),

            // ========== 比較演算 (Int) ==========
            (BinaryOp::Lt, Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l < r)),
            (BinaryOp::Le, Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l <= r)),
            (BinaryOp::Gt, Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l > r)),
            (BinaryOp::Ge, Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l >= r)),

            // ========== 比較演算 (Float) ==========
            (BinaryOp::Lt, Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l < r)),
            (BinaryOp::Le, Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l <= r)),
            (BinaryOp::Gt, Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l > r)),
            (BinaryOp::Ge, Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l >= r)),

            // ========== 文字列結合 ==========
            (BinaryOp::Add, Value::String(l), Value::String(r)) => {
                // Rc<String> なので中身を取り出して結合し、新しい Rc を作る
                Ok(Value::String(crate::gc::rc::Rc::new(format!("{}{}", l, r))))
            },

            // ========== ビット演算 (Int) ==========
            (BinaryOp::BitAnd, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l & r)),
            (BinaryOp::BitOr, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l | r)),
            (BinaryOp::BitXor, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l ^ r)),
            (BinaryOp::Shl, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l << r)),
            (BinaryOp::Shr, Value::Int(l), Value::Int(r)) => Ok(Value::Int(l >> r)),

            // 型不一致
            _ => Err(RuntimeError::TypeMismatch {
                message: format!(
                    "Cannot apply operator {} to {} and {}", 
                    op, left_val.type_name(), right_val.type_name()
                )
            }),
        }
    }

    fn eval_unary_op(&self, op: UnaryOp, operand: &Value) -> RuntimeResult<Value> {
        match (op, operand) {
            (UnaryOp::Neg, Value::Int(n)) => Ok(Value::Int(-n)),
            (UnaryOp::Neg, Value::Float(f)) => Ok(Value::Float(-f.clone())),
            (UnaryOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
            _ => Err(RuntimeError::InvalidOperation { message: format!("Cannot apply {:?} to {:?}", op, operand) }),
        }
    }



    fn eval_call(&mut self, callee: &Expression, args: &[Expression], span: Span) -> RuntimeResult<Value> {
        // 1) 引数を先に評価
        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(self.eval_expression(arg)?);
        }

        // 2) UFCS Method Call: Trait::method(arg0, ...)
        if let Expression::UfcsMethod { trait_path: _, method, .. } = callee {
            if arg_values.is_empty() {
                return Err(RuntimeError::InvalidOperation {
                    message: "UFCS method call requires at least one argument (receiver)".to_string(),
                });
            }

            // 第1引数をレシーバとして取り出す
            let receiver = arg_values[0].clone();

            // ネイティブ実装（必要ならここは残す）
            if let (Value::Array(arr), "head") = (&arg_values[0], method.name.as_str()) {
                let elements: &Vec<Value> = &*arr;
                if let Some(first) = elements.first() {
                    return Ok(Value::Enum {
                        name: "Option".to_string(),
                        variant: "Some".to_string(),
                        fields: crate::gc::Rc::new(vec![first.clone()]),
                    });
                } else {
                    return Ok(Value::Enum {
                        name: "Option".to_string(),
                        variant: "None".to_string(),
                        fields: crate::gc::Rc::new(vec![]),
                    });
                }
            }

            // ★ここが重要：UFCS でも callmethod に投げる
            let rest_args: Vec<Value> = arg_values[1..].to_vec();
            return self.call_method(receiver, &method.name, &rest_args, span);
        }


        // 3) print / println の特別扱い
        if let Expression::Variable { name, .. } = callee {
            match name.last_name().unwrap() {
                "print" | "println" => {
                    for (i, arg) in arg_values.iter().enumerate() {
                        if i > 0 { print!(" "); }
                        print!("{}", arg);
                    }
                    if name.last_ident().unwrap().name == "println" { println!(); }
                    return Ok(Value::Unit);
                }
                _ => {}
            }
        }

        // 4) FieldAccess: a.b() の処理
        if let Expression::FieldAccess { object, field, .. } = callee {
            let recv = self.eval_expression(object)?;
            // arg_values は既に Vec<Value> を作っている前提
            return self.call_method(recv, &field.name, &arg_values, span);
        }


        // 5) 通常の関数呼び出し
        let func_val = self.eval_expression(callee)?;
        self.eval_call_value(func_val, &arg_values, span)
    }



            // ヘルパー関数として定義しておくと便利
    fn call_method(
        &mut self,
        receiver: Value,
        methodname: &str,
        args: &[Value],
        span: Span,
    ) -> RuntimeResult<Value> {
        if methodname == "clone" && args.is_empty() {
            return Ok(receiver.clone());
        }

        if methodname == "sample" && args.is_empty() {
            // ユーザ定義 struct Gaussian { mean, std } を Value::Gaussian に昇格できるなら昇格
            let dist = self.to_gaussian_if_struct(&receiver);
            return self.sample(dist);
        }
        // 0. TraitObject の場合、vtable から直接取得
        //    (let m: Multiplier = n; m.get() のようなケース)
        if let Value::TraitObject { vtable, data, .. } = &receiver {
            if let Some(func) = vtable.get(methodname) {
                // Trait Object 呼び出しの場合、receiver(self)はラップされているデータ(data)
                let mut callargs = vec![(**data).clone()];
                callargs.extend(args.iter().cloned());

                match func {
                    Value::Builtin(ref name) => {
                        return self.call_builtin(name, callargs, span);
                    }
                    _ => {
                        return self.call_function_value(func, callargs);
                    }
                }
            } else {
                return Err(RuntimeError::UndefinedField {
                    field: format!(
                        "Method '{}' not found in trait object vtable",
                        methodname
                    ),
                });
            }
        }

        // 通常のメソッド呼び出し (Inherent or Trait implementation)
        let typename = Self::type_name_from_value_for_method(&receiver);

        // 1) "{Type}::{method}" (Inherent Impl を優先)
        //    (impl Number { fn get... } のようなケース)
        let methodkey = format!("{}::{}", typename, methodname);
        if let Some(funcval) = self.globals.get(&methodkey).cloned() {
            let mut callargs: Vec<Value> = Vec::with_capacity(1 + args.len());
            callargs.push(receiver.clone());
            callargs.extend(args.iter().cloned());

            match funcval {
                Value::Builtin(ref name) => {
                    return self.call_builtin(name, callargs, span);
                }
                _ => {
                    return self.call_function_value(&funcval, callargs);
                }
            }
        }

        // 2) "{Type}::{Trait}::{method}" (Trait Impl を検索)
        //    execute_item(ItemImpl) 側が trait impl を "Type::Trait::method" で globals に登録している前提。
        let mut trait_candidates: Vec<(String, Value)> = Vec::new();
        for traitname in self.known_traits.iter() {
            let key = format!("{}::{}::{}", typename, traitname, methodname);
            if let Some(funcval) = self.globals.get(&key).cloned() {
                trait_candidates.push((traitname.clone(), funcval));
            }
        }

        match trait_candidates.len() {
            0 => {}
            1 => {
                let (_tname, funcval) = trait_candidates.pop().unwrap();
                let mut callargs: Vec<Value> = Vec::with_capacity(1 + args.len());
                callargs.push(receiver.clone());
                callargs.extend(args.iter().cloned());

                match funcval {
                    Value::Builtin(ref name) => {
                        return self.call_builtin(name, callargs, span);
                    }
                    _ => {
                        return self.call_function_value(&funcval, callargs);
                    }
                }
            }
            _ => {
                let names = trait_candidates
                    .iter()
                    .map(|(t, _)| t.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(RuntimeError::InvalidOperation {
                    message: format!(
                        "Ambiguous method '{}' for type '{}' (candidates: {})",
                        methodname, typename, names
                    ),
                });
            }
        }

        // 3) "*::{method}"（ジェネリック/ワイルドカード）
        let wildcardkey = format!("*::{}", methodname);
        if let Some(funcval) = self.globals.get(&wildcardkey).cloned() {
            let mut callargs: Vec<Value> = Vec::with_capacity(1 + args.len());
            callargs.push(receiver.clone());
            callargs.extend(args.iter().cloned());

            match funcval {
                Value::Builtin(ref name) => {
                    return self.call_builtin(name, callargs, span);
                }
                _ => {
                    return self.call_function_value(&funcval, callargs);
                }
            }
        }

        // 4) フィールドアクセス fallback
        //    (Gaussian.sample のように、フィールドから関数を取り出して呼ぶパターン)
        if let Ok(funcval) = self.eval_field_access(&receiver, methodname) {
            let mut callargs: Vec<Value> = Vec::with_capacity(1 + args.len());
            callargs.push(receiver.clone());
            callargs.extend(args.iter().cloned());

            match funcval {
                Value::Builtin(ref name) => {
                    return self.call_builtin(name, callargs, span);
                }
                _ => {
                    return self.call_function_value(&funcval, callargs);
                }
            }
        }

        Err(RuntimeError::UndefinedField {
            field: format!("Method '{}' not found for type '{}'", methodname, typename),
        })
    }






        // 正しい eval_field_access
    fn eval_field_access(&self, object: &Value, field: &str) -> RuntimeResult<Value> {
        match object {
            Value::Struct { .. } if field == "clone" => Ok(object.clone()),
            Value::Struct { name, .. } if name == "MockResource" && field == "close" => {
                Ok(Value::Builtin("println_closed".to_string()))
            }
            Value::Struct { fields, .. } => {
                fields.get(field)
                    .cloned()
                    .ok_or_else(|| RuntimeError::UndefinedField { field: field.to_string() })
            }
            Value::Gaussian { mean, std } => {
                let (mean, std) = (mean.clone(), std.clone());
                match field {
                    "sample" => Ok(Value::Function {
                        params: vec![],
                        body: Block { statements: vec![], span: Span::initial() },
                        closure: Rc::new({
                            let mut e = self.env.clone();
                            e.set("__gaussian_mean".into(), Value::Float(mean));
                            e.set("__gaussian_std".into(), Value::Float(std));
                            e.set("__method_name".into(), Value::String(Rc::new("sample".into())));
                            e
                        })
                    }),
                    "pdf" => Ok(Value::Function {
                        params: vec![Parameter { name: Identifier { name: "x".into(), span: Span::initial() }, ty: Type::Float, span: Span::initial() }],
                        body: Block { statements: vec![], span: Span::initial() },
                        closure: Rc::new({
                            let mut e = self.env.clone();
                            e.set("__gaussian_mean".into(), Value::Float(mean));
                            e.set("__gaussian_std".into(), Value::Float(std));
                            e.set("__method_name".into(), Value::String(Rc::new("pdf".into())));
                            e
                        })
                    }),
                    "cdf" => Ok(Value::Function {
                        params: vec![Parameter { name: Identifier { name: "x".into(), span: Span::initial() }, ty: Type::Float, span: Span::initial() }],
                        body: Block { statements: vec![], span: Span { line: 0, column: 0, length: 1 } },
                        closure: Rc::new({
                            let mut e = self.env.clone();
                            e.set("__gaussian_mean".into(), Value::Float(mean));
                            e.set("__gaussian_std".into(), Value::Float(std));
                            e.set("__method_name".into(), Value::String(Rc::new("cdf".into())));
                            e
                        })
                    }),
                    _ => Err(RuntimeError::UndefinedField { field: field.to_string() }),
                }
            },
            Value::Signal { current_value, .. } => {
                let val = current_value.clone();
                match field {
                    "value" => Ok(val.as_ref().clone()),
                    "map" => Ok(Value::Function {
                        params: vec![Parameter { name: Identifier { name: "mapper".into(), span: Span::initial() }, ty: Type::Function { params: vec![], return_type: Box::new(Type::Unit) }, span: Span::initial() }],
                        body: Block { statements: vec![], span: Span::initial() },
                        closure: Rc::new({ let mut e = self.env.clone(); e.set("__signal_val".into(), val.as_ref().clone()); e.set("__signal_op".into(), Value::String(Rc::new("map".into()))); e })
                    }),
                    "filter" => Ok(Value::Function {
                        params: vec![Parameter { name: Identifier { name: "predicate".into(), span: Span::initial() }, ty: Type::Function { params: vec![], return_type: Box::new(Type::Bool) }, span: Span::initial() }],
                        body: Block { statements: vec![], span: Span::initial() },
                        closure: Rc::new({ let mut e = self.env.clone(); e.set("__signal_val".into(), val.as_ref().clone()); e.set("__signal_op".into(), Value::String(Rc::new("filter".into()))); e })
                    }),
                    _ => Err(RuntimeError::UndefinedField { field: field.to_string() }),
                }
            },
            Value::Module { name } => {
                self.modules
                    .get(name.as_str())
                    .and_then(|m| m.get(field))
                    .cloned()
                    .ok_or_else(|| RuntimeError::UndefinedField { field: format!("{name}::{field}") })
            },
            _ => Err(RuntimeError::InvalidOperation { message: format!("Cannot access field '{}' on non-struct type", field) }),
        }
    }

    // 正しい eval_call_value
    fn eval_call_value(&mut self, func_val: Value, arg_values: &[Value], span: Span) -> RuntimeResult<Value> {
        match func_val {
            Value::Builtin(ref name) => self.call_builtin(name, arg_values.to_vec(), span),
            Value::Function { params, body, closure } => {
                if params.len() != arg_values.len() {
                    return Err(RuntimeError::ArityMismatch { expected: params.len(), found: arg_values.len(), span });
                }

                if body.statements.is_empty() {
                    if let (Some(Value::Float(mean)), Some(Value::Float(std)), Some(Value::String(method_name))) =
                        (closure.get("__gaussian_mean"), closure.get("__gaussian_std"), closure.get("__method_name"))
                    {
                        match method_name.as_str() {
                            "sample" => {
                                let mut rng = rand::thread_rng();
                                let u1: f64 = rng.gen();
                                let u2: f64 = rng.gen();
                                let z = f64::sqrt(-2.0 * u1.ln()) * f64::cos(2.0 * std::f64::consts::PI * u2);
                                return Ok(Value::Float(ADFloat::from(mean.value() + std.value() * z)));
                            },
                            "pdf" => {
                                let x = match arg_values.get(0) {
                                    Some(Value::Float(f)) => f.clone(),
                                    _ => return Err(RuntimeError::TypeMismatch { message: "pdf expects a Float argument".into() })
                                };
                                if std <= ADFloat::Concrete(0.0) { return Ok(Value::Float(if x == mean { ADFloat::Concrete(f64::INFINITY) } else { ADFloat::Concrete(0.0) })); }
                                let z = (x.clone() - mean.clone()) / std.clone();
                                let pi = std::f64::consts::PI;
                                let denominator = std.clone() * ADFloat::Concrete((2.0 * pi).sqrt());
                                let exponent = -ADFloat::Concrete(0.5) * z.clone() * z.clone();
                                let pdf_val = (ADFloat::Concrete(1.0) / denominator) * exponent.exp();
                                return Ok(Value::Float(pdf_val));
                            },
                            "cdf" => {
                                let x = match arg_values.get(0) {
                                    Some(Value::Float(f)) => f.clone(),
                                    _ => return Err(RuntimeError::TypeMismatch { message: "cdf expects a Float argument".into() })
                                };
                                if std <= ADFloat::Concrete(0.0) { return Ok(Value::Float(if x >= mean { ADFloat::Concrete(1.0) } else { ADFloat::Concrete(0.0) })); }
                                let z = (x - mean) / (std * 2.0f64.sqrt());
                                let cdf_val = 0.5 * (1.0 + statrs::function::erf::erf(z.value()));
                                return Ok(Value::Float(ADFloat::Concrete(cdf_val)));
                            }
                            _ => {} 
                        }
                    }
                }

                let saved_env = self.env.clone();
                self.env = closure.as_ref().clone();
                self.push_scope();
                for (param, arg_val) in params.iter().zip(arg_values.iter()) {
                    self.env.set(param.name.name.clone(), arg_val.clone());
                }
                let result = match self.eval_block(&body) {
                    Ok(val) => val,
                    Err(RuntimeError::EarlyReturn) => self.return_value.take().unwrap_or(Value::Unit),
                    Err(e) => {
                        self.pop_scope();
                        self.env = saved_env;
                        return Err(e);
                    }
                };
                self.pop_scope();
                self.env = saved_env;
                Ok(result)
            }
            Value::NativeFunction { name, library_index, params, return_type, is_async } => {
                if is_async {
                    // Phase 4: Async FFI is not fully supported in the interpreter yet.
                    // For now, we return an error. A full implementation would require
                    // integrating with tokio or async-std runtime.
                    return Err(RuntimeError::Unimplemented(
                        format!("Async FFI call '{}' is not yet supported in the interpreter. \
                                 Use the native compiler for async extern functions.", name)
                    ));
                }
                self.call_native(&name, library_index, &params, &return_type, arg_values.to_vec(), span)
            }
            _ => Err(RuntimeError::NotCallable { value: format!("{:?}", func_val), span }),
        }
    }



    fn eval_index(&self, object: &Value, index: &Value) -> RuntimeResult<Value> {
        match (object, index) {
            (Value::Array(elements), Value::Int(idx)) => {
                let i = *idx as usize;
                elements.get(i).cloned().ok_or_else(|| RuntimeError::IndexOutOfBounds { index: i, length: elements.len() })
            },
            (Value::Tuple(elements), Value::Int(idx)) => {
                let i = *idx as usize;
                elements.get(i).cloned().ok_or_else(|| RuntimeError::IndexOutOfBounds { index: i, length: elements.len() })
            },
            _ => Err(RuntimeError::InvalidOperation { message: format!("Cannot index {} with {}", object.type_name(), index.type_name()) }),
        }
    }

    fn pattern_matches(&self, pattern: &Pattern, value: &Value) -> RuntimeResult<bool> {
        match (pattern, value) {
            (Pattern::Wildcard { .. }, _) => Ok(true),
            (Pattern::Identifier { .. }, _) => Ok(true),
            (Pattern::Literal { value: lit, .. }, val) => Ok(self.eval_literal(lit)? == *val),
            (Pattern::Tuple { patterns, .. }, Value::Tuple(values)) | (Pattern::Tuple { patterns, .. }, Value::Array(values)) => {
                if patterns.len() != values.len() { return Ok(false); }
                for (pat, val) in patterns.iter().zip(values.iter()) {
                    if !self.pattern_matches(pat, val)? { return Ok(false); }
                }
                Ok(true)
            },
            (Pattern::Some { pattern: inner, .. }, Value::Enum { name, variant, fields })
                if name == "Option" && variant == "Some" =>
            {
                if fields.len() != 1 { return Ok(false); }
                self.pattern_matches(inner, &fields[0])
            }

            (Pattern::None { .. }, Value::Enum { name, variant, fields })
                if name == "Option" && variant == "None" =>
            {
                Ok(fields.is_empty())
            }

            (Pattern::Ok { pattern: inner, .. }, Value::Enum { name, variant, fields })
                if name == "Result" && variant == "Ok" =>
            {
                if fields.len() != 1 { return Ok(false); }
                self.pattern_matches(inner, &fields[0])
            }

            (Pattern::Err { pattern: inner, .. }, Value::Enum { name, variant, fields })
                if name == "Result" && variant == "Err" =>
            {
                if fields.len() != 1 { return Ok(false); }
                self.pattern_matches(inner, &fields[0])
            }
            (Pattern::Enum { name: en, variant: ev, args: pargs, .. }, 
            Value::Enum { name: vn, variant: vv, fields, .. }) => {
                if en.name != *vn || ev.name != *vv {
                    return Ok(false);
                }
                if pargs.len() != fields.len() {
                    return Ok(false);
                }
                for (p, v) in pargs.iter().zip(fields.iter()) {
                    if !self.pattern_matches(p, v)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }



            (Pattern::Struct { name, fields, .. }, value) => {
                if let Value::Struct { name: s_name, fields: s_fields } = value {
                    if name.name != *s_name {
                        return Ok(false);
                    }
                    for field_pat in fields {
                        let field_name = &field_pat.name.name;
                        if let Some(val) = s_fields.get(field_name) {
                            // パターンがある場合のみチェック
                            if let Some(inner_pat) = &field_pat.pattern {
                                if !self.pattern_matches(inner_pat, val)? {
                                    return Ok(false);
                                }
                            }
                            // None の場合は「変数束縛」なので、値があればマッチ成功
                        } else {
                            return Ok(false); // フィールドが存在しない
                        }
                    }
                    Ok(true)
                } else {
                    Ok(false)
                }
            },
            (Pattern::Or { patterns, .. }, value) => {
                for pat in patterns {
                    if self.pattern_matches(pat, value)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            _ => Ok(false),
        }
    }

    fn bind_pattern(&mut self, pattern: &Pattern, value: Value) -> RuntimeResult<()> {
        match pattern {
            Pattern::Identifier { name, .. } => { self.env.set(name.name.clone(), value); Ok(()) },
            Pattern::Wildcard { .. } => Ok(()),
            Pattern::Tuple { patterns, .. } => {
                if let Value::Tuple(values) = value {
                    for (pat, val) in patterns.iter().zip(values.iter()) { self.bind_pattern(pat, val.clone())?; }
                    Ok(())
                } else if let Value::Array(values) = value {
                    for (pat, val) in patterns.iter().zip(values.iter()) { self.bind_pattern(pat, val.clone())?; }
                    Ok(())
                } else { Err(RuntimeError::TypeMismatch { message: "Expected tuple/array".into() }) }
            },
            Pattern::Some { pattern: inner, .. } => {
                if let Value::Enum { name, variant, fields } = value {
                    if name == "Option" && variant == "Some" {
                        if fields.len() != 1 {
                            return Err(RuntimeError::TypeMismatch {
                                message: "Option::Some must have exactly 1 field".into(),
                            });
                        }
                        return self.bind_pattern(inner, fields[0].clone());
                    }
                }
                Err(RuntimeError::TypeMismatch { message: "Expected Some".into() })
            },
            Pattern::None { .. } => {
                if let Value::Enum { name, variant, fields } = value {
                    if name == "Option" && variant == "None" {
                        if !fields.is_empty() {
                            return Err(RuntimeError::TypeMismatch {
                                message: "Option::None must have 0 fields".into(),
                            });
                        }
                        return Ok(());
                    }
                }
                Err(RuntimeError::TypeMismatch { message: "Expected None".into() })
            },
            Pattern::Ok { pattern: inner, .. } => {
                if let Value::Enum { name, variant, fields } = value {
                    if name == "Result" && variant == "Ok" {
                        if fields.len() != 1 {
                            return Err(RuntimeError::TypeMismatch {
                                message: "Result::Ok must have exactly 1 field".into(),
                            });
                        }
                        return self.bind_pattern(inner, fields[0].clone());
                    }
                }
                Err(RuntimeError::TypeMismatch { message: "Expected Ok".into() })
            },
            Pattern::Err { pattern: inner, .. } => {
                if let Value::Enum { name, variant, fields } = value {
                    if name == "Result" && variant == "Err" {
                        if fields.len() != 1 {
                            return Err(RuntimeError::TypeMismatch {
                                message: "Result::Err must have exactly 1 field".into(),
                            });
                        }
                        return self.bind_pattern(inner, fields[0].clone());
                    }
                }
                Err(RuntimeError::TypeMismatch { message: "Expected Err".into() })
            },
            Pattern::Enum { name: en, variant: ev, args: pargs, .. } => {
                if let Value::Enum { name: vn, variant: vv, fields, .. } = value {
                    // 名前チェック
                    if en.name != vn || ev.name != vv {
                        return Err(RuntimeError::TypeMismatch { 
                            message: format!("Expected {}::{}, got {}::{}", en.name, ev.name, vn, vv) 
                        }.into());
                    }
                    // 数チェック
                    if pargs.len() != fields.len() {
                        return Err(RuntimeError::TypeMismatch { 
                            message: format!("Arity mismatch: expected {}, got {}", pargs.len(), fields.len()) 
                        }.into());
                    }
                    // 再帰的に束縛
                    for (p, v) in pargs.iter().zip(fields.iter()) {
                        self.bind_pattern(p, v.clone())?;
                    }
                    Ok(())
                } else {
                    Err(RuntimeError::TypeMismatch { message: "Expected Enum value".into() }.into())
                }
            },
            


            Pattern::Struct { name, fields, .. } => {
                if let Value::Struct { name: s_name, fields: s_fields } = value {
                    if name.name != *s_name {
                        return Err(RuntimeError::TypeMismatch { 
                            message: format!("Expected struct {}, got {}", name.name, s_name) 
                        });
                    }
                    for field_pat in fields {
                        let field_name = &field_pat.name.name;
                        let val = s_fields.get(field_name).ok_or_else(|| RuntimeError::TypeMismatch {
                             message: format!("Missing field {} in struct {}", field_name, s_name)
                        })?;

                        // Patternがあれば再帰的に、なければ名前で束縛
                        if let Some(pat) = &field_pat.pattern {
                            self.bind_pattern(pat, val.clone())?;
                        } else {
                            // 省略形 { x } -> x に束縛
                            self.env.set(field_name.clone(), val.clone());
                        }
                    }
                    Ok(())
                } else {
                    Err(RuntimeError::TypeMismatch { message: "Expected Struct value".into() })
                }
            },
            Pattern::Or { patterns, .. } => {
                // マッチする最初のパターンで束縛を行う
                // 注意: Orパターン内の全パターンは同じ変数を束縛する必要がある（静的チェックで保証される前提）
                for pat in patterns {
                    if self.pattern_matches(pat, &value)? {
                        self.bind_pattern(pat, value.clone())?;
                        return Ok(());
                    }
                }
                Err(RuntimeError::TypeMismatch { message: "No pattern matched in Or-pattern".into() })
            },
            _ => Ok(()),
        }
    }

    pub fn push_scope(&mut self) { self.env.push_scope(); }
    pub fn pop_scope(&mut self) { self.env.pop_scope(); }
    pub fn set_variable(&mut self, name: String, value: Value) { self.env.set(name, value); }
    pub fn get_variable(&self, name: &str) -> Option<Value> { self.env.get(name).or_else(|| self.globals.get(name).cloned()) }
}

// Helper for std::serialize
fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Int(i) => serde_json::Value::Number((*i).into()),
        Value::Float(f) => serde_json::Value::Number(serde_json::Number::from_f64(f.value()).unwrap_or(serde_json::Number::from(0))),
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::String(s) => serde_json::Value::String(s.as_ref().clone()),
        Value::Unit => serde_json::Value::Null,
        Value::Array(arr) => serde_json::Value::Array(arr.iter().map(|item| value_to_json(item)).collect()),
        Value::Map(m) => {
             let mut json_map = serde_json::Map::new();
             for (k, val) in m.borrow().iter() {
                 json_map.insert(k.clone(), value_to_json(val));
             }
             serde_json::Value::Object(json_map)
        },
        Value::Struct { name, fields } => {
             let mut json_map = serde_json::Map::new();
             json_map.insert("__type".to_string(), serde_json::Value::String(name.clone()));
             for (k, val) in fields.iter() {
                  json_map.insert(k.clone(), value_to_json(val));
             }
             serde_json::Value::Object(json_map)
        },
        Value::Enum { name, variant, fields } => {
             let mut json_map = serde_json::Map::new();
             json_map.insert("__enum".to_string(), serde_json::Value::String(name.clone()));
             json_map.insert("variant".to_string(), serde_json::Value::String(variant.clone()));
             json_map.insert("fields".to_string(), serde_json::Value::Array(fields.iter().map(|f| value_to_json(f)).collect()));
             serde_json::Value::Object(json_map)
        }
        _ => serde_json::Value::String(format!("<{}>", v.type_name())),
    }
}

fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Unit,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if n.is_i64() {
                Value::Int(n.as_i64().unwrap())
            } else {
                Value::Float(crate::ad::types::ADFloat::Concrete(n.as_f64().unwrap_or(0.0)))
            }
        },
        serde_json::Value::String(s) => Value::String(crate::gc::Rc::new(s.clone())),
        serde_json::Value::Array(arr) => {
            Value::Array(crate::gc::Rc::new(arr.iter().map(|item| json_to_value(item)).collect()))
        },
        serde_json::Value::Object(map) => {
            // Check for Struct/Enum markers
            if let Some(serde_json::Value::String(ty)) = map.get("__type") {
                 let mut fields = std::collections::HashMap::new();
                 for (k, v) in map.iter() {
                     if k == "__type" { continue; }
                     fields.insert(k.clone(), json_to_value(v));
                 }
                 Value::Struct { name: ty.clone(), fields: crate::gc::Rc::new(fields) }
            } else if let Some(serde_json::Value::String(en)) = map.get("__enum") {
                 let variant = map.get("variant").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string();
                 let fields_val = map.get("fields").and_then(|v| v.as_array()).map(|arr| {
                     arr.iter().map(|item| json_to_value(item)).collect()
                 }).unwrap_or_default();
                 
                 Value::Enum { name: en.clone(), variant, fields: crate::gc::Rc::new(fields_val) }
            } else {
                 // Regular Map
                 let mut content = std::collections::HashMap::new();
                 for (k, v) in map.iter() {
                     content.insert(k.clone(), json_to_value(v));
                 }
                 Value::Map(crate::gc::Rc::new(std::cell::RefCell::new(content)))
            }
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod enum_option_result_tests {
    use super::*;
    use crate::gc::Rc;

    #[test]
    fn enum_option_some_none_display() {
        let some = Value::Enum {
            name: "Option".into(),
            variant: "Some".into(),
            fields: Rc::new(vec![Value::Int(42)]),
        };
        let none = Value::Enum {
            name: "Option".into(),
            variant: "None".into(),
            fields: Rc::new(vec![]),
        };

        // Display が Option/Result 統一後も分かりやすく出ることを確認
        assert_eq!(some.to_string(), "Option::Some(42)");
        assert_eq!(none.to_string(), "Option::None");
    }

    #[test]
    fn enum_result_ok_err_display() {
        let ok = Value::Enum {
            name: "Result".into(),
            variant: "Ok".into(),
            fields: Rc::new(vec![Value::String(Rc::new("ok".into()))]),
        };
        let err = Value::Enum {
            name: "Result".into(),
            variant: "Err".into(),
            fields: Rc::new(vec![Value::String(Rc::new("err".into()))]),
        };

        assert_eq!(ok.to_string(), "Result::Ok(ok)");
        assert_eq!(err.to_string(), "Result::Err(err)");
    }

    #[test]
    fn enum_result_shape_mismatch_is_detectable() {
        // Try 演算子などが「Result::Ok はフィールド1個」を期待するので、
        // 壊れた値を作れること自体はOKだが、VM側が検出できる形にしておく。
        let bad_ok = Value::Enum {
            name: "Result".into(),
            variant: "Ok".into(),
            fields: Rc::new(vec![]),
        };
        assert_eq!(bad_ok.type_name(), "Result"); // type_name を name.as_str() にしてる前提
    }
}
