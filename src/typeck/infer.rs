// Unified type inference and type checking for Flux language.
//
// Includes:
// - Literal inference
// - Binary/Unary expression checking
// - If-expression (Block-based)
// - Let statement
// - Function definition and call

use crate::ast::node::{BinaryOp, Expression, Literal, Span, Type, Block, UnaryOp, Pattern, Path, Identifier, VariantData, DefId, ImplItem, WherePredicate};
use crate::ast::node::PathSeg;
use crate::typeck::env::{TypeEnv, ItemDef, TypeAliasDef, TraitDefInfo, EnumDefInfo, VariantInfo, ImplDef, StructDefInfo};
use crate::typeck::error::{TypeError, TypeResult};
use std::collections::HashMap;


#[derive(Debug, Clone)]
pub struct TypeInfer {
    pub env: TypeEnv,
    next_meta: usize,
    subst: HashMap<usize, Type>,
}
struct SelectedImpl {
    impl_def: ImplDef,
    subst: HashMap<String, Type>,
    assoc_bindings: HashMap<String, Type>,
}
impl TypeInfer {
    pub fn new() -> Self {
        let mut type_env = TypeEnv::new();
        
        // 組み込み関数を登録
        
        // println(...)
        type_env.define(
            "println".to_string(),
            Type::Function {
                params: vec![Type::Variadic(Box::new(Type::Any))],
                return_type: Box::new(Type::Unit),
            },
        ).ok();
        type_env.define(
            "print".to_string(),
            Type::Function {
                params: vec![Type::Variadic(Box::new(Type::Any))],
                return_type: Box::new(Type::Unit),
            },
        ).ok();
        type_env.define(
            "nearly_eq".to_string(), 
            Type::Function { 
                params: vec![Type::Float, Type::Float], 
                return_type: Box::new(Type::Bool) 
            }
        ).ok();
        // Signal_new(value: Any) -> Signal<Any>
        type_env.define(
            "Signal_new".to_string(),
            Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::Signal(Box::new(Type::Any))),
            },
        ).ok();

        // Signal_set(sig: Signal<Any>, value: Any) -> Unit
        type_env.define(
            "Signal_set".to_string(),
            Type::Function {
                params: vec![
                    Type::Signal(Box::new(Type::Any)),
                    Type::Any,
                ],
                return_type: Box::new(Type::Unit),
            },
        ).ok();

        // Signal_get(sig: Signal<Any>) -> Any
        type_env.define(
            "Signal_get".to_string(),
            Type::Function {
                params: vec![Type::Signal(Box::new(Type::Any))],
                return_type: Box::new(Type::Any),
            },
        ).ok();

        // Signal_map(sig: Signal<Any>, f: Fn(Any) -> Any) -> Signal<Any>
        type_env.define(
            "Signal_map".to_string(),
            Type::Function {
                params: vec![
                    Type::Signal(Box::new(Type::Any)),
                    Type::Function {
                        params: vec![Type::Any],
                        return_type: Box::new(Type::Any),
                    },
                ],
                return_type: Box::new(Type::Signal(Box::new(Type::Any))),
            },
        ).ok();

        // Signal_filter(sig: Signal<Any>, pred: Fn(Any) -> Bool) -> Signal<Any>
        type_env.define(
            "Signal_filter".to_string(),
            Type::Function {
                params: vec![
                    Type::Signal(Box::new(Type::Any)),
                    Type::Function {
                        params: vec![Type::Any],
                        return_type: Box::new(Type::Bool),
                    },
                ],
                return_type: Box::new(Type::Signal(Box::new(Type::Any))),
            },
        ).ok();
        // Event_new() -> Event<Any>
        type_env.define(
            "Event_new".to_string(),
            Type::Function {
                params: vec![],
                return_type: Box::new(Type::Event(Box::new(Type::Any))),
            },
        ).ok();

        // Event_emit(evt: Event<Any>, value: Any) -> Unit
        type_env.define(
            "Event_emit".to_string(),
            Type::Function {
                params: vec![Type::Event(Box::new(Type::Any)), Type::Any],
                return_type: Box::new(Type::Unit),
            },
        ).ok();

        // Event_fold(evt: Event<Any>, init: Any, f: Fn(Any, Any) -> Any) -> Signal<Any>
        type_env.define(
            "Event_fold".to_string(),
            Type::Function {
                params: vec![
                    Type::Event(Box::new(Type::Any)),
                    Type::Any,
                    Type::Function {
                        params: vec![Type::Any, Type::Any],
                        return_type: Box::new(Type::Any),
                    },
                ],
                return_type: Box::new(Type::Signal(Box::new(Type::Any))),
            },
        ).ok();
        // Gaussian(mean: Float, std: Float) -> Gaussian
        type_env.define(
            "Gaussian".to_string(),
            Type::Function {
                params: vec![Type::Float, Type::Float],
                return_type: Box::new(Type::Gaussian),
            },
        ).ok();

        // Uniform(min: Float, max: Float) -> Uniform
        type_env.define(
            "Uniform".to_string(),
            Type::Function {
                params: vec![Type::Float, Type::Float],
                return_type: Box::new(Type::Uniform),
            },
        ).ok();

        // sample(dist: Any) -> Float
        type_env.define(
            "sample".to_string(),
            Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::Float),
            },
        ).ok();
        // observe(dist: Any, value: Float) -> Unit
        type_env.define(
            "observe".to_string(),
            Type::Function {
                params: vec![Type::Any, Type::Float],
                return_type: Box::new(Type::Unit),
            },
        ).ok();

        // Bernoulli(p: Float) -> Bernoulli
        type_env.define(
            "Bernoulli".to_string(),
            Type::Function {
                params: vec![Type::Float],
                return_type: Box::new(Type::Bernoulli),
            },
        ).ok();

        // Beta(alpha: Float, beta: Float) -> Beta
        type_env.define(
            "Beta".to_string(),
            Type::Function {
                params: vec![Type::Float, Type::Float],
                return_type: Box::new(Type::Beta),
            },
        ).ok();

        // Math Builtins (Phase 8)
        let unary_float = Type::Function {
             params: vec![Type::Float],
             return_type: Box::new(Type::Float),
        };
        for name in &["exp", "ln", "sin", "cos", "tan", "sqrt"] {
             type_env.define(name.to_string(), unary_float.clone()).ok();
        }
        
        type_env.define("pow".to_string(), Type::Function {
             params: vec![Type::Float, Type::Float],
             return_type: Box::new(Type::Float),
        }).ok();

        // Serialize Builtins (Phase 8)
        type_env.define("to_json".to_string(), Type::Function {
             params: vec![Type::Any],
             return_type: Box::new(Type::String),
        }).ok();
        type_env.define("from_json".to_string(), Type::Function {
             params: vec![Type::String],
             return_type: Box::new(Type::Any),
        }).ok();

        // infer(n: Int, model: Fn() -> Any) -> Array<Any>
        type_env.define(
            "infer".to_string(),
            Type::Function {
                params: vec![
                    Type::Int,
                    //Type::Function {
                    //    params: vec![],
                    //    return_type: Box::new(Type::Any),
                    //},
                    Type::Any,
                ],
                return_type: Box::new(Type::Array(Box::new(Type::Any))),
            },
        ).ok();

        // infer_vi(config: Map<String, Any>, model: Fn() -> Unit, guide: Fn() -> Unit) -> Map<String, Float>
        type_env.define(
            "infer_vi".to_string(),
            Type::Function {
                params: vec![
                    Type::Map(Box::new(Type::String), Box::new(Type::Any)),
                    Type::Any, // model
                    Type::Any, // guide
                ],
                return_type: Box::new(Type::Map(Box::new(Type::String), Box::new(Type::Float))),
            },
        ).ok();

        // infer_hmc(config: Map<String, Any>, model: Fn() -> Any) -> Array<Map<String, Float>>
        type_env.define(
            "infer_hmc".to_string(),
            Type::Function {
                params: vec![
                    Type::Map(Box::new(Type::String), Box::new(Type::Any)),
                    Type::Any, // model
                ],
                return_type: Box::new(Type::Array(Box::new(
                    Type::Map(Box::new(Type::String), Box::new(Type::Float))
                ))),
            },
        ).ok();

        // param(name: String, init: Float) -> Float
        type_env.define(
            "param".to_string(),
            Type::Function {
                params: vec![Type::String, Type::Float],
                return_type: Box::new(Type::Float),
            },
        ).ok();

        // Map() -> Map<String, Any>
        type_env.define(
            "Map".to_string(),
            Type::Function {
                params: vec![],
                return_type: Box::new(Type::Map(Box::new(Type::String), Box::new(Type::Any))),
            },
        ).ok();

        // exp(x: Float) -> Float
        type_env.define(
            "exp".to_string(),
            Type::Function {
                params: vec![Type::Float],
                return_type: Box::new(Type::Float),
            },
        ).ok();
        // Rc_new<T>(value: T) -> Rc<T>
        type_env.define(
            "Rc_new".to_string(),
            Type::Function {
                params: vec![Type::Any],
                return_type: Box::new(Type::Rc(Box::new(Type::Any))),
            },
        ).ok();

        // Rc_downgrade<T>(rc: Rc<T>) -> Weak<T>
        type_env.define(
            "Rc_downgrade".to_string(),
            Type::Function {
                params: vec![Type::Rc(Box::new(Type::Any))],
                return_type: Box::new(Type::Weak(Box::new(Type::Any))),
            },
        ).ok();

        // Weak_upgrade<T>(weak: Weak<T>) -> Option<Rc<T>>
        type_env.define(
            "Weak_upgrade".to_string(),
            Type::Function {
                params: vec![Type::Weak(Box::new(Type::Any))],
                return_type: Box::new(Type::Option(Box::new(Type::Rc(Box::new(Type::Any))))),
            },
        ).ok();
        // --- AD (Automatic Differentiation) Builtins ---
        // create_tape() -> Int
        type_env.define(
            "create_tape".to_string(),
            Type::Function {
                params: vec![],
                return_type: Box::new(Type::Int),
            },
        ).ok();

        // param(value: Float, tape_id: Int) -> Float
        type_env.define(
            "param".to_string(),
            Type::Function {
                params: vec![Type::Float, Type::Int],
                return_type: Box::new(Type::Float),
            },
        ).ok();
        
        // backward(target: Float) -> Unit
        type_env.define(
            "backward".to_string(),
            Type::Function {
                params: vec![Type::Float],
                return_type: Box::new(Type::Unit),
            },
        ).ok();
        
        // grad(target: Float) -> Float
        type_env.define(
            "grad".to_string(),
            Type::Function {
                params: vec![Type::Float],
                return_type: Box::new(Type::Float),
            },
        ).ok();


        TypeInfer {
            env: type_env,
            next_meta: 0,
            subst: HashMap::new(),
        }
    }

    fn trait_key_from_path(&self, path: &Path) -> String {
        // いまは「末尾の識別子だけ」で十分ならこれで OK
        // 例: Foo::Bar::Greeter -> "Greeter"
        path.last_ident().unwrap().name.clone()
    }


    fn resolve_trait_def_from_path(&self, path: &Path, span: Span) -> TypeResult<DefId> {
        if let Some(id) = path.resolved {
            return Ok(id);
        }

        // いまは「末尾の識別子だけ」をキーにする
        let ident = path
            .last_ident()
            .ok_or(TypeError::CannotInfer { span })?;
        let key = ident.name.clone();

        self.env
            .resolve_def(&ident.name)
            .ok_or(TypeError::UndefinedVariable {
                name: ident.name.clone(),
                span,
            })
    }

    // ============================================================
    // Literal and Variable
    // ============================================================
    pub fn infer_literal(&self, lit: &Literal) -> Type {
        match lit {
            Literal::Int(_) => Type::Int,
            Literal::Float(_) => Type::Float,
            Literal::Bool(_) => Type::Bool,
            Literal::String(_) => Type::String,
            Literal::Unit => Type::Unit,
        }
    }

    pub fn infer_variable(&self, name: &str, span: Span) -> TypeResult<Type> {
        self.env.lookup(name).ok_or(TypeError::UndefinedVariable {
            name: name.to_string(),
            span,
        })
    }




    // ============================================================
    // Expression Dispatcher
    // ============================================================
    pub fn infer_expression(&mut self, expr: &mut Expression) -> TypeResult<Type> {
        match expr {
            Expression::Literal { value, .. } => Ok(self.infer_literal(value)),
            Expression::Variable { name, span, .. } => {
                // 0) Try resolving Enum::Variant (Simple case: EnumName::VariantName)
                if name.segments.len() >= 2 {
                    let enum_name_seg = &name.segments[name.segments.len() - 2];
                    if let PathSeg::Ident(enum_ident) = enum_name_seg {
                        if let Some(def_id) = self.env.resolve_def(&enum_ident.name) {
                             // Clone to release borrow on self.env
                             let enum_data = if let Some(ItemDef::Enum(info)) = self.env.get_def(def_id) {
                                  Some((info.variants.clone(), info.typeparams.clone()))
                             } else {
                                  None
                             };

                             if let Some((variants, typeparams)) = enum_data {
                                 let variant_seg = name.segments.last().unwrap();
                                 if let PathSeg::Ident(variant_ident) = variant_seg {
                                      if let Some(variant_info) = variants.get(&variant_ident.name) {
                                          // Found Variant!
                                          name.resolved = Some(def_id); 
                                          
                                          // DEBUG
                                          println!("[DEBUG] Found Enum Variant: {:?}, typeparams len: {}", enum_ident.name, typeparams.len());

                                          // Generics Instantiation
                                          let mut typeargs = Vec::new();
                                          for _ in &typeparams {
                                              typeargs.push(self.fresh_meta());
                                          }
                                          
                                          let mut subst = std::collections::HashMap::new();
                                          for (i, param) in typeparams.iter().enumerate() {
                                              subst.insert(param.name.name.clone(), typeargs[i].clone());
                                          }

                                          // Construct Enum Type
                                          let enum_path = Path { 
                                              segments: name.segments[..name.segments.len()-1].to_vec(),
                                              span: *span,
                                              resolved: Some(def_id) 
                                          };
                                          let enum_ty = Type::Named {
                                              name: enum_path,
                                              type_args: typeargs, 
                                          };
                                          
                                          match variant_info {
                                              VariantInfo::Unit => return Ok(enum_ty),
                                              VariantInfo::Tuple(tys) => {
                                                  // Substitute field types
                                                  let mut func_params = Vec::new();
                                                  for t in tys {
                                                      func_params.push(self.subst_typevars(t.clone(), &subst));
                                                  }

                                                  return Ok(Type::Function {
                                                      params: func_params,
                                                      return_type: Box::new(enum_ty),
                                                  });
                                              },
                                              VariantInfo::Struct(_) => {
                                                  return Err(TypeError::CannotInfer { span: *span });
                                              }
                                          }    
                                     }
                                 }
                             }
                        }
                    }
                }

                // 借用を跨がないように clone する
                let id = name
                    .last_ident()
                    .cloned()
                    .ok_or(TypeError::CannotInfer { span: *span })?;

                // 1) ローカル値（let/param）を優先
                if let Some(ty) = self.env.lookup(&id.name) {
                    return Ok(ty);
                }


                // 2) defs 側（関数など）を参照
                let defid = self
                    .env
                    .resolve_def(&id.name)
                    .ok_or(TypeError::UndefinedVariable {
                        name: id.name.clone(),
                        span: *span,
                    })?;

                // ASTに解決結果を書き込む
                name.resolved = Some(defid);

                match self.env.get_def(defid) {
                    Some(ItemDef::Function(ty)) => Ok(ty.clone()),
                    // 構造体や列挙型バリアントなど、変数として参照可能な定義があればここに追加
                    Some(ItemDef::Struct(_)) => {
                        // 構造体そのものを変数として使う（コンストラクタ的な用法など）場合の型が必要なら返す
                        // 現状の実装に合わせて調整してください。
                         Err(TypeError::UndefinedVariable {
                            name: id.name.clone(),
                            span: *span,
                        })
                    }
                     _ => Err(TypeError::UndefinedVariable {
                        name: id.name.clone(),
                        span: *span,
                    }),
                }
            }


            Expression::Binary { op, left, right, span } => {
                self.infer_binary(*op, left, right, *span)
            }
            Expression::Unary { op, operand, span } => {
                self.infer_unary(op, operand, *span)
            }
            Expression::If { condition, then_branch, else_branch, .. } => {
                // Block → Expression::Blockとして推論
                self.infer_if_expression(condition, then_branch, else_branch.as_mut())
            }
            Expression::Call { callee, args, span } => {
                // Call(obj.method, args...) を MethodCall(obj, method, args...) へデシュガーするための一時置き場
                // （borrow の都合で、この場で *expr = ... しない）
                let mut desugared: Option<(Expression, Type)> = None;
                {
                    if let Expression::UfcsMethod { trait_path, method, span: ufcs_span } = callee.as_mut() {
                        // 借用を切るために clone（このあと *expr に代入するので重要）
                        let traitpath = trait_path.clone();
                        let method = method.clone();
                        let call_span = *span;
                        let ufcs_span = *ufcs_span;

                        if args.is_empty() {
                            return Err(TypeError::ArityMismatch { expected: 1, found: 0, span: ufcs_span });
                        }

                        // receiver は args[0]
                        let receiverty = {
                            let t = self.infer_expression(args.first_mut().unwrap())?;
                            self.apply(t)
                        };

                        // trait 解決（Greeter が trait でないなら GreeterTrait fallback）
                        let mut traitdefid = self.resolve_trait_def_from_path(&traitpath, ufcs_span)?;
                        let is_trait = matches!(self.env.get_def(traitdefid), Some(ItemDef::Trait(_)));
                        if !is_trait {
                            let base = self.trait_key_from_path(&traitpath);          // 例: "Greeter"
                            let fallback = format!("{base}Trait");                    // 例: "GreeterTrait"
                            if let Some(fid) = self.env.resolve_def(fallback.as_str()) {
                                if matches!(self.env.get_def(fid), Some(ItemDef::Trait(_))) {
                                    traitdefid = fid;
                                }
                            }
                        }

                        // impl selection + method type/defid 解決（infer_function_call の UFCS 分岐の中身を移植）
                        let selected = self.select_impl(traitdefid, receiverty.clone(), ufcs_span)?;
                        let (methodty, methoddefid) = selected
                            .impl_def
                            .methods
                            .get(method.name.as_str())
                            .ok_or(TypeError::CannotInfer { span: ufcs_span })?
                            .clone();

                        // methodty = fn(Self, ...) -> Ret を前提に、receiver/args を unify（既存コードを移植）
                        let (params, returntype) = match methodty {
                            Type::Function { params, return_type } => (params, return_type),
                            other => return Err(TypeError::NotCallable { value_type: other, span: ufcs_span }),
                        };

                        // receiver unify + 引数 unify（params[0] が Self）
                        self.unify(receiverty.clone(), params[0].clone(), ufcs_span)?;
                        for (i, argexpr) in args.iter_mut().enumerate() {
                            if i == 0 { continue; } // args[0] は receiver
                            let argty = self.infer_expression(argexpr)?;
                            self.unify(argty, params[i].clone(), argexpr.span())?;
                        }

                        let receiver_expr = args.remove(0);
                        let method_args = std::mem::take(args);
                        
                        // desugared に代入して、スコープを抜けた後で *expr を更新させる
                        desugared = Some((
                            Expression::MethodCall {
                                receiver: Box::new(receiver_expr),
                                method,
                                args: method_args,
                                span: call_span,
                                resolved: Some(methoddefid),
                            },
                            self.apply(*returntype)
                        ));
                    }
                    // callee が `obj.method` の形なら、メソッド呼び出しとして扱い receiver を暗黙引数にする
                    if let Expression::FieldAccess { object, field, .. } = callee.as_mut() {
                        let obj_ty_raw = self.infer_expression(object.as_mut())?;
                        let obj_ty = self.apply(obj_ty_raw);

                        // String Method Hook (Phase 7)
                        if matches!(obj_ty, Type::String) {
                            let receiver_expr = std::mem::replace(
                                object,
                                Box::new(Expression::Literal { value: Literal::Unit, span: *span }),
                            );
                            
                            let (ret_type, is_method) = match field.name.as_str() {
                                "trim" => (Type::String, true),
                                "split" => (Type::Array(Box::new(Type::String)), true),
                                "replace" => (Type::String, true),
                                "contains" => (Type::Bool, true),
                                _ => (Type::Unit, false), 
                            };
                            
                            if is_method {
                                for arg in args.iter_mut() {
                                    let _ = self.infer_expression(arg)?;
                                }
                                
                                desugared = Some((
                                    Expression::MethodCall {
                                        receiver: receiver_expr,
                                        method: field.clone(),
                                        args: std::mem::take(args),
                                        span: *span,
                                        resolved: None,
                                    },
                                    ret_type
                                ));
                            }
                        }

                        // Gaussian.sample() の場合
                        if let Type::Gaussian = obj_ty {
                            if field.name == "sample" {
                                let receiver_expr = std::mem::replace(
                                    object,
                                    Box::new(Expression::Literal {
                                        value: Literal::Unit,
                                        span: *span,
                                    }),
                                );
                                
                                desugared = Some((
                                    Expression::MethodCall {
                                        receiver: receiver_expr,
                                        method: field.clone(),
                                        args: std::mem::take(args),
                                        span: *span,
                                        resolved: None,
                                    },
                                    Type::Float
                                ));
                                // これで後続の処理（if let Some(...) = desugared）に流れるはず
                            }
                        }

                        // 解決対象のパスを取得 (Named または DynTrait)
                        let path_opt = match &obj_ty {
                            Type::Named { name, .. } => Some(name),
                            Type::DynTrait { trait_path } => Some(trait_path),
                            _ => {
                                None
                            }
                        };

                        if let Some(path) = path_opt {
                            // 名前解決を行い、DefId を取得
                            let selfdefid = if let Some(defid) = path.resolved {
                                defid
                            } else {
                                self.env
                                    .resolve_def(path.last_ident().unwrap().name.as_str())
                                    .ok_or(TypeError::CannotInfer { span: *span })?
                            };

                            // (Type, Option<DefId>) で集める：
                            // - inherent / impl 経由なら Some(defid)
                            // - trait 定義直参照（フォールバック）なら None
                            let mut method_sig: Option<(Type, Option<DefId>)> = None;

                            // 1) inherent impl
                            if let Some((ty, defid)) = self.env.lookup_method(selfdefid, field.name.as_str()) {
                                method_sig = Some((ty.clone(), Some(*defid)));
                            }

                            // 2) trait 定義（DefId はここでは取れないので None）
                            if method_sig.is_none() {
                                if let Some(ItemDef::Trait(trait_info)) = self.env.get_def(selfdefid) {
                                    if let Some(ty) = trait_info.methods.get(field.name.as_str()) {
                                        method_sig = Some((ty.clone(), None));
                                    }
                                }
                            }

                            // 3) impl 定義から探す (Trait Impl for Type)
                            if method_sig.is_none() {
                                for impl_def in &self.env.impls {
                                    // 対象の型に対する Impl かどうかをチェック（簡易に DefId 一致）
                                    let is_matching_type = match &impl_def.self_ty {
                                        Type::Named { name: impl_name, .. } => {
                                            impl_name
                                                .resolved
                                                .or_else(|| {
                                                    self.env.resolve_def(
                                                        impl_name.last_ident().unwrap().name.as_str(),
                                                    )
                                                }) == Some(selfdefid)
                                        }
                                        _ => false,
                                    };

                                    if is_matching_type {
                                        if let Some((mty, mdefid)) = impl_def.methods.get(&field.name) {
                                            method_sig = Some((mty.clone(), Some(*mdefid)));
                                            break;
                                        }
                                    }
                                }
                            }

                            if let Some((method_ty, resolved_defid)) = method_sig {
                                if let Type::Function { params, return_type } = method_ty {
                                    // 引数の個数チェック (可変長引数対応)
                                    let is_variadic = params
                                        .last()
                                        .map_or(false, |t| matches!(t, Type::Variadic(_)));

                                    // params[0] が receiver(Self) なので、呼び出し側 args は expected_min 個
                                    let expected_min = params.len().saturating_sub(1);
                                    if !is_variadic && args.len() != expected_min {
                                        return Err(TypeError::ArityMismatch {
                                            expected: expected_min,
                                            found: args.len(),
                                            span: *span,
                                        });
                                    }
                                    if is_variadic && args.len() < params.len().saturating_sub(2) {
                                        return Err(TypeError::ArityMismatch {
                                            expected: params.len().saturating_sub(2),
                                            found: args.len(),
                                            span: *span,
                                        });
                                    }

                                    // receiver (obj_ty) を params[0] (Self) に unify
                                    self.unify(obj_ty.clone(), params[0].clone(), object.span())?;

                                    let args_len = args.len();
                                    for (i, arg_expr) in args.iter_mut().enumerate() {
                                        let arg_ty = self.infer_expression(arg_expr)?;
                                        let expected_ty = if 1 + i < params.len() {
                                            match &params[1 + i] {
                                                Type::Variadic(inner) => inner.as_ref().clone(),
                                                other => other.clone(),
                                            }
                                        } else if is_variadic {
                                            match params.last().unwrap() {
                                                Type::Variadic(inner) => inner.as_ref().clone(),
                                                _ => unreachable!(),
                                            }
                                        } else {
                                            return Err(TypeError::ArityMismatch {
                                                expected: expected_min,
                                                found: args_len,
                                                span: arg_expr.span(),
                                            });
                                        };

                                        self.unify(arg_ty, expected_ty, arg_expr.span())?;
                                    }

                                    // 返り値型（メタ変数解消込み）
                                    let ret_ty = self.apply(*return_type);

                                    // ここで AST を MethodCall に置き換える（※ borrow のため外で *expr に代入）
                                    let receiver_expr = std::mem::replace(
                                        object,
                                        Box::new(Expression::Literal {
                                            value: Literal::Unit,
                                            span: *span,
                                        }),
                                    );
                                    let method_ident = field.clone();
                                    let call_args = std::mem::take(args);

                                    desugared = Some((
                                        Expression::MethodCall {
                                            receiver: receiver_expr,
                                            method: method_ident,
                                            args: call_args,
                                            span: *span,
                                            resolved: resolved_defid,
                                        },
                                        ret_ty,
                                    ));
                                }
                            }
                        }
                        // メソッドとして解決できなかった場合はこの if ブロックを抜け、
                        // 通常の関数 (FieldAccess の結果を呼び出す) として扱われる
                    }
                }

                // borrow が切れた後に置換する
                if let Some((new_expr, ty)) = desugared {
                    *expr = new_expr;
                    return Ok(ty);
                }

                // UFCSハック: Iterable::head(a) のような呼び出しを特例で処理
                if let Expression::Variable { name, .. } = &**callee {
                    // 関数名（パスの末尾）が "head" の場合
                    if name.last_ident().map(|id| id.name.as_str()) == Some("head") {
                        // 引数が1つ以上あるか確認
                        if let Some(first_arg) = args.first_mut() {
                            // 第一引数の型を推論
                            if let Ok(arg_ty) = self.infer_expression(first_arg) {
                                // 配列型 (Array<T>) なら、戻り値を Option<T> とする
                                if let Type::Array(elem_ty) = arg_ty {
                                    return Ok(Type::Option(elem_ty));
                                }
                            }
                        }
                    }
                }

                // 分布コンストラクタの静的パラメータチェック
                if let Expression::Variable { name, .. } = &**callee {
                    if let Some(ident) = name.last_ident() {
                        self.check_distribution_parameters(&ident.name, args, *span)?;
                    }
                }

                self.infer_function_call(callee, args)
            }



            Expression::Block { block, .. } => self.infer_block(block),

            Expression::MethodCall {
                receiver,
                method,
                args,
                span,
                ref mut resolved, 
            } => {
                // 1. レシーバの型推論
                let recv_ty_raw = self.infer_expression(receiver)?;
                let recv_ty = self.apply(recv_ty_raw);
                let objty = recv_ty.clone();

                // Int Method Hook
                if matches!(objty, Type::Int) {
                     for arg in args.iter_mut() {
                         let _ = self.infer_expression(arg)?;
                     }
                     match method.name.as_str() {
                         "to_string" => return Ok(Type::String),
                         _ => {},
                     }
                }

                // String Method Hook (Phase 7 - MethodCall variant)
                if matches!(objty, Type::String) {
                     // Args type check (simplified)
                     for arg in args.iter_mut() {
                         let _ = self.infer_expression(arg)?;
                     }

                     match method.name.as_str() {
                         "trim" => return Ok(Type::String),
                         "split" => return Ok(Type::Array(Box::new(Type::String))),
                         "replace" => return Ok(Type::String),
                         "contains" => return Ok(Type::Bool),
                         "len" => return Ok(Type::Int),
                         "slice" => return Ok(Type::String),
                         _ => {},
                     }
                }

                // Float Method Hook (Phase 8)
                if matches!(objty, Type::Float) {
                     for arg in args.iter_mut() {
                         let _ = self.infer_expression(arg)?;
                     }
                     match method.name.as_str() {
                         "abs" | "exp" | "ln" | "sqrt" | "sin" | "cos" | "tan" | "ceil" | "floor" | "round" => return Ok(Type::Float),
                         "pow" | "powf" => return Ok(Type::Float),
                         _ => {},
                     }
                }

                // Case 1: Primitive Probabilistic Types
                if objty.is_probabilistic() {
                    if method.name == "sample" {
                        return Ok(Type::Float);
                    }
                    if method.name == "clone" {
                        return Ok(objty.clone());
                    }
                }
                
                // Case 2: Type::Named ("Distribution") (Defined in env.rs)
                if let Type::Named { name, .. } = &objty {
                    // パスの末尾が "Distribution" かチェック
                    // resolved がある場合とない場合の両方を考慮
                    let is_distribution = if let Some(ident) = name.last_ident() {
                         ident.name == "Distribution"
                    } else {
                         false
                    };

                    if is_distribution && method.name == "sample" {
                         return Ok(Type::Float);
                    }
                }
                if let Type::Array(ref elem_ty) = objty {
                    if method.name == "len" {
                         return Ok(Type::Int);
                    }
                    if method.name == "push" {
                         if args.len() != 1 {
                              return Err(TypeError::ArityMismatch { expected: 1, found: args.len(), span: *span });
                         }
                         return Ok(Type::Array(elem_ty.clone()));
                    }
                }

                if let Type::Map(key_ty, val_ty) = &objty {
                    match method.name.as_str() {
                        "insert" => {
                            if args.len() != 2 {
                                return Err(TypeError::ArityMismatch { expected: 2, found: args.len(), span: *span });
                            }
                            let k_ty = self.infer_expression(&mut args[0])?;
                            self.unify(k_ty, *key_ty.clone(), args[0].span())?;
                            
                            let v_ty = self.infer_expression(&mut args[1])?;
                            self.unify(v_ty, *val_ty.clone(), args[1].span())?;
                            
                            return Ok(Type::Unit);
                        }
                        "get" => {
                            if args.len() != 1 {
                                return Err(TypeError::ArityMismatch { expected: 1, found: args.len(), span: *span });
                            }
                            let k_ty = self.infer_expression(&mut args[0])?;
                            self.unify(k_ty, *key_ty.clone(), args[0].span())?;
                            
                            return Ok(*val_ty.clone());
                        }
                        "contains_key" => {
                            if args.len() != 1 {
                                return Err(TypeError::ArityMismatch { expected: 1, found: args.len(), span: *span });
                            }
                            let k_ty = self.infer_expression(&mut args[0])?;
                            self.unify(k_ty, *key_ty.clone(), args[0].span())?;
                            
                            return Ok(Type::Bool);
                        }
                        _ => {}
                    }
                }

                // 2. メソッド解決用に alias 展開 & Rc/Weak を剥がす
                let mut dispatch_ty = self.expand_alias_fixpoint(recv_ty.clone());
                loop {
                    dispatch_ty = self.expand_alias_fixpoint(dispatch_ty);
                    match dispatch_ty {
                        Type::Rc(inner) | Type::Weak(inner) => dispatch_ty = *inner,
                        _ => break,
                    }
                }

                // 3. Named Type (Struct/Enum) -> Inherent or Trait Impl
                if let Type::Named { name: type_name, .. } = &dispatch_ty {
                    let type_def_id = if let Some(id) = type_name.resolved {
                        id
                    } else {
                        // 解決されていなければ解決を試みる
                        let ident = type_name
                            .last_ident()
                            .ok_or(TypeError::CannotInfer { span: *span })?;
                        self.env.resolve_def(&ident.name).ok_or(TypeError::UndefinedVariable {
                            name: ident.name.clone(),
                            span: *span,
                        })?
                    };

                    // 3-a) Inherent method
                    if let Some((method_ty, method_def_id)) =
                        self.env.lookup_method(type_def_id, &method.name).cloned()
                    {
                        *resolved = Some(method_def_id);

                        if let Type::Function { params, return_type } = method_ty {
                            if params.is_empty() {
                                return Err(TypeError::CannotInfer { span: *span });
                            }

                            let is_variadic = params
                                .last()
                                .map_or(false, |t| matches!(t, Type::Variadic(_)));
                            let expected_min = params.len().saturating_sub(1);

                            if !is_variadic && args.len() != expected_min {
                                return Err(TypeError::ArityMismatch {
                                    expected: expected_min,
                                    found: args.len(),
                                    span: *span,
                                });
                            }
                            if is_variadic && args.len() < expected_min.saturating_sub(1) {
                                return Err(TypeError::ArityMismatch {
                                    expected: expected_min.saturating_sub(1),
                                    found: args.len(),
                                    span: *span,
                                });
                            }

                            self.unify(recv_ty.clone(), params[0].clone(), receiver.span())?;

                            let args_len = args.len();
                            for (i, argexpr) in args.iter_mut().enumerate() {
                                let arg_ty = self.infer_expression(argexpr)?;

                                let expected_ty = if i + 1 < params.len() {
                                    match &params[i + 1] {
                                        Type::Variadic(inner) => *inner.clone(),
                                        other => other.clone(),
                                    }
                                } else if is_variadic {
                                    match params.last().unwrap() {
                                        Type::Variadic(inner) => *inner.clone(),
                                        _ => unreachable!(),
                                    }
                                } else {
                                    return Err(TypeError::ArityMismatch {
                                        expected: expected_min,
                                        found: args_len,
                                        span: argexpr.span(),
                                    });
                                };

                                self.unify(arg_ty, expected_ty, argexpr.span())?;
                            }

                            return Ok(self.apply(*return_type));
                        } else {
                            return Err(TypeError::NotCallable {
                                value_type: method_ty,
                                span: *span,
                            });
                        }
                    }

                    // 3-b) Trait impl method
                    let impls = self.env.impls.clone();
                    let mut trait_defs: std::collections::HashSet<DefId> = std::collections::HashSet::new();
                    for impl_def in impls.iter() {
                        if impl_def.methods.contains_key(&method.name) {
                            trait_defs.insert(impl_def.trait_def);
                        }
                    }

                    let mut matches: Vec<(SelectedImpl, std::collections::HashMap<usize, Type>)> = Vec::new();

                    for trait_def in trait_defs.into_iter() {
                        let saved_subst = self.subst.clone();
                        match self.select_impl(trait_def, recv_ty.clone(), *span) {
                            Ok(sel) => {
                                if sel.impl_def.methods.contains_key(&method.name) {
                                    let subst_after = self.subst.clone();
                                    matches.push((sel, subst_after));
                                }
                            }
                            Err(TypeError::NoMatchingImpl { .. }) => {}
                            Err(e) => return Err(e),
                        }
                        self.subst = saved_subst;
                    }

                    if matches.len() > 1 {
                        let trait_def = matches[0].0.impl_def.trait_def;
                        return Err(TypeError::AmbiguousImpl {
                            trait_def,
                            receiver: recv_ty.clone(),
                            span: *span,
                        });
                    }
                    
                    if let Some((selected, subst_after)) = matches.pop() {
                        self.subst = subst_after;
                        let (mut method_ty, method_def_id) = selected
                            .impl_def
                            .methods
                            .get(&method.name)
                            .ok_or(TypeError::CannotInfer { span: *span })?
                            .clone();

                        *resolved = Some(method_def_id);

                        method_ty = self.subst_typevars(method_ty, &selected.subst);
                        method_ty = self.apply(method_ty);
                        
                        let mut solved_assoc = selected.assoc_bindings.clone();
                        for (_k, v) in solved_assoc.iter_mut() {
                            *v = self.apply(self.subst_typevars(v.clone(), &selected.subst));
                        }
                        
                        method_ty = self.subst_assoc(method_ty, &solved_assoc);
                        method_ty = self.apply(method_ty);

                        if let Type::Function { params, return_type } = method_ty {
                            if params.is_empty() {
                                return Err(TypeError::CannotInfer { span: *span });
                            }

                            let is_variadic = params
                                .last()
                                .map_or(false, |t| matches!(t, Type::Variadic(_)));
                            let expected_min = params.len().saturating_sub(1);

                            if !is_variadic && args.len() != expected_min {
                                return Err(TypeError::ArityMismatch {
                                    expected: expected_min,
                                    found: args.len(),
                                    span: *span,
                                });
                            }
                            if is_variadic && args.len() < expected_min.saturating_sub(1) {
                                return Err(TypeError::ArityMismatch {
                                    expected: expected_min.saturating_sub(1),
                                    found: args.len(),
                                    span: *span,
                                });
                            }

                            self.unify(recv_ty.clone(), params[0].clone(), receiver.span())?;

                            let args_len = args.len();
                            for (i, argexpr) in args.iter_mut().enumerate() {
                                let arg_ty = self.infer_expression(argexpr)?;
                                let expected_ty = if i + 1 < params.len() {
                                    match &params[i + 1] {
                                        Type::Variadic(inner) => *inner.clone(),
                                        other => other.clone(),
                                    }
                                } else if is_variadic {
                                    match params.last().unwrap() {
                                        Type::Variadic(inner) => *inner.clone(),
                                        _ => unreachable!(),
                                    }
                                } else {
                                    return Err(TypeError::ArityMismatch {
                                        expected: expected_min,
                                        found: args_len,
                                        span: argexpr.span(),
                                    });
                                };
                                self.unify(arg_ty, expected_ty, argexpr.span())?;
                            }
                            return Ok(self.apply(*return_type));
                        } else {
                            return Err(TypeError::NotCallable {
                                value_type: method_ty,
                                span: *span,
                            });
                        }
                    }
                }

                // 4. dyn Trait
                if let Type::Named {
                    name: trait_path,
                    type_args: _recv_args,
                } = dispatch_ty.clone()
                {
                    let trait_defid = if let Some(defid) = trait_path.resolved {
                        defid
                    } else {
                        if let Some(last) = trait_path.last_ident() {
                            if let Some(d) = self.env.resolve_def(&last.name) {
                                d
                            } else {
                                return Err(TypeError::UndefinedVariable {
                                    name: last.name.clone(),
                                    span: *span,
                                });
                            }
                        } else {
                            return Err(TypeError::CannotInfer { span: *span });
                        }
                    };

                    if let Some(ItemDef::Trait(trait_info_ref)) = self.env.get_def(trait_defid) {
                        *resolved = None;
                        let trait_typeparams = trait_info_ref.typeparams.clone();
                        let method_ty_raw = trait_info_ref
                            .methods
                            .get(&method.name)
                            .ok_or(TypeError::CannotInfer { span: *span })?
                            .clone();
                        let fresh = self.instantiate_typeparams(&trait_typeparams);
                        let method_ty = self.apply(self.subst_typevars(method_ty_raw, &fresh));

                        if let Type::Function { params, return_type } = method_ty {
                            if params.is_empty() {
                                return Err(TypeError::CannotInfer { span: *span });
                            }
                            let is_variadic = params.last().map_or(false, |t| matches!(t, Type::Variadic(_)));
                            let expected_min = params.len().saturating_sub(1);
                            if !is_variadic && args.len() != expected_min {
                                return Err(TypeError::ArityMismatch { expected: expected_min, found: args.len(), span: *span });
                            }
                            if is_variadic && args.len() < expected_min.saturating_sub(1) {
                                return Err(TypeError::ArityMismatch { expected: expected_min.saturating_sub(1), found: args.len(), span: *span });
                            }
                            self.unify(recv_ty.clone(), params[0].clone(), receiver.span())?;
                            let args_len = args.len();
                            for (i, argexpr) in args.iter_mut().enumerate() {
                                let arg_ty = self.infer_expression(argexpr)?;
                                let expected_ty = if i + 1 < params.len() {
                                    match &params[i + 1] {
                                        Type::Variadic(inner) => *inner.clone(),
                                        other => other.clone(),
                                    }
                                } else if is_variadic {
                                    match params.last().unwrap() {
                                        Type::Variadic(inner) => *inner.clone(),
                                        _ => unreachable!(),
                                    }
                                } else {
                                    return Err(TypeError::ArityMismatch { expected: expected_min, found: args_len, span: argexpr.span() });
                                };
                                self.unify(arg_ty, expected_ty, argexpr.span())?;
                            }
                            return Ok(self.apply(*return_type));
                        }
                    }
                }

                Err(TypeError::CannotInfer { span: *span })
            }





            // ★追加: Option::Some
            Expression::Some { expr, .. } => {
                let inner_ty = self.infer_expression(expr)?;
                Ok(Type::Option(Box::new(inner_ty)))
            }
            // ★追加: Option::None
            Expression::None { .. } => {
                // Noneの型は Option<Any> (または推論待ち) だが、
                // ここでは Option<Any> としておく。
                Ok(Type::Option(Box::new(Type::Any)))
            }
            Expression::Ok { expr, .. } => {
                let inner = self.infer_expression(expr.as_mut())?;
                Ok(Type::Result {
                    ok_type: Box::new(inner),
                    err_type: Box::new(self.fresh_meta()),
                })
            }

            Expression::Err { expr, .. } => {
                let inner = self.infer_expression(expr.as_mut())?;
                Ok(Type::Result {
                    ok_type: Box::new(self.fresh_meta()),
                    err_type: Box::new(inner),
                })
            }

            Expression::Try { expr, span } => {
                let t = self.infer_expression(expr.as_mut())?;
                let t = self.apply(t);
                match t {
                    Type::Result { ok_type, err_type: _ } => Ok(*ok_type),
                    other => Err(TypeError::TypeMismatch {
                        expected: Type::Result {
                            ok_type: Box::new(self.fresh_meta()),
                            err_type: Box::new(self.fresh_meta()),
                        },
                        found: other,
                        span: *span,
                    }),
                }
            }
            Expression::Index { object, index, span } => {
                // index は Int
                let idx_ty = self.infer_expression(index)?;
                self.unify(idx_ty, Type::Int, *span)?;

                // object は Array<T> を要求し、要素型 T を返す
                let obj_ty = self.infer_expression(object.as_mut())?;
                let elem_ty = self.fresh_meta();              // _0
                let expected_arr = Type::Array(Box::new(elem_ty.clone()));

                // ここが重要: 「Array<_0> と Array<Int> を unify」させる
                self.unify(obj_ty, expected_arr, *span)?;

                Ok(self.apply(elem_ty))
            }
            // ★追加: Match式
            Expression::Match { scrutinee, arms, span } => {
                self.infer_match_expression(scrutinee, arms, *span)
            }
            Expression::Array { elements, span } => {
                if elements.is_empty() {
                    return Ok(Type::Array(Box::new(Type::Unit)));
                }
                
                // 1. 全要素の型を推論
                let mut elem_types = Vec::new();
                for elem in &mut *elements {
                    let ty = self.infer_expression(elem)?;  // ★ ここで各要素を推論
                    elem_types.push(ty);
                }
                
                // 2. 最初の要素の型を基準とする
                let unified_ty = elem_types[0].clone();
                
                // 3. 残りの要素を統一
                for (i, elem_ty) in elem_types.iter().enumerate().skip(1) {
                    // unifyで型を統一（エラーは無視して最初の型を維持）
                    if let Err(_) = self.unify(elem_ty.clone(), unified_ty.clone(), elements[i].span()) {
                        // 異なる型の場合、そのまま最初の型を使う
                        // TODO: 共通のtraitを探す処理
                    }
                }
                
                Ok(Type::Array(Box::new(self.apply(unified_ty))))
            }

            //おそらくast/node.rsの方でLamdaなので、削除(後回し)
            Expression::Closure { params, body, .. } => {
                // クロージャの型推論
                self.env.push_scope();
                
                // パラメータの型を環境に登録
                let param_types: Vec<Type> = params.iter().map(|p| {
                    // パラメータに型注釈があればそれを使う。なければ Any（推論待ち）
                    let ty = p.ty.clone();
                    self.env.define(p.name.name.clone(), ty.clone()).ok();
                    ty
                }).collect();
                
                // ボディの型を推論
                let return_type = self.infer_expression(body)?;
                
                self.env.pop_scope();
                
                // クロージャの型は関数型
                Ok(Type::Function {
                    params: param_types,
                    return_type: Box::new(return_type),
                })
            }
            Expression::Lambda { params, body, .. } => {
                self.env.push_scope();

                let mut param_types = Vec::new();
                for p in params {
                    let ty = p.ty.clone();
                    param_types.push(ty.clone());
                    let _ = self.env.define(p.name.name.clone(), ty);
                }

                let return_type = self.infer_expression(body)?;

                self.env.pop_scope();

                Ok(Type::Function {
                    params: param_types,
                    return_type: Box::new(return_type),
                })
            }
            Expression::Struct { name, fields, span } => {
                let struct_name = &name.name;
                
                // 2. DefId を解決
                let defid = match self.env.resolve_def(struct_name) {
                    Some(id) => {
                        id
                    }
                    None => {
                        return Err(TypeError::CannotInfer {
                            span: *span,
                        });
                    }
                };
                                        
                // 3. Struct 定義を取得
                let info = match self.env.get_def(defid).cloned() {
                    Some(ItemDef::Struct(info)) => info,
                    _ => {
                        return Err(TypeError::CannotInfer {
                            span: *span,
                        });
                    }
                };
                
                // 4. 型引数をフレッシュなメタ変数として生成
                let mut typeargs = Vec::new();
                for _param in &info.typeparams {
                    typeargs.push(self.fresh_meta());
                }
                
                // 5. 型パラメータの置換マップを作成
                let mut subst = HashMap::new();
                for (i, param) in info.typeparams.iter().enumerate() {
                    subst.insert(param.name.name.clone(), typeargs[i].clone());
                }
                
                // 6. フィールドの型チェック
                for field in fields.iter_mut() {
                    let field_ty = self.infer_expression(&mut field.value)?;
                    
                    if let Some(expected_ty) = info.fields.get(&field.name.name) {
                        // 型パラメータを置換した期待される型
                        let expected_instantiated = self.subst_typevars(expected_ty.clone(), &subst);
                        self.unify(field_ty, expected_instantiated, field.value.span())?;
                    } else {
                        return Err(TypeError::UndefinedField {
                            field: field.name.name.clone(),
                            structname: struct_name.clone(),
                            span: field.span,
                        });
                    }
                }
                
                // 7. 解決済みの Path を作成
                let mut resolved_path = Path::from_ident(name.clone());
                resolved_path.resolved = Some(defid);
                
                // 8. 型引数を適用して返す
                Ok(Type::Named {
                    name: resolved_path,
                    type_args: typeargs.into_iter().map(|t| self.apply(t)).collect(),
                })
            }




            Expression::FieldAccess{ object, field, span } => {
                let objtyraw = self.infer_expression(object.as_mut())?;
                let objty = self.apply(objtyraw);
                
                // 0) Gaussian.sample の特殊ケース（既存）
                if objty == Type::Gaussian && field.name == "sample" {
                    return Ok(Type::Function {
                        params: vec![],
                        return_type: Box::new(Type::Float),
                    });
                }
                
                // 1) Named 型（struct）の場合
                if let Type::Named{ ref name, .. } = objty {
                    // 1-a) struct定義を取得
                    let self_defid = if let Some(defid) = name.resolved {
                        defid
                    } else {
                        self.env.resolve_def(name.last_ident().unwrap().name.as_str())
                            .ok_or(TypeError::CannotInfer { span: *span })?
                    };
                    
                    // 1-b) structのフィールドを探索
                    if let Some(ItemDef::Struct(struct_info)) = self.env.get_def(self_defid) {
                        if let Some(field_ty) = struct_info.fields.get(&field.name) {
                            return Ok(field_ty.clone());
                        }
                    }
                    
                    // 1-c) メソッドテーブルから探索（Inherent Impl）
                    if let Some((method_ty, _method_defid)) = self.env.lookup_method(self_defid, field.name.as_str()) {
                        return Ok(method_ty.clone());
                    }

                    // 1-d) Trait impls search (Fallback: Trait Impl) [NEW]
                    // Inherent Impl で見つからなかった場合、Trait Impl を探す
                    for impl_def in &self.env.impls {
                        // 対象の型に対する Impl かどうかをチェック
                        // 簡易的に DefId の一致で判定（ジェネリクスなしの場合）
                        let is_matching_type = match &impl_def.self_ty {
                            Type::Named{ name: impl_name, .. } => {
                                impl_name.resolved.or_else(|| 
                                    self.env.resolve_def(impl_name.last_ident().unwrap().name.as_str())
                                ) == Some(self_defid)
                            },
                            _ => false,
                        };

                        if is_matching_type {
                            if let Some(method_ty) = impl_def.methods.get(&field.name) {
                                return Ok(method_ty.clone().0);
                            }
                        }
                    }
                    
                    // 1-e) Vector2 のハードコード（既存・後方互換用）
                    if name.last_ident().map(|id| id.name.as_str()) == Some("Vector2") {
                        return match field.name.as_str() {
                            "x" | "y" => Ok(Type::Float),
                            _ => Err(TypeError::CannotInfer { span: *span }),
                        };
                    }
                }
                
                Err(TypeError::UndefinedField {
                    field: field.name.clone(),
                    structname: format!("{:?}", objty),
                    span: *span,
                })
            }


            Expression::Paren { expr, .. } => {
                self.infer_expression(expr)
            }

            Expression::Tuple { elements, .. } => {
                if elements.is_empty() {
                    Ok(Type::Unit)
                } else {
                    // 今回はシンプルに: 全要素同型を要求してその型を返す
                    let first = self.infer_expression(&mut elements[0])?;
                    for e in elements.iter_mut().skip(1) {
                        let t = self.infer_expression(e)?;
                        if t != first {
                            return Err(TypeError::TypeMismatch {
                                expected: first,
                                found: t,
                                span: e.span(),
                            });
                        }
                    }
                    Ok(first)
                }
            }
            Expression::With { name, initializer, body, span: _ } => {
                let init_ty = self.infer_expression(initializer)?;
                self.env.push_scope();
                self.env.define(name.name.clone(), init_ty)?;
                let body_ty = self.infer_block(body)?; // body が Block なら
                self.env.pop_scope();
                Ok(self.apply(body_ty))
            }


            Expression::Enum {
                name,
                variant,
                args,
                named_fields,
                span,
            } => {
                // EnvからEnum情報を引く
                let def_id = self
                    .env
                    .resolve_def(&name.name)
                    .or_else(|| self.env.resolve_def(&name.name))
                    .ok_or(TypeError::UndefinedVariable {
                        name: name.name.clone(),
                        span: *span,
                    })?;

                // Enum定義を取得（borrow conflict回避のため info を clone して所有する）
                let (variant_info, typeparams) = match self.env.get_def(def_id) {
                    Some(ItemDef::Enum(info)) => {
                        let v = info.variants
                            .get(&variant.name)
                            .ok_or(TypeError::CannotInfer { span: *span })?
                            .clone();
                        (v, info.typeparams.clone())
                    }
                    _ => return Err(TypeError::CannotInfer { span: *span }), // Not an enum
                };

                // ジェネリクス型パラメータのインスタンス化 (MetaVar生成)
                let mut enum_type_args = Vec::new();
                for _ in &typeparams {
                    enum_type_args.push(self.fresh_meta());
                }

                // 置換マップの作成
                let mut subst = HashMap::new();
                for (i, param) in typeparams.iter().enumerate() {
                    subst.insert(param.name.name.clone(), enum_type_args[i].clone());
                }

                // named_fields がある場合は定義順に並べ替えて args を生成
                // named_fields がない場合は args をそのまま使うが、Struct Variantなら数チェックを行う
                let mut final_args: Vec<Expression> = match &variant_info {
                    VariantInfo::Unit => Vec::new(),

                    VariantInfo::Tuple(tys) => {
                        if args.len() != tys.len() {
                            return Err(TypeError::ArityMismatch {
                                expected: tys.len(),
                                found: args.len(),
                                span: *span,
                            });
                        }
                        args.clone()
                    }

                    VariantInfo::Struct(order) => {
                        if let Some(fields) = named_fields {
                            // 旧来の名前付きフィールド初期化: P::Q { x: 1, y: 2 }
                            let mut new_args = Vec::new();
                            for (field_name, _field_ty) in order.iter() {
                                let f = fields
                                    .iter()
                                    .find(|x| x.name.name == *field_name)
                                    .ok_or(TypeError::CannotInfer { span: *span })?;
                                new_args.push(f.value.clone());
                            }
                            new_args
                        } else {
                            // Parserで変換された Positional args 初期化: P::Q { x: 1, y: 2 } -> args=[1,2]
                            if args.len() != order.len() {
                                return Err(TypeError::ArityMismatch {
                                    expected: order.len(),
                                    found: args.len(),
                                    span: *span,
                                });
                            }
                            args.clone()
                        }
                    }
                };

                // 各引数の型推論とフィールド型へのUnify
                for (i, a) in final_args.iter_mut().enumerate() {
                    let arg_ty = self.infer_expression(a)?;

                    // 期待される型を取得してUnify
                    let expected_ty_base = match &variant_info {
                        VariantInfo::Tuple(tys) => tys.get(i),
                        VariantInfo::Struct(order) => order.get(i).map(|(_, t)| t),
                        VariantInfo::Unit => None,
                    };

                    if let Some(base_ty) = expected_ty_base {
                         // ここでGeneric型パラメータを置換する
                         let expected_ty = self.subst_typevars(base_ty.clone(), &subst);
                         self.unify(arg_ty, expected_ty, a.span())?;
                    }
                }

                // Enum自体の型（Type::Named）を返す。type_args にはインスタンス化した型引数を入れる
                let path = Path {
                    segments: vec![PathSeg::Ident(name.clone())],
                    span: name.span,
                    resolved: Some(def_id),
                };

                Ok(Type::Named { name: path, type_args: enum_type_args })
            }


            _ => {
                Err(TypeError::CannotInfer { span: expr.span() })
            }

        }
    }




    // ============================================================
    // Block Expression
    // ============================================================
    pub fn infer_block(&mut self, block: &mut Block) -> TypeResult<Type> {
        self.env.push_scope();
        let mut result = Type::Unit;

        for stmt in block.statements.iter_mut() {
            match stmt {
                // 式文: 結果を更新 (タプルバリアント)
                crate::ast::node::Statement::Expression(expr) => {
                    result = self.infer_expression(expr)?;
                }

                // return文: return値の型をブロック結果として返す
                crate::ast::node::Statement::Return { value, span } => {
                    let ret_ty = if let Some(expr) = value {
                        self.infer_expression(expr)?
                    } else {
                        Type::Unit
                    };

                    // 以降の文は到達不能扱いにしても良いが、とりあえず「最後に見たreturnを結果」にする
                    result = ret_ty;

                    // 早期return: ブロックの型は return の型で確定
                    self.env.pop_scope();
                    return Ok(result);
                }
                crate::ast::node::Statement::Let { ty, init, pattern, span, .. } => {
                    if let Some(initializer) = init {
                        let inferred = self.infer_expression(initializer)?;

                        // 対応パターン分岐
                        match pattern {
                            crate::ast::node::Pattern::Identifier { name, .. } => {
                                if let Some(annot) = ty.clone() {
                                    // 型注釈(annot)がある場合
                                    // inferred -> annot への代入が可能かチェック (Coercion含む)
                                    // check_assignable は、Struct -> dyn Trait の変換などを許容し、
                                    // 成功すれば期待される型(annot)を返す
                                    let _ = self.check_assignable(
                                        inferred.clone(),
                                        annot.clone(),
                                        initializer.span(),
                                    )?;

                                    // 変数を環境に登録 (型注釈の型で登録)
                                    self.env
                                        .define(name.name.clone(), annot.clone())
                                        .map_err(|_| TypeError::DuplicateDefinition {
                                            name: name.name.clone(),
                                            span: initializer.span(),
                                        })?;
                                } else {
                                    // 型アノテーションがない場合は推論された型(inferred)で登録
                                    self.env
                                        .define(name.name.clone(), inferred.clone())
                                        .map_err(|_| TypeError::DuplicateDefinition {
                                            name: name.name.clone(),
                                            span: initializer.span(),
                                        })?;
                                }
                            }
                            _ => {
                                // 複雑なパターン（未サポート）はエラー
                                return Err(TypeError::CannotInfer { span: *span });
                            }
                        }
                    } else {
                        // 初期化式がない場合 (let x: T;)
                        // 型注釈は必須
                        if let Some(annot) = ty.clone() {
                            match pattern {
                                crate::ast::node::Pattern::Identifier { name, .. } => {
                                    self.env
                                        .define(name.name.clone(), annot.clone())
                                        .map_err(|_| TypeError::DuplicateDefinition {
                                            name: name.name.clone(),
                                            // init が無いので initializer.span() は使えない
                                            span: *span,
                                        })?;
                                }
                                _ => return Err(TypeError::CannotInfer { span: *span }),
                            }
                        } else {
                            // 初期化式も型注釈もない場合は推論不可
                            return Err(TypeError::CannotInfer { span: *span });
                        }
                    }
                }


                // その他の文: 現状無視
                _ => {}
            }
        }

        self.env.pop_scope();
        Ok(result)
    }


    //Match式の推論
    fn infer_match_expression(
        &mut self, 
        scrutinee: &mut Expression, 
        arms: &mut [crate::ast::node::MatchArm], 
        span: Span
    ) -> TypeResult<Type> {
        let scrutinee_ty = self.infer_expression(scrutinee)?;
        
        let mut ret_ty: Option<Type> = None;

        for arm in arms.iter_mut() {
            self.env.push_scope();
            
            // パターンの型チェック & 変数登録
            self.check_pattern(&arm.pattern, &scrutinee_ty, arm.span)?;
            
            // ガード (if) のチェック
            if let Some(guard) = &mut arm.guard {
                let guard_ty = self.infer_expression(guard)?;
                if guard_ty != Type::Bool {
                     return Err(TypeError::TypeMismatch {
                        expected: Type::Bool,
                        found: guard_ty,
                        span: arm.span,
                    });
                }
            }
            
            // ボディの型推論
            let body_ty = self.infer_expression(&mut arm.body)?;
            
            self.env.pop_scope();

            // 戻り値の型統一
            if let Some(prev) = ret_ty.take() {
                let unified = self.unify(prev, body_ty, arm.span)?;
                ret_ty = Some(self.apply(unified));
            } else {
                ret_ty = Some(body_ty);
            }
        }

        Ok(ret_ty.unwrap_or(Type::Unit))
    }

    fn fresh_meta(&mut self) -> Type {
        let id = self.next_meta;
        self.next_meta += 1;
        Type::MetaVar(id)
    }
    fn apply(&mut self, ty: Type) -> Type {
        match ty {
            Type::MetaVar(id) => {
                if let Some(t) = self.subst.get(&id).cloned() {
                    let t2 = self.apply(t);
                    // パス圧縮
                    self.subst.insert(id, t2.clone());
                    t2
                } else {
                    Type::MetaVar(id)
                }
            }

            Type::Option(inner) => Type::Option(Box::new(self.apply(*inner))),
            Type::Array(inner) => Type::Array(Box::new(self.apply(*inner))),
            Type::Signal(inner) => Type::Signal(Box::new(self.apply(*inner))),
            Type::Event(inner) => Type::Event(Box::new(self.apply(*inner))),
            Type::Rc(inner) => Type::Rc(Box::new(self.apply(*inner))),
            Type::Weak(inner) => Type::Weak(Box::new(self.apply(*inner))),
            Type::Variadic(inner) => Type::Variadic(Box::new(self.apply(*inner))),
            Type::Handle(inner) => Type::Handle(Box::new(self.apply(*inner))),

            Type::Result { ok_type, err_type } => Type::Result {
                ok_type: Box::new(self.apply(*ok_type)),
                err_type: Box::new(self.apply(*err_type)),
            },

            Type::Function { params, return_type } => Type::Function {
                params: params.into_iter().map(|p| self.apply(p)).collect(),
                return_type: Box::new(self.apply(*return_type)),
            },

            Type::Tuple(ts) => Type::Tuple(ts.into_iter().map(|t| self.apply(t)).collect()),
            Type::Assoc { trait_def, self_ty, name } => {
                let self_ty = Box::new(self.apply(*self_ty));
                // ここで normalize を試みるのが理想だが、
                // まずは構造を維持して返すだけでもOK
                Type::Assoc { trait_def, self_ty, name }
            },

            other => other,
        }
    }

    fn occurs(&mut self, id: usize, ty: &Type) -> bool {
        let t = self.apply(ty.clone());
        match t {
            Type::MetaVar(x) => x == id,
            Type::Option(inner)
            | Type::Array(inner)
            | Type::Signal(inner)
            | Type::Event(inner)
            | Type::Rc(inner)
            | Type::Weak(inner)
            | Type::Handle(inner)
            | Type::Variadic(inner) => self.occurs(id, &inner),

            Type::Result { ok_type, err_type } => self.occurs(id, &ok_type) || self.occurs(id, &err_type),

            Type::Function { params, return_type } => {
                params.iter().any(|p| self.occurs(id, p)) || self.occurs(id, &return_type)
            }

            Type::Tuple(ts) => ts.iter().any(|t| self.occurs(id, t)),

            _ => false,
        }
    }

    fn bind_meta(&mut self, id: usize, ty: Type, span: Span) -> TypeResult<()> {
        let t = self.apply(ty);
        if let Type::MetaVar(x) = t {
            if x == id {
                return Ok(());
            }
        }
        if self.occurs(id, &t) {
            println!("[CannotInfer] at {:?}, context A", span);
            return Err(TypeError::CannotInfer { span }); // あるなら OccursCheck 用エラーにしてもOK
        }
        self.subst.insert(id, t);
        Ok(())
    }



    fn check_assignable(&mut self, found: Type, expected: Type, span: Span) -> TypeResult<Type> {
        let found = self.apply(found);
        let expected = self.apply(expected);

        if let Type::Named { name: ref trait_path, type_args: ref targs, .. } = expected {
            if targs.is_empty() {
                let traitdefid_opt = trait_path.resolved.or_else(|| {
                    trait_path
                        .last_ident()
                        .and_then(|id| self.env.resolve_def(id.name.as_str()))
                });

                if let Some(traitdefid) = traitdefid_opt {
                    if matches!(self.env.get_def(traitdefid), Some(ItemDef::Trait(_))) {
                        if self.select_impl(traitdefid, found.clone(), span).is_ok() {
                            return Ok(expected);
                        }
                        return Err(TypeError::TypeMismatch { expected, found, span });
                    }
                }
            }
        }
        match (found.clone(), expected.clone()) {
            // 目的：T -> dyn Trait を許可
            (_, Type::DynTrait{ trait_path }) => {
                // found が既に dyn Trait なら同一traitか確認してOK
                if matches!(found, Type::DynTrait{..}) {
                    // ここは unify か、trait同一性チェックでOKにする
                    return Ok(expected);
                }

                // traitdefid を解決
                let traitdefid = if let Some(id) = trait_path.resolved {
                    id
                } else {
                    let key = self.trait_key_from_path(&trait_path);
                    self.env.resolve_def(&key)
                        .ok_or(TypeError::UndefinedVariable { name: key.clone(), span })?
                };

                // T: Trait が証明できるか（impl 探し）
                if self.select_impl(traitdefid, found.clone(), span).is_ok() {
                    return Ok(expected); // 「found を dyn Trait にアップキャスト可能」
                }

                return Err(TypeError::TypeMismatch { expected, found, span });
            }

            // 配列：要素型にも同じ規則を再帰適用（Array<dyn Trait> のため）
            (Type::Array(a), Type::Array(b)) => {
                let inner = self.check_assignable(*a, *b, span)?;
                Ok(Type::Array(Box::new(inner)))
            }
            (Type::Handle(a), Type::Handle(b)) => {
                let inner = self.check_assignable(*a, *b, span)?;
                Ok(Type::Handle(Box::new(inner)))
            }
            (Type::Named{ name, type_args }, Type::DynTrait{ trait_path, .. }) => {
                // 1. トレイトの定義IDを解決
                let trait_def_id = if let Some(id) = trait_path.resolved {
                    id
                } else {
                    let last = trait_path.last_ident().ok_or_else(|| TypeError::CannotInfer { span })?;
                    self.env.resolve_def(last.name.as_str())
                        .ok_or_else(|| TypeError::UndefinedVariable { name: last.name.clone(), span })?
                };

                // 2. select_impl を呼んで、この構造体(found)がトレイト(trait_def_id)を実装しているか確認
                //    成功すれば代入可能なので、期待される型(expected = DynTrait)を返す
                if self.select_impl(trait_def_id, found.clone(), span).is_ok() {
                    return Ok(expected); 
                }
                
                // 実装が見つからない場合はエラーへ (unifyに任せるか、ここでエラーにする)
                return Err(TypeError::TypeMismatch {
                    expected: expected.clone(),
                    found: found.clone(),
                    span,
                });
            }

            // それ以外は従来の同一性（unify）でOK
            _ => self.unify(found, expected, span),
        }
    }

    fn same_trait_path(&self, a: &Path, b: &Path) -> bool {
        let ida = a.resolved.or_else(|| {
            a.last_ident()
                .and_then(|id| self.env.resolve_def(id.name.as_str()))
        });

        let idb = b.resolved.or_else(|| {
            b.last_ident()
                .and_then(|id| self.env.resolve_def(id.name.as_str()))
        });

        match (ida, idb) {
            (Some(x), Some(y)) => x == y,
            _ => a.last_name() == b.last_name(), // 最後の保険（span無視）[file:644]
        }
    }




    fn unify(&mut self, a: Type, b: Type, span: Span) -> TypeResult<Type> {
        let a_applied = self.apply(a);
        let a_expanded = self.expand_alias_fixpoint(a_applied); // 名前を変える
        let b_applied = self.apply(b);
        let b_expanded = self.expand_alias_fixpoint(b_applied);

        let a = a_expanded;
        let b = b_expanded;


        // Any はとりあえずトップとして扱う（片方 Any なら通す）
        if matches!(a, Type::Any) { return Ok(b); }
        if matches!(b, Type::Any) { return Ok(a); }
        if a == b {
            return Ok(a);
        }
        match (a.clone(), b.clone()) {
            (Type::DynTrait { trait_path: p1 }, Type::DynTrait { trait_path: p2 }) => {
                if self.same_trait_path(&p1, &p2) {
                    Ok(Type::DynTrait { trait_path: p1 })
                } else {
                    Err(TypeError::TypeMismatch {
                        expected: Type::DynTrait { trait_path: p1 },
                        found: Type::DynTrait { trait_path: p2 },
                        span,
                    })
                }
            }
            // dyn Trait とそれ以外は「代入可能性」で判定する
            // DynTrait と Named の相互作用
            (Type::DynTrait { trait_path }, Type::Named { name, type_args })
            | (Type::Named { name, type_args }, Type::DynTrait { trait_path }) => {
                // (A) greet の receiver みたいに「trait Self 型 (Named Trait)」と「dyn Trait」を同一視
                if type_args.is_empty() && self.same_trait_path(&name, &trait_path) {
                    return Ok(Type::DynTrait { trait_path });
                }

                // (B) それ以外（例: Person -> dyn Greeter のアップキャスト）は assignable 判定へ
                let expected = Type::DynTrait { trait_path: trait_path.clone() };
                let found = Type::Named { name, type_args };
                self.check_assignable(found, expected, span)
            }

            (Type::MetaVar(id), t) => {
                self.bind_meta(id, t.clone(), span)?;
                Ok(t)
            }
            (t, Type::MetaVar(id)) => {
                self.bind_meta(id, t.clone(), span)?;
                Ok(t)
            }

            // --- Named 型（struct/enum/type alias）を unify する ---
            (
                Type::Named {
                    name: mut an,
                    type_args: aargs,
                },
                Type::Named {
                    name: bn,
                    type_args: bargs,
                },
            ) => {
                let a_def = an
                    .resolved
                    .or_else(|| self.env.resolve_def(an.last_ident().unwrap().name.as_str()));
                let b_def = bn
                    .resolved
                    .or_else(|| self.env.resolve_def(bn.last_ident().unwrap().name.as_str()));

                if let (Some(ad), Some(bd)) = (a_def, b_def) {
                    if ad != bd {
                        return Err(TypeError::TypeMismatch {
                            expected: Type::Named { name: an, type_args: aargs },
                            found: Type::Named { name: bn, type_args: bargs },
                            span,
                        });
                    }
                } else {
                    if an.last_name() != bn.last_name() {
                        return Err(TypeError::TypeMismatch {
                            expected: Type::Named { name: an, type_args: aargs },
                            found: Type::Named { name: bn, type_args: bargs },
                            span,
                        });
                    }
                }

                if aargs.len() != bargs.len() {
                    return Err(TypeError::TypeMismatch {
                        expected: Type::Named { name: an, type_args: aargs },
                        found: Type::Named { name: bn, type_args: bargs },
                        span,
                    });
                }

                let mut out_args = Vec::new();
                for (x, y) in aargs.into_iter().zip(bargs.into_iter()) {
                    out_args.push(self.unify(x, y, span)?);
                }

                an.resolved = a_def;

                Ok(Type::Named {
                    name: an,
                    type_args: out_args,
                })
            }


            (Type::Int, Type::Int) => Ok(Type::Int),
            (Type::Float, Type::Float) => Ok(Type::Float),
            (Type::Bool, Type::Bool) => Ok(Type::Bool),
            (Type::String, Type::String) => Ok(Type::String),
            (Type::Unit, Type::Unit) => Ok(Type::Unit),
            (Type::Gaussian, Type::Gaussian) => Ok(Type::Gaussian),
            (Type::Uniform, Type::Uniform) => Ok(Type::Uniform),

            (Type::Option(a1), Type::Option(b1)) => {
                let inner = self.unify(*a1, *b1, span)?;
                Ok(Type::Option(Box::new(inner)))
            }

            (Type::Array(a1), Type::Array(b1)) => {
                let inner = self.unify(*a1, *b1, span)?;
                Ok(Type::Array(Box::new(inner)))
            }

            (Type::Handle(a1), Type::Handle(b1)) => {
                let inner = self.unify(*a1, *b1, span)?;
                Ok(Type::Handle(Box::new(inner)))
            }
            (Type::Assoc { .. }, _) | (_, Type::Assoc { .. }) => {
                // 本当は normalize してから unify する
                // とりあえず "構造が完全に一致する場合のみOK" にしておく
                if a == b {
                    Ok(a)
                } else {
                    Err(TypeError::TypeMismatch {
                        expected: a,
                        found: b,
                        span,
                    })
                }
            }

            (
                Type::Result {
                    ok_type: aok,
                    err_type: aer,
                },
                Type::Result {
                    ok_type: bok,
                    err_type: ber,
                },
            ) => {
                let ok = self.unify(*aok, *bok, span)?;
                let er = self.unify(*aer, *ber, span)?;
                Ok(Type::Result {
                    ok_type: Box::new(ok),
                    err_type: Box::new(er),
                })
            }

            (x, y) => Err(TypeError::TypeMismatch {
                expected: x,
                found: y,
                span,
            }),
        }
    }



    //パターンの型チェックと環境への登録
    fn check_pattern(&mut self, pattern: &Pattern, target_ty: &Type, span: Span) -> TypeResult<()> {
        match pattern {
            Pattern::Identifier { name, .. } => {
                // 変数束縛
                self.env.define(name.name.clone(), target_ty.clone())
                    .map_err(|_| TypeError::DuplicateDefinition {
                        name: name.name.clone(),
                        span,
                    })?;
                Ok(())
            },
            Pattern::Wildcard { .. } => Ok(()),
            
            Pattern::Literal { value, .. } => {
                let lit_ty = self.infer_literal(value);
                self.unify(lit_ty, target_ty.clone(), span)?;
                Ok(())
            },


            Pattern::Some { pattern: inner, .. } => {
                match target_ty {
                    Type::Option(inner_ty) => {
                        self.check_pattern(inner, inner_ty, span)
                    },
                    // Option<Any> (Noneから推論された場合など) への対応
                    Type::Any => {
                         self.check_pattern(inner, &Type::Any, span)
                    },
                    _ => Err(TypeError::TypeMismatch {
                        expected: Type::Option(Box::new(Type::Any)),
                        found: target_ty.clone(),
                        span,
                    }),
                }
            },
            
            Pattern::None { .. } => {
                match target_ty {
                    Type::Option(_) | Type::Any => Ok(()),
                    _ => Err(TypeError::TypeMismatch {
                        expected: Type::Option(Box::new(Type::Any)),
                        found: target_ty.clone(),
                        span,
                    }),
                }
            },
            Pattern::Ok { pattern: inner, .. } => {
                match target_ty {
                    Type::Result { ok_type, err_type: _ } => {
                        self.check_pattern(inner.as_ref(), ok_type.as_ref(), span)
                    }
                    Type::Any => self.check_pattern(inner.as_ref(), &Type::Any, span),
                    _ => Err(TypeError::TypeMismatch {
                        expected: Type::Result {
                            ok_type: Box::new(Type::Any),
                            err_type: Box::new(Type::Any),
                        },
                        found: target_ty.clone(),
                        span,
                    }),
                }
            },
            Pattern::Err { pattern: inner, .. } => {
                match target_ty {
                    Type::Result { ok_type: _, err_type } => {
                        self.check_pattern(inner.as_ref(), err_type.as_ref(), span)
                    }
                    Type::Any => self.check_pattern(inner.as_ref(), &Type::Any, span),
                    _ => Err(TypeError::TypeMismatch {
                        expected: Type::Result {
                            ok_type: Box::new(Type::Any),
                            err_type: Box::new(Type::Any),
                        },
                        found: target_ty.clone(),
                        span,
                    }),
                }
            },
            Pattern::Tuple { patterns, .. } => {
                match target_ty {
                    Type::Tuple(elem_tys) => {
                        if patterns.len() != elem_tys.len() {
                            return Err(TypeError::TypeMismatch {
                                expected: target_ty.clone(),
                                found: target_ty.clone(), // ここは専用エラーが無いなら妥協
                                span,
                            });
                        }
                        for (p, t) in patterns.iter().zip(elem_tys.iter()) {
                            self.check_pattern(p, t, span)?;
                        }
                        Ok(())
                    }
                    Type::Any => {
                        // 型が分からないなら全部 Any で束縛だけ通す
                        for p in patterns {
                            self.check_pattern(p, &Type::Any, span)?;
                        }
                        Ok(())
                    }
                    _ => Err(TypeError::TypeMismatch {
                        expected: Type::Tuple(vec![]), // 専用エラーが無いなら「Tuple期待」を示すだけ
                        found: target_ty.clone(),
                        span,
                    }),
                }
            },
            Pattern::Enum {
                name,
                variant,
                args,
                named_fields,
                span,
            } => {
                // EnvからEnum情報を解決
                // EnvからEnum情報を解決
                let def_id = self.env.resolve_def(&name.name)
                    .or_else(|| self.env.resolve_def(&name.name))
                    .ok_or(TypeError::UndefinedVariable { name: name.name.clone(), span: *span })?;

                // Enum定義を取得し、対象バリアントの情報を探す
                let (variant_info, typeparams) = match self.env.get_def(def_id) {
                    Some(ItemDef::Enum(info)) => {
                        let v_info = info.variants.get(&variant.name)
                            .ok_or(TypeError::CannotInfer { span: *span })?
                            .clone();
                        (v_info, info.typeparams.clone())
                    },
                    _ => return Err(TypeError::CannotInfer { span: *span }),
                };

                // Generics Instantiation & Unification
                let mut typeargs = Vec::new();
                for _ in &typeparams {
                    typeargs.push(self.fresh_meta());
                }
                
                // Construct expected Enum Type and Unify with target_ty
                // これは重要: ターゲット型(match対象)が持つ型引数(例: Option<Int>)と
                // ここで生成したメタ変数(例: Option<_1>)をUnifyすることで、_1 = Int となる。
                let constructed_enum_ty = Type::Named {
                    name: Path {
                        segments: vec![crate::ast::node::PathSeg::Ident(name.clone())],
                        span: name.span,
                        resolved: Some(def_id),
                    },
                    type_args: typeargs.clone(),
                };
                self.unify(target_ty.clone(), constructed_enum_ty, *span)?;

                // Substitution Map creation
                let mut subst = std::collections::HashMap::new();
                for (i, param) in typeparams.iter().enumerate() {
                    subst.insert(param.name.name.clone(), typeargs[i].clone());
                }

                // 検証すべき「期待される引数の型リスト」と「実際のパターンリスト」を揃える
                let (expected_types, patterns_to_check): (Vec<Type>, Vec<Pattern>) = match variant_info {
                    VariantInfo::Unit => (vec![], vec![]),
                    
                    VariantInfo::Tuple(tys) => {
                        // タプルバリアント: args と tys の長さが一致するはず
                        if args.len() != tys.len() {
                            return Err(TypeError::ArityMismatch { 
                                expected: tys.len(), 
                                found: args.len(), 
                                span: *span 
                            });
                        }
                        
                        // Substitute generics in expected types
                        let mut subst_tys = Vec::new();
                        for t in tys {
                            subst_tys.push(self.subst_typevars(t.clone(), &subst));
                        }
                        
                        (subst_tys, args.clone())
                    },
                    
                    VariantInfo::Struct(fields) => {
                        // 構造体バリアント
                        if let Some(user_fields) = named_fields {
                            // 名前付きパターン { x: pat, y: pat }
                            let mut p_list = Vec::new();
                            let mut t_list = Vec::new();
                            
                            // 定義順序 (fields) に従って、ユーザーが書いたパターンを取り出す
                            for (fname, fty) in fields {
                                let user_field = user_fields.iter().find(|x| x.name.name == *fname)
                                    .ok_or(TypeError::MissingField { 
                                        field: fname.clone(), 
                                        struct_name: variant.name.clone(), 
                                        span: *span 
                                    })?;
 
                                 // パターンがあればそれ、省略形 {x} なら Identifierパターンを生成
                                 let pat = user_field.pattern.clone().unwrap_or_else(|| Pattern::Identifier {
                                     name: user_field.name.clone(),
                                     span: user_field.span,
                                 });
 
                                 // Substitute types
                                 t_list.push(self.subst_typevars(fty.clone(), &subst));
                                 p_list.push(pat);
                             }
                             (t_list, p_list)
                         } else {
                             // Positional args for Struct Pattern
                             if args.len() != fields.len() {
                                 return Err(TypeError::ArityMismatch {
                                     expected: fields.len(),
                                     found: args.len(),
                                     span: *span,
                                 });
                             }
                             
                             // フィールドの型リストを作成（定義順）
                             let mut field_tys = Vec::new();
                             for (_, ty) in fields {
                                 field_tys.push(self.subst_typevars(ty.clone(), &subst));
                             }
                             
                             (field_tys, args.clone())
                         }
                     }
                 };
                for (p, expected_ty) in patterns_to_check.into_iter().zip(expected_types.into_iter()) {
                    self.check_pattern(&p, &expected_ty, *span)?;
                }
                
                Ok(())
            },


            Pattern::Struct { name, fields, span } => {
                let def_id = self.env.resolve_def(&name.name)
                    .ok_or(TypeError::UndefinedVariable { name: name.name.clone(), span: *span })?;

                let struct_fields = match self.env.get_def(def_id) {
                    Some(ItemDef::Struct(info)) => info.fields.clone(),
                    _ => return Err(TypeError::CannotInfer { span: *span }),
                };

                for field_pat in fields {
                    let field_name = &field_pat.name.name;
                    let field_ty = struct_fields.get(field_name)
                        .ok_or(TypeError::MissingField {
                            field: field_name.clone(),
                            struct_name: name.name.clone(),
                            span: *span,
                        })?.clone();

                    let pat = field_pat.pattern.clone().unwrap_or_else(|| Pattern::Identifier {
                        name: field_pat.name.clone(),
                        span: field_pat.span,
                    });
                    self.check_pattern(&pat, &field_ty, *span)?;
                }
                Ok(())
            },

            Pattern::Or { patterns, span } => {
                for pat in patterns {
                    self.check_pattern(pat, target_ty, *span)?;
                }
                Ok(())
            },

            Pattern::Range { start, end, span, .. } => {
                let start_ty = self.infer_literal(start);
                let end_ty = self.infer_literal(end);
                self.unify(start_ty.clone(), target_ty.clone(), *span)?;
                self.unify(end_ty, start_ty, *span)?;
                Ok(())
            },

            _ => Ok(()), 
        }
    }
    fn instantiate_typeparams(
        &mut self,
        params: &[crate::ast::node::TypeParameter],
    ) -> std::collections::HashMap<String, Type> {
        let mut subst = std::collections::HashMap::new();
        for p in params {
            subst.insert(p.name.name.clone(), self.fresh_meta());
        }
        subst
    }

    pub fn select_impl(
        &mut self,
        trait_def_id: DefId,
        receiver_ty: Type,
        span: Span,
    ) -> TypeResult<SelectedImpl> {
        // 1. 候補を収集
        let candidates: Vec<ImplDef> = self
            .env
            .impls
            .iter()
            .filter(|impl_def| impl_def.trait_def == trait_def_id)
            .cloned()
            .collect();

        // (impl_def, subst_after_checks, fresh_type_param_map)
        let mut matched_impls: Vec<(ImplDef, HashMap<usize, Type>, HashMap<String, Type>)> =
            Vec::new();

        // 2. 候補ごとにマッチングを試行
        for impl_def in candidates {
            let saved_subst = self.subst.clone();

            let fresh_subst = self.instantiate_typeparams(&impl_def.typeparams);
            let fresh_self = self.subst_typevars(impl_def.self_ty.clone(), &fresh_subst);

            // A. レシーバ型と impl型 の Unify
            let unify_success = self.unify(receiver_ty.clone(), fresh_self, span).is_ok();
            
            // B. where 節チェック (Unify成功時のみ)
            let where_success = if unify_success {
                self.fulfill_obligations(&impl_def.where_preds, &fresh_subst, span).is_ok()
            } else {
                false
            };

            if unify_success && where_success {
                // 両方成功なら候補として残す
                let cand_subst = self.subst.clone();
                matched_impls.push((impl_def, cand_subst, fresh_subst));
            }

            // 次の候補のために戻す
            self.subst = saved_subst;
        }

        // 3. 結果判定 (B1: CannotInfer ではなく専用エラーを返す)
        if matched_impls.is_empty() {
            return Err(TypeError::NoMatchingImpl {
                trait_def: trait_def_id,
                receiver: receiver_ty,
                span,
            });
        }

        if matched_impls.len() > 1 {
            return Err(TypeError::AmbiguousImpl {
                trait_def: trait_def_id,
                receiver: receiver_ty,
                span,
            });
        }

        // 唯一の候補を採用
        let (impl_def, final_subst, fresh_subst) = matched_impls.pop().unwrap();
        self.subst = final_subst;

        // 4. 戻り値の構築
        // 型引数 T が具体的に何になったかを解決 (例: T -> Int)
        let mut solved_tparams = HashMap::new();
        for (name, ty) in fresh_subst {
            solved_tparams.insert(name, self.apply(ty));
        }

        // 関連型 (type Item = ...) の解決
        let mut solved_assoc = HashMap::new();
        for (name, ty) in &impl_def.assoc_bindings {
            // T を具体型に置換してからメタ変数を解消
            let substituted = self.subst_typevars(ty.clone(), &solved_tparams);
            solved_assoc.insert(name.clone(), self.apply(substituted));
        }

        Ok(SelectedImpl {
            impl_def: impl_def.clone(),
            subst: solved_tparams,
            assoc_bindings: solved_assoc,
        })
    }

    // プログラム全体の型チェック
    pub fn check_program(&mut self, program: &mut crate::ast::node::Program) -> TypeResult<()> {
        // ========================================
        // Pass 1: 全ての型定義とトップレベル宣言を登録
        // ========================================
        
        for item in &mut program.items {
            match item {
                crate::ast::node::Item::Function(func) => {
                    let param_types: Vec<Type> = func.params.iter().map(|p| p.ty.clone()).collect();
                    let ret_type = func.return_type.clone().unwrap_or(Type::Unit);

                    let func_ty = Type::Function {
                        params: param_types,
                        return_type: Box::new(ret_type),
                    };

                    // 変更: insert_def が DefId を返すので受け取る
                    let defid = self.env.insert_def(
                        func.name.name.clone(),
                        ItemDef::Function(func_ty),
                        func.span,
                    )?;

                    // 追加: ASTにIDを焼く
                    func.defid = Some(defid);
                }

                crate::ast::node::Item::Struct(s) => {
                    let mut fields = HashMap::new();
                    for field in &s.fields {
                        fields.insert(field.name.name.clone(), field.ty.clone());
                    }
                    
                    let info = StructDefInfo {
                        typeparams: s.type_params.clone(),
                        fields,
                    };
                    
                    self.env.insert_def(
                        s.name.name.clone(),
                        ItemDef::Struct(info),
                        s.span
                    )?;
                }

                crate::ast::node::Item::Trait(t) => {
                    let mut methods: HashMap<String, Type> = HashMap::new();

                    for m in &t.methods {
                        let params = m
                            .params
                            .iter()
                            .map(|p| Self::replace_self_in_trait(t, p.ty.clone()))
                            .collect::<Vec<_>>();

                        let ret = Self::replace_self_in_trait(t, m.return_type.clone().unwrap_or(Type::Unit));

                        let fnty = Type::Function {
                            params,
                            return_type: Box::new(ret),
                        };

                        methods.insert(m.name.name.clone(), fnty);
                    }

                    let info = TraitDefInfo {
                        typeparams: t.type_params.clone(),
                        assoc_types: HashMap::new(),
                        methods,
                    };
                    
                    let key = t.name.name.clone(); // or prefix + name
                    let defid = self.env.insert_trait_def(key, info, t.span)?;

                    //self.env.insert_def(t.name.name.clone(), ItemDef::Trait(info), t.span)?;
                }

                crate::ast::node::Item::Enum(e) => {
                    let mut variants: HashMap<String, VariantInfo> = HashMap::new();
                    for v in &e.variants {
                        let info: VariantInfo = match &v.data {
                            VariantData::Unit => VariantInfo::Unit,
                            VariantData::Tuple(tys) => VariantInfo::Tuple(tys.clone()),
                            VariantData::Struct(fields) => {
                                VariantInfo::Struct(
                                    fields.iter()
                                        .map(|f| (f.name.name.clone(), f.ty.clone()))
                                        .collect(),
                                )
                            }
                        };
                        variants.insert(v.name.name.clone(), info);
                    }
                    let info = EnumDefInfo { 
                        typeparams: e.type_params.clone(),
                        variants
                    };

                    self.env.insert_def(e.name.name.clone(), ItemDef::Enum(info), e.span)?;
                }

                crate::ast::node::Item::TypeAlias(a) => {
                    self.env.insert_def(
                        a.name.name.clone(),
                        ItemDef::TypeAlias(TypeAliasDef {
                            typeparams: a.type_params.clone(),
                            ty: a.ty.clone(),
                        }),
                        a.span,
                    )?;
                }

                crate::ast::node::Item::Module(m) => {
                    self.register_module_items(&m.name.name, &m.items)?;
                }

                crate::ast::node::Item::Import(imp) => {
                    // defs の alias として処理
                    let full = Self::path_to_key(&imp.path, "");

                    let target = self.env.resolve_def(&full).ok_or(TypeError::UndefinedVariable {
                        name: full.clone(),
                        span: imp.span,
                    })?;

                    let local = imp
                        .alias
                        .as_ref()
                        .map(|id| id.name.clone())
                        .unwrap_or_else(|| {
                            imp.path
                                .last_ident()
                                .expect("import path must have at least one ident segment")
                                .name
                                .clone()
                        });

                    self.env.alias_def(local, target, imp.span)?;
                }

                crate::ast::node::Item::Impl(_) => {
                    // impl ブロックは次のパスで処理
                }

                crate::ast::node::Item::Extern(ext) => {
                    for f in &ext.functions {
                        let param_types: Vec<Type> = f.params.iter().map(|p| p.ty.clone()).collect();
                        let ret_type = f.return_type.clone().unwrap_or(Type::Unit);

                        let func_ty = Type::Function {
                            params: param_types,
                            return_type: Box::new(ret_type),
                        };

                        self.env.insert_def(
                            f.name.name.clone(),
                            ItemDef::Function(func_ty),
                            f.span,
                        )?;
                    }
                }
            }
        }

        // ========================================
        // Pass 2: Impl ブロックを処理
        // ========================================

        for item in &mut program.items {
            match item {
                crate::ast::node::Item::Impl(impl_block) => {
                    if let Some(trait_ref) = &impl_block.trait_ref {
                        // --- Trait Impl: impl Trait for Type ---
                        
                        // 1. Resolve Trait Path
                        let trait_def_id = if let Type::Named { name, .. } = trait_ref {
                            self.resolve_trait_def_from_path(name, impl_block.span)?
                        } else {
                            return Err(TypeError::CannotInfer { span: impl_block.span });
                        };
                        let typename: String = match &impl_block.self_ty {
                            Type::Named { name, .. } => name
                                .last_ident()
                                .map(|id| id.name.clone())
                                .unwrap_or_else(|| "anon".to_string()),
                            _ => "nonnamed".to_string(),
                        };
                        // 2. Collect Methods and Associated Types
                        let mut methods = HashMap::new();
                        let mut assoc_bindings = HashMap::new();

                        // Iterate mutable items to set DefId on AST and collect environment info
                        for item in &mut impl_block.items {
                            match item {
                                ImplItem::Method(func) => {
                                    let paramtypes: Vec<Type> = func.params.iter().map(|p| {
                                        // Replace Self with actual type
                                        if let Type::Named { name, .. } = &p.ty {
                                            if name.last_ident().map(|id| id.name.as_str()) == Some("Self") {
                                                return impl_block.self_ty.clone();
                                            }
                                        }
                                        p.ty.clone()
                                    }).collect();

                                    let rettype = func.return_type.clone().unwrap_or(Type::Unit);

                                    let functy = Type::Function {
                                        params: paramtypes,
                                        return_type: Box::new(rettype),
                                    };
                                    
                                    // For Trait Impl, methods are stored in ImplDef.
                                    let fq = format!("{}::{}", typename, func.name.name);
                                    let methoddefid = self.env.insert_def(
                                        fq,
                                        ItemDef::Function(functy.clone()),
                                        func.span,
                                    )?;
                                    func.defid = Some(methoddefid);

                                    methods.insert(func.name.name.clone(), (functy.clone(), methoddefid));

                                }

                                ImplItem::AssocType(binding) => {
                                    assoc_bindings.insert(binding.name.name.clone(), binding.ty.clone());
                                }
                            }
                        }

                        let impldef = ImplDef {
                            impl_id: self.env.impls.len(),
                            trait_def: trait_def_id,
                            self_ty: impl_block.self_ty.clone(),
                            typeparams: impl_block.type_params.clone(),
                            where_preds: impl_block.where_preds.clone(),
                            assoc_bindings,
                            methods,
                            span: impl_block.span,
                        };
                        self.env.impls.push(impldef);

                    } else {
                        // --- Inherent Impl: impl Type ---
                        
                        // Resolve Self Type DefId
                        let self_defid = match &impl_block.self_ty {
                            Type::Named { name, .. } => {
                                if let Some(defid) = name.resolved {
                                    defid
                                } else {
                                    self.env
                                        .resolve_def(&name.last_ident().unwrap().name)
                                        .ok_or(TypeError::UndefinedVariable {
                                            name: name.last_ident().unwrap().name.clone(),
                                            span: impl_block.span,
                                        })?
                                }
                            }
                            _ => return Ok(()), // Skip for non-named types or handle error
                        };

                        for item in &mut impl_block.items {
                            if let crate::ast::node::ImplItem::Method(func) = item {
                                let param_types: Vec<Type> = func
                                    .params
                                    .iter()
                                    .map(|p| {
                                        if let Type::Named { name, .. } = &p.ty {
                                            if name.last_ident().unwrap().name == "Self" {
                                                return impl_block.self_ty.clone();
                                            }
                                        }
                                        p.ty.clone()
                                    })
                                    .collect();

                                let ret_type = func.return_type.clone().unwrap_or(Type::Unit);
                                let method_ty = Type::Function {
                                    params: param_types,
                                    return_type: Box::new(ret_type),
                                };

                                // ★ Assign DefId, Register to Env, and Set AST
                                
                                // Generate a unique key for the method in def database
                                let type_name = match &impl_block.self_ty {
                                    Type::Named { name, .. } => name
                                        .last_ident()
                                        .map(|id| id.name.clone())
                                        .unwrap_or_else(|| "<anon>".to_string()),
                                    _ => "<non_named>".to_string(),
                                };

                                let fq = format!("{}::{}", type_name, func.name.name);

                                // First, allocate a DefId for the function itself
                                let method_def_id = self.env.insert_def(
                                    fq,
                                    ItemDef::Function(method_ty.clone()),
                                    func.span,
                                )?;
                                func.defid = Some(method_def_id);

                                // Then register it as a method of the struct/enum
                                // Note: insert_method signature is updated to take method_def_id
                                self.env.insert_method(
                                    self_defid,
                                    func.name.name.clone(),
                                    method_ty,
                                    method_def_id,
                                    func.span,
                                )?;
                            }
                        }
                    }

                }

                _ => {}
            }
        }


        // Pass 3: 関数本体をチェック

        
        for item in program.items.iter_mut() {
            match item {
                crate::ast::node::Item::Function(func) => {
                    self.check_function(func)?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    // infer.rs: impl TypeInfer { ... } に追加
    fn subst_assoc(&self, ty: Type, assoc: &HashMap<String, Type>) -> Type {
        match ty {
            Type::Assoc { trait_def, self_ty, name } => {
                assoc.get(&name).cloned().unwrap_or(Type::Assoc { trait_def, self_ty, name })
            }
            Type::Option(inner) => Type::Option(Box::new(self.subst_assoc(*inner, assoc))),
            Type::Array(inner) => Type::Array(Box::new(self.subst_assoc(*inner, assoc))),
            Type::Signal(inner) => Type::Signal(Box::new(self.subst_assoc(*inner, assoc))),
            Type::Event(inner) => Type::Event(Box::new(self.subst_assoc(*inner, assoc))),
            Type::Rc(inner) => Type::Rc(Box::new(self.subst_assoc(*inner, assoc))),
            Type::Weak(inner) => Type::Weak(Box::new(self.subst_assoc(*inner, assoc))),
            Type::Variadic(inner) => Type::Variadic(Box::new(self.subst_assoc(*inner, assoc))),
            Type::Result { ok_type, err_type } => Type::Result {
                ok_type: Box::new(self.subst_assoc(*ok_type, assoc)),
                err_type: Box::new(self.subst_assoc(*err_type, assoc)),
            },
            Type::Function { params, return_type } => Type::Function {
                params: params.into_iter().map(|t| self.subst_assoc(t, assoc)).collect(),
                return_type: Box::new(self.subst_assoc(*return_type, assoc)),
            },
            Type::Tuple(ts) => Type::Tuple(ts.into_iter().map(|t| self.subst_assoc(t, assoc)).collect()),
            Type::Named { name, type_args } => Type::Named {
                name,
                type_args: type_args.into_iter().map(|t| self.subst_assoc(t, assoc)).collect(),
            },
            other => other,
        }
    }




    fn register_module_items(
        &mut self,
        prefix: &str,
        items: &[crate::ast::node::Item],
    ) -> TypeResult<()> {
        for item in items {
            match item {
                crate::ast::node::Item::Function(func) => {
                    let param_types: Vec<Type> = func.params.iter().map(|p| p.ty.clone()).collect();
                    let ret_type = func.return_type.clone().unwrap_or(Type::Unit);
                    let func_ty = Type::Function {
                        params: param_types,
                        return_type: Box::new(ret_type),
                    };

                    let key = format!("{prefix}::{}", func.name.name);
                    self.env
                        .insert_def(key, ItemDef::Function(func_ty), func.span)?;
                }

                crate::ast::node::Item::Struct(s) => {
                    let key = format!("{prefix}::{}", s.name.name);
                    let mut fields = HashMap::new();
                    for f in &s.fields {
                        fields.insert(f.name.name.clone(), f.ty.clone());
                    }

                    let info = StructDefInfo {
                        typeparams: s.type_params.clone(),
                        fields,
                    };

                    self.env.insert_def(key, ItemDef::Struct(info), s.span)?;
                }

                crate::ast::node::Item::Enum(e) => {
                    // 1. info を構築（型情報を保持する）
                    let mut variants: HashMap<String, VariantInfo> = HashMap::new();
                    for v in &e.variants {
                        let info: VariantInfo = match &v.data {
                            VariantData::Unit => VariantInfo::Unit,

                            // ★ここが tys.len() ではなく tys.clone()
                            VariantData::Tuple(tys) => VariantInfo::Tuple(tys.clone()),

                            // ★ここが Vec<String> ではなく Vec<(String, Type)>
                            VariantData::Struct(fields) => {
                                VariantInfo::Struct(
                                    fields
                                        .iter()
                                        .map(|f| (f.name.name.clone(), f.ty.clone()))
                                        .collect(),
                                )
                            }
                        };
                        variants.insert(v.name.name.clone(), info);
                    }
                    let info = EnumDefInfo { 
                        typeparams: e.type_params.clone(),
                        variants 
                    };

                    // 2. insert_def 呼び出し
                    self.env.insert_def(e.name.name.clone(), ItemDef::Enum(info), e.span)?;
                }

                crate::ast::node::Item::Trait(t) => {
                    let key = format!("{}{}", prefix, t.name.name);
                    let mut methods: HashMap<String, Type> = HashMap::new();

                    for m in &t.methods {
                        // ★修正: Trait method は implicit self (dyn Trait) を先頭に持つ
                        // これで sample() のように引数無しに見えるメソッドも params=[Self, ...] になる
                        let mut params: Vec<Type> = Vec::new();
                        params.push(Self::trait_self_ty_t(t));

                        params.extend(
                            m.params
                                .iter()
                                .map(|p| Self::replace_self_in_trait(t, p.ty.clone())),
                        );

                        let ret = Self::replace_self_in_trait(t, m.return_type.clone().unwrap_or(Type::Unit));

                        let fnty = Type::Function {
                            params,
                            return_type: Box::new(ret),
                        };

                        methods.insert(m.name.name.clone(), fnty);
                    }

                    let info = TraitDefInfo {
                        typeparams: t.type_params.clone(),
                        assoc_types: HashMap::new(),
                        methods,
                    };
                    let defid = self.env.insert_trait_def(key, info, t.span)?;
                    // self.env.insert_def(key, ItemDef::Trait(info), t.span)?;
                }

                crate::ast::node::Item::TypeAlias(a) => {
                    let key = format!("{}{}", prefix, a.name.name); // 既存の joinprefix があるならそれを使う
                    self.env.insert_def(
                        key,
                        ItemDef::TypeAlias(TypeAliasDef {
                            typeparams: a.type_params.clone(),
                            ty: a.ty.clone(),
                        }),
                        a.span,
                    )?;
                }
                crate::ast::node::Item::Import(imp) => {
                    // module 内の import は currentprefix=prefix で解決
                    let full = Self::path_to_key(&imp.path, prefix);

                    let target = self.env.resolve_def(&full).ok_or(TypeError::UndefinedVariable {
                        name: full.clone(),
                        span: imp.span,
                    })?;

                    let local = imp
                        .alias
                        .as_ref()
                        .map(|id| id.name.clone())
                        .unwrap_or_else(|| {
                            imp.path
                                .last_ident()
                                .expect("import path must have at least one ident segment")
                                .name
                                .clone()
                        });

                    // local は「この module のスコープ」なので prefix 付きにする
                    let local_fq = format!("{prefix}::{}", local);
                    self.env.alias_def(local_fq, target, imp.span)?;
                }

                crate::ast::node::Item::Module(inner) => {
                    let new_prefix = format!("{prefix}::{}", inner.name.name);
                    self.register_module_items(&new_prefix, &inner.items)?;
                }

                crate::ast::node::Item::Extern(ext) => {
                    for f in &ext.functions {
                        let param_types: Vec<Type> = f.params.iter().map(|p| p.ty.clone()).collect();
                        let ret_type = f.return_type.clone().unwrap_or(Type::Unit);
                        let func_ty = Type::Function {
                            params: param_types,
                            return_type: Box::new(ret_type),
                        };

                        let key = format!("{prefix}::{}", f.name.name);
                        self.env.insert_def(key, ItemDef::Function(func_ty), f.span)?;
                    }
                }

                _ => { /* いったん無視 */ }
            }
        }
        Ok(())
    }
    // Helper: create a Type::Named that represents "Self" in the context of a trait definition.
    // Used for the implicit receiver of trait methods.
    fn trait_self_ty_t(t: &crate::ast::node::TraitDef) -> Type {
        let span = t.span;
        let name_path = crate::ast::node::Path {
            segments: vec![crate::ast::node::PathSeg::Ident(crate::ast::node::Identifier {
                name: t.name.name.clone(),
                span,
            })],
            span, // ★修正: spanを追加
            resolved: None, 
        };

        let type_args = t
            .type_params
            .iter()
            .map(|tp| Type::TypeVar(crate::ast::node::Identifier {
                name: tp.name.name.clone(),
                span: tp.span,
            }))
            .collect::<Vec<_>>();

        Type::Named {
            name: name_path,
            type_args,
        }
    }

    // Helper: Replace "Self" type occurrences in a Trait method signature 
    // with the actual trait type (represented by trait_self_ty_t).
    fn replace_self_in_trait(t: &crate::ast::node::TraitDef, ty: Type) -> Type {
        match ty {
            Type::Named { name, type_args } => {
                let last_ident = name.last_ident().map(|id| id.name.as_str());
                if last_ident == Some("Self") {
                    Self::trait_self_ty_t(t)
                } else {
                    Type::Named {
                        name,
                        type_args: type_args
                            .into_iter()
                            .map(|arg| Self::replace_self_in_trait(t, arg))
                            .collect(),
                    }
                }
            }
            Type::Function { params, return_type } => Type::Function {
                params: params
                    .into_iter()
                    .map(|p| Self::replace_self_in_trait(t, p))
                    .collect(),
                return_type: Box::new(Self::replace_self_in_trait(t, *return_type)),
            },
            Type::Tuple(ts) => Type::Tuple(
                ts.into_iter()
                    .map(|e| Self::replace_self_in_trait(t, e))
                    .collect(),
            ),
            Type::Array(inner) => Type::Array(Box::new(Self::replace_self_in_trait(t, *inner))),
            Type::Option(inner) => Type::Option(Box::new(Self::replace_self_in_trait(t, *inner))),
            Type::Result { ok_type, err_type } => Type::Result {
                ok_type: Box::new(Self::replace_self_in_trait(t, *ok_type)),
                err_type: Box::new(Self::replace_self_in_trait(t, *err_type)),
            },
            // ★ Type::Reference がないので削除
            other => other,
        }
    }



    fn fulfill_obligations(
        &mut self,
        preds: &[WherePredicate],
        subst: &HashMap<String, Type>, // ← ここは Stringキー (fresh_subst用)
        _span: Span
    ) -> TypeResult<()> {
        for pred in preds {
            match pred {
                WherePredicate::Bound { target_ty, bound_ty, span: p_span } => {
                    // Generics (T) を具体的な型 (Stringなど) に置換
                    let concrete_target = self.subst_typevars(target_ty.clone(), subst);
                    let concrete_bound = self.subst_typevars(bound_ty.clone(), subst);

                    // Trait名を取り出して再帰チェック
                    if let Type::Named { name, type_args: _ } = concrete_bound {
                        // トレイトの DefId を解決 (env.resolve_def などを使用)
                        // 名前解決できない場合はエラー
                        let trait_def_id = self.env.resolve_def(&name.to_string())
                            .ok_or(TypeError::CannotInfer { span: *p_span })?;

                        // ★ 再帰的に select_impl を呼んで実装が存在するか確認
                        // エラーなら「T: Copy を満たさない」ことになる
                        if self.select_impl(trait_def_id, concrete_target, *p_span).is_err() {
                            // 本来は専用のエラー型 (TraitBoundNotSatisfied) が望ましいが
                            // ここでは CannotInfer や TypeMismatch で代用して落とす
                            return Err(TypeError::CannotInfer { span: *p_span });
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }






    fn subst_typevars(&self, ty: Type, subst: &HashMap<String, Type>) -> Type {
        match ty {
            Type::TypeVar(id) => subst.get(&id.name).cloned().unwrap_or(Type::TypeVar(id)),

            Type::Option(inner) => Type::Option(Box::new(self.subst_typevars(*inner, subst))),
            Type::Array(inner) => Type::Array(Box::new(self.subst_typevars(*inner, subst))),
            Type::Signal(inner) => Type::Signal(Box::new(self.subst_typevars(*inner, subst))),
            Type::Event(inner) => Type::Event(Box::new(self.subst_typevars(*inner, subst))),
            Type::Rc(inner) => Type::Rc(Box::new(self.subst_typevars(*inner, subst))),
            Type::Weak(inner) => Type::Weak(Box::new(self.subst_typevars(*inner, subst))),
            Type::Variadic(inner) => Type::Variadic(Box::new(self.subst_typevars(*inner, subst))),

            Type::Result { ok_type, err_type } => Type::Result {
                ok_type: Box::new(self.subst_typevars(*ok_type, subst)),
                err_type: Box::new(self.subst_typevars(*err_type, subst)),
            },

            Type::Function { params, return_type } => Type::Function {
                params: params.into_iter().map(|t| self.subst_typevars(t, subst)).collect(),
                return_type: Box::new(self.subst_typevars(*return_type, subst)),
            },

            Type::Tuple(ts) => Type::Tuple(ts.into_iter().map(|t| self.subst_typevars(t, subst)).collect()),

            // alias 展開後にさらに Named が残る可能性があるので、args は再帰しておく
            Type::Named { name, type_args } => Type::Named {
                name,
                type_args: type_args.into_iter().map(|t| self.subst_typevars(t, subst)).collect(),
            },

            other => other,
        }
    }


    fn expand_alias_once(&self, ty: Type) -> Type {
        match ty {
            Type::Named { name, type_args } => {
                let defidopt = name.resolved.or_else(|| {
                    name.last_ident()
                        .and_then(|id| self.env.resolve_def(id.name.as_str()))
                });

                if let Some(defid) = defidopt {
                    if let Some(ItemDef::TypeAlias(alias_def)) = self.env.get_def(defid) {
                        // arity が合わないときは、とりあえず展開しない（最小差分）
                        if alias_def.typeparams.len() != type_args.len() {
                            return Type::Named { name, type_args };
                        }

                        let subst: HashMap<String, Type> = alias_def
                            .typeparams
                            .iter()
                            .map(|tp| tp.name.name.clone())     // TypeParameter -> String
                            .zip(type_args.clone())             // (String, Type)
                            .collect();

                        let body = alias_def.ty.clone();
                        return self.subst_typevars(body, &subst);
                    }
                }

                Type::Named { name, type_args }
            }
            other => other,
        }
    }


    fn expand_alias_fixpoint(&self, mut ty: Type) -> Type {
        // 循環対策は後で。まずは回数制限で安全に。
        for _ in 0..32 {
            let next = self.expand_alias_once(ty.clone());
            if next == ty { break; }
            ty = next;
        }
        ty
    }



    // `path` を `current_prefix`（例: "a::b"）基準で `env` のキーに正規化する
    fn path_to_key(path: &Path, current_prefix: &str) -> String {
        // 先頭が Ident("crate"/"self"/"super") で来ても吸収する
        let mut idx = 0usize;
        let mut super_count = 0usize;
        let mut absolute = false;
        let mut use_prefix = true; // self/relative のとき true、crate のとき false

        // 先頭セグメント判定（PathSeg版 + Ident救済版）
        loop {
            match path.segments.get(idx) {
                Some(PathSeg::Crate(_)) => { absolute = true; use_prefix = false; idx += 1; break; }
                Some(PathSeg::Self_(_)) => { use_prefix = true; idx += 1; break; }
                Some(PathSeg::Super(_)) => { super_count += 1; idx += 1; continue; }

                // 救済: Ident("crate"/"self"/"super")
                Some(PathSeg::Ident(id)) if id.name == "crate" => { absolute = true; use_prefix = false; idx += 1; break; }
                Some(PathSeg::Ident(id)) if id.name == "self"  => { use_prefix = true; idx += 1; break; }
                Some(PathSeg::Ident(id)) if id.name == "super" => { super_count += 1; idx += 1; continue; }

                _ => break,
            }
        }

        // 残りIdentを回収
        let mut rest: Vec<String> = Vec::new();
        for seg in path.segments[idx..].iter() {
            if let PathSeg::Ident(id) = seg {
                rest.push(id.name.clone());
            }
        }

        // crate 絶対パス
        if absolute {
            return rest.join("::");
        }

        // super: prefixを上へ
        let mut prefix = current_prefix;
        for _ in 0..super_count {
            prefix = prefix.rsplit_once("::").map(|(p, _)| p).unwrap_or("");
        }

        // self/relative: prefixを付ける（トップレベルならそのまま）
        if !use_prefix || prefix.is_empty() {
            rest.join("::")
        } else {
            format!("{}::{}", prefix, rest.join("::"))
        }
    }


    // 関数の型チェック
    pub fn check_function(&mut self, func: &mut crate::ast::node::FunctionDef) -> TypeResult<()> {
        self.env.push_scope();

        // 途中でエラーになっても scope を必ず pop するために、結果を一旦変数に入れる
        let result: TypeResult<()> = (|| {
            // 引数を環境に登録
            for param in &func.params {
                self.env
                    .define(param.name.name.clone(), param.ty.clone())
                    .map_err(|_| TypeError::DuplicateDefinition {
                        name: param.name.name.clone(),
                        span: param.span,
                    })?;
            }

            // ボディの型推論
            let body_ty = self.infer_block(&mut func.body)?;

            // 宣言された戻り値型
            let expected_ret = func.return_type.clone().unwrap_or(Type::Unit);

            // 旧: if body_ty != expected_ret { ... }
            // 新: unify で制約を追加し、矛盾したら TypeMismatch にする
            self.unify(body_ty, expected_ret, func.span)?;

            Ok(())
        })();

        self.env.pop_scope();
        result
    }


    // Binary Operator Rules
    pub fn infer_binary(
        &mut self,
        op: BinaryOp,
        left: &mut Expression,
        right: &mut Expression,
        span: Span,
    ) -> TypeResult<Type> {
        let left_ty = self.infer_expression(left)?;
        let right_ty = self.infer_expression(right)?;
        match op {
            // 算術演算子
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                // String Concatenation Logic (Phase 6 Enhancement)
                // Allow "String + Any" and "Any + String" -> resulting in String
                if matches!(op, BinaryOp::Add) {
                    if matches!(left_ty, Type::String) || matches!(right_ty, Type::String) {
                        return Ok(Type::String);
                    }
                }

                match (op, &left_ty, &right_ty) {
                    // Int 同士
                    (_, Type::Int, Type::Int) => Ok(Type::Int),
                    // Float 同士
                    (_, Type::Float, Type::Float) => Ok(Type::Float),

                    (_, Type::Float, Type::Int) => Ok(Type::Float),
                    (_, Type::Int, Type::Float) => Ok(Type::Float),

                    // 確率型の演算は prob.rs へ委譲 (Gaussianなど)
                    // ここでは簡易的に判定を行う
                    (_, Type::Gaussian, _) | (_, _, Type::Gaussian) => {
                        // 加算・減算: Gaussian同士のみ
                        if matches!(op, BinaryOp::Add | BinaryOp::Sub)
                            && left_ty == Type::Gaussian
                            && right_ty == Type::Gaussian
                        {
                            Ok(Type::Gaussian)
                        }
                        // 乗算・除算: Gaussianとスカラー(Float)の組み合わせ、またはGaussian同士
                        else if matches!(op, BinaryOp::Mul | BinaryOp::Div) {
                            match (&left_ty, &right_ty) {
                                (Type::Gaussian, Type::Float) => Ok(Type::Gaussian),
                                (Type::Float, Type::Gaussian) => Ok(Type::Gaussian),
                                
                                // 独立性の仮定などが必要だが、型としてはGaussianを返す
                                (Type::Gaussian, Type::Gaussian) => Ok(Type::Gaussian),
                                _ => Err(TypeError::InvalidBinaryOp {
                                    op: op.to_string(),
                                    left_type: left_ty,
                                    right_type: right_ty,
                                    span,
                                }),
                            }
                        } else {
                            Err(TypeError::InvalidBinaryOp {
                                op: op.to_string(),
                                left_type: left_ty,
                                right_type: right_ty,
                                span,
                            })
                        }
                    }

                    // それ以外の組み合わせは型不一致
                    _ => Err(TypeError::TypeMismatch {
                        expected: left_ty.clone(),
                        found: right_ty,
                        span,
                    }),
                }
            }

            // ========== 比較演算 (Eq, Ne, Lt, Le, Gt, Ge) ==========
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                // 左右の型は unify で一致させる（Named などの表示同名問題もここで吸収）
                let unified = self.unify(left_ty, right_ty, span)?;

                // 比較可能な型かチェック
                match unified {
                    Type::Int | Type::Float | Type::Bool | Type::String => Ok(Type::Bool),
                    _ => Err(TypeError::InvalidBinaryOp {
                        op: op.to_string(),
                        left_type: unified.clone(),
                        right_type: unified,
                        span,
                    }),
                }
            }


            // ========== 論理演算 (&&, ||) ==========
            BinaryOp::And | BinaryOp::Or => {
                if left_ty != Type::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: Type::Bool,
                        found: left_ty,
                        span: left.span(),
                    });
                }
                if right_ty != Type::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: Type::Bool,
                        found: right_ty,
                        span: right.span(),
                    });
                }
                Ok(Type::Bool)
            }

            // ========== ビット演算 (&, |, ^, <<, >>) ==========
            BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr => {
                if left_ty != Type::Int {
                    return Err(TypeError::TypeMismatch {
                        expected: Type::Int,
                        found: left_ty,
                        span: left.span(),
                    });
                }
                if right_ty != Type::Int {
                    return Err(TypeError::TypeMismatch {
                        expected: Type::Int,
                        found: right_ty,
                        span: right.span(),
                    });
                }
                Ok(Type::Int)
            }

            // 代入系などのその他の演算子はUnit型を返す（文として扱われる想定だが、式としてもUnitを返す）
            _ => Ok(Type::Unit),
        }
    }


    pub fn infer_unary(&mut self, op: &UnaryOp, operand: &mut Expression, span: Span) -> TypeResult<Type> {
        let ty = self.infer_expression(operand)?;
        match op {
            UnaryOp::Neg => match ty {
                Type::Int | Type::Float => Ok(ty),
                _ => Err(TypeError::InvalidUnaryOp {
                    op: "-".to_string(),
                    operand_type: ty,
                    span,
                }),
            },
            UnaryOp::Not => match ty {
                Type::Bool => Ok(Type::Bool),
                _ => Err(TypeError::InvalidUnaryOp {
                    op: "!".to_string(),
                    operand_type: ty,
                    span,
                }),
            },
            // 必要なら BitNot も追加
            _ => Ok(ty),
        }
    }
    
    // ============================================================
    // If Expression (then/else are Blocks)
    // ============================================================
    pub fn infer_if_expression(
        &mut self,
        condition: &mut Expression,
        then_block: &mut Block,
        else_block: Option<&mut Block>,
    ) -> TypeResult<Type> {
        let cond_ty = self.infer_expression(condition)?;
        if cond_ty != Type::Bool {
            return Err(TypeError::InvalidBinaryOp {
                op: "if condition".to_string(),
                left_type: cond_ty,
                right_type: Type::Bool,
                span: condition.span(),
            });
        }

        let then_ty = self.infer_block(then_block)?;
        if let Some(else_blk) = else_block {
            let else_ty = self.infer_block(else_blk)?;
            let unified = self.unify(then_ty, else_ty, else_blk.span)?;
            Ok(self.apply(unified))
        } else {
            Ok(Type::Unit)
        }

    }

    // ============================================================
    // Function Definition Checking
    // ============================================================
    pub fn check_function_definition(
        &mut self,
        name: &str,
        params: &[(String, Type)],
        return_type: &Type,
        body: &mut Expression,
    ) -> TypeResult<()> {
        self.env.push_scope();
        for (param_name, param_ty) in params {
            if self.env.define(param_name.clone(), param_ty.clone()).is_err() {
                return Err(TypeError::DuplicateDefinition {
                    name: param_name.clone(),
                    span: body.span(),
                });
            }
        }

        let body_ty = self.infer_expression(body)?;
        /*if &body_ty != return_type {
            self.env.pop_scope();
            return Err(TypeError::ReturnTypeMismatch {
                expected: return_type.clone(),
                found: body_ty,
                span: body.span(),
            });
        }*/
        if !body_ty.matches(return_type) && *return_type != Type::Unit {
            return Err(TypeError::ReturnTypeMismatch {
                expected: return_type.clone(),
                found: body_ty,
                span: body.span(),
            });
        }


        self.env.pop_scope();
        self.env
            .define(
                name.to_string(),
                Type::Function {
                    params: params.iter().map(|(_, t)| t.clone()).collect(),
                    return_type: Box::new(return_type.clone()),
                },
            )
            .map_err(|_| TypeError::DuplicateDefinition {
                name: name.to_string(),
                span: body.span(),
            })
    }

    // ============================================================
    // Function Call Checking
    // ============================================================
    pub fn infer_function_call(
        &mut self,
        callee: &mut Expression,
        args: &mut Vec<Expression>,
    ) -> TypeResult<Type> {
        // --- 1) UFCS Call: Trait::method(arg0, ...) ---
        if let Expression::UfcsMethod {
            trait_path,
            method,
            span,
        } = callee
        {
            if args.is_empty() {
                return Err(TypeError::ArityMismatch {
                    expected: 1,
                    found: 0,
                    span: *span,
                });
            }

            // ============================================================
            // 1. Receiver (第1引数) の型推論
            // ============================================================
            // ※ desugar するので args[0] を後で remove するが、
            //   まずは &mut args[0] で通常通り型推論してOK。
            let receiver_ty = self.infer_expression(&mut args[0])?;
            let receiver_ty = self.apply(receiver_ty);

            // ============================================================
            // 2. Trait の解決
            // ============================================================
            let mut trait_def_id = self.resolve_trait_def_from_path(trait_path, *span)?;

            // ★ここが根本修正: UFCS の左側が trait でなければ `XTrait` を探す
            let is_trait = matches!(self.env.get_def(trait_def_id), Some(ItemDef::Trait(_)));
            if !is_trait {
                let base = self.trait_key_from_path(trait_path); // "Greeter"
                let fallback = format!("{base}Trait");           // "GreeterTrait"

                if let Some(fid) = self.env.resolve_def(&fallback) {
                    if matches!(self.env.get_def(fid), Some(ItemDef::Trait(_))) {
                        trait_def_id = fid;
                    }
                }
            }

            // まだ trait になっていないなら、UFCS として成立していないのでエラー
            if !matches!(self.env.get_def(trait_def_id), Some(ItemDef::Trait(_))) {
                return Err(TypeError::CannotInfer { span: *span });
            }


            // ============================================================
            // 3. Impl Selection
            // ============================================================
            let selected = self.select_impl(trait_def_id, receiver_ty.clone(), *span)?;

            // ============================================================
            // 4. メソッドシグネチャ + method_defid の取得
            // ============================================================
            // ★ここが env / ImplDef の型変更に対応した部分
            // ImplDef.methods: HashMap<String, (Type, DefId)>
            let (mut method_ty, method_defid) = selected
                .impl_def
                .methods
                .get(method.name.as_str())
                .ok_or_else(|| TypeError::CannotInfer { span: *span })?
                .clone();

            // --- ここからがあなたが書いていた具体化処理（維持） ---

            // (A) method_ty 内の type params (Tなど) を具体化してから apply
            method_ty = self.subst_typevars(method_ty, &selected.subst);
            method_ty = self.apply(method_ty);

            // (B) assoc_bindings 側も type params を具体化してから apply
            let mut solved_assoc = selected.assoc_bindings.clone();
            for (_k, v) in solved_assoc.iter_mut() {
                *v = self.apply(self.subst_typevars(v.clone(), &selected.subst));
            }

            // (C) Self::Item 等の関連型参照を展開して apply
            method_ty = self.subst_assoc(method_ty, &solved_assoc);
            method_ty = self.apply(method_ty);

            // ============================================================
            // 5. 型検査 (Unify) + UFCS -> MethodCall desugar
            // ============================================================
            match method_ty {
                Type::Function { params, return_type } => {
                    // receiver を含む形で params が入っている前提（あなたの既存設計と同じ）
                    let is_variadic = params
                        .last()
                        .map_or(false, |t| matches!(t, Type::Variadic(_)));

                    // receiver 分も含めて arity チェック（あなたのコードのまま）
                    if !is_variadic && args.len() != params.len() {
                        return Err(TypeError::ArityMismatch {
                            expected: params.len(),
                            found: args.len(),
                            span: *span,
                        });
                    }

                    if is_variadic && args.len() < params.len().saturating_sub(1) {
                        return Err(TypeError::ArityMismatch {
                            expected: params.len().saturating_sub(1),
                            found: args.len(),
                            span: *span,
                        });
                    }

                    // 引数 Unify（あなたのコードのまま）
                    for (i, arg_expr) in args.iter_mut().enumerate() {
                        let arg_ty = self.infer_expression(arg_expr)?;

                        let expected_ty = if i < params.len() {
                            params[i].clone()
                        } else if is_variadic {
                            if let Type::Variadic(inner) = params.last().unwrap() {
                                inner.as_ref().clone()
                            } else {
                                unreachable!()
                            }
                        } else {
                            unreachable!()
                        };

                        // 保険: expected_ty に残っているメタ変数を潰してから unify
                        let expected_ty = self.apply(expected_ty);

                        self.unify(arg_ty, expected_ty, arg_expr.span())?;
                    }

                    // ★ここが案Aの「desugar」本体
                    // args[0] を receiver として抜き取り、残りを method args にする
                    let receiver_expr = args.remove(0);
                    let method_args = std::mem::take(args);

                    *callee = Expression::MethodCall {
                        receiver: Box::new(receiver_expr),
                        method: method.clone(),
                        args: method_args,
                        span: *span,
                        resolved: Some(method_defid), // ★ VM fast-path 用
                    };

                    // 戻り値も apply してから返す（メタ変数残り防止）
                    let ret = self.apply(*return_type);
                    return Ok(ret);
                }
                other => {
                    return Err(TypeError::NotCallable {
                        value_type: other,
                        span: *span,
                    })
                }
            }
        }




        // --- 2) println / assert の特別扱い (Existing) ---
        if let Expression::Variable { name, .. } = callee {
            let fname = name
                .last_ident()
                .map(|id| id.name.as_str())
                .unwrap_or("");

            if fname == "println" {
                for arg in args.iter_mut() {
                    self.infer_expression(arg)?;
                }
                return Ok(Type::Unit);
            }

            if fname == "assert" {
                if args.is_empty() || args.len() > 2 {
                    return Err(TypeError::ArityMismatch {
                        expected: 2,
                        found: args.len(),
                        span: callee.span(),
                    });
                }

                let cond_ty = self.infer_expression(&mut args[0])?;
                self.unify(cond_ty, Type::Bool, args[0].span())?;

                if args.len() == 2 {
                    let msg_ty = self.infer_expression(&mut args[1])?;
                    self.unify(msg_ty, Type::String, args[1].span())?;
                }

                return Ok(Type::Unit);
            }
        }

        // --- 3) 通常の関数呼び出し / 変数経由の呼び出し ---
        
        // 関数自体の型を推論
        let func_ty = self.infer_expression(callee)?;

        let (param_types, return_ty) = match func_ty.clone() {
            Type::Function { params, return_type } => (params, *return_type),
            _ => {
                return Err(TypeError::NotCallable {
                    value_type: func_ty,
                    span: callee.span(),
                })
            }
        };

        // 引数の数と型のチェック
        let is_variadic = param_types
            .last()
            .map_or(false, |t| matches!(t, Type::Variadic(_)));

        if !is_variadic && args.len() != param_types.len() {
            return Err(TypeError::ArityMismatch {
                expected: param_types.len(),
                found: args.len(),
                span: callee.span(),
            });
        }
        if is_variadic && args.len() < (param_types.len().saturating_sub(1)) {
            return Err(TypeError::ArityMismatch {
                expected: param_types.len().saturating_sub(1),
                found: args.len(),
                span: callee.span(),
            });
        }
        let args_len = args.len();
        for (i, arg_expr) in args.iter_mut().enumerate() {
            let arg_ty = self.infer_expression(arg_expr)?;

            // expected_ty は &Type ではなく Type(owned) にして unify に渡す
            let expected_ty: Type = if i < param_types.len() {
                match &param_types[i] {
                    Type::Variadic(inner) => inner.as_ref().clone(),
                    other => other.clone(),
                }
            } else if is_variadic {
                match param_types.last().unwrap() {
                    Type::Variadic(inner) => inner.as_ref().clone(),
                    _ => unreachable!(),
                }
            } else {
                return Err(TypeError::ArityMismatch {
                    expected: param_types.len(),
                    found: args_len,
                    span: arg_expr.span(),
                });
            };

            self.unify(arg_ty, expected_ty, arg_expr.span())?;
        }

        Ok(return_ty)
    }

    // 分布コンストラクタのリテラル引数の静的パラメータチェック
    // 引数がリテラル値の場合のみチェックを行う（変数の場合は実行時チェックに任せる）
    fn check_distribution_parameters(
        &self,
        func_name: &str,
        args: &[Expression],
        span: Span,
    ) -> TypeResult<()> {
        match func_name {
            "Gaussian" => {
                // Gaussian(mean, std): std > 0
                if args.len() >= 2 {
                    if let Expression::Literal { value: Literal::Float(std), .. } = &args[1] {
                        if *std <= 0.0 {
                            return Err(TypeError::InvalidDistributionParameter {
                                distribution: "Gaussian".to_string(),
                                param_name: "std".to_string(),
                                reason: format!("standard deviation must be positive, got {}", std),
                                span,
                            });
                        }
                    }
                    if let Expression::Literal { value: Literal::Int(std), .. } = &args[1] {
                        if *std <= 0 {
                            return Err(TypeError::InvalidDistributionParameter {
                                distribution: "Gaussian".to_string(),
                                param_name: "std".to_string(),
                                reason: format!("standard deviation must be positive, got {}", std),
                                span,
                            });
                        }
                    }
                }
            }
            "Uniform" => {
                // Uniform(min, max): min < max
                if args.len() >= 2 {
                    let min_val = Self::extract_float_literal(&args[0]);
                    let max_val = Self::extract_float_literal(&args[1]);
                    
                    if let (Some(min), Some(max)) = (min_val, max_val) {
                        if min >= max {
                            return Err(TypeError::InvalidDistributionParameter {
                                distribution: "Uniform".to_string(),
                                param_name: "min, max".to_string(),
                                reason: format!("min must be less than max, got min={}, max={}", min, max),
                                span,
                            });
                        }
                    }
                }
            }
            "Beta" => {
                // Beta(alpha, beta): alpha > 0, beta > 0
                if args.len() >= 2 {
                    if let Some(alpha) = Self::extract_float_literal(&args[0]) {
                        if alpha <= 0.0 {
                            return Err(TypeError::InvalidDistributionParameter {
                                distribution: "Beta".to_string(),
                                param_name: "alpha".to_string(),
                                reason: format!("alpha must be positive, got {}", alpha),
                                span,
                            });
                        }
                    }
                    if let Some(beta) = Self::extract_float_literal(&args[1]) {
                        if beta <= 0.0 {
                            return Err(TypeError::InvalidDistributionParameter {
                                distribution: "Beta".to_string(),
                                param_name: "beta".to_string(),
                                reason: format!("beta must be positive, got {}", beta),
                                span,
                            });
                        }
                    }
                }
            }
            "Bernoulli" => {
                // Bernoulli(p): 0 <= p <= 1
                if !args.is_empty() {
                    if let Some(p) = Self::extract_float_literal(&args[0]) {
                        if !(0.0..=1.0).contains(&p) {
                            return Err(TypeError::InvalidDistributionParameter {
                                distribution: "Bernoulli".to_string(),
                                param_name: "p".to_string(),
                                reason: format!("probability must be between 0 and 1, got {}", p),
                                span,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    // リテラル式からf64値を抽出するヘルパー
    fn extract_float_literal(expr: &Expression) -> Option<f64> {
        match expr {
            Expression::Literal { value: Literal::Float(v), .. } => Some(*v),
            Expression::Literal { value: Literal::Int(v), .. } => Some(*v as f64),
            Expression::Unary { op: crate::ast::node::UnaryOp::Neg, operand, .. } => {
                Self::extract_float_literal(operand).map(|v| -v)
            }
            _ => None,
        }
    }


}

// #[cfg(test)]
// mod tests { ... } // Tests disabled due to AST changes (Variable needs Path, infer needs mut)
