// src/typeck/infer.rs

use crate::ast::node::PathSeg;
use crate::ast::node::{
    BinaryOp, Block, DefId, Expression, ImplItem, Literal, Path, Pattern, Span, Type, UnaryOp,
    VariantData, WherePredicate,
};
use crate::typeck::env::{
    EnumDefInfo, ImplDef, ItemDef, StructDefInfo, TraitDefInfo, TypeAliasDef, TypeEnv, VariantInfo,
};
use crate::typeck::error::{TypeError, TypeResult};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TypeInfer {
    pub env: TypeEnv,
    next_meta: usize,
    subst: HashMap<usize, Type>,
}
pub struct SelectedImpl {
    impl_def: ImplDef,
    subst: HashMap<String, Type>,
    assoc_bindings: HashMap<String, Type>,
}
impl TypeInfer {
    pub fn new() -> Self {
        let mut type_env = TypeEnv::new();

        type_env
            .define(
                "println".to_string(),
                Type::Function {
                    params: vec![Type::Variadic(Box::new(Type::Any))],
                    return_type: Box::new(Type::Unit),
                },
            )
            .ok();
        type_env
            .define(
                "print".to_string(),
                Type::Function {
                    params: vec![Type::Variadic(Box::new(Type::Any))],
                    return_type: Box::new(Type::Unit),
                },
            )
            .ok();
        type_env
            .define(
                "nearly_eq".to_string(),
                Type::Function {
                    params: vec![Type::Float, Type::Float],
                    return_type: Box::new(Type::Bool),
                },
            )
            .ok();
        type_env
            .define(
                "Signal_new".to_string(),
                Type::Function {
                    params: vec![Type::Any],
                    return_type: Box::new(Type::Signal(Box::new(Type::Any))),
                },
            )
            .ok();
        type_env
            .define(
                "Signal_set".to_string(),
                Type::Function {
                    params: vec![Type::Signal(Box::new(Type::Any)), Type::Any],
                    return_type: Box::new(Type::Unit),
                },
            )
            .ok();
        type_env
            .define(
                "Signal_get".to_string(),
                Type::Function {
                    params: vec![Type::Signal(Box::new(Type::Any))],
                    return_type: Box::new(Type::Any),
                },
            )
            .ok();
        type_env
            .define(
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
            )
            .ok();

        type_env
            .define(
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
            )
            .ok();

        type_env
            .define(
                "Signal_combine".to_string(),
                Type::Function {
                    params: vec![
                        Type::Signal(Box::new(Type::Any)),
                        Type::Signal(Box::new(Type::Any)),
                        Type::Function {
                            params: vec![Type::Any, Type::Any],
                            return_type: Box::new(Type::Any),
                        },
                    ],
                    return_type: Box::new(Type::Signal(Box::new(Type::Any))),
                },
            )
            .ok();
        type_env
            .define(
                "Event_new".to_string(),
                Type::Function {
                    params: vec![],
                    return_type: Box::new(Type::Event(Box::new(Type::Any))),
                },
            )
            .ok();

        type_env
            .define(
                "Event_emit".to_string(),
                Type::Function {
                    params: vec![Type::Event(Box::new(Type::Any)), Type::Any],
                    return_type: Box::new(Type::Unit),
                },
            )
            .ok();

        type_env
            .define(
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
            )
            .ok();
        type_env
            .define(
                "Gaussian".to_string(),
                Type::Function {
                    params: vec![Type::Float, Type::Float],
                    return_type: Box::new(Type::Gaussian),
                },
            )
            .ok();

        type_env
            .define(
                "Uniform".to_string(),
                Type::Function {
                    params: vec![Type::Float, Type::Float],
                    return_type: Box::new(Type::Uniform),
                },
            )
            .ok();

        type_env
            .define(
                "sample".to_string(),
                Type::Function {
                    params: vec![Type::Any],
                    return_type: Box::new(Type::Float),
                },
            )
            .ok();
        type_env
            .define(
                "observe".to_string(),
                Type::Function {
                    params: vec![Type::Any, Type::Float],
                    return_type: Box::new(Type::Unit),
                },
            )
            .ok();

        type_env
            .define(
                "Bernoulli".to_string(),
                Type::Function {
                    params: vec![Type::Float],
                    return_type: Box::new(Type::Bernoulli),
                },
            )
            .ok();

        type_env
            .define(
                "Beta".to_string(),
                Type::Function {
                    params: vec![Type::Float, Type::Float],
                    return_type: Box::new(Type::Beta),
                },
            )
            .ok();

        type_env
            .define(
                "VonMises".to_string(),
                Type::Function {
                    params: vec![Type::Float, Type::Float], // mu, kappa
                    return_type: Box::new(Type::VonMises),
                },
            )
            .ok();

        let unary_float = Type::Function {
            params: vec![Type::Float],
            return_type: Box::new(Type::Float),
        };
        for name in &["exp", "ln", "sin", "cos", "tan", "sqrt"] {
            type_env.define(name.to_string(), unary_float.clone()).ok();
        }

        type_env
            .define(
                "pow".to_string(),
                Type::Function {
                    params: vec![Type::Float, Type::Float],
                    return_type: Box::new(Type::Float),
                },
            )
            .ok();

        type_env
            .define(
                "to_json".to_string(),
                Type::Function {
                    params: vec![Type::Any],
                    return_type: Box::new(Type::String),
                },
            )
            .ok();
        type_env
            .define(
                "from_json".to_string(),
                Type::Function {
                    params: vec![Type::String],
                    return_type: Box::new(Type::Any),
                },
            )
            .ok();

        type_env
            .define(
                "infer".to_string(),
                Type::Function {
                    params: vec![Type::Int, Type::Any],
                    return_type: Box::new(Type::Array(Box::new(Type::Any))),
                },
            )
            .ok();

        type_env
            .define(
                "infer_vi".to_string(),
                Type::Function {
                    params: vec![
                        Type::Map(Box::new(Type::String), Box::new(Type::Any)),
                        Type::Any,
                        Type::Any,
                    ],
                    return_type: Box::new(Type::Map(Box::new(Type::String), Box::new(Type::Float))),
                },
            )
            .ok();

        type_env
            .define(
                "infer_hmc".to_string(),
                Type::Function {
                    params: vec![
                        Type::Map(Box::new(Type::String), Box::new(Type::Any)),
                        Type::Any,
                    ],
                    return_type: Box::new(Type::Array(Box::new(Type::Map(
                        Box::new(Type::String),
                        Box::new(Type::Float),
                    )))),
                },
            )
            .ok();

        type_env
            .define(
                "param".to_string(),
                Type::Function {
                    params: vec![Type::String, Type::Float],
                    return_type: Box::new(Type::Float),
                },
            )
            .ok();

        type_env
            .define(
                "Map".to_string(),
                Type::Function {
                    params: vec![],
                    return_type: Box::new(Type::Map(Box::new(Type::String), Box::new(Type::Any))),
                },
            )
            .ok();

        type_env
            .define(
                "exp".to_string(),
                Type::Function {
                    params: vec![Type::Float],
                    return_type: Box::new(Type::Float),
                },
            )
            .ok();
        type_env
            .define(
                "Rc_new".to_string(),
                Type::Function {
                    params: vec![Type::Any],
                    return_type: Box::new(Type::Rc(Box::new(Type::Any))),
                },
            )
            .ok();

        type_env
            .define(
                "Rc_downgrade".to_string(),
                Type::Function {
                    params: vec![Type::Rc(Box::new(Type::Any))],
                    return_type: Box::new(Type::Weak(Box::new(Type::Any))),
                },
            )
            .ok();

        type_env
            .define(
                "Weak_upgrade".to_string(),
                Type::Function {
                    params: vec![Type::Weak(Box::new(Type::Any))],
                    return_type: Box::new(Type::Option(Box::new(Type::Rc(Box::new(Type::Any))))),
                },
            )
            .ok();

        type_env
            .define(
                "create_tape".to_string(),
                Type::Function {
                    params: vec![],
                    return_type: Box::new(Type::Int),
                },
            )
            .ok();

        type_env
            .define(
                "param".to_string(),
                Type::Function {
                    params: vec![Type::Float, Type::Int],
                    return_type: Box::new(Type::Float),
                },
            )
            .ok();

        type_env
            .define(
                "backward".to_string(),
                Type::Function {
                    params: vec![Type::Float],
                    return_type: Box::new(Type::Unit),
                },
            )
            .ok();

        type_env
            .define(
                "grad".to_string(),
                Type::Function {
                    params: vec![Type::Float],
                    return_type: Box::new(Type::Float),
                },
            )
            .ok();

        TypeInfer {
            env: type_env,
            next_meta: 0,
            subst: HashMap::new(),
        }
    }

    fn trait_key_from_path(&self, path: &Path) -> String {
        path.last_ident().unwrap().name.clone()
    }

    fn resolve_trait_def_from_path(&self, path: &Path, span: Span) -> TypeResult<DefId> {
        if let Some(id) = path.resolved {
            return Ok(id);
        }

        let ident = path.last_ident().ok_or(TypeError::CannotInfer { span })?;
        let _key = ident.name.clone();

        self.env
            .resolve_def(&ident.name)
            .ok_or(TypeError::UndefinedVariable {
                name: ident.name.clone(),
                span,
            })
    }

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

    pub fn infer_expression(&mut self, expr: &mut Expression) -> TypeResult<Type> {
        match expr {
            Expression::Literal { value, .. } => Ok(self.infer_literal(value)),
            Expression::Variable { name, span, .. } => {
                if name.segments.len() >= 2 {
                    let enum_name_seg = &name.segments[name.segments.len() - 2];
                    if let PathSeg::Ident(enum_ident) = enum_name_seg {
                        if let Some(def_id) = self.env.resolve_def(&enum_ident.name) {
                            let enum_data =
                                if let Some(ItemDef::Enum(info)) = self.env.get_def(def_id) {
                                    Some((info.variants.clone(), info.typeparams.clone()))
                                } else {
                                    None
                                };

                            if let Some((variants, typeparams)) = enum_data {
                                let variant_seg = name.segments.last().unwrap();
                                if let PathSeg::Ident(variant_ident) = variant_seg {
                                    if let Some(variant_info) = variants.get(&variant_ident.name) {
                                        name.resolved = Some(def_id);

                                        println!(
                                            "[DEBUG] Found Enum Variant: {:?}, typeparams len: {}",
                                            enum_ident.name,
                                            typeparams.len()
                                        );

                                        let mut typeargs = Vec::new();
                                        for _ in &typeparams {
                                            typeargs.push(self.fresh_meta());
                                        }

                                        let mut subst = std::collections::HashMap::new();
                                        for (i, param) in typeparams.iter().enumerate() {
                                            subst.insert(
                                                param.name.name.clone(),
                                                typeargs[i].clone(),
                                            );
                                        }

                                        let enum_path = Path {
                                            segments: name.segments[..name.segments.len() - 1]
                                                .to_vec(),
                                            span: *span,
                                            resolved: Some(def_id),
                                        };
                                        let enum_ty = Type::Named {
                                            name: enum_path,
                                            type_args: typeargs,
                                        };

                                        match variant_info {
                                            VariantInfo::Unit => return Ok(enum_ty),
                                            VariantInfo::Tuple(tys) => {
                                                let mut func_params = Vec::new();
                                                for t in tys {
                                                    func_params.push(
                                                        self.subst_typevars(t.clone(), &subst),
                                                    );
                                                }

                                                return Ok(Type::Function {
                                                    params: func_params,
                                                    return_type: Box::new(enum_ty),
                                                });
                                            }
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

                let id = name
                    .last_ident()
                    .cloned()
                    .ok_or(TypeError::CannotInfer { span: *span })?;
                if let Some(ty) = self.env.lookup(&id.name) {
                    return Ok(ty);
                }

                let defid = self
                    .env
                    .resolve_def(&id.name)
                    .ok_or(TypeError::UndefinedVariable {
                        name: id.name.clone(),
                        span: *span,
                    })?;

                name.resolved = Some(defid);

                match self.env.get_def(defid) {
                    Some(ItemDef::Function(ty)) => Ok(ty.clone()),
                    Some(ItemDef::Struct(_)) => Err(TypeError::UndefinedVariable {
                        name: id.name.clone(),
                        span: *span,
                    }),
                    _ => Err(TypeError::UndefinedVariable {
                        name: id.name.clone(),
                        span: *span,
                    }),
                }
            }

            Expression::Binary {
                op,
                left,
                right,
                span,
            } => self.infer_binary(*op, left, right, *span),
            Expression::Unary { op, operand, span } => self.infer_unary(op, operand, *span),
            Expression::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => self.infer_if_expression(condition, then_branch, else_branch.as_mut()),
            Expression::Call { callee, args, span } => {
                let mut desugared: Option<(Expression, Type)> = None;
                {
                    if let Expression::UfcsMethod {
                        trait_path,
                        method,
                        span: ufcs_span,
                    } = callee.as_mut()
                    {
                        let traitpath = trait_path.clone();
                        let method = method.clone();
                        let call_span = *span;
                        let ufcs_span = *ufcs_span;

                        if args.is_empty() {
                            return Err(TypeError::ArityMismatch {
                                expected: 1,
                                found: 0,
                                span: ufcs_span,
                            });
                        }

                        let receiverty = {
                            let t = self.infer_expression(args.first_mut().unwrap())?;
                            self.apply(t)
                        };
                        let mut traitdefid =
                            self.resolve_trait_def_from_path(&traitpath, ufcs_span)?;
                        let is_trait =
                            matches!(self.env.get_def(traitdefid), Some(ItemDef::Trait(_)));
                        if !is_trait {
                            let base = self.trait_key_from_path(&traitpath);
                            let fallback = format!("{base}Trait");
                            if let Some(fid) = self.env.resolve_def(fallback.as_str()) {
                                if matches!(self.env.get_def(fid), Some(ItemDef::Trait(_))) {
                                    traitdefid = fid;
                                }
                            }
                        }

                        let selected =
                            self.select_impl(traitdefid, receiverty.clone(), ufcs_span)?;
                        let (methodty, methoddefid) = selected
                            .impl_def
                            .methods
                            .get(method.name.as_str())
                            .ok_or(TypeError::CannotInfer { span: ufcs_span })?
                            .clone();

                        let (params, returntype) = match methodty {
                            Type::Function {
                                params,
                                return_type,
                            } => (params, return_type),
                            other => {
                                return Err(TypeError::NotCallable {
                                    value_type: other,
                                    span: ufcs_span,
                                })
                            }
                        };
                        self.unify(receiverty.clone(), params[0].clone(), ufcs_span)?;
                        for (i, argexpr) in args.iter_mut().enumerate() {
                            if i == 0 {
                                continue;
                            }
                            let argty = self.infer_expression(argexpr)?;
                            self.unify(argty, params[i].clone(), argexpr.span())?;
                        }

                        let receiver_expr = args.remove(0);
                        let method_args = std::mem::take(args);

                        desugared = Some((
                            Expression::MethodCall {
                                receiver: Box::new(receiver_expr),
                                method,
                                args: method_args,
                                span: call_span,
                                resolved: Some(methoddefid),
                            },
                            self.apply(*returntype),
                        ));
                    }
                    if let Expression::FieldAccess { object, field, .. } = callee.as_mut() {
                        let obj_ty_raw = self.infer_expression(object.as_mut())?;
                        let obj_ty = self.apply(obj_ty_raw);

                        if matches!(obj_ty, Type::String) {
                            let receiver_expr = std::mem::replace(
                                object,
                                Box::new(Expression::Literal {
                                    value: Literal::Unit,
                                    span: *span,
                                }),
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
                                    ret_type,
                                ));
                            }
                        }

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
                                    Type::Float,
                                ));
                            }
                        }

                        if let Type::Map(_, _) = &obj_ty {
                            let (ret_type, found) = match field.name.as_str() {
                                "insert" => (Type::Unit, true),
                                "get" => (Type::Any, true),
                                "contains_key" => (Type::Bool, true),
                                _ => (Type::Unit, false),
                            };
                            if found {
                                let receiver_expr = std::mem::replace(
                                    object,
                                    Box::new(Expression::Literal {
                                        value: Literal::Unit,
                                        span: *span,
                                    }),
                                );
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
                                    ret_type,
                                ));
                            }
                        }

                        let path_opt = match &obj_ty {
                            Type::Named { name, .. } => Some(name),
                            Type::DynTrait { trait_path } => Some(trait_path),
                            _ => None,
                        };

                        if let Some(path) = path_opt {
                            let selfdefid = if let Some(defid) = path.resolved {
                                defid
                            } else {
                                self.env
                                    .resolve_def(path.last_ident().unwrap().name.as_str())
                                    .ok_or(TypeError::CannotInfer { span: *span })?
                            };

                            let mut method_sig: Option<(Type, Option<DefId>)> = None;
                            if let Some((ty, defid)) =
                                self.env.lookup_method(selfdefid, field.name.as_str())
                            {
                                method_sig = Some((ty.clone(), Some(*defid)));
                            }
                            if method_sig.is_none() {
                                if let Some(ItemDef::Trait(trait_info)) =
                                    self.env.get_def(selfdefid)
                                {
                                    if let Some(ty) = trait_info.methods.get(field.name.as_str()) {
                                        method_sig = Some((ty.clone(), None));
                                    }
                                }
                            }

                            if method_sig.is_none() {
                                for impl_def in &self.env.impls {
                                    let is_matching_type = match &impl_def.self_ty {
                                        Type::Named {
                                            name: impl_name, ..
                                        } => {
                                            impl_name.resolved.or_else(|| {
                                                self.env.resolve_def(
                                                    impl_name.last_ident().unwrap().name.as_str(),
                                                )
                                            }) == Some(selfdefid)
                                        }
                                        _ => false,
                                    };

                                    if is_matching_type {
                                        if let Some((mty, mdefid)) =
                                            impl_def.methods.get(&field.name)
                                        {
                                            method_sig = Some((mty.clone(), Some(*mdefid)));
                                            break;
                                        }
                                    }
                                }
                            }

                            if let Some((method_ty, resolved_defid)) = method_sig {
                                if let Type::Function {
                                    params,
                                    return_type,
                                } = method_ty
                                {
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
                                    if is_variadic && args.len() < params.len().saturating_sub(2) {
                                        return Err(TypeError::ArityMismatch {
                                            expected: params.len().saturating_sub(2),
                                            found: args.len(),
                                            span: *span,
                                        });
                                    }

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

                                    let ret_ty = self.apply(*return_type);

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
                    }
                }
                if let Some((new_expr, ty)) = desugared {
                    *expr = new_expr;
                    return Ok(ty);
                }

                if let Expression::Variable { name, .. } = &**callee {
                    if name.last_ident().map(|id| id.name.as_str()) == Some("head") {
                        if let Some(first_arg) = args.first_mut() {
                            if let Ok(arg_ty) = self.infer_expression(first_arg) {
                                if let Type::Array(elem_ty) = arg_ty {
                                    return Ok(Type::Option(elem_ty));
                                }
                            }
                        }
                    }
                }

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
                let recv_ty_raw = self.infer_expression(receiver)?;
                let recv_ty = self.apply(recv_ty_raw);
                let objty = recv_ty.clone();
                if matches!(objty, Type::Int) {
                    for arg in args.iter_mut() {
                        let _ = self.infer_expression(arg)?;
                    }
                    match method.name.as_str() {
                        "to_string" => return Ok(Type::String),
                        _ => {}
                    }
                }

                if matches!(objty, Type::String) {
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
                        _ => {}
                    }
                }

                if matches!(objty, Type::Float) {
                    for arg in args.iter_mut() {
                        let _ = self.infer_expression(arg)?;
                    }
                    match method.name.as_str() {
                        "abs" | "exp" | "ln" | "sqrt" | "sin" | "cos" | "tan" | "ceil"
                        | "floor" | "round" => return Ok(Type::Float),
                        "pow" | "powf" => return Ok(Type::Float),
                        _ => {}
                    }
                }

                if objty.is_probabilistic() {
                    let mut arg_types = Vec::new();
                    for arg in args.iter_mut() {
                        arg_types.push(self.infer_expression(arg)?);
                    }
                    return crate::typeck::prob::infer_distribution_method(
                        &method.name,
                        &objty,
                        &arg_types,
                        *span,
                    );
                }

                if let Type::Named { name, .. } = &objty {
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
                            return Err(TypeError::ArityMismatch {
                                expected: 1,
                                found: args.len(),
                                span: *span,
                            });
                        }
                        return Ok(Type::Array(elem_ty.clone()));
                    }
                }

                if let Type::Map(key_ty, val_ty) = &objty {
                    match method.name.as_str() {
                        "insert" => {
                            if args.len() != 2 {
                                return Err(TypeError::ArityMismatch {
                                    expected: 2,
                                    found: args.len(),
                                    span: *span,
                                });
                            }
                            let k_ty = self.infer_expression(&mut args[0])?;
                            self.unify(k_ty, *key_ty.clone(), args[0].span())?;

                            let v_ty = self.infer_expression(&mut args[1])?;
                            self.unify(v_ty, *val_ty.clone(), args[1].span())?;

                            return Ok(Type::Unit);
                        }
                        "get" => {
                            if args.len() != 1 {
                                return Err(TypeError::ArityMismatch {
                                    expected: 1,
                                    found: args.len(),
                                    span: *span,
                                });
                            }
                            let k_ty = self.infer_expression(&mut args[0])?;
                            self.unify(k_ty, *key_ty.clone(), args[0].span())?;

                            return Ok(*val_ty.clone());
                        }
                        "contains_key" => {
                            if args.len() != 1 {
                                return Err(TypeError::ArityMismatch {
                                    expected: 1,
                                    found: args.len(),
                                    span: *span,
                                });
                            }
                            let k_ty = self.infer_expression(&mut args[0])?;
                            self.unify(k_ty, *key_ty.clone(), args[0].span())?;

                            return Ok(Type::Bool);
                        }
                        _ => {}
                    }
                }

                let mut dispatch_ty = self.expand_alias_fixpoint(recv_ty.clone());
                loop {
                    dispatch_ty = self.expand_alias_fixpoint(dispatch_ty);
                    match dispatch_ty {
                        Type::Rc(inner) | Type::Weak(inner) => dispatch_ty = *inner,
                        _ => break,
                    }
                }

                if let Type::Named {
                    name: type_name, ..
                } = &dispatch_ty
                {
                    let type_def_id = if let Some(id) = type_name.resolved {
                        id
                    } else {
                        let ident = type_name
                            .last_ident()
                            .ok_or(TypeError::CannotInfer { span: *span })?;
                        self.env
                            .resolve_def(&ident.name)
                            .ok_or(TypeError::UndefinedVariable {
                                name: ident.name.clone(),
                                span: *span,
                            })?
                    };

                    if let Some((method_ty, method_def_id)) =
                        self.env.lookup_method(type_def_id, &method.name).cloned()
                    {
                        *resolved = Some(method_def_id);

                        if let Type::Function {
                            params,
                            return_type,
                        } = method_ty
                        {
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

                    let impls = self.env.impls.clone();
                    let mut trait_defs: std::collections::HashSet<DefId> =
                        std::collections::HashSet::new();
                    for impl_def in impls.iter() {
                        if impl_def.methods.contains_key(&method.name) {
                            trait_defs.insert(impl_def.trait_def);
                        }
                    }

                    let mut matches: Vec<(SelectedImpl, std::collections::HashMap<usize, Type>)> =
                        Vec::new();

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

                        if let Type::Function {
                            params,
                            return_type,
                        } = method_ty
                        {
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

                        if let Type::Function {
                            params,
                            return_type,
                        } = method_ty
                        {
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
                        }
                    }
                }

                Err(TypeError::CannotInfer { span: *span })
            }

            Expression::Some { expr, .. } => {
                let inner_ty = self.infer_expression(expr)?;
                Ok(Type::Option(Box::new(inner_ty)))
            }
            Expression::None { .. } => Ok(Type::Option(Box::new(Type::Any))),
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
                    Type::Result {
                        ok_type,
                        err_type: _,
                    } => Ok(*ok_type),
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
            Expression::Index {
                object,
                index,
                span,
            } => {
                let idx_ty = self.infer_expression(index)?;
                self.unify(idx_ty, Type::Int, *span)?;

                let obj_ty = self.infer_expression(object.as_mut())?;
                let elem_ty = self.fresh_meta();
                let expected_arr = Type::Array(Box::new(elem_ty.clone()));

                self.unify(obj_ty, expected_arr, *span)?;

                Ok(self.apply(elem_ty))
            }
            Expression::Match {
                scrutinee,
                arms,
                span,
            } => self.infer_match_expression(scrutinee, arms, *span),
            Expression::Array {
                elements,
                span: _span,
            } => {
                if elements.is_empty() {
                    return Ok(Type::Array(Box::new(self.fresh_meta())));
                }

                let mut elem_types = Vec::new();
                for elem in &mut *elements {
                    let ty = self.infer_expression(elem)?;
                    elem_types.push(ty);
                }

                let unified_ty = elem_types[0].clone();

                for (i, elem_ty) in elem_types.iter().enumerate().skip(1) {
                    if let Err(_) =
                        self.unify(elem_ty.clone(), unified_ty.clone(), elements[i].span())
                    {
                    }
                }

                Ok(Type::Array(Box::new(self.apply(unified_ty))))
            }

            Expression::Closure { params, body, .. } => {
                self.env.push_scope();

                let param_types: Vec<Type> = params
                    .iter()
                    .map(|p| {
                        let ty = p.ty.clone();
                        self.env.define(p.name.name.clone(), ty.clone()).ok();
                        ty
                    })
                    .collect();

                let return_type = self.infer_expression(body)?;

                self.env.pop_scope();

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

                let defid = match self.env.resolve_def(struct_name) {
                    Some(id) => id,
                    None => {
                        return Err(TypeError::CannotInfer { span: *span });
                    }
                };

                let info = match self.env.get_def(defid).cloned() {
                    Some(ItemDef::Struct(info)) => info,
                    _ => {
                        return Err(TypeError::CannotInfer { span: *span });
                    }
                };

                let mut typeargs = Vec::new();
                for _param in &info.typeparams {
                    typeargs.push(self.fresh_meta());
                }

                let mut subst = HashMap::new();
                for (i, param) in info.typeparams.iter().enumerate() {
                    subst.insert(param.name.name.clone(), typeargs[i].clone());
                }

                for field in fields.iter_mut() {
                    let field_ty = self.infer_expression(&mut field.value)?;

                    if let Some(expected_ty) = info.fields.get(&field.name.name) {
                        let expected_instantiated =
                            self.subst_typevars(expected_ty.clone(), &subst);
                        self.unify(field_ty, expected_instantiated, field.value.span())?;
                    } else {
                        return Err(TypeError::UndefinedField {
                            field: field.name.name.clone(),
                            structname: struct_name.clone(),
                            span: field.span,
                        });
                    }
                }

                let mut resolved_path = Path::from_ident(name.clone());
                resolved_path.resolved = Some(defid);

                Ok(Type::Named {
                    name: resolved_path,
                    type_args: typeargs.into_iter().map(|t| self.apply(t)).collect(),
                })
            }

            Expression::FieldAccess {
                object,
                field,
                span,
            } => {
                let objtyraw = self.infer_expression(object.as_mut())?;
                let objty = self.apply(objtyraw);

                if objty == Type::Gaussian && field.name == "sample" {
                    return Ok(Type::Function {
                        params: vec![],
                        return_type: Box::new(Type::Float),
                    });
                }

                if let Type::Named { ref name, .. } = objty {
                    let self_defid = if let Some(defid) = name.resolved {
                        defid
                    } else {
                        self.env
                            .resolve_def(name.last_ident().unwrap().name.as_str())
                            .ok_or(TypeError::CannotInfer { span: *span })?
                    };

                    if let Some(ItemDef::Struct(struct_info)) = self.env.get_def(self_defid) {
                        if let Some(field_ty) = struct_info.fields.get(&field.name) {
                            return Ok(field_ty.clone());
                        }
                    }

                    if let Some((method_ty, _method_defid)) =
                        self.env.lookup_method(self_defid, field.name.as_str())
                    {
                        return Ok(method_ty.clone());
                    }
                    for impl_def in &self.env.impls {
                        let is_matching_type = match &impl_def.self_ty {
                            Type::Named {
                                name: impl_name, ..
                            } => {
                                impl_name.resolved.or_else(|| {
                                    self.env
                                        .resolve_def(impl_name.last_ident().unwrap().name.as_str())
                                }) == Some(self_defid)
                            }
                            _ => false,
                        };

                        if is_matching_type {
                            if let Some(method_ty) = impl_def.methods.get(&field.name) {
                                return Ok(method_ty.clone().0);
                            }
                        }
                    }

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

            Expression::Paren { expr, .. } => self.infer_expression(expr),

            Expression::Tuple { elements, .. } => {
                if elements.is_empty() {
                    Ok(Type::Unit)
                } else {
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
            Expression::With {
                name,
                initializer,
                body,
                span: _,
            } => {
                let init_ty = self.infer_expression(initializer)?;
                self.env.push_scope();
                self.env.define(name.name.clone(), init_ty)?;
                let body_ty = self.infer_block(body)?;
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
                let def_id = self
                    .env
                    .resolve_def(&name.name)
                    .or_else(|| self.env.resolve_def(&name.name))
                    .ok_or(TypeError::UndefinedVariable {
                        name: name.name.clone(),
                        span: *span,
                    })?;

                let (variant_info, typeparams) = match self.env.get_def(def_id) {
                    Some(ItemDef::Enum(info)) => {
                        let v = info
                            .variants
                            .get(&variant.name)
                            .ok_or(TypeError::CannotInfer { span: *span })?
                            .clone();
                        (v, info.typeparams.clone())
                    }
                    _ => return Err(TypeError::CannotInfer { span: *span }),
                };

                let mut enum_type_args = Vec::new();
                for _ in &typeparams {
                    enum_type_args.push(self.fresh_meta());
                }

                let mut subst = HashMap::new();
                for (i, param) in typeparams.iter().enumerate() {
                    subst.insert(param.name.name.clone(), enum_type_args[i].clone());
                }
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

                for (i, a) in final_args.iter_mut().enumerate() {
                    let arg_ty = self.infer_expression(a)?;

                    let expected_ty_base = match &variant_info {
                        VariantInfo::Tuple(tys) => tys.get(i),
                        VariantInfo::Struct(order) => order.get(i).map(|(_, t)| t),
                        VariantInfo::Unit => None,
                    };

                    if let Some(base_ty) = expected_ty_base {
                        let expected_ty = self.subst_typevars(base_ty.clone(), &subst);
                        self.unify(arg_ty, expected_ty, a.span())?;
                    }
                }
                let path = Path {
                    segments: vec![PathSeg::Ident(name.clone())],
                    span: name.span,
                    resolved: Some(def_id),
                };

                Ok(Type::Named {
                    name: path,
                    type_args: enum_type_args,
                })
            }

            Expression::Spawn { expr, span: _span } => {
                let inner_ty = self.infer_expression(expr)?;
                Ok(Type::Future(Box::new(inner_ty)))
            }

            Expression::Await { expr, span } => {
                let future_ty = self.infer_expression(expr)?;
                match self.apply(future_ty) {
                    Type::Future(inner) => Ok(*inner),
                    other => Err(TypeError::TypeMismatch {
                        expected: Type::Future(Box::new(Type::Any)),
                        found: other,
                        span: *span,
                    }),
                }
            }

            _ => Err(TypeError::CannotInfer { span: expr.span() }),
        }
    }

    pub fn infer_block(&mut self, block: &mut Block) -> TypeResult<Type> {
        self.env.push_scope();
        let mut result = Type::Unit;

        for stmt in block.statements.iter_mut() {
            match stmt {
                crate::ast::node::Statement::Expression(expr) => {
                    result = self.infer_expression(expr)?;
                }
                crate::ast::node::Statement::Return { value, span: _span } => {
                    let ret_ty = if let Some(expr) = value {
                        self.infer_expression(expr)?
                    } else {
                        Type::Unit
                    };

                    result = ret_ty;

                    self.env.pop_scope();
                    return Ok(result);
                }
                crate::ast::node::Statement::Let {
                    ty,
                    init,
                    pattern,
                    span,
                    ..
                } => {
                    if let Some(initializer) = init {
                        let inferred = self.infer_expression(initializer)?;

                        match pattern {
                            crate::ast::node::Pattern::Identifier { name, .. } => {
                                if let Some(annot) = ty.clone() {
                                    let _ = self.check_assignable(
                                        inferred.clone(),
                                        annot.clone(),
                                        initializer.span(),
                                    )?;

                                    self.env.define(name.name.clone(), annot.clone()).map_err(
                                        |_| TypeError::DuplicateDefinition {
                                            name: name.name.clone(),
                                            span: initializer.span(),
                                        },
                                    )?;
                                } else {
                                    self.env
                                        .define(name.name.clone(), inferred.clone())
                                        .map_err(|_| TypeError::DuplicateDefinition {
                                            name: name.name.clone(),
                                            span: initializer.span(),
                                        })?;
                                }
                            }
                            _ => {
                                return Err(TypeError::CannotInfer { span: *span });
                            }
                        }
                    } else {
                        if let Some(annot) = ty.clone() {
                            match pattern {
                                crate::ast::node::Pattern::Identifier { name, .. } => {
                                    self.env.define(name.name.clone(), annot.clone()).map_err(
                                        |_| TypeError::DuplicateDefinition {
                                            name: name.name.clone(),
                                            span: *span,
                                        },
                                    )?;
                                }
                                _ => return Err(TypeError::CannotInfer { span: *span }),
                            }
                        } else {
                            return Err(TypeError::CannotInfer { span: *span });
                        }
                    }
                }
                _ => {}
            }
        }

        self.env.pop_scope();
        Ok(result)
    }

    fn infer_match_expression(
        &mut self,
        scrutinee: &mut Expression,
        arms: &mut [crate::ast::node::MatchArm],
        _span: Span,
    ) -> TypeResult<Type> {
        let scrutinee_ty = self.infer_expression(scrutinee)?;

        let mut ret_ty: Option<Type> = None;

        for arm in arms.iter_mut() {
            self.env.push_scope();

            self.check_pattern(&arm.pattern, &scrutinee_ty, arm.span)?;

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

            let body_ty = self.infer_expression(&mut arm.body)?;

            self.env.pop_scope();
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

            Type::Function {
                params,
                return_type,
            } => Type::Function {
                params: params.into_iter().map(|p| self.apply(p)).collect(),
                return_type: Box::new(self.apply(*return_type)),
            },

            Type::Tuple(ts) => Type::Tuple(ts.into_iter().map(|t| self.apply(t)).collect()),
            Type::Assoc {
                trait_def,
                self_ty,
                name,
            } => {
                let self_ty = Box::new(self.apply(*self_ty));
                Type::Assoc {
                    trait_def,
                    self_ty,
                    name,
                }
            }

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

            Type::Result { ok_type, err_type } => {
                self.occurs(id, &ok_type) || self.occurs(id, &err_type)
            }

            Type::Function {
                params,
                return_type,
            } => params.iter().any(|p| self.occurs(id, p)) || self.occurs(id, &return_type),

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
            return Err(TypeError::CannotInfer { span });
        }
        self.subst.insert(id, t);
        Ok(())
    }

    fn check_assignable(&mut self, found: Type, expected: Type, span: Span) -> TypeResult<Type> {
        let found = self.apply(found);
        let expected = self.apply(expected);

        if let Type::Named {
            name: ref trait_path,
            type_args: ref targs,
            ..
        } = expected
        {
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
                        return Err(TypeError::TypeMismatch {
                            expected,
                            found,
                            span,
                        });
                    }
                }
            }
        }
        match (found.clone(), expected.clone()) {
            (_, Type::DynTrait { trait_path }) => {
                if matches!(found, Type::DynTrait { .. }) {
                    return Ok(expected);
                }
                let traitdefid = if let Some(id) = trait_path.resolved {
                    id
                } else {
                    let key = self.trait_key_from_path(&trait_path);
                    self.env
                        .resolve_def(&key)
                        .ok_or(TypeError::UndefinedVariable {
                            name: key.clone(),
                            span,
                        })?
                };

                if self.select_impl(traitdefid, found.clone(), span).is_ok() {
                    return Ok(expected);
                }

                return Err(TypeError::TypeMismatch {
                    expected,
                    found,
                    span,
                });
            }

            (Type::Array(a), Type::Array(b)) => {
                let inner = self.check_assignable(*a, *b, span)?;
                Ok(Type::Array(Box::new(inner)))
            }
            (Type::Handle(a), Type::Handle(b)) => {
                let inner = self.check_assignable(*a, *b, span)?;
                Ok(Type::Handle(Box::new(inner)))
            }

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
            _ => a.last_name() == b.last_name(),
        }
    }

    fn unify(&mut self, a: Type, b: Type, span: Span) -> TypeResult<Type> {
        let a_applied = self.apply(a);
        let a_expanded = self.expand_alias_fixpoint(a_applied);
        let b_applied = self.apply(b);
        let b_expanded = self.expand_alias_fixpoint(b_applied);

        let a = a_expanded;
        let b = b_expanded;

        if matches!(a, Type::Any) || matches!(a, Type::Infer) {
            return Ok(b);
        }
        if matches!(b, Type::Any) || matches!(b, Type::Infer) {
            return Ok(a);
        }
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
            (Type::DynTrait { trait_path }, Type::Named { name, type_args })
            | (Type::Named { name, type_args }, Type::DynTrait { trait_path }) => {
                if type_args.is_empty() && self.same_trait_path(&name, &trait_path) {
                    return Ok(Type::DynTrait { trait_path });
                }

                let expected = Type::DynTrait {
                    trait_path: trait_path.clone(),
                };
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
                            expected: Type::Named {
                                name: an,
                                type_args: aargs,
                            },
                            found: Type::Named {
                                name: bn,
                                type_args: bargs,
                            },
                            span,
                        });
                    }
                } else {
                    if an.last_name() != bn.last_name() {
                        return Err(TypeError::TypeMismatch {
                            expected: Type::Named {
                                name: an,
                                type_args: aargs,
                            },
                            found: Type::Named {
                                name: bn,
                                type_args: bargs,
                            },
                            span,
                        });
                    }
                }

                if aargs.len() != bargs.len() {
                    return Err(TypeError::TypeMismatch {
                        expected: Type::Named {
                            name: an,
                            type_args: aargs,
                        },
                        found: Type::Named {
                            name: bn,
                            type_args: bargs,
                        },
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
            (Type::Signal(a1), Type::Signal(b1)) => {
                let inner = self.unify(*a1, *b1, span)?;
                Ok(Type::Signal(Box::new(inner)))
            }
            (Type::Event(a1), Type::Event(b1)) => {
                let inner = self.unify(*a1, *b1, span)?;
                Ok(Type::Event(Box::new(inner)))
            }
            (Type::Map(ak, av), Type::Map(bk, bv)) => {
                let rk = self.unify(*ak, *bk, span)?;
                let rv = self.unify(*av, *bv, span)?;
                Ok(Type::Map(Box::new(rk), Box::new(rv)))
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
                if p1.len() != p2.len() {
                    return Err(TypeError::TypeMismatch {
                        expected: Type::Function {
                            params: p1,
                            return_type: r1,
                        },
                        found: Type::Function {
                            params: p2,
                            return_type: r2,
                        },
                        span,
                    });
                }
                let mut rp = Vec::new();
                for (x, y) in p1.into_iter().zip(p2.into_iter()) {
                    rp.push(self.unify(x, y, span)?);
                }
                let rr = self.unify(*r1, *r2, span)?;
                Ok(Type::Function {
                    params: rp,
                    return_type: Box::new(rr),
                })
            }
            (Type::Assoc { .. }, _) | (_, Type::Assoc { .. }) => {
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

    fn check_pattern(&mut self, pattern: &Pattern, target_ty: &Type, span: Span) -> TypeResult<()> {
        match pattern {
            Pattern::Identifier { name, .. } => {
                self.env
                    .define(name.name.clone(), target_ty.clone())
                    .map_err(|_| TypeError::DuplicateDefinition {
                        name: name.name.clone(),
                        span,
                    })?;
                Ok(())
            }
            Pattern::Wildcard { .. } => Ok(()),

            Pattern::Literal { value, .. } => {
                let lit_ty = self.infer_literal(value);
                self.unify(lit_ty, target_ty.clone(), span)?;
                Ok(())
            }

            Pattern::Some { pattern: inner, .. } => match target_ty {
                Type::Option(inner_ty) => self.check_pattern(inner, inner_ty, span),
                Type::Any => self.check_pattern(inner, &Type::Any, span),
                _ => Err(TypeError::TypeMismatch {
                    expected: Type::Option(Box::new(Type::Any)),
                    found: target_ty.clone(),
                    span,
                }),
            },

            Pattern::None { .. } => match target_ty {
                Type::Option(_) | Type::Any => Ok(()),
                _ => Err(TypeError::TypeMismatch {
                    expected: Type::Option(Box::new(Type::Any)),
                    found: target_ty.clone(),
                    span,
                }),
            },
            Pattern::Ok { pattern: inner, .. } => match target_ty {
                Type::Result {
                    ok_type,
                    err_type: _,
                } => self.check_pattern(inner.as_ref(), ok_type.as_ref(), span),
                Type::Any => self.check_pattern(inner.as_ref(), &Type::Any, span),
                _ => Err(TypeError::TypeMismatch {
                    expected: Type::Result {
                        ok_type: Box::new(Type::Any),
                        err_type: Box::new(Type::Any),
                    },
                    found: target_ty.clone(),
                    span,
                }),
            },
            Pattern::Err { pattern: inner, .. } => match target_ty {
                Type::Result {
                    ok_type: _,
                    err_type,
                } => self.check_pattern(inner.as_ref(), err_type.as_ref(), span),
                Type::Any => self.check_pattern(inner.as_ref(), &Type::Any, span),
                _ => Err(TypeError::TypeMismatch {
                    expected: Type::Result {
                        ok_type: Box::new(Type::Any),
                        err_type: Box::new(Type::Any),
                    },
                    found: target_ty.clone(),
                    span,
                }),
            },
            Pattern::Tuple { patterns, .. } => match target_ty {
                Type::Tuple(elem_tys) => {
                    if patterns.len() != elem_tys.len() {
                        return Err(TypeError::TypeMismatch {
                            expected: target_ty.clone(),
                            found: target_ty.clone(),
                            span,
                        });
                    }
                    for (p, t) in patterns.iter().zip(elem_tys.iter()) {
                        self.check_pattern(p, t, span)?;
                    }
                    Ok(())
                }
                Type::Any => {
                    for p in patterns {
                        self.check_pattern(p, &Type::Any, span)?;
                    }
                    Ok(())
                }
                _ => Err(TypeError::TypeMismatch {
                    expected: Type::Tuple(vec![]),
                    found: target_ty.clone(),
                    span,
                }),
            },
            Pattern::Enum {
                name,
                variant,
                args,
                named_fields,
                span,
            } => {
                let def_id = self
                    .env
                    .resolve_def(&name.name)
                    .or_else(|| self.env.resolve_def(&name.name))
                    .ok_or(TypeError::UndefinedVariable {
                        name: name.name.clone(),
                        span: *span,
                    })?;
                let (variant_info, typeparams) = match self.env.get_def(def_id) {
                    Some(ItemDef::Enum(info)) => {
                        let v_info = info
                            .variants
                            .get(&variant.name)
                            .ok_or(TypeError::CannotInfer { span: *span })?
                            .clone();
                        (v_info, info.typeparams.clone())
                    }
                    _ => return Err(TypeError::CannotInfer { span: *span }),
                };

                let mut typeargs = Vec::new();
                for _ in &typeparams {
                    typeargs.push(self.fresh_meta());
                }

                let constructed_enum_ty = Type::Named {
                    name: Path {
                        segments: vec![crate::ast::node::PathSeg::Ident(name.clone())],
                        span: name.span,
                        resolved: Some(def_id),
                    },
                    type_args: typeargs.clone(),
                };
                self.unify(target_ty.clone(), constructed_enum_ty, *span)?;

                let mut subst = std::collections::HashMap::new();
                for (i, param) in typeparams.iter().enumerate() {
                    subst.insert(param.name.name.clone(), typeargs[i].clone());
                }
                let (expected_types, patterns_to_check): (Vec<Type>, Vec<Pattern>) =
                    match variant_info {
                        VariantInfo::Unit => (vec![], vec![]),

                        VariantInfo::Tuple(tys) => {
                            if args.len() != tys.len() {
                                return Err(TypeError::ArityMismatch {
                                    expected: tys.len(),
                                    found: args.len(),
                                    span: *span,
                                });
                            }

                            let mut subst_tys = Vec::new();
                            for t in tys {
                                subst_tys.push(self.subst_typevars(t.clone(), &subst));
                            }

                            (subst_tys, args.clone())
                        }

                        VariantInfo::Struct(fields) => {
                            if let Some(user_fields) = named_fields {
                                let mut p_list = Vec::new();
                                let mut t_list = Vec::new();

                                for (fname, fty) in fields {
                                    let user_field = user_fields
                                        .iter()
                                        .find(|x| x.name.name == *fname)
                                        .ok_or(TypeError::MissingField {
                                            field: fname.clone(),
                                            struct_name: variant.name.clone(),
                                            span: *span,
                                        })?;

                                    let pat = user_field.pattern.clone().unwrap_or_else(|| {
                                        Pattern::Identifier {
                                            name: user_field.name.clone(),
                                            span: user_field.span,
                                        }
                                    });

                                    t_list.push(self.subst_typevars(fty.clone(), &subst));
                                    p_list.push(pat);
                                }
                                (t_list, p_list)
                            } else {
                                if args.len() != fields.len() {
                                    return Err(TypeError::ArityMismatch {
                                        expected: fields.len(),
                                        found: args.len(),
                                        span: *span,
                                    });
                                }

                                let mut field_tys = Vec::new();
                                for (_, ty) in fields {
                                    field_tys.push(self.subst_typevars(ty.clone(), &subst));
                                }

                                (field_tys, args.clone())
                            }
                        }
                    };
                for (p, expected_ty) in patterns_to_check
                    .into_iter()
                    .zip(expected_types.into_iter())
                {
                    self.check_pattern(&p, &expected_ty, *span)?;
                }

                Ok(())
            }

            Pattern::Struct { name, fields, span } => {
                let def_id =
                    self.env
                        .resolve_def(&name.name)
                        .ok_or(TypeError::UndefinedVariable {
                            name: name.name.clone(),
                            span: *span,
                        })?;

                let struct_fields = match self.env.get_def(def_id) {
                    Some(ItemDef::Struct(info)) => info.fields.clone(),
                    _ => return Err(TypeError::CannotInfer { span: *span }),
                };

                for field_pat in fields {
                    let field_name = &field_pat.name.name;
                    let field_ty = struct_fields
                        .get(field_name)
                        .ok_or(TypeError::MissingField {
                            field: field_name.clone(),
                            struct_name: name.name.clone(),
                            span: *span,
                        })?
                        .clone();

                    let pat = field_pat
                        .pattern
                        .clone()
                        .unwrap_or_else(|| Pattern::Identifier {
                            name: field_pat.name.clone(),
                            span: field_pat.span,
                        });
                    self.check_pattern(&pat, &field_ty, *span)?;
                }
                Ok(())
            }

            Pattern::Or { patterns, span } => {
                for pat in patterns {
                    self.check_pattern(pat, target_ty, *span)?;
                }
                Ok(())
            }

            Pattern::Range {
                start, end, span, ..
            } => {
                let start_ty = self.infer_literal(start);
                let end_ty = self.infer_literal(end);
                self.unify(start_ty.clone(), target_ty.clone(), *span)?;
                self.unify(end_ty, start_ty, *span)?;
                Ok(())
            }
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
        let candidates: Vec<ImplDef> = self
            .env
            .impls
            .iter()
            .filter(|impl_def| impl_def.trait_def == trait_def_id)
            .cloned()
            .collect();

        let mut matched_impls: Vec<(ImplDef, HashMap<usize, Type>, HashMap<String, Type>)> =
            Vec::new();

        for impl_def in candidates {
            let saved_subst = self.subst.clone();

            let fresh_subst = self.instantiate_typeparams(&impl_def.typeparams);
            let fresh_self = self.subst_typevars(impl_def.self_ty.clone(), &fresh_subst);

            let unify_success = self.unify(receiver_ty.clone(), fresh_self, span).is_ok();

            let where_success = if unify_success {
                self.fulfill_obligations(&impl_def.where_preds, &fresh_subst, span)
                    .is_ok()
            } else {
                false
            };

            if unify_success && where_success {
                let cand_subst = self.subst.clone();
                matched_impls.push((impl_def, cand_subst, fresh_subst));
            }
            self.subst = saved_subst;
        }

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

        let (impl_def, final_subst, fresh_subst) = matched_impls.pop().unwrap();
        self.subst = final_subst;
        let mut solved_tparams = HashMap::new();
        for (name, ty) in fresh_subst {
            solved_tparams.insert(name, self.apply(ty));
        }

        let mut solved_assoc = HashMap::new();
        for (name, ty) in &impl_def.assoc_bindings {
            let substituted = self.subst_typevars(ty.clone(), &solved_tparams);
            solved_assoc.insert(name.clone(), self.apply(substituted));
        }

        Ok(SelectedImpl {
            impl_def: impl_def.clone(),
            subst: solved_tparams,
            assoc_bindings: solved_assoc,
        })
    }

    pub fn check_program(&mut self, program: &mut crate::ast::node::Program) -> TypeResult<()> {
        for item in &mut program.items {
            match item {
                crate::ast::node::Item::Function(func) => {
                    let param_types: Vec<Type> = func.params.iter().map(|p| p.ty.clone()).collect();
                    let ret_type = func.return_type.clone().unwrap_or(Type::Unit);

                    let return_type = if func.is_async {
                        Type::Future(Box::new(ret_type))
                    } else {
                        ret_type
                    };

                    let func_ty = Type::Function {
                        params: param_types,
                        return_type: Box::new(return_type),
                    };

                    let defid = self.env.insert_def(
                        func.name.name.clone(),
                        ItemDef::Function(func_ty),
                        func.span,
                    )?;

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

                    self.env
                        .insert_def(s.name.name.clone(), ItemDef::Struct(info), s.span)?;
                }

                crate::ast::node::Item::Trait(t) => {
                    let mut methods: HashMap<String, Type> = HashMap::new();

                    for m in &t.methods {
                        let params = m
                            .params
                            .iter()
                            .map(|p| Self::replace_self_in_trait(t, p.ty.clone()))
                            .collect::<Vec<_>>();

                        let ret = Self::replace_self_in_trait(
                            t,
                            m.return_type.clone().unwrap_or(Type::Unit),
                        );

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

                    let key = t.name.name.clone();
                    let _defid = self.env.insert_trait_def(key, info, t.span)?;
                }

                crate::ast::node::Item::Enum(e) => {
                    let mut variants: HashMap<String, VariantInfo> = HashMap::new();
                    for v in &e.variants {
                        let info: VariantInfo = match &v.data {
                            VariantData::Unit => VariantInfo::Unit,
                            VariantData::Tuple(tys) => VariantInfo::Tuple(tys.clone()),
                            VariantData::Struct(fields) => VariantInfo::Struct(
                                fields
                                    .iter()
                                    .map(|f| (f.name.name.clone(), f.ty.clone()))
                                    .collect(),
                            ),
                        };
                        variants.insert(v.name.name.clone(), info);
                    }
                    let info = EnumDefInfo {
                        typeparams: e.type_params.clone(),
                        variants,
                    };

                    self.env
                        .insert_def(e.name.name.clone(), ItemDef::Enum(info), e.span)?;
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
                    self.register_module_items(&m.name.name, &mut m.items)?;
                }

                crate::ast::node::Item::Import(imp) => {
                    let full = Self::path_to_key(&imp.path, "");

                    let target =
                        self.env
                            .resolve_def(&full)
                            .ok_or(TypeError::UndefinedVariable {
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

                crate::ast::node::Item::Impl(_) => {}

                crate::ast::node::Item::Extern(ext) => {
                    for f in &ext.functions {
                        let param_types: Vec<Type> =
                            f.params.iter().map(|p| p.ty.clone()).collect();
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

        for item in &mut program.items {
            match item {
                crate::ast::node::Item::Impl(impl_block) => {
                    if let Some(trait_ref) = &impl_block.trait_ref {
                        let trait_def_id = if let Type::Named { name, .. } = trait_ref {
                            self.resolve_trait_def_from_path(name, impl_block.span)?
                        } else {
                            return Err(TypeError::CannotInfer {
                                span: impl_block.span,
                            });
                        };
                        let typename: String = match &impl_block.self_ty {
                            Type::Named { name, .. } => name
                                .last_ident()
                                .map(|id| id.name.clone())
                                .unwrap_or_else(|| "anon".to_string()),
                            _ => "nonnamed".to_string(),
                        };
                        let mut methods = HashMap::new();
                        let mut assoc_bindings = HashMap::new();

                        for item in &mut impl_block.items {
                            match item {
                                ImplItem::Method(func) => {
                                    let paramtypes: Vec<Type> = func
                                        .params
                                        .iter()
                                        .map(|p| {
                                            if let Type::Named { name, .. } = &p.ty {
                                                if name.last_ident().map(|id| id.name.as_str())
                                                    == Some("Self")
                                                {
                                                    return impl_block.self_ty.clone();
                                                }
                                            }
                                            p.ty.clone()
                                        })
                                        .collect();

                                    let rettype = func.return_type.clone().unwrap_or(Type::Unit);

                                    let return_type = if func.is_async {
                                        Type::Future(Box::new(rettype))
                                    } else {
                                        rettype
                                    };

                                    let functy = Type::Function {
                                        params: paramtypes,
                                        return_type: Box::new(return_type),
                                    };

                                    let fq = format!("{}::{}", typename, func.name.name);
                                    let methoddefid = self.env.insert_def(
                                        fq,
                                        ItemDef::Function(functy.clone()),
                                        func.span,
                                    )?;
                                    func.defid = Some(methoddefid);

                                    methods.insert(
                                        func.name.name.clone(),
                                        (functy.clone(), methoddefid),
                                    );
                                }

                                ImplItem::AssocType(binding) => {
                                    assoc_bindings
                                        .insert(binding.name.name.clone(), binding.ty.clone());
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
                            _ => return Ok(()),
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

                                let type_name = match &impl_block.self_ty {
                                    Type::Named { name, .. } => name
                                        .last_ident()
                                        .map(|id| id.name.clone())
                                        .unwrap_or_else(|| "<anon>".to_string()),
                                    _ => "<non_named>".to_string(),
                                };

                                let fq = format!("{}::{}", type_name, func.name.name);

                                let method_def_id = self.env.insert_def(
                                    fq,
                                    ItemDef::Function(method_ty.clone()),
                                    func.span,
                                )?;
                                func.defid = Some(method_def_id);

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

        for item in program.items.iter_mut() {
            match item {
                crate::ast::node::Item::Function(func) => {
                    self.check_function(func)?;
                }
                crate::ast::node::Item::Impl(impl_block) => {
                    for impl_item in &mut impl_block.items {
                        if let crate::ast::node::ImplItem::Method(func) = impl_item {
                            self.check_impl_method(func, &impl_block.self_ty)?;
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn subst_assoc(&self, ty: Type, assoc: &HashMap<String, Type>) -> Type {
        match ty {
            Type::Assoc {
                trait_def,
                self_ty,
                name,
            } => assoc.get(&name).cloned().unwrap_or(Type::Assoc {
                trait_def,
                self_ty,
                name,
            }),
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
            Type::Function {
                params,
                return_type,
            } => Type::Function {
                params: params
                    .into_iter()
                    .map(|t| self.subst_assoc(t, assoc))
                    .collect(),
                return_type: Box::new(self.subst_assoc(*return_type, assoc)),
            },
            Type::Tuple(ts) => {
                Type::Tuple(ts.into_iter().map(|t| self.subst_assoc(t, assoc)).collect())
            }
            Type::Named { name, type_args } => Type::Named {
                name,
                type_args: type_args
                    .into_iter()
                    .map(|t| self.subst_assoc(t, assoc))
                    .collect(),
            },
            other => other,
        }
    }

    fn register_module_items(
        &mut self,
        prefix: &str,
        items: &mut [crate::ast::node::Item],
    ) -> TypeResult<()> {
        // Pass 1: Types and Functions
        for item in items.iter_mut() {
            match item {
                crate::ast::node::Item::Function(func) => {
                    let param_types: Vec<Type> = func.params.iter().map(|p| p.ty.clone()).collect();
                    let ret_type = func.return_type.clone().unwrap_or(Type::Unit);
                    let func_ty = Type::Function {
                        params: param_types,
                        return_type: Box::new(ret_type),
                    };

                    let key = format!("{prefix}::{}", func.name.name);
                    let defid = self
                        .env
                        .insert_def(key, ItemDef::Function(func_ty), func.span)?;
                    func.defid = Some(defid);
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
                    let mut variants: HashMap<String, VariantInfo> = HashMap::new();
                    for v in &e.variants {
                        let info: VariantInfo = match &v.data {
                            VariantData::Unit => VariantInfo::Unit,
                            VariantData::Tuple(tys) => VariantInfo::Tuple(tys.clone()),
                            VariantData::Struct(fields) => VariantInfo::Struct(
                                fields
                                    .iter()
                                    .map(|f| (f.name.name.clone(), f.ty.clone()))
                                    .collect(),
                            ),
                        };
                        variants.insert(v.name.name.clone(), info);
                    }
                    let info = EnumDefInfo {
                        typeparams: e.type_params.clone(),
                        variants,
                    };

                    self.env
                        .insert_def(e.name.name.clone(), ItemDef::Enum(info), e.span)?;
                }

                crate::ast::node::Item::Trait(t) => {
                    let key = format!("{}{}", prefix, t.name.name);
                    let mut methods: HashMap<String, Type> = HashMap::new();

                    for m in &t.methods {
                        let mut params: Vec<Type> = Vec::new();
                        params.push(Self::trait_self_ty_t(t));

                        params.extend(
                            m.params
                                .iter()
                                .map(|p| Self::replace_self_in_trait(t, p.ty.clone())),
                        );

                        let ret = Self::replace_self_in_trait(
                            t,
                            m.return_type.clone().unwrap_or(Type::Unit),
                        );

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
                    self.env.insert_trait_def(key, info, t.span)?;
                }

                crate::ast::node::Item::TypeAlias(a) => {
                    let key = format!("{}{}", prefix, a.name.name);
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
                    let full = Self::path_to_key(&imp.path, prefix);

                    let target =
                        self.env
                            .resolve_def(&full)
                            .ok_or(TypeError::UndefinedVariable {
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

                    let local_fq = format!("{prefix}::{}", local);
                    self.env.alias_def(local_fq, target, imp.span)?;
                }

                crate::ast::node::Item::Module(inner) => {
                    let new_prefix = format!("{prefix}::{}", inner.name.name);
                    self.register_module_items(&new_prefix, &mut inner.items)?;
                }

                crate::ast::node::Item::Extern(ext) => {
                    for f in &ext.functions {
                        let param_types: Vec<Type> =
                            f.params.iter().map(|p| p.ty.clone()).collect();
                        let ret_type = f.return_type.clone().unwrap_or(Type::Unit);
                        let func_ty = Type::Function {
                            params: param_types,
                            return_type: Box::new(ret_type),
                        };

                        let key = format!("{prefix}::{}", f.name.name);
                        self.env
                            .insert_def(key, ItemDef::Function(func_ty), f.span)?;
                    }
                }

                _ => {}
            }
        }

        // Pass 2: Impl
        for item in items {
            if let crate::ast::node::Item::Impl(impl_block) = item {
                if let Some(trait_ref) = &impl_block.trait_ref {
                    let trait_def_id = if let Type::Named { name, .. } = trait_ref {
                        self.resolve_trait_def_from_path(name, impl_block.span)?
                    } else {
                        return Err(TypeError::CannotInfer {
                            span: impl_block.span,
                        });
                    };
                    let typename: String = match &impl_block.self_ty {
                        Type::Named { name, .. } => name
                            .last_ident()
                            .map(|id| id.name.clone())
                            .unwrap_or_else(|| "anon".to_string()),
                        _ => "nonnamed".to_string(),
                    };
                    let mut methods = HashMap::new();
                    let mut assoc_bindings = HashMap::new();

                    for item in &impl_block.items {
                        match item {
                            ImplItem::Method(func) => {
                                let paramtypes: Vec<Type> = func
                                    .params
                                    .iter()
                                    .map(|p| {
                                        if let Type::Named { name, .. } = &p.ty {
                                            if name.last_ident().map(|id| id.name.as_str())
                                                == Some("Self")
                                            {
                                                return impl_block.self_ty.clone();
                                            }
                                        }
                                        p.ty.clone()
                                    })
                                    .collect();
                                let rettype = func.return_type.clone().unwrap_or(Type::Unit);
                                let functy = Type::Function {
                                    params: paramtypes,
                                    return_type: Box::new(rettype),
                                };

                                let fq = format!("{}::{}", typename, func.name.name);
                                // Note: Trait impl methods in modules might need prefix,
                                // but usually they are looked up via Trait tables, not by FQ name directly.
                                // However, unique DefId is needed.
                                // Let's use prefix + typename + impl method name
                                let fq_impl = format!("{prefix}::{fq}");

                                let methoddefid = self.env.insert_def(
                                    fq_impl,
                                    ItemDef::Function(functy.clone()),
                                    func.span,
                                )?;
                                // func.defid = Some(methoddefid); // Cannot mutate immutable reference
                                // Note: register_module_items takes &[Item], so we cannot mutate func.defid easily without interior mutability or changing signature.
                                // But check_program takes &mut Program.
                                // register_module_items signature is `items: &[Item]`.
                                // We cannot set defid here! This is a problem.
                                // However, if we simply register them in env, lookup might work if we don't rely on func.defid being set in AST?
                                // lookup_method/lookup_trait_method uses env tables.

                                methods
                                    .insert(func.name.name.clone(), (functy.clone(), methoddefid));
                            }
                            ImplItem::AssocType(binding) => {
                                assoc_bindings
                                    .insert(binding.name.name.clone(), binding.ty.clone());
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
                    // Inherent Impl
                    let self_defid = match &impl_block.self_ty {
                        Type::Named { name, .. } => {
                            // Trying resolve with prefix logic if necessary?
                            // Usually names are local or absolute.
                            if let Some(defid) = name.resolved {
                                defid
                            } else {
                                // Simplified resolution try
                                let ident = name.last_ident().unwrap();
                                let key = format!("{}::{}", prefix, ident.name);
                                self.env
                                    .resolve_def(&key)
                                    .or_else(|| self.env.resolve_def(&ident.name))
                                    .ok_or(TypeError::UndefinedVariable {
                                        name: ident.name.clone(),
                                        span: impl_block.span,
                                    })?
                            }
                        }
                        _ => continue,
                    };

                    for item in &impl_block.items {
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

                            let type_name = match &impl_block.self_ty {
                                Type::Named { name, .. } => name.last_ident().unwrap().name.clone(),
                                _ => "anon".to_string(),
                            };

                            let fq = format!("{}::{}::{}", prefix, type_name, func.name.name);

                            let method_def_id = self.env.insert_def(
                                fq,
                                ItemDef::Function(method_ty.clone()),
                                func.span,
                            )?;

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
        }
        Ok(())
    }
    fn trait_self_ty_t(t: &crate::ast::node::TraitDef) -> Type {
        let span = t.span;
        let name_path = crate::ast::node::Path {
            segments: vec![crate::ast::node::PathSeg::Ident(
                crate::ast::node::Identifier {
                    name: t.name.name.clone(),
                    span,
                },
            )],
            span,
            resolved: None,
        };

        let type_args = t
            .type_params
            .iter()
            .map(|tp| {
                Type::TypeVar(crate::ast::node::Identifier {
                    name: tp.name.name.clone(),
                    span: tp.span,
                })
            })
            .collect::<Vec<_>>();

        Type::Named {
            name: name_path,
            type_args,
        }
    }

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
            Type::Function {
                params,
                return_type,
            } => Type::Function {
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
            other => other,
        }
    }

    fn fulfill_obligations(
        &mut self,
        preds: &[WherePredicate],
        subst: &HashMap<String, Type>,
        _span: Span,
    ) -> TypeResult<()> {
        for pred in preds {
            match pred {
                WherePredicate::Bound {
                    target_ty,
                    bound_ty,
                    span: p_span,
                } => {
                    let concrete_target = self.subst_typevars(target_ty.clone(), subst);
                    let concrete_bound = self.subst_typevars(bound_ty.clone(), subst);
                    if let Type::Named { name, type_args: _ } = concrete_bound {
                        let trait_def_id = self
                            .env
                            .resolve_def(&name.to_string())
                            .ok_or(TypeError::CannotInfer { span: *p_span })?;

                        if self
                            .select_impl(trait_def_id, concrete_target, *p_span)
                            .is_err()
                        {
                            return Err(TypeError::CannotInfer { span: *p_span });
                        }
                    }
                }
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

            Type::Function {
                params,
                return_type,
            } => Type::Function {
                params: params
                    .into_iter()
                    .map(|t| self.subst_typevars(t, subst))
                    .collect(),
                return_type: Box::new(self.subst_typevars(*return_type, subst)),
            },

            Type::Tuple(ts) => Type::Tuple(
                ts.into_iter()
                    .map(|t| self.subst_typevars(t, subst))
                    .collect(),
            ),

            Type::Named { name, type_args } => Type::Named {
                name,
                type_args: type_args
                    .into_iter()
                    .map(|t| self.subst_typevars(t, subst))
                    .collect(),
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
                        if alias_def.typeparams.len() != type_args.len() {
                            return Type::Named { name, type_args };
                        }

                        let subst: HashMap<String, Type> = alias_def
                            .typeparams
                            .iter()
                            .map(|tp| tp.name.name.clone())
                            .zip(type_args.clone())
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
        for _ in 0..32 {
            let next = self.expand_alias_once(ty.clone());
            if next == ty {
                break;
            }
            ty = next;
        }
        ty
    }

    fn path_to_key(path: &Path, current_prefix: &str) -> String {
        let mut idx = 0usize;
        let mut super_count = 0usize;
        let mut absolute = false;
        let mut use_prefix = true;

        loop {
            match path.segments.get(idx) {
                Some(PathSeg::Crate(_)) => {
                    absolute = true;
                    use_prefix = false;
                    idx += 1;
                    break;
                }
                Some(PathSeg::Self_(_)) => {
                    use_prefix = true;
                    idx += 1;
                    break;
                }
                Some(PathSeg::Super(_)) => {
                    super_count += 1;
                    idx += 1;
                    continue;
                }

                Some(PathSeg::Ident(id)) if id.name == "crate" => {
                    absolute = true;
                    use_prefix = false;
                    idx += 1;
                    break;
                }
                Some(PathSeg::Ident(id)) if id.name == "self" => {
                    use_prefix = true;
                    idx += 1;
                    break;
                }
                Some(PathSeg::Ident(id)) if id.name == "super" => {
                    super_count += 1;
                    idx += 1;
                    continue;
                }

                _ => break,
            }
        }

        let mut rest: Vec<String> = Vec::new();
        for seg in path.segments[idx..].iter() {
            if let PathSeg::Ident(id) = seg {
                rest.push(id.name.clone());
            }
        }

        if absolute {
            return rest.join("::");
        }
        let mut prefix = current_prefix;
        for _ in 0..super_count {
            prefix = prefix.rsplit_once("::").map(|(p, _)| p).unwrap_or("");
        }

        if !use_prefix || prefix.is_empty() {
            rest.join("::")
        } else {
            format!("{}::{}", prefix, rest.join("::"))
        }
    }

    pub fn check_function(&mut self, func: &mut crate::ast::node::FunctionDef) -> TypeResult<()> {
        self.env.push_scope();

        let result: TypeResult<()> = (|| {
            for param in &func.params {
                self.env
                    .define(param.name.name.clone(), param.ty.clone())
                    .map_err(|_| TypeError::DuplicateDefinition {
                        name: param.name.name.clone(),
                        span: param.span,
                    })?;
            }

            let body_ty = self.infer_block(&mut func.body)?;

            let expected_ret = func.return_type.clone().unwrap_or(Type::Unit);
            self.unify(body_ty, expected_ret, func.span)?;

            Ok(())
        })();

        self.env.pop_scope();
        result
    }

    pub fn check_impl_method(
        &mut self,
        func: &mut crate::ast::node::FunctionDef,
        self_ty: &Type,
    ) -> TypeResult<()> {
        self.env.push_scope();

        let result: TypeResult<()> = (|| {
            for param in &func.params {
                let param_ty = if param.name.name == "self" {
                    self_ty.clone()
                } else if let Type::Named { name, .. } = &param.ty {
                    if name.last_ident().map(|id| id.name.as_str()) == Some("Self") {
                        self_ty.clone()
                    } else {
                        param.ty.clone()
                    }
                } else {
                    param.ty.clone()
                };

                self.env
                    .define(param.name.name.clone(), param_ty)
                    .map_err(|_| TypeError::DuplicateDefinition {
                        name: param.name.name.clone(),
                        span: param.span,
                    })?;
            }

            let body_ty = self.infer_block(&mut func.body)?;
            let expected_ret = func.return_type.clone().unwrap_or(Type::Unit);
            self.unify(body_ty, expected_ret, func.body.span)?;

            Ok(())
        })();

        self.env.pop_scope();
        result
    }

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
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                if matches!(op, BinaryOp::Add) {
                    if matches!(left_ty, Type::String) || matches!(right_ty, Type::String) {
                        return Ok(Type::String);
                    }
                }

                match (op, &left_ty, &right_ty) {
                    (_, Type::Int, Type::Int) => Ok(Type::Int),
                    (_, Type::Float, Type::Float) => Ok(Type::Float),

                    (_, Type::Float, Type::Int) => Ok(Type::Float),
                    (_, Type::Int, Type::Float) => Ok(Type::Float),

                    (_, Type::Gaussian, _) | (_, _, Type::Gaussian) => {
                        if matches!(op, BinaryOp::Add | BinaryOp::Sub)
                            && left_ty == Type::Gaussian
                            && right_ty == Type::Gaussian
                        {
                            Ok(Type::Gaussian)
                        } else if matches!(op, BinaryOp::Mul | BinaryOp::Div) {
                            match (&left_ty, &right_ty) {
                                (Type::Gaussian, Type::Float) => Ok(Type::Gaussian),
                                (Type::Float, Type::Gaussian) => Ok(Type::Gaussian),
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

                    _ => Err(TypeError::TypeMismatch {
                        expected: left_ty.clone(),
                        found: right_ty,
                        span,
                    }),
                }
            }

            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => {
                let unified = self.unify(left_ty, right_ty, span)?;
                match unified {
                    Type::Int
                    | Type::Float
                    | Type::Bool
                    | Type::String
                    | Type::Any
                    | Type::Unit => Ok(Type::Bool),
                    _ => Err(TypeError::InvalidBinaryOp {
                        op: op.to_string(),
                        left_type: unified.clone(),
                        right_type: unified,
                        span,
                    }),
                }
            }

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

            BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr => {
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

            _ => Ok(Type::Unit),
        }
    }

    pub fn infer_unary(
        &mut self,
        op: &UnaryOp,
        operand: &mut Expression,
        span: Span,
    ) -> TypeResult<Type> {
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
            _ => Ok(ty),
        }
    }

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

    pub fn check_function_definition(
        &mut self,
        name: &str,
        params: &[(String, Type)],
        return_type: &Type,
        body: &mut Expression,
    ) -> TypeResult<()> {
        self.env.push_scope();
        for (param_name, param_ty) in params {
            if self
                .env
                .define(param_name.clone(), param_ty.clone())
                .is_err()
            {
                return Err(TypeError::DuplicateDefinition {
                    name: param_name.clone(),
                    span: body.span(),
                });
            }
        }

        let body_ty = self.infer_expression(body)?;

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

    pub fn infer_function_call(
        &mut self,
        callee: &mut Expression,
        args: &mut Vec<Expression>,
    ) -> TypeResult<Type> {
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

            let receiver_ty = self.infer_expression(&mut args[0])?;
            let receiver_ty = self.apply(receiver_ty);
            let mut trait_def_id = self.resolve_trait_def_from_path(trait_path, *span)?;

            let is_trait = matches!(self.env.get_def(trait_def_id), Some(ItemDef::Trait(_)));
            if !is_trait {
                let base = self.trait_key_from_path(trait_path);
                let fallback = format!("{base}Trait");

                if let Some(fid) = self.env.resolve_def(&fallback) {
                    if matches!(self.env.get_def(fid), Some(ItemDef::Trait(_))) {
                        trait_def_id = fid;
                    }
                }
            }

            if !matches!(self.env.get_def(trait_def_id), Some(ItemDef::Trait(_))) {
                return Err(TypeError::CannotInfer { span: *span });
            }

            let selected = self.select_impl(trait_def_id, receiver_ty.clone(), *span)?;
            let (mut method_ty, method_defid) = selected
                .impl_def
                .methods
                .get(method.name.as_str())
                .ok_or_else(|| TypeError::CannotInfer { span: *span })?
                .clone();

            method_ty = self.subst_typevars(method_ty, &selected.subst);
            method_ty = self.apply(method_ty);

            let mut solved_assoc = selected.assoc_bindings.clone();
            for (_k, v) in solved_assoc.iter_mut() {
                *v = self.apply(self.subst_typevars(v.clone(), &selected.subst));
            }

            method_ty = self.subst_assoc(method_ty, &solved_assoc);
            method_ty = self.apply(method_ty);
            match method_ty {
                Type::Function {
                    params,
                    return_type,
                } => {
                    let is_variadic = params
                        .last()
                        .map_or(false, |t| matches!(t, Type::Variadic(_)));

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

                        let expected_ty = self.apply(expected_ty);

                        self.unify(arg_ty, expected_ty, arg_expr.span())?;
                    }
                    let receiver_expr = args.remove(0);
                    let method_args = std::mem::take(args);

                    *callee = Expression::MethodCall {
                        receiver: Box::new(receiver_expr),
                        method: method.clone(),
                        args: method_args,
                        span: *span,
                        resolved: Some(method_defid),
                    };

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

        if let Expression::Variable { name, .. } = callee {
            let fname = name.last_ident().map(|id| id.name.as_str()).unwrap_or("");

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

        let func_ty = self.infer_expression(callee)?;

        let (param_types, return_ty) = match func_ty.clone() {
            Type::Function {
                params,
                return_type,
            } => (params, *return_type),
            _ => {
                return Err(TypeError::NotCallable {
                    value_type: func_ty,
                    span: callee.span(),
                })
            }
        };

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

    fn check_distribution_parameters(
        &self,
        func_name: &str,
        args: &[Expression],
        span: Span,
    ) -> TypeResult<()> {
        match func_name {
            "Gaussian" => {
                if args.len() >= 2 {
                    if let Expression::Literal {
                        value: Literal::Float(std),
                        ..
                    } = &args[1]
                    {
                        if *std <= 0.0 {
                            return Err(TypeError::InvalidDistributionParameter {
                                distribution: "Gaussian".to_string(),
                                param_name: "std".to_string(),
                                reason: format!("standard deviation must be positive, got {}", std),
                                span,
                            });
                        }
                    }
                    if let Expression::Literal {
                        value: Literal::Int(std),
                        ..
                    } = &args[1]
                    {
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
                if args.len() >= 2 {
                    let min_val = Self::extract_float_literal(&args[0]);
                    let max_val = Self::extract_float_literal(&args[1]);

                    if let (Some(min), Some(max)) = (min_val, max_val) {
                        if min >= max {
                            return Err(TypeError::InvalidDistributionParameter {
                                distribution: "Uniform".to_string(),
                                param_name: "min, max".to_string(),
                                reason: format!(
                                    "min must be less than max, got min={}, max={}",
                                    min, max
                                ),
                                span,
                            });
                        }
                    }
                }
            }
            "Beta" => {
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

    fn extract_float_literal(expr: &Expression) -> Option<f64> {
        match expr {
            Expression::Literal {
                value: Literal::Float(v),
                ..
            } => Some(*v),
            Expression::Literal {
                value: Literal::Int(v),
                ..
            } => Some(*v as f64),
            Expression::Unary {
                op: crate::ast::node::UnaryOp::Neg,
                operand,
                ..
            } => Self::extract_float_literal(operand).map(|v| -v),
            _ => None,
        }
    }
}
