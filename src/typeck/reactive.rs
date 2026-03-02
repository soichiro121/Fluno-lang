// src/typeck/reactive.rs

use crate::ast::node::{Type, Span};
use crate::typeck::{TypeError, TypeResult};

pub fn infer_signal_static(method: &str, args: &[Type], span: Span) -> TypeResult<Type> {
    match method {
        "new" => {
            if args.len() != 1 {
                return Err(TypeError::ArgumentMismatch { expected: 1, found: args.len(), span });
            }
            Ok(Type::Signal(Box::new(args[0].clone())))
        }
        "combine" => {
            if args.len() != 3 {
                return Err(TypeError::ArgumentMismatch { expected: 3, found: args.len(), span });
            }
            let type_a = match &args[0] {
                Type::Signal(inner) => inner,
                t => return Err(TypeError::TypeMismatch { message: format!("Arg 1 must be Signal, got {:?}", t) }),
            };
            let type_b = match &args[1] {
                Type::Signal(inner) => inner,
                t => return Err(TypeError::TypeMismatch { message: format!("Arg 2 must be Signal, got {:?}", t) }),
            };
            match &args[2] {
                Type::Function { params, return_type } => {
                    if params.len() != 2 {
                        return Err(TypeError::TypeMismatch {
                            message: "Combiner function must take 2 arguments".into(),
                        });
                    }

                    if params[0] != *type_a.clone() || params[1] != *type_b.clone() {
                        return Err(TypeError::TypeMismatch {
                            message: "Combiner function arg types mismatch signal types".into(),
                        });
                    }

                    Ok(Type::Signal(return_type.clone()))
                }
                t => Err(TypeError::TypeMismatch {
                    message: format!("Arg 3 must be Function, got {:?}", t),
                }),
            }

        }
        _ => Err(TypeError::UnknownMethod { method: method.into(), receiver: "Signal".into(), span }),
    }
}

pub fn infer_signal_method(method: &str, receiver_inner_ty: &Type, args: &[Type], span: Span) -> TypeResult<Type> {
    match method {
        "map" => {
            if args.len() != 1 {
                return Err(TypeError::ArgumentMismatch { expected: 1, found: args.len(), span });
            }
            match &args[0] {
                Type::Function { params, return_type } => {
                    if params.len() != 1 || params[0] != *receiver_inner_ty {
                        return Err(TypeError::TypeMismatch {
                            message: "Map function arg type mismatch".into(),
                        });
                    }
                    Ok(Type::Signal(return_type.clone()))
                }
                t => Err(TypeError::TypeMismatch {
                    message: format!("Map arg must be Function, got {:?}", t),
                }),
            }

        }
        "filter" => {
            if args.len() != 1 {
                return Err(TypeError::ArgumentMismatch { expected: 1, found: args.len(), span });
            }
            match &args[0] {
                Type::Function { params, return_type } => {
                    if params.len() != 1 || params[0] != *receiver_inner_ty {
                        return Err(TypeError::TypeMismatch {
                            message: "Filter function arg type mismatch".into(),
                        });
                    }
                    if return_type.as_ref() != &Type::Bool {
                        return Err(TypeError::TypeMismatch {
                            message: "Filter function must return Bool".into(),
                        });
                    }
                    Ok(Type::Signal(Box::new(receiver_inner_ty.clone())))
                }
                t => Err(TypeError::TypeMismatch {
                    message: format!("Filter arg must be Function, got {:?}", t),
                }),
            }

        }
        _ => Err(TypeError::UnknownMethod { method: method.into(), receiver: "Signal".into(), span }),
    }
}
