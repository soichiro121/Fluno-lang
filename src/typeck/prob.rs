// src/typeck/prob.rs

use crate::ast::node::{BinaryOp, Span, Type};
use crate::typeck::error::{TypeError, TypeResult};

pub fn infer_gaussian_binop(
    left_ty: &Type,
    op: &BinaryOp,
    right_ty: &Type,
    span: Span,
) -> TypeResult<Type> {
    match (left_ty, op, right_ty) {
        (Type::Gaussian, BinaryOp::Add, Type::Gaussian) => Ok(Type::Gaussian),
        (Type::Gaussian, BinaryOp::Mul, Type::Float) => Ok(Type::Gaussian),
        (Type::Float, BinaryOp::Mul, Type::Gaussian) => Ok(Type::Gaussian),
        (Type::Gaussian, BinaryOp::Sub, Type::Gaussian) => Ok(Type::Gaussian),
        (Type::Gaussian, op, other) | (other, op, Type::Gaussian) => {
            Err(TypeError::InvalidBinaryOp {
                op: format!("{:?}", op),
                left_type: Type::Gaussian,
                right_type: other.clone(),
                span,
            })
        }
        _ => Err(TypeError::InvalidBinaryOp {
            op: format!("{:?}", op),
            left_type: left_ty.clone(),
            right_type: right_ty.clone(),
            span,
        }),
    }
}

pub fn infer_distribution_method(
    method: &str,
    receiver_ty: &Type,
    args: &[Type],
    span: Span,
) -> TypeResult<Type> {
    match receiver_ty {
        Type::Gaussian => infer_gaussian_method_impl(method, args, span),
        Type::Uniform => infer_uniform_method(method, args, span),
        Type::Bernoulli => infer_bernoulli_method(method, args, span),
        Type::Beta => infer_beta_method(method, args, span),
        Type::VonMises => infer_von_mises_method(method, args, span),
        _ => Err(TypeError::UnknownMethod {
            type_name: "Method call on non-distribution type".into(),
            method: method.into(),
            span,
        }),
    }
}

fn infer_gaussian_method_impl(method: &str, args: &[Type], span: Span) -> TypeResult<Type> {
    match method {
        "pdf" | "cdf" => {
            if args.len() == 1 && matches!(args[0], Type::Float) {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "Gaussian pdf/cdf requires exactly one Float argument".into(),
                    span,
                })
            }
        }
        "sample" => {
            if args.is_empty() {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "sample takes no arguments".into(),
                    span,
                })
            }
        }
        "clone" => {
            if args.is_empty() {
                Ok(Type::Gaussian)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "clone takes no arguments".into(),
                    span,
                })
            }
        }
        _ => Err(TypeError::UnknownMethod {
            type_name: "Gaussian".into(),
            method: method.into(),
            span,
        }),
    }
}

fn infer_uniform_method(method: &str, args: &[Type], span: Span) -> TypeResult<Type> {
    match method {
        "pdf" | "cdf" => {
            if args.len() == 1 && matches!(args[0], Type::Float) {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "Uniform pdf/cdf requires exactly one Float argument".into(),
                    span,
                })
            }
        }
        "sample" => {
            if args.is_empty() {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "sample takes no arguments".into(),
                    span,
                })
            }
        }
        "clone" => {
            if args.is_empty() {
                Ok(Type::Uniform)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "clone takes no arguments".into(),
                    span,
                })
            }
        }
        _ => Err(TypeError::UnknownMethod {
            type_name: "Uniform".into(),
            method: method.into(),
            span,
        }),
    }
}

fn infer_bernoulli_method(method: &str, args: &[Type], span: Span) -> TypeResult<Type> {
    match method {
        "pmf" => {
            if args.len() == 1 && matches!(args[0], Type::Bool | Type::Int) {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "Bernoulli pmf requires exactly one Bool or Int argument".into(),
                    span,
                })
            }
        }
        "sample" => {
            if args.is_empty() {
                Ok(Type::Bool)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "sample takes no arguments".into(),
                    span,
                })
            }
        }
        "clone" => {
            if args.is_empty() {
                Ok(Type::Bernoulli)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "clone takes no arguments".into(),
                    span,
                })
            }
        }
        _ => Err(TypeError::UnknownMethod {
            type_name: "Bernoulli".into(),
            method: method.into(),
            span,
        }),
    }
}

fn infer_beta_method(method: &str, args: &[Type], span: Span) -> TypeResult<Type> {
    match method {
        "pdf" => {
            if args.len() == 1 && matches!(args[0], Type::Float) {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "Beta pdf requires exactly one Float argument".into(),
                    span,
                })
            }
        }
        "sample" => {
            if args.is_empty() {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "sample takes no arguments".into(),
                    span,
                })
            }
        }
        "clone" => {
            if args.is_empty() {
                Ok(Type::Beta)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "clone takes no arguments".into(),
                    span,
                })
            }
        }
        _ => Err(TypeError::UnknownMethod {
            type_name: "Beta".into(),
            method: method.into(),
            span,
        }),
    }
}

fn infer_von_mises_method(method: &str, args: &[Type], span: Span) -> TypeResult<Type> {
    match method {
        "pdf" | "log_pdf" => {
            if args.len() == 1 && matches!(args[0], Type::Float) {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "VonMises pdf/log_pdf requires exactly one Float argument".into(),
                    span,
                })
            }
        }
        "sample" => {
            if args.is_empty() {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "sample takes no arguments".into(),
                    span,
                })
            }
        }
        "clone" => {
            if args.is_empty() {
                Ok(Type::VonMises)
            } else {
                Err(TypeError::InvalidArgument {
                    message: "clone takes no arguments".into(),
                    span,
                })
            }
        }
        _ => Err(TypeError::UnknownMethod {
            type_name: "VonMises".into(),
            method: method.into(),
            span,
        }),
    }
}

pub fn infer_gaussian_method(
    method: &str,
    receiver_ty: &Type,
    args: &[Type],
    span: Span,
) -> TypeResult<Type> {
    if receiver_ty != &Type::Gaussian {
        return Err(TypeError::UnknownMethod {
            type_name: "Method call on non-Gaussian type".into(),
            method: method.into(),
            span,
        });
    }
    infer_gaussian_method_impl(method, args, span)
}
