// src/typeck/prob.rs

use crate::ast::node::{Type, BinaryOp, Span};
use crate::typeck::{TypeError, TypeResult};

pub fn infer_gaussian_binop(left_ty: &Type, op: &BinaryOp, right_ty: &Type, span: Span) -> TypeResult<Type> {
    match (left_ty, op, right_ty) {
        (Type::Gaussian, BinaryOp::Add, Type::Gaussian) => Ok(Type::Gaussian),
        (Type::Gaussian, BinaryOp::Mul, Type::Float) => Ok(Type::Gaussian),
        (Type::Float, BinaryOp::Mul, Type::Gaussian) => Ok(Type::Gaussian),
        (Type::Gaussian, BinaryOp::Sub, Type::Gaussian) => Ok(Type::Gaussian),
        (Type::Gaussian, op, other) | (other, op, Type::Gaussian) => {
            Err(TypeError::InvalidBinaryOp(op.clone(), format!("Unsupported operation between Gaussian and {:?}", other), span))
        }
        _ => Err(TypeError::InvalidBinaryOp(op.clone(), "Non-Gaussian type in Gaussian operation".into(), span)),
    }
}

pub fn infer_distribution_method(method: &str, receiver_ty: &Type, args: &[Type], span: Span) -> TypeResult<Type> {
    match receiver_ty {
        Type::Gaussian => infer_gaussian_method_impl(method, args, span),
        Type::Uniform => infer_uniform_method(method, args, span),
        Type::Bernoulli => infer_bernoulli_method(method, args, span),
        Type::Beta => infer_beta_method(method, args, span),
        _ => Err(TypeError::UnknownMethod("Method call on non-distribution type".into(), method.into(), span)),
    }
}

fn infer_gaussian_method_impl(method: &str, args: &[Type], span: Span) -> TypeResult<Type> {
    match method {
        "pdf" | "cdf" => {
            if args.len() == 1 && matches!(args[0], Type::Float) {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument("Gaussian pdf/cdf requires exactly one Float argument".into(), span))
            }
        }
        "sample" => {
            if args.is_empty() {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument("sample takes no arguments".into(), span))
            }
        }
        "clone" => {
            if args.is_empty() {
                Ok(Type::Gaussian)
            } else {
                Err(TypeError::InvalidArgument("clone takes no arguments".into(), span))
            }
        }
        _ => Err(TypeError::UnknownMethod("Gaussian".into(), method.into(), span)),
    }
}

fn infer_uniform_method(method: &str, args: &[Type], span: Span) -> TypeResult<Type> {
    match method {
        "pdf" | "cdf" => {
            if args.len() == 1 && matches!(args[0], Type::Float) {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument("Uniform pdf/cdf requires exactly one Float argument".into(), span))
            }
        }
        "sample" => {
            if args.is_empty() {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument("sample takes no arguments".into(), span))
            }
        }
        "clone" => {
            if args.is_empty() {
                Ok(Type::Uniform)
            } else {
                Err(TypeError::InvalidArgument("clone takes no arguments".into(), span))
            }
        }
        _ => Err(TypeError::UnknownMethod("Uniform".into(), method.into(), span)),
    }
}

fn infer_bernoulli_method(method: &str, args: &[Type], span: Span) -> TypeResult<Type> {
    match method {
        "pmf" => {
            if args.len() == 1 && matches!(args[0], Type::Bool | Type::Int) {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument("Bernoulli pmf requires exactly one Bool or Int argument".into(), span))
            }
        }
        "sample" => {
            if args.is_empty() {
                Ok(Type::Bool)
            } else {
                Err(TypeError::InvalidArgument("sample takes no arguments".into(), span))
            }
        }
        "clone" => {
            if args.is_empty() {
                Ok(Type::Bernoulli)
            } else {
                Err(TypeError::InvalidArgument("clone takes no arguments".into(), span))
            }
        }
        _ => Err(TypeError::UnknownMethod("Bernoulli".into(), method.into(), span)),
    }
}

fn infer_beta_method(method: &str, args: &[Type], span: Span) -> TypeResult<Type> {
    match method {
        "pdf" => {
            if args.len() == 1 && matches!(args[0], Type::Float) {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument("Beta pdf requires exactly one Float argument".into(), span))
            }
        }
        "sample" => {
            if args.is_empty() {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument("sample takes no arguments".into(), span))
            }
        }
        "clone" => {
            if args.is_empty() {
                Ok(Type::Beta)
            } else {
                Err(TypeError::InvalidArgument("clone takes no arguments".into(), span))
            }
        }
        _ => Err(TypeError::UnknownMethod("Beta".into(), method.into(), span)),
    }
}

pub fn infer_gaussian_method(method: &str, receiver_ty: &Type, args: &[Type], span: Span) -> TypeResult<Type> {
    if receiver_ty != &Type::Gaussian {
        return Err(TypeError::UnknownMethod("Method call on non-Gaussian type".into(), method.into(), span));
    }
    infer_gaussian_method_impl(method, args, span)
}
