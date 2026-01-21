// Gaussian確率型の演算・型推論専用モジュール

use crate::ast::node::{Type, Expression, BinaryOp, Span};
use crate::typeck::{TypeError, TypeResult};

// Gaussian型の二項演算・メソッド型推論ルール
pub fn infer_gaussian_binop(left_ty: &Type, op: &BinaryOp, right_ty: &Type, span: Span) -> TypeResult<Type> {
    match (left_ty, op, right_ty) {
        (Type::Gaussian, BinaryOp::Add, Type::Gaussian) => Ok(Type::Gaussian),
        (Type::Gaussian, BinaryOp::Mul, Type::Float) => Ok(Type::Gaussian),
        (Type::Float, BinaryOp::Mul, Type::Gaussian) => Ok(Type::Gaussian),
        (Type::Gaussian, BinaryOp::Sub, Type::Gaussian) => Ok(Type::Gaussian),
        (Type::Gaussian, op, other) | (other, op, Type::Gaussian) => {
            Err(TypeError::InvalidBinaryOp(op.clone(), format!("Gaussianと{:?}の演算は未対応", other), span))
        }
        _ => Err(TypeError::InvalidBinaryOp(op.clone(), "非Gaussian型にGaussian推論".into(), span)),
    }
}

// Gaussian型のメソッド呼び出し型検査
pub fn infer_gaussian_method(method: &str, receiver_ty: &Type, args: &[Type], span: Span) -> TypeResult<Type> {
    if receiver_ty != &Type::Gaussian {
        return Err(TypeError::UnknownMethod("Gaussianでない型にメソッド呼び出し".into(), method.into(), span));
    }
    match method {
        "pdf" | "cdf" => {
            if args.len() == 1 && matches!(args[0], Type::Float) {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument("Gaussianのpdf/cdfはFloat引数1つのみ".into(), span))
            }
        }
        "sample" => {
            if args.is_empty() {
                Ok(Type::Float)
            } else {
                Err(TypeError::InvalidArgument("sampleは引数なし".into(), span))
            }
        }
        _ => Err(TypeError::UnknownMethod("Gaussian未定義メソッド".into(), method.into(), span)),
    }
}
