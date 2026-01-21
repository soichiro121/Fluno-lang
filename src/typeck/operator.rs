use crate::ast::operator::{OperatorTrait, OverloadNode};
use crate::ast::node::*;

pub fn check_overload_node(node: &OverloadNode, ctx: &TypeContext) -> TypeResult<Type> {
    let lhs_ty = ctx.infer_expr_type(&node.lhs)?;
    let rhs_ty = ctx.infer_expr_type(&node.rhs)?;

    // 型に対応する演算子トレイトが実装されているか
    if !ctx.env.type_implements_trait(&lhs_ty, node.trait_kind.trait_name()) {
        return Err(TypeError::NoOperatorTrait {
            op: node.trait_kind,
            ty: lhs_ty.clone(),
            span: node.span,
        });
    }

    // メソッドの戻り値型推論
    let result_ty = ctx.env.trait_method_return_type(&lhs_ty, node.trait_kind.method_name(), &[rhs_ty])?;
    Ok(result_ty)
}
