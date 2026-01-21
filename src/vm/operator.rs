use crate::ast::operator::{OperatorTrait, OverloadNode};
use crate::vm::value::Value;

pub fn eval_overload_node(node: &OverloadNode, vm: &mut Interpreter) -> RuntimeResult<Value> {
    let lhs_val = vm.eval_expression(&node.lhs)?;
    let rhs_val = vm.eval_expression(&node.rhs)?;

    // もしプリミティブ型(Int, Float)なら高速分岐
    match (node.trait_kind, &lhs_val, &rhs_val) {
        (OperatorTrait::Add, Value::Int(a), Value::Int(b)) => return Ok(Value::Int(a + b)),
        (OperatorTrait::Add, Value::Float(a), Value::Float(b)) => return Ok(Value::Float(a + b)),
        // ... 他の標準型デフォルト...
        _ => {}
    }

    // 自作型やPoint/Gaussianなどはトレイトテーブルを参照してメソッド呼び出し
    if let Some(method) = vm.env.lookup_trait_method(&lhs_val, node.trait_kind.method_name()) {
        return method.call(vec![lhs_val, rhs_val], vm);
    }

    Err(RuntimeError::NoOperatorImpl {
        op: node.trait_kind,
        ty: lhs_val.type_name(),
        span: node.span,
    })
}
