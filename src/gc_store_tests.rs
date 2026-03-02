#[cfg(test)]
mod tests {
    use crate::ad::types::ADFloat;
    use crate::vm::region::{extract_handles, ContainerHandle, ContainerStore};
    use crate::vm::value::Value;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_sweep_removes_unreachable() {
        let mut store = ContainerStore::new();

        let h1 = store.alloc_array(vec![Value::Int(1), Value::Int(2)]);
        let h2 = store.alloc_array(vec![Value::Int(3)]);
        let h3 = store.alloc_dict(HashMap::new());

        assert_eq!(store.handle_count(), 3);

        // h1 のみを到達可能とする
        let reachable: HashSet<ContainerHandle> = [h1].into_iter().collect();
        store.sweep(&reachable);

        assert_eq!(store.handle_count(), 1);
        assert!(store.contains(h1));
        assert!(!store.contains(h2));
        assert!(!store.contains(h3));
    }

    #[test]
    fn test_sweep_all_unreachable() {
        let mut store = ContainerStore::new();

        store.alloc_array(vec![Value::Int(10)]);
        store.alloc_array(vec![Value::Int(20)]);
        store.alloc_dict(HashMap::new());

        assert_eq!(store.handle_count(), 3);

        let reachable: HashSet<ContainerHandle> = HashSet::new();
        store.sweep(&reachable);

        assert_eq!(store.handle_count(), 0);
    }

    #[test]
    fn test_trace_follows_nested_containers() {
        let mut store = ContainerStore::new();

        let inner = store.alloc_array(vec![Value::Int(42)]);
        let outer = store.alloc_array(vec![Value::Array(inner)]);
        let _orphan = store.alloc_array(vec![Value::Int(99)]);

        assert_eq!(store.handle_count(), 3);

        // outer をルートとして走査 → inner にも到達可能
        let reachable = store.trace_from(&[outer]);
        store.sweep(&reachable);

        assert_eq!(store.handle_count(), 2);
        assert!(store.contains(outer));
        assert!(store.contains(inner));
    }

    #[test]
    fn test_trace_handles_cycles() {
        let mut store = ContainerStore::new();

        // 循環参照を作る: arr_a -> arr_b -> arr_a
        let h_a = store.alloc_array(vec![Value::Int(1)]);
        let h_b = store.alloc_array(vec![Value::Array(h_a)]);
        // arr_a の中身を arr_b を指すように更新 (循環)
        store.with_array_mut(h_a, |arr| {
            arr.push(Value::Array(h_b));
        });

        let _orphan = store.alloc_array(vec![Value::Int(999)]);
        assert_eq!(store.handle_count(), 3);

        // h_a をルートに走査 → h_b にも到達可能、循環しても無限ループしない
        let reachable = store.trace_from(&[h_a]);
        assert!(reachable.contains(&h_a));
        assert!(reachable.contains(&h_b));
        assert_eq!(reachable.len(), 2);

        store.sweep(&reachable);
        assert_eq!(store.handle_count(), 2);
        assert!(store.contains(h_a));
        assert!(store.contains(h_b));
    }

    #[test]
    fn test_sweep_collects_cyclic_garbage() {
        let mut store = ContainerStore::new();

        // 循環参照を作るが、どこからもルートとして参照されていない
        let h_a = store.alloc_array(vec![Value::Int(1)]);
        let h_b = store.alloc_array(vec![Value::Array(h_a)]);
        store.with_array_mut(h_a, |arr| {
            arr.push(Value::Array(h_b));
        });

        // 別の到達可能なコンテナ
        let h_live = store.alloc_array(vec![Value::Int(42)]);

        assert_eq!(store.handle_count(), 3);

        // h_live のみがルート → 循環参照している h_a, h_b は回収される
        let reachable = store.trace_from(&[h_live]);
        store.sweep(&reachable);

        assert_eq!(store.handle_count(), 1);
        assert!(store.contains(h_live));
        assert!(
            !store.contains(h_a),
            "Cyclic garbage h_a should be collected"
        );
        assert!(
            !store.contains(h_b),
            "Cyclic garbage h_b should be collected"
        );
    }

    #[test]
    fn test_trace_through_struct_and_enum() {
        let mut store = ContainerStore::new();

        let inner_arr = store.alloc_array(vec![Value::Float(ADFloat::Concrete(3.14))]);
        let mut fields = HashMap::new();
        fields.insert("data".to_string(), Value::Array(inner_arr));
        let struct_handle = store.alloc_dict(fields);

        let _orphan = store.alloc_dict(HashMap::new());

        assert_eq!(store.handle_count(), 3);

        // struct_handle をルートに → 中の inner_arr にも到達可能
        let reachable = store.trace_from(&[struct_handle]);
        store.sweep(&reachable);

        assert_eq!(store.handle_count(), 2);
        assert!(store.contains(struct_handle));
        assert!(store.contains(inner_arr));
    }

    #[test]
    fn test_extract_handles_from_value() {
        let h1: ContainerHandle = 10;
        let h2: ContainerHandle = 20;

        let val = Value::Tuple(crate::gc::Rc::new(vec![
            Value::Array(h1),
            Value::Int(5),
            Value::Map(h2),
        ]));

        let handles = extract_handles(&val);
        assert!(handles.contains(&h1));
        assert!(handles.contains(&h2));
        assert_eq!(handles.len(), 2);
    }

    #[test]
    fn test_gc_threshold_trigger() {
        let mut store = ContainerStore::new();

        assert!(!store.should_gc());

        // 閾値(256)まで割り当て続ける
        for i in 0..256 {
            store.alloc_array(vec![Value::Int(i)]);
        }

        assert!(store.should_gc(), "GC should trigger after 256 allocations");

        // sweep (全部到達可能)
        let all: HashSet<ContainerHandle> = (1..=256).collect();
        store.sweep(&all);

        assert!(!store.should_gc(), "GC should reset after sweep");
    }
}
