// src/ad/mod.rs

pub mod backend;
pub mod backward;
pub mod cpu_backend;
pub mod graph;
pub mod optimizer;
pub mod pool;
pub mod tensor;
pub mod types;

use crate::ad::graph::Tape;
pub use backward::backward;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

pub type ForwardFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;
pub type VjpFn = Box<dyn Fn(&[f64], f64, f64) -> Vec<f64> + Send + Sync>;

pub struct CustomVjpDef {
    pub forward: ForwardFn,
    pub vjp: VjpFn,
}

static VJP_REGISTRY: LazyLock<RwLock<HashMap<String, CustomVjpDef>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

pub fn register_custom_vjp<F, V>(name: &str, forward: F, vjp: V)
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    V: Fn(&[f64], f64, f64) -> Vec<f64> + Send + Sync + 'static,
{
    let mut registry = VJP_REGISTRY.write().unwrap();
    registry.insert(
        name.to_string(),
        CustomVjpDef {
            forward: Box::new(forward),
            vjp: Box::new(vjp),
        },
    );
}

pub fn get_vjp(
    _name: &str,
) -> Option<std::sync::RwLockReadGuard<'static, HashMap<String, CustomVjpDef>>> {
    Some(VJP_REGISTRY.read().unwrap())
}

pub fn call_vjp(name: &str, inputs: &[f64], output: f64, grad_output: f64) -> Option<Vec<f64>> {
    let registry = VJP_REGISTRY.read().unwrap();
    registry
        .get(name)
        .map(|def| (def.vjp)(inputs, output, grad_output))
}

pub fn call_forward(name: &str, inputs: &[f64]) -> Option<f64> {
    let registry = VJP_REGISTRY.read().unwrap();
    registry.get(name).map(|def| (def.forward)(inputs))
}

thread_local! {
    static TAPE_STORAGE: RefCell<HashMap<usize, Tape>> = RefCell::new(HashMap::new());
    static NEXT_TAPE_ID: RefCell<usize> = RefCell::new(1);
}

pub fn create_tape() -> usize {
    TAPE_STORAGE.with(|storage| {
        let mut map = storage.borrow_mut();
        let id = NEXT_TAPE_ID.with(|n| {
            let mut i = n.borrow_mut();
            let res = *i;
            *i += 1;
            res
        });
        map.insert(id, Tape::new());
        id
    })
}

pub fn remove_tape(tape_id: usize) {
    TAPE_STORAGE.with(|storage| {
        storage.borrow_mut().remove(&tape_id);
    });
}

pub fn with_tape<F, R>(tape_id: usize, f: F) -> R
where
    F: FnOnce(&Tape) -> R,
{
    TAPE_STORAGE.with(|storage| {
        let map = storage.borrow();
        let tape = map
            .get(&tape_id)
            .expect("Tape accessing error: Tape ID not found in current thread storage.");
        f(tape)
    })
}
