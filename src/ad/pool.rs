// src/ad/pool.rs

use ndarray::ArrayD;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static TENSOR_POOL: RefCell<TensorPool> = RefCell::new(TensorPool::new());
}

struct TensorPool {
    pool: HashMap<Vec<usize>, Vec<ArrayD<f64>>>,
}

impl TensorPool {
    fn new() -> Self {
        Self {
            pool: HashMap::new(),
        }
    }

    fn get(&mut self, shape: &[usize], init_val: f64) -> ArrayD<f64> {
        if let Some(vec) = self.pool.get_mut(shape) {
            if let Some(mut arr) = vec.pop() {
                arr.fill(init_val);
                return arr;
            }
        }
        ArrayD::from_elem(shape, init_val)
    }

    fn recycle(&mut self, arr: ArrayD<f64>) {
        let shape = arr.shape().to_vec();
        let entry = self.pool.entry(shape).or_insert_with(Vec::new);
        if entry.len() < 20 {
            entry.push(arr);
        }
    }
}

pub fn get_pooled_tensor(shape: &[usize], init_val: f64) -> ArrayD<f64> {
    TENSOR_POOL.with(|pool| pool.borrow_mut().get(shape, init_val))
}

pub fn recycle_tensor(tensor: ArrayD<f64>) {
    TENSOR_POOL.with(|pool| pool.borrow_mut().recycle(tensor))
}
