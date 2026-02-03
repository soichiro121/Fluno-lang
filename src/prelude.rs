// src/prelude.rs

use crate::vm::prob_runtime::{ProbContext, InferenceMode};
use std::cell::RefCell;

thread_local! {
    pub static NATIVE_CONTEXT: RefCell<Option<ProbContext>> = RefCell::new(None);
}

pub use crate::ad::types::ADFloat;
pub use crate::vm::distributions::{Gaussian, Uniform, Bernoulli, Beta, DistributionTrait};
pub use std::ops::{Add, Sub, Mul, Div};
pub use std::rc::Rc; 

pub fn sample<D: DistributionTrait>(dist: D) -> ADFloat {
    let result = NATIVE_CONTEXT.with(|ctx| {
        let mut ctx_borrow = ctx.borrow_mut();
        if let Some(context) = ctx_borrow.as_mut() {
            let id = context.sample_counter;
            context.sample_counter += 1;
            let addr = format!("sample_{}", id);

            if let Some(val) = context.trace.get(&addr) {
                 if let crate::vm::Value::Float(ad_val) = val {
                     return Some(ad_val.clone());
                 }
            }

            let mut rng = rand::thread_rng();
            let sample_val = match context.mode {
                InferenceMode::Sampling => ADFloat::Concrete(dist.sample(&mut rng).expect("Sample failed")),
                _ => dist.sample_ad(&mut rng).expect("SampleAD failed"),
            };

            context.trace.insert(addr, crate::vm::Value::Float(sample_val.clone()));

            let log_p = dist.log_pdf(&sample_val);
            context.accumulated_log_prob = context.accumulated_log_prob.clone() + log_p;

            Some(sample_val)
        } else {
            None
        }
    });

    if let Some(val) = result {
        val
    } else {
        let mut rng = rand::thread_rng();
        ADFloat::Concrete(dist.sample(&mut rng).expect("Runtime Error during sampling"))
    }
}

pub fn observe<D: DistributionTrait>(dist: D, value: ADFloat) {
    NATIVE_CONTEXT.with(|ctx| {
        if let Some(c) = ctx.borrow_mut().as_mut() {
             let log_p = dist.log_pdf(&value);
             c.accumulated_log_prob = c.accumulated_log_prob.clone() + log_p;
        }
    });
}

#[allow(non_snake_case)]
pub fn Map() -> std::collections::HashMap<String, crate::vm::Value> {
    std::collections::HashMap::new()
}

pub fn clone<T: Clone>(x: T) -> T {
    x.clone()
}

#[allow(non_snake_case)]
pub fn Gaussian<T: Into<ADFloat>, U: Into<ADFloat>>(mean: T, std: U) -> Gaussian {
    Gaussian {
        mean: mean.into(),
        std: std.into(),
    }
}

#[allow(non_snake_case)]
pub fn Uniform<T: Into<ADFloat>, U: Into<ADFloat>>(min: T, max: U) -> crate::vm::distributions::Uniform {
    crate::vm::distributions::Uniform {
        min: min.into(),
        max: max.into(),
    }
}

#[allow(non_snake_case)]
pub fn Beta<T: Into<ADFloat>, U: Into<ADFloat>>(alpha: T, beta: U) -> crate::vm::distributions::Beta {
    crate::vm::distributions::Beta {
        alpha: alpha.into(),
        beta: beta.into(),
    }
}

#[allow(non_snake_case)]
pub fn Bernoulli<T: Into<ADFloat>>(p: T) -> crate::vm::distributions::Bernoulli {
    crate::vm::distributions::Bernoulli {
        p: p.into(),
    }
}

pub fn println(s: impl std::fmt::Display) {
    std::println!("{}", s);
}

pub fn infer_hmc<F>(_config: std::collections::HashMap<String, crate::vm::Value>, model: F) -> Vec<std::collections::HashMap<String, ADFloat>>
where F: Fn() -> ADFloat 
{
    let _ = model();
    Vec::new()
}

#[derive(Clone)]
pub struct Signal<T> {
    compute: Rc<dyn Fn() -> T>,
}

impl<T: Clone + 'static> Signal<T> {
    pub fn new(val: T) -> Self {
        Signal { compute: Rc::new(move || val.clone()) }
    }

    pub fn get(&self) -> T {
        (self.compute)()
    }

    pub fn map<U: Clone + 'static, F: Fn(T) -> U + 'static>(&self, f: F) -> Signal<U> {
        let prev = self.compute.clone();
        Signal {
            compute: Rc::new(move || f(prev()))
        }
    }
}

#[allow(non_snake_case)]
pub fn Signal_new<T: Clone + 'static>(val: T) -> Signal<T> {
    Signal::new(val)
}
