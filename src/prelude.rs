use crate::vm::prob_runtime::{ProbContext, InferenceMode};
use std::cell::RefCell;

// NATIVE_CONTEXT for thread-local probabilistic state
thread_local! {
    pub static NATIVE_CONTEXT: RefCell<Option<ProbContext>> = RefCell::new(None);
}

pub use crate::ad::types::ADFloat;
pub use crate::vm::distributions::{Gaussian, Uniform, Bernoulli, Beta, DistributionTrait};
pub use std::ops::{Add, Sub, Mul, Div};
pub use std::rc::Rc; 

pub fn sample<D: DistributionTrait>(dist: D) -> ADFloat {
    // Check if we are in a probabilistic context
    let result = NATIVE_CONTEXT.with(|ctx| {
        let mut ctx_borrow = ctx.borrow_mut();
        if let Some(context) = ctx_borrow.as_mut() {
            let id = context.sample_counter;
            context.sample_counter += 1;
            let addr = format!("sample_{}", id);

            // Trace lookup
            if let Some(val) = context.trace.get(&addr) {
                 if let crate::vm::Value::Float(ad_val) = val {
                     return Some(ad_val.clone());
                 }
            }

            // Sample from distribution
            let mut rng = rand::thread_rng();
            let sample_val = match context.mode {
                InferenceMode::Sampling => ADFloat::Concrete(dist.sample(&mut rng).expect("Sample failed")),
                _ => dist.sample_ad(&mut rng).expect("SampleAD failed"),
            };

            // Store in trace
            context.trace.insert(addr, crate::vm::Value::Float(sample_val.clone()));

            // Update log probability
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

pub fn println(s: impl std::fmt::Display) {
    std::println!("{}", s);
}

// Stub infer_hmc with proper signature
pub fn infer_hmc<F>(_config: std::collections::HashMap<String, crate::vm::Value>, model: F) -> Vec<std::collections::HashMap<String, ADFloat>>
where F: Fn() -> ADFloat 
{
    // Minimal logic to execute model once so we can verify compilation
    let _ = model();
    Vec::new()
}
