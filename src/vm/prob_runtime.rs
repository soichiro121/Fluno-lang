// src/vm/prob_runtime.rs

use crate::ad::types::ADFloat;
use crate::ast::node::Block;
use crate::vm::value::Value;
use crate::vm::RuntimeResult;
use crate::vm::{Interpreter, RuntimeError};
use rand::Rng;
use rand::RngCore;
use std::collections::HashMap;

pub trait DistributionImpl {
    fn sample_rng<R: Rng + RngCore>(&self, rng: &mut R) -> f64;
    fn log_pdf(&self, x: f64) -> f64;
    fn log_pdf_ad(&self, x: &ADFloat) -> ADFloat;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceMode {
    Sampling,
    Gradient { tape_id: usize },
    Guide { tape_id: usize },
}
#[derive(Debug, Clone)]
pub struct ProbContext {
    pub mode: InferenceMode,
    pub trace: HashMap<String, Value>,
    pub sample_counter: usize,

    pub tape_id: usize,

    pub accumulated_log_prob: ADFloat,
    pub param_nodes: HashMap<String, usize>,

    pub log_joint: ADFloat,
    pub vi_params: HashMap<String, ADFloat>,
}
impl ProbContext {
    pub fn new(mode: InferenceMode) -> Self {
        let tape_id = match mode {
            InferenceMode::Gradient { tape_id } | InferenceMode::Guide { tape_id } => tape_id,
            _ => crate::ad::create_tape(),
        };

        Self {
            mode,
            trace: HashMap::new(),
            sample_counter: 0,
            tape_id,
            accumulated_log_prob: ADFloat::Concrete(0.0),
            param_nodes: HashMap::new(),
            log_joint: ADFloat::Concrete(0.0),
            vi_params: HashMap::new(),
        }
    }
    pub fn current_log_prob_value(&self) -> f64 {
        self.accumulated_log_prob.value()
    }

    pub fn compute_gradients(&self) {
        self.accumulated_log_prob.backward();
    }
    pub fn log_joint(&self) -> f64 {
        self.accumulated_log_prob.value()
    }
}

fn sample_standard_normal<R: Rng + RngCore>(rng: &mut R) -> f64 {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.sample(rng)
}

pub fn get_distribution_sample<R: Rng + RngCore>(
    dist: &Value,
    rng: &mut R,
    mode: InferenceMode,
) -> RuntimeResult<ADFloat> {
    if let Some(d) = crate::vm::distributions::get_distribution(dist) {
        match mode {
            InferenceMode::Sampling => Ok(ADFloat::Concrete(d.sample(rng)?)),
            _ => d.sample_ad(rng),
        }
    } else {
        Err(RuntimeError::TypeMismatch {
            message: "Value is not a distribution".into(),
        })
    }
}

pub fn register_param(ctx: &mut ProbContext, name: &str, init_val: f64) -> ADFloat {
    if let Some(existing) = ctx.vi_params.get(name) {
        return existing.clone();
    }

    let val_ad = match ctx.mode {
        InferenceMode::Guide { tape_id } | InferenceMode::Gradient { tape_id } => {
            ADFloat::new_input(init_val, tape_id)
        }
        InferenceMode::Sampling => ADFloat::Concrete(init_val),
    };

    ctx.vi_params.insert(name.to_string(), val_ad.clone());

    if let ADFloat::Dual { node_id, .. } = val_ad {
        ctx.param_nodes.insert(name.to_string(), node_id);
    }

    val_ad
}

pub struct HMC {
    pub num_samples: usize,
    pub burn_in: usize,
    pub epsilon: f64,
    pub l_steps: usize,
}

impl HMC {
    pub fn new(num_samples: usize, burn_in: usize, epsilon: f64, l_steps: usize) -> Self {
        Self {
            num_samples,
            burn_in,
            epsilon,
            l_steps,
        }
    }

    pub fn infer(
        &self,
        model: &Block,
        interpreter: &mut Interpreter,
        init_params: HashMap<String, f64>,
    ) -> RuntimeResult<Vec<HashMap<String, f64>>> {
        let mut samples = Vec::new();
        let mut rng = rand::thread_rng();

        let (mut current_log_prob, mut current_grads, mut current_q) =
            self.run_model_and_get_grads(interpreter, model, &init_params)?;

        if current_q.is_empty() {
            return Ok(vec![]);
        }

        for i in 0..(self.num_samples + self.burn_in) {
            if i % 10 == 0 {
                println!("HMC Step {}: LogProb={:.4}", i, current_log_prob);
            }

            let mut current_p: HashMap<String, f64> = HashMap::new();
            for (key, _) in &current_q {
                use rand_distr::StandardNormal;
                let p_val: f64 = rng.sample(StandardNormal);
                current_p.insert(key.clone(), p_val);
            }

            let current_k: f64 = current_p.values().map(|p| 0.5 * p * p).sum();
            let current_h = -current_log_prob + current_k;

            let mut q_new = current_q.clone();
            let mut p_new = current_p.clone();
            let mut grads_new = current_grads.clone();
            let mut log_prob_new = current_log_prob;

            for (key, p) in p_new.iter_mut() {
                if let Some(grad) = grads_new.get(key) {
                    *p -= (self.epsilon / 2.0) * (-grad);
                }
            }

            for step in 0..self.l_steps {
                for (key, q) in q_new.iter_mut() {
                    if let Some(p) = p_new.get(key) {
                        *q += self.epsilon * (*p);
                    }
                }

                let (lp, grads, _) = self.run_model_and_get_grads(interpreter, model, &q_new)?;
                log_prob_new = lp;
                grads_new = grads;
                if step != self.l_steps - 1 {
                    for (key, p) in p_new.iter_mut() {
                        if let Some(grad) = grads_new.get(key) {
                            *p -= self.epsilon * (-grad);
                        }
                    }
                }
            }

            for (key, p) in p_new.iter_mut() {
                if let Some(grad) = grads_new.get(key) {
                    *p -= (self.epsilon / 2.0) * (-grad);
                }
            }

            let new_k: f64 = p_new.values().map(|p| 0.5 * p * p).sum();
            let new_h = -log_prob_new + new_k;

            let acceptance_prob = (current_h - new_h).exp();

            use rand::Rng;
            if rng.gen::<f64>() < acceptance_prob {
                current_q = q_new;
                current_log_prob = log_prob_new;
                current_grads = grads_new;
            }

            if i >= self.burn_in {
                samples.push(current_q.clone());
            }
        }
        Ok(samples)
    }

    fn run_model_and_get_grads(
        &self,
        interpreter: &mut Interpreter,
        model: &Block,
        q_values: &HashMap<String, f64>,
    ) -> RuntimeResult<(f64, HashMap<String, f64>, HashMap<String, f64>)> {
        interpreter.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
        let current_tape_id = interpreter.prob_context.as_ref().unwrap().tape_id;

        if let Some(ctx) = interpreter.prob_context.as_mut() {
            for (k, v) in q_values {
                ctx.trace
                    .insert(k.clone(), Value::Float(ADFloat::Concrete(*v)));
            }

            if q_values.is_empty() {
                ctx.mode = InferenceMode::Sampling;
            } else {
                ctx.mode = InferenceMode::Gradient {
                    tape_id: ctx.tape_id,
                };
            }
        }
        interpreter.prob_mode = true;
        interpreter.prob_id_counter = 0;

        let _result = match interpreter.eval_block(model) {
            Ok(v) => v,
            Err(RuntimeError::EarlyReturn) => {
                interpreter.return_value.take().unwrap_or(Value::Unit)
            }
            Err(_e) => Value::Unit,
        };

        let res = if let Some(ctx) = interpreter.prob_context.as_ref() {
            let log_prob = ctx.accumulated_log_prob.value();

            let mut new_q = HashMap::new();
            for (k, v) in &ctx.trace {
                if let Value::Float(ad_val) = v {
                    new_q.insert(k.clone(), ad_val.value());
                }
            }

            let mut grads = HashMap::new();

            if let Some(loss_node_id) = ctx.accumulated_log_prob.node_id() {
                let all_grads = crate::ad::with_tape(ctx.tape_id, |tape| {
                    crate::ad::backward::backward(tape, loss_node_id)
                });

                for (name, id) in &ctx.param_nodes {
                    if let Some(grad_enum) = all_grads.get(id) {
                        if let ADGradient::Scalar(g) = grad_enum {
                            grads.insert(name.clone(), *g);
                        }
                    }
                }
            }

            Ok((log_prob, grads, new_q))
        } else {
            Ok((0.0, HashMap::new(), HashMap::new()))
        };

        crate::ad::remove_tape(current_tape_id);

        interpreter.prob_mode = false;
        interpreter.prob_context = None;

        res
    }
}

pub struct MetropolisHastings {
    pub num_samples: usize,
    pub burn_in: usize,
}

impl MetropolisHastings {
    pub fn new(num_samples: usize, burn_in: usize) -> Self {
        Self {
            num_samples,
            burn_in,
        }
    }

    pub fn infer(
        &self,
        model: &Block,
        interpreter: &mut Interpreter,
    ) -> Result<Vec<Value>, RuntimeError> {
        let mut samples = Vec::new();

        let mut current_trace = HashMap::new();
        let mut current_log_prob = std::f64::NEG_INFINITY;
        let mut current_result;

        let mut rng = rand::thread_rng();
        let mut accepted_count = 0;

        interpreter.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
        interpreter.prob_mode = true;
        interpreter.prob_id_counter = 0;

        let result = match interpreter.eval_block(model) {
            Ok(val) => val,
            Err(RuntimeError::EarlyReturn) => {
                if let Some(ret_val) = interpreter.return_value.take() {
                    ret_val
                } else {
                    Value::Unit
                }
            }
            Err(e) => return Err(e),
        };

        current_result = result;

        if let Some(ctx) = &interpreter.prob_context {
            current_trace = ctx.trace.clone();
            current_log_prob = ctx.accumulated_log_prob.value();
        }

        interpreter.prob_mode = false;
        interpreter.prob_context = None;

        for i in 0..(self.num_samples + self.burn_in) {
            if i % 100 == 0 {
                println!(
                    "MCMC step: {}, Accepted: {}, LogProb: {:.2}",
                    i, accepted_count, current_log_prob
                );
            }

            let mut proposal_trace = current_trace.clone();

            let keys: Vec<String> = proposal_trace.keys().cloned().collect();
            let log_proposal_ratio = 0.0;

            if !keys.is_empty() {
                let idx = rng.gen_range(0..keys.len());
                let target_key = &keys[idx];

                if let Some(old_val) = proposal_trace.get(target_key) {
                    let new_val = match old_val {
                        Value::Float(v) => Value::Float(
                            v.clone() + ADFloat::Concrete(sample_standard_normal(&mut rng)),
                        ),
                        Value::Int(v) => Value::Int(v + rng.gen_range(-1..=1)),
                        _ => old_val.clone(),
                    };
                    proposal_trace.insert(target_key.clone(), new_val);
                }
            }

            interpreter.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
            interpreter.prob_mode = true;
            interpreter.prob_id_counter = 0;

            if let Some(ctx) = interpreter.prob_context.as_mut() {
                ctx.trace = proposal_trace.clone();
            }
            let proposal_result = match interpreter.eval_block(model) {
                Ok(val) => val,
                Err(RuntimeError::EarlyReturn) => {
                    if let Some(ret_val) = interpreter.return_value.take() {
                        ret_val
                    } else {
                        Value::Unit
                    }
                }
                Err(e) => return Err(e),
            };

            let proposal_log_prob = if let Some(ctx) = &interpreter.prob_context {
                ctx.accumulated_log_prob.value()
            } else {
                0.0
            };

            interpreter.prob_mode = false;
            interpreter.prob_context = None;

            let log_alpha = proposal_log_prob - current_log_prob + log_proposal_ratio;

            let rand_val = rng.gen::<f64>();
            let log_threshold = if rand_val <= 0.0 {
                -std::f64::INFINITY
            } else {
                rand_val.ln()
            };

            if !log_alpha.is_nan() && log_alpha > log_threshold {
                current_trace = proposal_trace;
                current_log_prob = proposal_log_prob;
                current_result = proposal_result;
                accepted_count += 1;
            }

            if i >= self.burn_in {
                let clean = match &current_result {
                    Value::Float(ad) => Value::Float(ADFloat::Concrete(ad.value())),
                    other => other.clone(),
                };
                samples.push(clean);
            }
        }

        interpreter.prob_mode = false;
        interpreter.prob_context = None;

        Ok(samples)
    }
}

pub fn score_gaussian(x: ADFloat, mean: ADFloat, std: ADFloat) -> ADFloat {
    let pi = ADFloat::Concrete(std::f64::consts::PI);
    let two = ADFloat::Concrete(2.0);
    let _half = ADFloat::Concrete(0.5);

    let var = std.clone() * std.clone();
    let diff = x - mean;

    let term1 = -(two.clone() * pi).ln() / two.clone();
    let term2 = -std.ln();
    let term3 = -(diff.clone() * diff) / (two * var);

    term1 + term2 + term3
}

#[allow(dead_code)]

pub fn calculate_score_ad(dist: &Value, val_ad: &ADFloat) -> RuntimeResult<ADFloat> {
    get_distribution_log_pdf_ad(dist, val_ad)
}

pub fn get_distribution_log_pdf(dist: &Value, x: f64) -> RuntimeResult<f64> {
    if let Some(d) = crate::vm::distributions::get_distribution(dist) {
        Ok(d.log_pdf(&ADFloat::Concrete(x)).value())
    } else {
        Err(RuntimeError::TypeMismatch {
            message: "Value is not a distribution".into(),
        })
    }
}

pub fn get_distribution_log_pdf_ad(dist: &Value, x: &ADFloat) -> RuntimeResult<ADFloat> {
    if let Some(d) = crate::vm::distributions::get_distribution(dist) {
        Ok(d.log_pdf(x))
    } else {
        Err(RuntimeError::TypeMismatch {
            message: "Value is not a distribution".into(),
        })
    }
}

use crate::ad::types::ADGradient;

pub trait Optimizer {
    fn step(
        &mut self,
        params: &mut HashMap<String, f64>,
        grads: &HashMap<usize, ADGradient>,
        param_nodes: &HashMap<String, usize>,
    );
}

pub struct SGD {
    pub lr: f64,
}

impl Optimizer for SGD {
    fn step(
        &mut self,
        params: &mut HashMap<String, f64>,
        grads: &HashMap<usize, ADGradient>,
        param_nodes: &HashMap<String, usize>,
    ) {
        for (name, node_id) in param_nodes {
            if let Some(grad_enum) = grads.get(node_id) {
                if let ADGradient::Scalar(grad) = grad_enum {
                    if let Some(val) = params.get_mut(name) {
                        *val += self.lr * grad;
                    }
                }
            }
        }
    }
}

pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    m: HashMap<String, f64>,
    v: HashMap<String, f64>,
    t: usize,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(
        &mut self,
        params: &mut HashMap<String, f64>,
        grads: &HashMap<usize, ADGradient>,
        param_nodes: &HashMap<String, usize>,
    ) {
        self.t += 1;
        let t = self.t as i32;
        let bias_correction1 = 1.0 - self.beta1.powi(t);
        let bias_correction2 = 1.0 - self.beta2.powi(t);

        let step_size = self.lr * bias_correction2.sqrt() / bias_correction1;

        for (name, node_id) in param_nodes {
            if let Some(grad_enum) = grads.get(node_id) {
                if let ADGradient::Scalar(grad) = grad_enum {
                    let g = *grad;

                    let m = self.m.entry(name.clone()).or_insert(0.0);
                    let v = self.v.entry(name.clone()).or_insert(0.0);

                    *m = self.beta1 * *m + (1.0 - self.beta1) * g;
                    *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;

                    if let Some(val) = params.get_mut(name) {
                        *val += step_size * *m / (v.sqrt() + self.eps);
                    }
                }
            }
        }
    }
}

pub struct ADVI {
    pub num_iters: usize,
    pub optimizer: Box<dyn Optimizer>,
}

impl ADVI {
    pub fn new(num_iters: usize, learning_rate: f64) -> Self {
        Self {
            num_iters,
            optimizer: Box::new(SGD { lr: learning_rate }),
        }
    }

    pub fn with_adam(num_iters: usize, learning_rate: f64) -> Self {
        Self {
            num_iters,
            optimizer: Box::new(Adam::new(learning_rate)),
        }
    }

    pub fn infer(
        &mut self,
        model: &Block,
        guide: &Block,
        interpreter: &mut Interpreter,
    ) -> RuntimeResult<HashMap<String, f64>> {
        let mut vi_params: HashMap<String, f64> = HashMap::new();

        {
            interpreter.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
            if let Some(ctx) = interpreter.prob_context.as_mut() {
                ctx.mode = InferenceMode::Guide {
                    tape_id: ctx.tape_id,
                };
            }
            interpreter.prob_mode = true;

            let _ = interpreter.eval_block(guide).ok();

            if let Some(ctx) = interpreter.prob_context.as_ref() {
                for (name, val) in &ctx.vi_params {
                    vi_params.insert(name.clone(), val.value());
                }
            }

            interpreter.prob_context = None;
            interpreter.prob_mode = false;
        }

        for i in 0..self.num_iters {
            interpreter.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
            let tape_id = interpreter.prob_context.as_mut().unwrap().tape_id;

            if let Some(ctx) = interpreter.prob_context.as_mut() {
                ctx.mode = InferenceMode::Guide { tape_id };
                for (name, val) in &vi_params {
                    let ad_val = ADFloat::new_input(*val, tape_id);
                    ctx.vi_params.insert(name.clone(), ad_val.clone());
                    if let ADFloat::Dual { node_id, .. } = ad_val {
                        ctx.param_nodes.insert(name.clone(), node_id);
                    }
                }
            }
            interpreter.prob_mode = true;
            interpreter.prob_id_counter = 0;

            let _ = interpreter.eval_block(guide).unwrap_or(Value::Unit);

            let (_trace_z, log_q) = if let Some(ctx) = interpreter.prob_context.as_ref() {
                (ctx.trace.clone(), ctx.accumulated_log_prob.clone())
            } else {
                (HashMap::new(), ADFloat::Concrete(0.0))
            };

            if let Some(ctx) = interpreter.prob_context.as_mut() {
                ctx.mode = InferenceMode::Gradient { tape_id };
                ctx.accumulated_log_prob = ADFloat::Concrete(0.0);
                ctx.sample_counter = 0;
            }

            let _ = interpreter.eval_block(model).unwrap_or(Value::Unit);

            let log_p = if let Some(ctx) = interpreter.prob_context.as_ref() {
                ctx.accumulated_log_prob.clone()
            } else {
                ADFloat::Concrete(0.0)
            };

            let elbo = log_p - log_q;

            if i % 100 == 0 || i == self.num_iters - 1 {
                // println!("VI Step {}: ELBO = {}, log_p = {}, log_q = {}", i, elbo.value(), log_p_val, log_q_val);
            }

            let grads = elbo.backward();

            if let Some(ctx) = interpreter.prob_context.as_ref() {
                self.optimizer
                    .step(&mut vi_params, &grads, &ctx.param_nodes);
            }

            crate::ad::remove_tape(tape_id);
            interpreter.prob_context = None;
            interpreter.prob_mode = false;
        }

        Ok(vi_params)
    }
}
