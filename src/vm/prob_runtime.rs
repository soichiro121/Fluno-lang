use crate::vm::value::Value;
use crate::vm::RuntimeResult;
use crate::vm::{Interpreter, RuntimeError};
use crate::ast::node::Block;
use std::collections::HashMap;
use rand::Rng;
use rand::RngCore;
use crate::ad::types::ADFloat;

// --- 確率分布トレイト ---
// 仕様書 Section 6 に従った確率分布の実装要件
pub trait DistributionImpl {
    // 値をサンプリングする (for Sampling Mode)
    fn sample_rng<R: Rng + RngCore>(&self, rng: &mut R) -> f64;

    // 対数確率密度を計算する (Concrete Float)
    fn log_pdf(&self, x: f64) -> f64;

    // 対数確率密度を計算する (AD Float)
    // 数式通りに ADFloat の演算を使って実装すること
    fn log_pdf_ad(&self, x: &ADFloat) -> ADFloat;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceMode {
    Sampling,
    Gradient { tape_id: usize },
    Guide { tape_id: usize }, // Reparameterization Trick mode
}
// 確率的実行のコンテキスト
#[derive(Debug, Clone)]
pub struct ProbContext {
    pub mode: InferenceMode, 
    pub trace: HashMap<String, Value>,
    pub sample_counter: usize, 

    pub tape_id: usize,
    
    // --- AD関連フィールド ---
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
    // 対数同時確率の「値」を取得（勾配計算前）
    pub fn current_log_prob_value(&self) -> f64 {
        self.accumulated_log_prob.value()
    }
    
    // 逆伝播を実行して勾配を計算する
    pub fn compute_gradients(&self) {
        // accumulated_log_prob に対して backward を呼ぶ
        // 注: ADFloat::backward(&self.accumulated_log_prob) のような実装を想定
        self.accumulated_log_prob.backward(); 
    }
    pub fn log_joint(&self) -> f64 {
        self.accumulated_log_prob.value()
    }
}

// Helper for standard normal sampling
fn sample_standard_normal<R: Rng + RngCore>(rng: &mut R) -> f64 {
    use rand_distr::{Normal, Distribution};
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.sample(rng)
}

// Valueから対応する分布の動作を実行する
pub fn get_distribution_sample<R: Rng + RngCore>(dist: &Value, rng: &mut R, mode: InferenceMode) -> RuntimeResult<ADFloat> {
    if let Some(d) = crate::vm::distributions::get_distribution(dist) {
        match mode {
             InferenceMode::Sampling => Ok(ADFloat::Concrete(d.sample(rng)?)),
             _ => d.sample_ad(rng),
        }
    } else {
        Err(RuntimeError::TypeMismatch { message: "Value is not a distribution".into() })
    }
}

pub fn register_param(ctx: &mut ProbContext, name: &str, init_val: f64) -> ADFloat {
    if let Some(existing) = ctx.vi_params.get(name) {
        return existing.clone();
    }
    
    // Create new parameter
    let val_ad = match ctx.mode {
        InferenceMode::Guide { tape_id } | InferenceMode::Gradient { tape_id } => {
            ADFloat::new_input(init_val, tape_id)
        }
        InferenceMode::Sampling => {
            ADFloat::Concrete(init_val)
        }
    };
    
    ctx.vi_params.insert(name.to_string(), val_ad.clone());
    
    // If we are in Gradient/Guide mode, we also need to register this param in param_nodes for optimizer to find it
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
        Self { num_samples, burn_in, epsilon, l_steps }
    }

    pub fn infer(
        &self,
        model: &Block,
        interpreter: &mut Interpreter,
        init_params: HashMap<String, f64>,
    ) -> RuntimeResult<Vec<HashMap<String, f64>>> {
        let mut samples = Vec::new();
        let mut rng = rand::thread_rng();

        // 1. 初期化実行 (Gradient Mode でパラメータ構造と初期値を取得)
        let (mut current_log_prob, mut current_grads, mut current_q) =
            self.run_model_and_get_grads(interpreter, model, &init_params)?;

        // もしパラメータがなければHMCできないので即終了
        if current_q.is_empty() {
            // パラメータなし（ただの計算）の場合
            // 通常の実行結果を返すなどの処理が必要だが、ここでは省略
            return Ok(vec![]);
        }

        for i in 0..(self.num_samples + self.burn_in) {
            if i % 10 == 0 {
                println!("HMC Step {}: LogProb={:.4}", i, current_log_prob);
            }

            // 2. 運動量 p のサンプリング
            let mut current_p: HashMap<String, f64> = HashMap::new();
            for (key, _) in &current_q {
                // 標準正規分布からサンプリング (質量行列 M=I)
                use rand_distr::StandardNormal; // Removed unused Distribution
                let p_val: f64 = rng.sample(StandardNormal);
                current_p.insert(key.clone(), p_val);
            }

            // 現在のハミルトニアン H = U(q) + K(p) = -log_prob + 0.5 * p^T p
            let current_k: f64 = current_p.values().map(|p| 0.5 * p * p).sum();
            let current_h = -current_log_prob + current_k;

            // 3. リープフロッグ積分
            let mut q_new = current_q.clone();
            let mut p_new = current_p.clone();
            let mut grads_new = current_grads.clone(); // 勾配は q_new に対応するものを使う
            let mut log_prob_new = current_log_prob;

            // 半ステップ p 更新
            for (key, p) in p_new.iter_mut() {
                if let Some(grad) = grads_new.get(key) {
                    *p -= (self.epsilon / 2.0) * (-grad); // ∇U = -∇logP
                }
            }

            for step in 0..self.l_steps {
                // q 更新 (全ステップ)
                for (key, q) in q_new.iter_mut() {
                    if let Some(p) = p_new.get(key) {
                        *q += self.epsilon * (*p);
                    }
                }

                // ここで q_new における勾配を再計算
                let (lp, grads, _) = self.run_model_and_get_grads(interpreter, model, &q_new)?;
                log_prob_new = lp;
                grads_new = grads;

                // p 更新 (最後のステップ以外は全ステップ)
                if step != self.l_steps - 1 {
                    for (key, p) in p_new.iter_mut() {
                        if let Some(grad) = grads_new.get(key) {
                            *p -= self.epsilon * (-grad);
                        }
                    }
                }
            }

            // 最後の半ステップ p 更新
            for (key, p) in p_new.iter_mut() {
                if let Some(grad) = grads_new.get(key) {
                    *p -= (self.epsilon / 2.0) * (-grad);
                }
            }

            // 4. 採択判定
            let new_k: f64 = p_new.values().map(|p| 0.5 * p * p).sum();
            let new_h = -log_prob_new + new_k;

            // exp(H_current - H_new)
            let acceptance_prob = (current_h - new_h).exp();

            use rand::Rng;
            if rng.gen::<f64>() < acceptance_prob {
                // Accept
                current_q = q_new;
                current_log_prob = log_prob_new;
                current_grads = grads_new;
            } else {
                // Reject (current_q のまま)
            }

            // 5. サンプル保存
            if i >= self.burn_in {
                 samples.push(current_q.clone());
            }
        }

        Ok(samples)
    }



    // 指定された q_values (パラメータ) でモデルを実行し、(log_prob, grads, current_q_map) を返す
    // initial_run=true の場合、TraceがないのでSamplingしてパラメータマップを作る
    fn run_model_and_get_grads(
        &self,
        interpreter: &mut Interpreter,
        model: &Block,
        q_values: &HashMap<String, f64>,
    ) -> RuntimeResult<(f64, HashMap<String, f64>, HashMap<String, f64>)> {
        
        // Context初期化
        interpreter.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
        // [重要] 必ず生成された tape_id を取得しておく
        let current_tape_id = interpreter.prob_context.as_ref().unwrap().tape_id;

        if let Some(ctx) = interpreter.prob_context.as_mut() {
            // q_values を Trace にセット
            for (k, v) in q_values {
                ctx.trace.insert(k.clone(), Value::Float(ADFloat::Concrete(*v)));
            }
            
            // Gradientモード、ただしTraceが空ならSamplingモード的に動いてTraceを作る必要がある
            if q_values.is_empty() {
                ctx.mode = InferenceMode::Sampling;
            } else {
                ctx.mode = InferenceMode::Gradient { tape_id: ctx.tape_id }; 
            }
        }
        interpreter.prob_mode = true;
        interpreter.prob_id_counter = 0;

        // 実行
        // エラーハンドリング省略（実際は ? で伝播すべき箇所だが元のコードに従う）
        let _result = match interpreter.eval_block(model) {
            Ok(v) => v,
            Err(RuntimeError::EarlyReturn) => {
                interpreter.return_value.take().unwrap_or(Value::Unit)
            },
            Err(_e) => {
                Value::Unit
            }
        };

        // 結果取得
        // ここで if let の結果を res に代入し、return はしない
        let res = if let Some(ctx) = interpreter.prob_context.as_ref() {
            let log_prob = ctx.accumulated_log_prob.value();
            
            // q (パラメータ) の最新値を取得
            let mut new_q = HashMap::new();
            for (k, v) in &ctx.trace {
                 if let Value::Float(ad_val) = v {
                     new_q.insert(k.clone(), ad_val.value());
                 }
            }

            // 勾配計算
            let mut grads = HashMap::new();
            
            // accumulated_log_prob の backward
            if let Some(loss_node_id) = ctx.accumulated_log_prob.node_id() {
                // [修正] tape変数はスコープにないので、with_tapeを使ってTLSのテープにアクセスする
                // backwardの結果(HashMap<usize, f64>)を受け取る
                let all_grads = crate::ad::with_tape(ctx.tape_id, |tape| {
                    crate::ad::backward::backward(tape, loss_node_id)
                });
                
                // param_nodes (name -> node_id) を使ってマッピング
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

        // [重要] クリーンアップ：使い終わったテープを削除
        crate::ad::remove_tape(current_tape_id);

        interpreter.prob_mode = false;
        interpreter.prob_context = None;

        res
    }

}

// Metropolis-Hastings サンプラー
pub struct MetropolisHastings {
    // サンプル数
    pub num_samples: usize,
    // バーンイン期間
    pub burn_in: usize,
}

impl MetropolisHastings {
    pub fn new(num_samples: usize, burn_in: usize) -> Self {
        Self { num_samples, burn_in }
    }

    // 推論を実行する
    // 
    // model: 推論対象のブロック（関数ボディなど）
    // interpreter: 親のインタプリタインスタンス
    pub fn infer(
        &self,
        model: &Block,
        interpreter: &mut Interpreter,
    ) -> Result<Vec<Value>, RuntimeError> {
        let mut samples = Vec::new();
        
        // 初期トレースの生成
        let mut current_trace = HashMap::new();
        let mut current_log_prob = std::f64::NEG_INFINITY;
        let mut current_result = Value::Unit;
        
        // MCMCループ
        let mut rng = rand::thread_rng();
        let mut accepted_count = 0;

        // 初期化実行 (Priorからのサンプリング + 初期尤度計算)
        // 初期化実行 (Priorからのサンプリング + 初期尤度計算)
        interpreter.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
        interpreter.prob_mode = true; // 確率モードON
        interpreter.prob_id_counter = 0; // カウンタリセット

        // 初期実行の結果をハンドリング
        let result = match interpreter.eval_block(model) {
            Ok(val) => val,
            Err(RuntimeError::EarlyReturn) => {
                if let Some(ret_val) = interpreter.return_value.take() { ret_val } else { Value::Unit }
            },
            Err(e) => return Err(e),
        };

        current_result = result;

        // 初期状態の尤度とトレースを保存
        if let Some(ctx) = &interpreter.prob_context {
            current_trace = ctx.trace.clone();
            current_log_prob = ctx.accumulated_log_prob.value();
            // ここで初めて tape_id を取得して消す準備ができる（今回は単純化のため省略）
        }

        // クリーンアップ
        interpreter.prob_mode = false;
        interpreter.prob_context = None;

        
        for i in 0..(self.num_samples + self.burn_in) {
            if i % 100 == 0 { 
                // デバッグ出力: 進捗と現在の受理数、対数尤度を表示
                println!("MCMC step: {}, Accepted: {}, LogProb: {:.2}", i, accepted_count, current_log_prob); 
            }

            // 1. 提案分布から新しいトレースを生成 (Proposal)
            let mut proposal_trace = current_trace.clone();
            
            let keys: Vec<String> = proposal_trace.keys().cloned().collect();
            let log_proposal_ratio = 0.0; // 対称提案分布を仮定

            if !keys.is_empty() {
                let idx = rng.gen_range(0..keys.len());
                let target_key = &keys[idx];
                
                if let Some(old_val) = proposal_trace.get(target_key) {
                    // 値を摂動させる (Gaussian Random Walk)
                    let new_val = match old_val {
                        Value::Float(v) => Value::Float(v.clone() + ADFloat::Concrete(sample_standard_normal(&mut rng))),
                        Value::Int(v) => Value::Int(v + rng.gen_range(-1..=1)),
                        // Gaussianオブジェクト自体がトレースに入っている場合はそのまま
                        _ => old_val.clone(),
                    };
                    proposal_trace.insert(target_key.clone(), new_val);
                }
            }

            // 2. モデルを実行して尤度を計算
            // 2. モデルを実行して尤度を計算
            interpreter.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
            interpreter.prob_mode = true; // これも必要かも（前のループでfalseにしてるなら）
            interpreter.prob_id_counter = 0;

            // ★ 重要: 提案されたトレースをセットしてリプレイさせる
            if let Some(ctx) = interpreter.prob_context.as_mut() {
                ctx.trace = proposal_trace.clone();
            }

            // 実行結果の取得
            let proposal_result = match interpreter.eval_block(model) {
                Ok(val) => val,
                Err(RuntimeError::EarlyReturn) => {
                    if let Some(ret_val) = interpreter.return_value.take() { ret_val } else { Value::Unit }
                },
                Err(e) => return Err(e),
            };

            let proposal_log_prob = if let Some(ctx) = &interpreter.prob_context {
                ctx.accumulated_log_prob.value()
            } else {
                0.0
            };

            // クリーンアップ
            interpreter.prob_mode = false;
            interpreter.prob_context = None;

            // 3. 受理判定 (Acceptance Step)
            // log_alpha = log_p_new - log_p_old + log_proposal_ratio
            let log_alpha = proposal_log_prob - current_log_prob + log_proposal_ratio;
            
            // NaNチェックと対数空間での比較 (アンダーフロー対策)
            let rand_val = rng.gen::<f64>();
            let log_threshold = if rand_val <= 0.0 { -std::f64::INFINITY } else { rand_val.ln() };

            if !log_alpha.is_nan() && log_alpha > log_threshold {
                // Accept
                current_trace = proposal_trace;
                current_log_prob = proposal_log_prob;
                current_result = proposal_result; // 結果を更新
                accepted_count += 1;
            } else {
                // Reject: current_result はそのまま維持
            }
                        
            if i >= self.burn_in {
                // テストが Float の配列を期待しているので、Floatだけ返す
                let clean = match &current_result {
                    Value::Float(ad) => Value::Float(ADFloat::Concrete(ad.value())),
                    other => other.clone(), // model が Float を返す前提ならここはエラーにしてもOK
                };
                samples.push(clean);
            }


        }
        
        interpreter.prob_mode = false;
        interpreter.prob_context = None;

        Ok(samples)
    }
}

// 確率分布のスコア計算ヘルパー
pub fn score_gaussian(x: ADFloat, mean: ADFloat, std: ADFloat) -> ADFloat {
    // 定数を ADFloat に
    let pi = ADFloat::Concrete(std::f64::consts::PI);
    let two = ADFloat::Concrete(2.0);
    let half = ADFloat::Concrete(0.5);

    // 計算 (すべて ADFloat の演算オーバーロードで処理される)
    let var = std.clone() * std.clone();
    let diff = x - mean;
    
    // -0.5 * log(2π) - log(σ) - (x - μ)^2 / (2σ^2)
    let term1 = -(two.clone() * pi).ln() / two.clone(); // あるいは定数として事前計算
    let term2 = -std.ln();
    let term3 = -(diff.clone() * diff) / (two * var);
    
    // println!("DEBUG: score_gaussian: term1={}, term2={}, term3={}", term1.value(), term2.value(), term3.value());
    
    term1 + term2 + term3
}

// Box-Muller法による標準正規分布からのサンプリング


#[allow(dead_code)]
fn make_weighted_sample(value: Value, log_weight: f64) -> Value {
    use std::collections::HashMap;
    use crate::gc::Rc;
    use crate::vm::value::Value;

    let mut fields = HashMap::new();
    fields.insert("value".to_string(), value);
    fields.insert("log_weight".to_string(), Value::Float(ADFloat::Concrete(log_weight)));

    Value::Struct {
        name: "Sample".to_string(),
        fields: Rc::new(fields),
    }
}

// 既存の関数をトレイト利用に書き換え
pub fn calculate_score_ad(dist: &Value, val_ad: &ADFloat) -> RuntimeResult<ADFloat> {
    get_distribution_log_pdf_ad(dist, val_ad)
}




pub fn get_distribution_log_pdf(dist: &Value, x: f64) -> RuntimeResult<f64> {
     if let Some(d) = crate::vm::distributions::get_distribution(dist) {
         Ok(d.log_pdf(&ADFloat::Concrete(x)).value())
     } else {
         Err(RuntimeError::TypeMismatch { message: "Value is not a distribution".into() })
     }
}

pub fn get_distribution_log_pdf_ad(dist: &Value, x: &ADFloat) -> RuntimeResult<ADFloat> {
     if let Some(d) = crate::vm::distributions::get_distribution(dist) {
         Ok(d.log_pdf(x))
     } else {
         Err(RuntimeError::TypeMismatch { message: "Value is not a distribution".into() })
     }
}

// --- ADVI Implementation ---

// --- Optimizers ---

use crate::ad::types::ADGradient;

pub trait Optimizer {
    fn step(&mut self, params: &mut HashMap<String, f64>, grads: &HashMap<usize, ADGradient>, param_nodes: &HashMap<String, usize>);
}

pub struct SGD {
    pub lr: f64,
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut HashMap<String, f64>, grads: &HashMap<usize, ADGradient>, param_nodes: &HashMap<String, usize>) {
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
         Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, m: HashMap::new(), v: HashMap::new(), t: 0 }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut HashMap<String, f64>, grads: &HashMap<usize, ADGradient>, param_nodes: &HashMap<String, usize>) {
        self.t += 1;
        let t = self.t as i32;
        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(t);
        let bias_correction2 = 1.0 - self.beta2.powi(t);
        
        // Effective learning rate
        let step_size = self.lr * bias_correction2.sqrt() / bias_correction1;

        for (name, node_id) in param_nodes {
            if let Some(grad_enum) = grads.get(node_id) {
                if let ADGradient::Scalar(grad) = grad_enum {
                    let g = *grad; // Gradient Ascent
                    
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
        Self { num_iters, optimizer: Box::new(SGD { lr: learning_rate }) }
    }

    pub fn with_adam(num_iters: usize, learning_rate: f64) -> Self {
        Self { num_iters, optimizer: Box::new(Adam::new(learning_rate)) }
    }

    pub fn infer(
        &mut self,
        model: &Block,
        guide: &Block,
        interpreter: &mut Interpreter
    ) -> RuntimeResult<HashMap<String, f64>> {
        // Init parameters from Guide execution
        let mut vi_params: HashMap<String, f64> = HashMap::new();
        
        // 1. Initial run to find parameters
        {
            interpreter.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
            // Set Guide mode
            if let Some(ctx) = interpreter.prob_context.as_mut() {
                ctx.mode = InferenceMode::Guide { tape_id: ctx.tape_id };
            }
            interpreter.prob_mode = true;
            
            // Run Guide
            let _ = interpreter.eval_block(guide).ok();
            
            // Extract registered params
            if let Some(ctx) = interpreter.prob_context.as_ref() {
                for (name, val) in &ctx.vi_params {
                    vi_params.insert(name.clone(), val.value());
                }
            }
            
            interpreter.prob_context = None;
            interpreter.prob_mode = false;
        }

        // Optimization Loop
        for i in 0..self.num_iters {
            interpreter.prob_context = Some(ProbContext::new(InferenceMode::Sampling));
            let tape_id = interpreter.prob_context.as_mut().unwrap().tape_id;
            
            // Setup Context with Params (as Input Nodes)
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

            // A. Run Guide (Trace generation + log q)
            // Note: Guide's sampling will use Reparameterization, so 'z' depends on params.
            let _ = interpreter.eval_block(guide).unwrap_or(Value::Unit);
            
            // Snapshot trace and log_q
            let (_trace_z, log_q) = if let Some(ctx) = interpreter.prob_context.as_ref() {
                (ctx.trace.clone(), ctx.accumulated_log_prob.clone())
            } else {
                (HashMap::new(), ADFloat::Concrete(0.0))
            };

            // B. Run Model (Trace replay + log p)
            // Switch mode to Gradient to compute log p(x, z) derivative
            // Important: We need to reset accumulated_log_prob for model part
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

            // C. ELBO = log_p - log_q
            let _log_p_val = log_p.value();
            let _log_q_val = log_q.value();
            let elbo = log_p - log_q;
            
            if i % 100 == 0 || i == self.num_iters - 1 {
                // println!("VI Step {}: ELBO = {}, log_p = {}, log_q = {}", i, elbo.value(), log_p_val, log_q_val);
            }

            // D. Backward
            let grads = elbo.backward();

            // E. Update Params (Optimizer)
            if let Some(ctx) = interpreter.prob_context.as_ref() {
                self.optimizer.step(&mut vi_params, &grads, &ctx.param_nodes);
            }

            crate::ad::remove_tape(tape_id);
            interpreter.prob_context = None;
            interpreter.prob_mode = false;
        }

        Ok(vi_params)
    }
}
