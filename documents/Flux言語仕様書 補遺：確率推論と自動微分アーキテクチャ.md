# Flux言語仕様書 補遺：確率推論と自動微分アーキテクチャ

**Version:** 1.0.0 (Draft)
**Date:** 2026-01-04
**Target:** Phase 3 Implementation (Gradient-based Inference)

***

## 1. 概要

本章では、Flux言語における確率的プログラミング（PPL）機能の核心となる、自動微分（Automatic Differentiation, AD）とそれを活用した高度な推論アルゴリズム（HMC, VI）の統合アーキテクチャを定義する。

**設計目標:**
1.  **Unified Runtime:** サンプリングベース（IS, SMC）と勾配ベース（HMC, VI）の推論を単一のランタイムでサポートする。
2.  **Differentiable Programming:** 確率モデル内のすべての浮動小数点演算を、透過的に微分可能にする。
3.  **Extensibility:** 新しい確率分布や推論アルゴリズムを容易に追加できる構造とする。

***

## 2. 自動微分（AD）サブシステム

Fluxは **Reverse-mode Automatic Differentiation (Tape-based)** を採用する。

### 2.1 データ構造：ADFloat

数値型 `Float` は、推論コンテキスト内では以下の列挙型として振る舞う。

```rust
// src/ad/types.rs

/// 自動微分対応浮動小数点数
#[derive(Debug, Clone)]
pub enum ADFloat {
    /// 具体値（通常の計算、サンプリングモード）
    Concrete(f64),
    
    /// 双対数（勾配計算モード）
    /// 計算グラフへの参照を保持する
    Dual {
        value: f64,
        tape_id: usize,  // 所属するテープID
        node_id: usize,  // 計算グラフ上のノードID
    },
}
```

### 2.2 計算グラフ：TapeとNode

計算履歴（Wengert List / Tape）をスレッドローカルストレージで管理する。

```rust
// src/ad/graph.rs

#[derive(Debug, Clone)]
pub enum ADNode {
    /// 独立変数（勾配計算の対象となるパラメータ）
    Input { value: f64 },
    
    /// 定数（計算には関与するが勾配は流れない）
    Constant { value: f64 },
    
    /// 二項演算 (z = x op y)
    Binary {
        op: BinaryOp,
        lhs_node: usize,
        rhs_node: usize,
        value: f64,
    },
    
    /// 単項演算 (y = op x)
    Unary {
        op: UnaryOp,
        arg_node: usize,
        value: f64,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp { Add, Sub, Mul, Div, Pow, Atan2, BetaSample }

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp { Neg, Exp, Log, Sin, Cos, Tan, Sqrt, Abs, Tanh, Sigmoid, LGamma }

pub struct Tape {
    pub nodes: Vec<ADNode>,
}
```

### 2.3 逆伝播（Backpropagation）アルゴリズム

**仕様:**
1.  テープ内のノード数分の `adjoints`（随伴変数）配列を `0.0` で初期化する。
2.  対象となる出力ノード（通常は `log_prob`）の `adjoint` を `1.0` に設定する。
3.  ノードIDの降順（作成の逆順）で走査し、各演算の連鎖律に基づいて `adjoint` を入力ノードに加算する。

**演算規則（一部抜粋）:**
- `z = x + y` → `dx += dz * 1.0`, `dy += dz * 1.0`
- `z = x * y` → `dx += dz * y`, `dy += dz * x`
- `z = exp(x)` → `dx += dz * z`
- `z = log(x)` → `dx += dz / x`

***

## 3. 推論コンテキスト (ProbContext)

実行時システム（VM）は、推論の状態を管理するために `ProbContext` を拡張する。

### 3.1 構造体定義

```rust
// src/vm/prob_context.rs

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceMode {
    /// 順方向サンプリング（Importance Sampling, SMC, MCMC）
    Sampling,
    
    /// 勾配計算（HMC, NUTS, VI）
    /// 特定のTapeに関連付けられる
    Gradient { tape_id: usize },
}

pub struct ProbContext {
    /// 現在の推論モード
    pub mode: InferenceMode,
    
    /// 実行トレース（アドレス -> 値）
    /// MCMCの提案やリプレイに使用
    pub trace: HashMap<String, Value>,
    
    /// サンプリングアドレス生成用カウンタ
    pub sample_counter: usize,
    
    /// 累積対数確率（スカラー）
    /// Samplingモードで使用
    pub log_prob_scalar: f64,
    
    /// 累積対数確率（計算グラフノード）
    /// Gradientモードで使用
    pub log_prob_node: Option<ADFloat>,
    
    /// パラメータ空間のマッピング
    /// 変数名 -> ノードID（Gradientモード用）
    pub param_map: HashMap<String, usize>,
}
```

***

## 4. 確率モデルの実行モデル

### 4.1 `sample(dist)` の挙動詳細

確率分布 `dist` から値をサンプリング、または対数確率を加算する。

**シグネチャ:** `fn sample(dist: Distribution) -> Value`

**アルゴリズム:**
1.  一意なアドレス `addr = format!("sample{}", ctx.sample_counter++)` を生成。
2.  `ctx.mode` に応じて分岐：

    **A. Sampling Mode:**
    1.  `ctx.trace[addr]` に値があれば（Replay）、それを `val` とする。
    2.  なければ、`dist.sample_rng()` で乱数生成し `val` とする。
    3.  `score = dist.log_pdf(val)` （`f64`）を計算。
    4.  `ctx.log_prob_scalar += score`。
    5.  `ctx.trace[addr] = val` を保存/更新。
    6.  `val` を返す。

    **B. Gradient Mode:**
    1.  `ctx.param_map[addr]` から対応するADノードを取得し、`val_ad`（`ADFloat::Dual`）とする。
        *   存在しない場合（初回実行時など）は、外部から与えられた初期値を `Input` ノードとして登録。
    2.  `score_ad = dist.log_pdf_ad(val_ad)` （`ADFloat::Dual`）を計算。
        *   分布内部の計算はすべてAD演算を使用。
    3.  `ctx.log_prob_node += score_ad` （計算グラフの構築）。
    4.  `val_ad.value()` （`f64`）を返す。

### 4.2 `observe(dist, value)` の挙動詳細

観測データ `value` に基づき、尤度（Likelihood）を累積する。

**シグネチャ:** `fn observe(dist: Distribution, val: Value) -> Unit`

**アルゴリズム:**
1.  `ctx.mode` に応じて分岐：

    **A. Sampling Mode:**
    1.  `score = dist.log_pdf(val)` （`f64`）を計算。
    2.  `ctx.log_prob_scalar += score`。

    **B. Gradient Mode:**
    1.  `val` を定数ノード `val_ad`（`ADFloat::Concrete`）に変換。
    2.  `score_ad = dist.log_pdf_ad(val_ad)` （`ADFloat::Dual`）を計算。
    3.  `ctx.log_prob_node += score_ad`。

***

## 5. 推論アルゴリズム API仕様

### 5.1 低レベルAPI: `TargetLogProb`

外部の推論エンジン（Rustで実装されたHMC等）が利用する、モデルの微分インターフェース。

```rust
// src/inference/interface.rs

pub trait TargetLogProb {
    /// パラメータ params における log p(x, z) と
    /// その勾配 ∇_z log p(x, z) を返す
    fn log_prob_and_grad(
        &self, 
        params: &HashMap<String, f64>
    ) -> (f64, HashMap<String, f64>);
}
```

### 5.2 Hamiltonian Monte Carlo (HMC)

Fluxの標準ライブラリとして提供されるHMC実装。

**コンフィグ構造体:**
```flux
struct HMCConfig {
    num_samples: Int,      // サンプル数
    step_size: Float,      // リープフロッグ積分のステップサイズ ε
    num_leapfrog: Int,     // リープフロッグのステップ数 L
    init_params: Map<String, Float> // 初期パラメータ
}
```

**関数シグネチャ:**
```flux
fn infer_hmc(
    config: HMCConfig,
    model: Fn(params: Map<String, Float>) -> Unit
) -> Array<Map<String, Float>>
```

**処理フロー:**
1.  Rust側で `Tape` を新規作成。
2.  `init_params` を `Input` ノードとしてテープに登録。
3.  `model` を `Gradient` モードで実行し、`log_prob` の計算グラフを構築。
4.  HMCのメインループ：
    a. 運動量 `p ~ N(0, I)` をサンプリング。
    b. `TargetLogProb` インターフェース経由で勾配を取得しながらリープフロッグ積分。
    c. Metropolis-Hastings 判定で採択/棄却。
5.  有効なサンプル配列を返す。

### 5.3 自動微分変分推論 (ADVI)

変分分布（Guide）を最適化する推論方式。

**コンフィグ構造体:**
```flux
struct VIConfig {
    num_iters: Int,        // 最適化ループ回数
    learning_rate: Float,  // 学習率
    optimizer: String      // "SGD", "Adam" 等
}
```

**関数シグネチャ:**
```flux
fn infer_vi(
    config: VIConfig,
    model: Fn() -> Unit,
    guide: Fn(params: Map<String, Float>) -> Unit
) -> Map<String, Float>
```

**処理フロー (Black Box Variational Inference):**
1.  変分パラメータ $\phi$ を初期化。
2.  最適化ループ：
    a. Guideからサンプル $z \sim q_\phi(z)$ を生成（Reparameterization Trickを使用）。
       *   $z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim N(0, I)$
       *   この操作自体をAD計算グラフに組み込む。
    b. `model(z)` を実行し、$\log p(x, z)$ を計算。
    c. `guide(z; \phi)` を実行し、$\log q_\phi(z)$ を計算。
    d. $\text{ELBO} = \log p(x, z) - \log q_\phi(z)$ を計算。
    e. Backpropagationにより $\nabla_\phi \text{ELBO}$ を取得。
    f. Optimizerにより $\phi$ を更新。
3.  最適化された $\phi$ を返す。

***

## 6. 確率分布の実装要件

すべての標準確率分布（`Gaussian`, `Uniform`, `Bernoulli` 等）は、以下の2つのメソッドを実装しなければならない。

```rust
// src/vm/distribution.rs

pub trait DistributionImpl {
    /// 値をサンプリングする (for Sampling Mode)
    fn sample_rng(&self, rng: &mut StdRng) -> f64;

    /// 対数確率密度を計算する (Concrete Float)
    fn log_pdf(&self, x: f64) -> f64;

    /// 対数確率密度を計算する (AD Float)
    /// 数式通りに ADFloat の演算を使って実装すること
    fn log_pdf_ad(&self, x: &ADFloat) -> ADFloat;
}
```

**Gaussianの実装例:**
```rust
fn log_pdf_ad(&self, x: &ADFloat) -> ADFloat {
    // 定数は Concrete として扱う
    let pi = ADFloat::Concrete(std::f64::consts::PI);
    let two = ADFloat::Concrete(2.0);
    
    // 計算グラフ構築: -0.5 * log(2π) - log(σ) - (x - μ)^2 / (2σ^2)
    let term1 = -(two.clone() * pi).log() / two.clone();
    let term2 = -self.std.log();
    let diff = x - self.mean;
    let term3 = -(diff.clone() * diff) / (two * self.std.clone() * self.std);
    
    term1 + term2 + term3
}
```

***

## 7. 実装マイルストーン

本仕様の実装は以下の順序で行う。

1.  **AD Core (Week 1):**
    *   `ADFloat`, `Tape`, `ADNode` の実装。
    *   基本四則演算と初等関数の `impl` オーバーロード。
    *   単体テスト: 単純な関数 $f(x) = x^2 + 3x$ などの微分が正しいか確認。

2.  **Context Integration (Week 2):**
    *   `ProbContext` への `Gradient` モード追加。
    *   `sample`, `observe` の分岐ロジック実装。
    *   `Distribution` トレイトへの `log_pdf_ad` 追加と `Gaussian` への実装。

3.  **HMC Implementation (Week 3):**
    *   `interface.rs` の整備。
    *   Rust側でのHMCアルゴリズム実装。
    *   Fluxの標準ライブラリ（`stdlib/inference.flux`）へのAPI露出。

4.  **VI & Refinement (Week 4):**
    *   VIの実装。
    *   パフォーマンスチューニング（Tapeの再利用、メモリ割り当て最適化）。
    *   ドキュメント整備と統合テスト。

5.  **Advanced AD & Reparameterization (Week 5):**
    *   `lgamma` プリミティブの実装（Beta, Gamma分布などで正規化定数の微分に必要）。
    *   `Beta` 分布の `Implicit Reparameterization Gradients` 実装。
        *   `BinaryOp::BetaSample` の追加。
        *   CDFの有限差分近似による勾配計算。
    *   `exp` 組み込み関数の追加。