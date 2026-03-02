# 第6章：確率的プログラミング (Probabilistic Programming)

Flunoの最大の特徴は、**確率的プログラミング (Probabilistic Programming)** を言語レベルでサポートしていることです。これにより、不確実性を含むモデルを直感的に記述し、データから未知のパラメータを推論することができます。

## 6.1 確率分布 (Probability Distributions)

Flunoにおいて、不確実性は「確率分布」として表現されます。標準ライブラリ `std::prob` (およびプレリュード) は、一般的な確率分布を提供しています。

### 基本的な分布

```rust
// 平均 10.0, 標準偏差 2.0 の正規分布
let g = Gaussian(10.0, 2.0);

// 0.0 から 1.0 の間の一様分布
let u = Uniform(0.0, 1.0);

// 成功確率 0.5 のベルヌーイ分布（コイン投げ）
let b = Bernoulli(0.5);
```

これらの分布自体は単なるオブジェクトであり、作成しただけでは何も起きません。これらを「サンプリング」することで初めて具体的な値が得られます。

## 6.2 sample と observe

確率的プログラミングの核心となる2つのプリミティブ操作があります。

### sample: 分布からの値の生成

`sample` 関数は、確率分布から値を一つ生成します。

```rust
let x = sample(Gaussian(0.0, 1.0));
println("Sampled value: {}", x);
```

しかし、Flunoにおいては `sample` は単なる乱数生成以上の意味を持ちます。**推論コンテキスト**の中で実行された場合、`sample` は確率変数として追跡され、その値が観測データと整合するように「推論」の対象となります。

### observe: 観測データの条件付け

`observe` 関数は、モデルから生成された値が、実際のデータ（観測値）とどの程度一致しているかをシステムに伝えます（尤度の計算）。

```rust
// モデル：真の値は平均10の正規分布に従うと仮定
let true_val = sample(Gaussian(10.0, 5.0));

// 観測：センサーの誤差は標準偏差1.0であり、実際に観測された値は 12.5 だった
observe(Gaussian(true_val, 1.0), 12.5);
```

このコードは、「`true_val` という未知の値があり、それをノイズ付きで測ったら 12.5 だった」という状況を記述しています。推論エンジンはこの情報を使って、`true_val` の事後分布（データを見た後の確からしい値の分布）を計算します。

## 6.3 推論の実践 (Inference)

確率モデルを記述したら、推論エンジンを使って事後分布を求めます。
Flunoは、HMC (Hamiltonian Monte Carlo) や VI (Variational Inference) などの高度な推論アルゴリズムをサポートしています。

### 例：コインのバイアス推定

コインの表が出る確率 `p` が未知だとします。コインを投げた結果（データ）から `p` を推定してみましょう。

```rust
fn coin_model(data: Array<Bool>) -> Float {
    // 事前分布：p は 0 から 1 の一様分布に従うと仮定（何も知らない状態）
    // Beta分布などもよく使われます
    let p = sample(Uniform(0.0, 1.0));

    // データへの適合
    for is_heads in data {
        // コイン投げの結果を観測
        // is_heads が true なら 1.0, false なら 0.0 として observe
        let obs_val = if is_heads { 1.0 } else { 0.0 };
        observe(Bernoulli(p), obs_val);
    }

    return p;
}

fn main() {
    // 観測データ：5回表、2回裏が出た
    let data = [true, true, true, false, true, false, true];

    // HMCを使って推論を実行
    // configは省略（デフォルト設定）
    let posterior_samples = infer_hmc(hmc_config(), || coin_model(data));

    // 結果の平均を表示
    let mean_p = posterior_samples.iter().sum() / posterior_samples.len();
    println("Estimated bias p: {}", mean_p);
}
```

このように、Flunoを使えば「モデルの記述」と「推論の実行」を明確に分離でき、複雑な統計モデルもシンプルに実装することが可能です。
