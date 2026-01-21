// ==========================================
// stdlib/prob.flux
// Gaussian分布ライブラリ
// ==========================================

// 円周率の定義 (組み込み定数がない場合のフォールバック)
let PI = 3.141592653589793

// 構造体定義
struct Gaussian {
    mean: Float,
    std: Float
}

// ==========================================
// 1. コンストラクタ & ファクトリ
// ==========================================

impl Gaussian {
    // コンストラクタ
    fn new(mean: Float, std: Float): Gaussian {
        Gaussian { mean: mean, std: std }
    }

    // 標準正規分布 N(0, 1)
    fn standard(): Gaussian {
        Gaussian { mean: 0.0, std: 1.0 }
    }
}

// ==========================================
// 2. 演算子オーバーロード
// ==========================================

// 加算: N(μ1, σ1) + N(μ2, σ2)
// 分散の加法性: σ^2 = σ1^2 + σ2^2
fn (g1: Gaussian) + (g2: Gaussian): Gaussian {
    Gaussian {
        mean: g1.mean + g2.mean,
        std: sqrt(g1.std * g1.std + g2.std * g2.std)
    }
}

// 減算: N(μ1, σ1) - N(μ2, σ2)
fn (g1: Gaussian) - (g2: Gaussian): Gaussian {
    Gaussian {
        mean: g1.mean - g2.mean,
        std: sqrt(g1.std * g1.std + g2.std * g2.std)
    }
}

// スカラー乗算 (右): Gaussian * Float
fn (g: Gaussian) * (scalar: Float): Gaussian {
    Gaussian {
        mean: g.mean * scalar,
        std: g.std * abs(scalar)
    }
}

// スカラー乗算 (左): Float * Gaussian
fn (scalar: Float) * (g: Gaussian): Gaussian {
    Gaussian {
        mean: g.mean * scalar,
        std: g.std * abs(scalar)
    }
}

// スカラー除算: Gaussian / Float
fn (g: Gaussian) / (scalar: Float): Gaussian {
    Gaussian {
        mean: g.mean / scalar,
        std: g.std / abs(scalar)
    }
}

// 積: N(μ1, σ1) * N(μ2, σ2)
// 積の分布は正規分布にならないが、モーメントマッチングによる近似を行う
fn (g1: Gaussian) * (g2: Gaussian): Gaussian {
    let mu = g1.mean * g2.mean
    // 変動係数の二乗和による近似
    let rel_var = (g1.std / g1.mean) * (g1.std / g1.mean) + 
                  (g2.std / g2.mean) * (g2.std / g2.mean)
    Gaussian {
        mean: mu,
        std: abs(mu) * sqrt(rel_var)
    }
}

// ==========================================
// 3. 確率・統計メソッド
// ==========================================

impl Gaussian {
    // PDF (確率密度関数)
    fn pdf(self, x: Float): Float {
        let z = (x - self.mean) / self.std
        (1.0 / (self.std * sqrt(2.0 * PI))) * exp(-0.5 * z * z)
    }

    // CDF (累積分布関数)
    // ※ erf (誤差関数) はVMのBuiltinとして実装されている前提
    fn cdf(self, x: Float): Float {
        let z = (x - self.mean) / (self.std * sqrt(2.0))
        0.5 * (1.0 + erf(z))
    }

    // サンプル生成 (Box-Muller法)
    // VMネイティブ実装がある場合はそれを呼ぶ形でも良いが、
    // ここではFluxコードでの実装例を示す
    fn sample(self): Float {
        // 0.0~1.0の一様乱数 (VM Builtin)
        let u1 = random()
        let u2 = random()
        
        let z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)
        self.mean + self.std * z
    }
}

// ==========================================
// 4. 信頼区間・統計値
// ==========================================

impl Gaussian {
    // 95%信頼区間 (±1.96σ)
    // 戻り値: Tuple (lower, upper)
    fn ci_95(self): (Float, Float) {
        let margin = 1.96 * self.std
        (self.mean - margin, self.mean + margin)
    }

    // 99%信頼区間 (±2.576σ)
    fn ci_99(self): (Float, Float) {
        let margin = 2.576 * self.std
        (self.mean - margin, self.mean + margin)
    }

    // 変動係数 (CV)
    fn cv(self): Float {
        self.std / self.mean
    }
    
    // 分散 (Variance)
    fn variance(self): Float {
        self.std * self.std
    }
}
