# Fluno Programming Language

<div align="center">
  <img src="assets/transparency-logo.png" alt="Fluno Logo" width="200" style="display: none;" onerror="this.style.display='none'">
  <p>
    <strong>Probabilistic, Reactive, and Differentiable Programming for Autonomous Systems</strong>
  </p>
  <p>
    <a href="documents/Fluno_Book/README.md">ドキュメント</a> |
    <a href="documents/fluxver2.md">仕様書</a>
  </p>
</div>

---

**Fluno** は、ロボティクス、自律システム、および高度なAIアプリケーションのために設計された、新しい静的型付けプログラミング言語です。
**不確実性 (Uncertainty)**、**時間変化 (Time-varying values)**、そして**学習 (Learning)** を言語のコアプリミティブとして統合することで、複雑な現実世界の問題をエレガントかつ安全に記述することを目指しています。

## 主な特徴

### 1. 確率的プログラミング (Probabilistic Programming)
Flunoは確率分布（`Gaussian`, `Uniform`, `Beta` 等）を第一級の型としてサポートします。ベイズ推論エンジン（HMC, VI）を内蔵しており、`sample` や `observe` 構文を用いて直感的に生成モデルを記述・推論できます。

```rust
let x = sample(Gaussian(10.0, 2.0));
observe(Gaussian(x, 1.0), data);
```

### 2. リアクティブプログラミング (Reactive Programming)
センサー入力などの時間とともに変化する値を `Signal<T>` 型として扱います。データフローグラフを自動的に構築・管理し、複雑な非同期イベント処理を宣言的に記述できます。

```rust
let distance: Signal<Float> = sensor.map(|v| v * scale);
```

### 3. 微分可能プログラミング (Differentiable Programming)
自動微分 (Automatic Differentiation, AD) エンジンを搭載。スカラーやテンソル計算の勾配を自動的に計算し、勾配降下法による最適化やニューラルネットワークのパラメータ学習を言語レベルでサポートします。

### 4. 安全で効率的
*   **静的型付け**: 強力な型推論機能を備え、コンパイル時に多くのエラーを検出します。
*   **メモリ管理**: ガベージコレクションを採用し、所有権の複雑さを隠蔽しつつ安全性を確保します。
*   **現代的な構文**: Rustライクな構文を採用し、構造体、列挙型、パターンマッチングなどをサポートします。

## ドキュメント (Documentation)

Flunoの学習には、公式ガイドブック **[The Fluno Programming Language](documents/Fluno_Book/README.md)** を参照してください。

*   **[第1章：導入](documents/Fluno_Book/01_introduction.md)**
*   **[第6章：確率的プログラミング](documents/Fluno_Book/06_probabilistic.md)**
*   **[第7章：リアクティブプログラミング](documents/Fluno_Book/07_reactive.md)**
*   **[第8章：自動微分](documents/Fluno_Book/08_autodiff.md)**

## セットアップ (Getting Started)

### 必要要件
*   Rust 1.70 以上
*   Cargo

### インストール
ソースコードからビルドします。

```bash
git clone https://github.com/your-org/flux.git
cd flux
cargo build --release
```

### 実行
現在の開発版では、Flunoのソースコードをインタプリタで実行できます。

```bash
# ヘルプの表示
cargo run -- --help

# ファイルの実行
cargo run -- examples/test.fln
```

## プロジェクト構造 (Project Structure)

Flunoのコンパイラとランタイムは、モジュール化されたRustプロジェクトとして構成されています。

```text
src/
├── ad/           # 自動微分 (Automatic Differentiation) エンジン
├── ast/          # 抽象構文木 (AST) 定義
├── bytecode/     # バイトコード定義とコンパイラ
├── compiler/     # ネイティブコンパイラ (LLVM/Cranelift等へのバックエンド)
├── gc/           # ガベージコレクション (Garbage Collection)
├── lexer/        # 字句解析器 (Lexer)
├── lsp/          # Language Server Protocol (LSP) 実装
├── parser/       # 構文解析器 (Parser)
├── stdlib/       # 標準ライブラリ (.fln ファイル群)
├── typeck/       # 型検査と型推論 (Type Checker)
├── vm/           # 仮想マシン (Runtime, PPL/FRPエンジンのコア)
├── main.rs       # エントリーポイント
└── lib.rs        # ライブラリルート
```

## 開発ステータス (Status)

現在は **Alpha** ステージです。コア機能の実装が進んでいますが、仕様変更の可能性があります。

*   [x] 基本的な文法と型システム
*   [x] インタプリタ (AST & Bytecode VM)
*   [x] ガベージコレクション
*   [x] 確率的プログラミング (サンプリング, HMC基礎)
*   [x] 自動微分 (スカラー)
*   [ ] リアクティブプログラミング (Signal伝播の最適化)
*   [ ] ネイティブコード生成
*   [ ] パッケージマネージャ

## 貢献 (Contributing)

バグ報告、機能提案、プルリクエストを歓迎します！

## ライセンス (License)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
