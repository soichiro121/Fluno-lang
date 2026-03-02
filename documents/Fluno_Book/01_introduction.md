# 第1章：導入 (Introduction)

Flunoの世界へようこそ。
Flunoは、信頼性が高く効率的なシステムを構築するために設計された、新しいプログラミング言語です。特に、**確率的プログラミング (Probabilistic Programming)**、**リアクティブプログラミング (Reactive Programming)**、そして**微分可能プログラミング (Differentiable Programming)** を第一級の市民として言語コアに統合している点が大きな特徴です。

ロボティクス、制御システム、AIエージェントの開発において、不確実性の扱いや時間変化する値の管理は避けて通れません。Flunoはこれらの課題に対し、型システムとランタイムレベルでの強力なサポートを提供します。

## Flunoの特徴

1.  **確率型 (Probabilistic Types)**:
    不確実な値を `Gaussian` や `Beta` などの確率分布として型安全に扱えます。ベイズ推論を言語レベルでサポートし、`sample` や `observe` といったプリミティブを通じて直感的にモデルを記述できます。

2.  **リアクティブ型 (Reactive Types)**:
    センサーデータのような時間とともに変化する値を `Signal<T>` 型として扱います。データフローグラフを構築し、値の変更を自動的に伝播させることで、複雑な非同期処理を宣言的に記述できます。

3.  **自動微分 (Automatic Differentiation)**:
    ニューラルネットワークや物理シミュレーションのために、勾配計算を自動化します。`ADFloat` 型を通じて、スカラー値だけでなくテンソル計算に対しても逆伝播 (Backpropagation) が可能です。

## インストール

Flunoを使用するには、まずコンパイラとツールチェーンをインストールする必要があります。
(※現在は開発中のため、ソースコードからのビルドを想定しています)

```bash
$ git clone https://github.com/your-org/flux.git
$ cd flux
$ cargo build --release
```

## Hello, World!

新しい言語を学ぶ伝統に従い、画面にテキストを表示するプログラムから始めましょう。

### プロジェクトディレクトリの作成

```bash
$ mkdir hello_world
$ cd hello_world
```

### ソースコードの記述

`main.fln` というファイルを作成し、以下のコードを記述します。

```rust
// main.fln
fn main() {
    println("Hello, world!");
}
```

### コンパイルと実行

```bash
$ fluno main.fln
Hello, world!
```

おめでとうございます！ これであなたもFlunoプログラマーの仲間入りです。
次の章では、Flunoの基本的な文法と概念について学んでいきます。
