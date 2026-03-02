# 付録B：標準ライブラリリファレンス (Standard Library)

Flunoの標準ライブラリは、言語のコア機能を拡張し、汎用的なツールを提供します。ここでは主要なモジュールの概要を説明します。

## `std::core`

言語の最も基本的なプリミティブ型やトレイトが含まれています。
*   `Int`, `Float`, `Bool`, `Char` などの基本型
*   `Option`, `Result` 型

## `std::prob`

確率的プログラミング (PPL) のための機能群です。
*   **分布**: `Gaussian`, `Uniform`, `Bernoulli`, `Beta` など
*   **トレイト**: `Distribution` (分布が実装すべきインターフェース)
*   **推論**: `infer_hmc`, `infer_vi` などの推論エンジンエントリーポイント

## `std::reactive`

リアクティブプログラミング (FRP) のための機能群です。
*   `Signal<T>`: 時間とともに変化する値
*   `Event<T>`: 離散的なイベントストリーム
*   演算子: `map`, `filter`, `fold` などを使ったデータフロー構築

## `std::collections`

汎用的なデータ構造を提供します。
*   `Vec<T>`: 可変長配列（ベクタ）
*   `HashMap<K, V>`: ハッシュマップ
*   `String`: UTF-8文字列

## `std::io`

入出力に関する機能です。
*   `println`: 標準出力への表示
*   ファイル操作（将来実装予定）

## `std::math`

数学的な関数や定数です。
*   三角関数 (`sin`, `cos`, `tan`)
*   指数・対数 (`exp`, `log`)
*   定数 (`PI`, `E`)
*   これらは `ADFloat` に対応しており、自動微分が適用可能です。
