# The Fluno Programming Language

このドキュメントは、Flunoプログラミング言語の公式ガイドブックです。"The Rust Programming Language" に倣い、言語の基礎から高度な機能までを網羅的かつ厳密に解説します。

## 目次

### 第1部：入門
- [第1章：導入](01_introduction.md)
    - Flunoとは
    - インストールとセットアップ
    - Hello, World!
- [第2章：基本概念](02_basic_concepts.md)
    - 変数と可変性
    - データ型
    - 関数
    - コメント
    - 制御フロー

### 第2部：所有権とメモリ管理
- [第3章：メモリ管理システム](03_memory_management.md)
    - ガベージコレクションと参照
    - 値のコピーとクローン

### 第3部：データ構造とモジュール
- [第4章：構造体と列挙型](04_structs_enums.md)
    - 構造体の定義
    - 列挙型とパターンマッチング
- [第5章：モジュールシステム](05_modules.md)
    - パッケージとクレート
    - スコープと可視性

### 第4部：Flunoの核心（Advanced）
- [第6章：確率的プログラミング (PPL)](06_probabilistic.md)
    - 確率分布とサンプリング
    - `sample` と `observe`
    - 推論エンジン
- [第7章：リアクティブプログラミング (FRP)](07_reactive.md)
    - Signalとデータフロー
    - 時間変化する値
- [第8章：自動微分 (AD)](08_autodiff.md)
    - 微分可能プログラミング
    - 勾配計算と最適化

### 第5部：付録
- [付録A：キーワード](appendix_a_keywords.md)
- [付録B：標準ライブラリリファレンス](appendix_b_stdlib.md)
