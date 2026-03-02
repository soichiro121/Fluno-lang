***

# Flux プログラミング言語 仕様書 v2.0

**要件定義・言語仕様・実装ガイド統合版**

***

## 文書情報

| 項目 | 内容 |
|------|------|
| 文書名 | Flux プログラミング言語 仕様書  |
| バージョン | 2.0 |
| 最終更新日 | 2025年10月9日 |
| 文書種別 | 言語仕様書・要件定義書・実装ガイド |
| 対象読者 | 言語設計者、コンパイラ実装者、ライブラリ開発者、ユーザー |
| ステータス | ドラフト |

## 改訂履歴

| 版 | 日付 | 変更内容 | 作成者 |
|----|------|----------|--------|
| 1.0 | 2023-8-07 | 初版作成 | soichiro_n |
| 2.0 | 2025-10-09 | 第２版作成 | soichiro_n |

***

## 目次

### 第1部：概要と設計思想
1. [Fluxとは](#第1章-fluxとは)
2. [設計思想と設計原則](#第2章-設計思想と設計原則)
3. [用語定義](#第3章-用語定義)

### 第2部：言語仕様
4. [字句構造](#第4章-字句構造)
5. [型システム](#第5章-型システム)
6. [式と文](#第6章-式と文)
7. [確率型](#第7章-確率型)
8. [リアクティブ型](#第8章-リアクティブ型)
9. [関数とクロージャ](#第9章-関数とクロージャ)
10. [構造体とenum](#第10章-構造体とenum)
11. [並行性と同期](#第11章-並行性と同期)
12. [メモリ管理](#第12章-メモリ管理)

### 第3部：標準ライブラリ
13. [標準ライブラリ概要](#第13章-標準ライブラリ概要)
14. [coreモジュール](#第14章-coreモジュール)
15. [probモジュール](#第15章-probモジュール)
16. [reactiveモジュール](#第16章-reactiveモジュール)
17. [collectionsモジュール](#第17章-collectionsモジュール)
18. [ioモジュール](#第18章-ioモジュール)
19. [syncモジュール](#第19章-syncモジュール)
20. [mathモジュール](#第20章-mathモジュール)

### 第4部：実行環境
21. [メモリレイアウト](#第21章-メモリレイアウト)
22. [実行時システム](#第22章-実行時システム)
23. [ガベージコレクション](#第23章-ガベージコレクション)
24. [FFI（外部関数インターフェース）](#第24章-ffi外部関数インターフェース)

### 第5部：ツールチェーン
25. [コンパイラ](#第25章-コンパイラ)
26. [パッケージマネージャ](#第26章-パッケージマネージャ)
27. [ビルドシステム](#第27章-ビルドシステム)
28. [テストフレームワーク](#第28章-テストフレームワーク)
29. [ドキュメント生成](#第29章-ドキュメント生成)
30. [デバッガ](#第30章-デバッガ)
31. [プロファイラ](#第31章-プロファイラ)
32. [Language Server Protocol](#第32章-language-server-protocol)

### 第6部：形式的仕様
33. [形式的意味論](#第33章-形式的意味論)
34. [型システムの健全性](#第34章-型システムの健全性)

### 第7部：実装要件
35. [実装ガイドライン](#第35章-実装ガイドライン)
36. [プラットフォーム要件](#第36章-プラットフォーム要件)
37. [パフォーマンス要件](#第37章-パフォーマンス要件)
38. [セキュリティ要件](#第38章-セキュリティ要件)

### 第8部：付録
39. [完全な文法（EBNF）](#第39章-完全な文法ebnf)
40. [標準エラーコード一覧](#第40章-標準エラーコード一覧)
41. [互換性ポリシー](#第41章-互換性ポリシー)
42. [移行ガイド](#第42章-移行ガイド)
43. [参考文献](#第43章-参考文献)

***

# 第1部：概要と設計思想

***

## 第1章 Fluxとは

### 1.1 概要

Flux（フラックス）は、**確率的プログラミング**と**リアクティブプログラミング**を統合した、ロボティクス・自律システム向けの静的型付けプログラミング言語である。

### 1.2 目的

Fluxは以下の問題を解決するために設計された：

1. **不確実性の扱い**：センサーノイズ、モデル誤差、環境の不確実性
2. **時間変化する値**：連続的に変化するセンサー値、状態遷移
3. **並行処理**：複数のセンサーとアクチュエータの協調動作
4. **実時間性**：厳格なタイミング制約

### 1.3 特徴

#### 1.3.1 核心的特徴

1. **確率型（Probabilistic Types）**
   - 不確実性を型として表現
   - 自動的な誤差伝播
   - ベイズ推論のサポート

2. **リアクティブ型（Reactive Types）**
   - 時変値を型として表現
   - 宣言的なデータフロー
   - 自動的な依存関係管理

3. **型安全な並行性**
   - データ競合の静的検出
   - デッドロックフリーの保証
   - 明示的な同期プリミティブ

4. **シンプルなメモリ管理**
   - ガベージコレクション
   - RAIIによるリソース管理
   - 手動メモリ管理不要

#### 1.3.2 言語レベルの特徴

- **静的型付け**：コンパイル時の型チェック
- **型推論**：冗長な型注釈の削減
- **パターンマッチング**：安全なデータ分解
- **高階関数**：関数を第一級オブジェクトとして扱う
- **ジェネリクス**：型パラメータによる汎用プログラミング

### 1.4 Hello World

```flux
fn main() {
    println("Hello, Flux!");
}
```

### 1.5 確率的プログラミングの例

```flux
fn sensor_reading() -> Gaussian {
    // センサー値：平均10.0、標準偏差0.5
    let reading = Gaussian(10.0, 0.5);
    
    // 自動的に誤差伝播
    let doubled = reading * 2.0;
    
    println("Mean: {}, Std: {}", doubled.mean, doubled.std);
    return doubled;
}
```

### 1.6 リアクティブプログラミングの例

```flux
fn temperature_monitor() {
    // 温度センサーからSignalを生成
    let celsius: Signal<Float> = Signal::from_sensor(thermometer);
    
    // 摂氏を華氏に変換
    let fahrenheit: Signal<Float> = celsius.map(|c| c * 1.8 + 32.0);
    
    // 閾値を超えたら警告
    fahrenheit
        .filter(|f| f > 100.0)
        .subscribe(|f| {
            println("Warning: Temperature {}°F", f);
        });
}
```

### 1.7 適用領域

Fluxは以下の領域での使用を想定している：

1. **自律移動ロボット**
2. **ドローン制御**
3. **産業用オートメーション**
4. **自動運転システム**
5. **センサーネットワーク**
6. **IoTデバイス**

### 1.8 非適用領域

Fluxは以下の用途には適していない：

1. **Webフロントエンド開発**
2. **汎用的なシステムプログラミング**（OS開発など）
3. **超高速数値計算**（専用のFortran/C++が優れる）
4. **組み込みシステム**（メモリ制約が厳しい場合）

***

## 第2章 設計思想と設計原則

### 2.1 設計思想

#### 2.1.1 不確実性の一級市民化

**原則**：不確実性は例外的なケースではなく、プログラムの通常の一部である。

**実現方法**：
- 確率分布を基本型として提供
- 演算子オーバーロードによる自然な記法
- 自動的な誤差伝播

**例**：
```flux
let sensor1 = Gaussian(10.0, 1.0);
let sensor2 = Gaussian(5.0, 2.0);
let fused = sensor1 + sensor2;  // 自動的に誤差伝播
```

#### 2.1.2 時間の明示的表現

**原則**：時間とともに変化する値は、通常の値とは異なる型で表現する。

**実現方法**：
- Signal型による時変値の表現
- 宣言的なデータフロー
- 自動的な依存関係追跡

**例**：
```flux
let position: Signal<Float> = ...;
let velocity: Signal<Float> = position.derivative();
```

#### 2.1.3 シンプルさ優先

**原則**：複雑な機能よりも、シンプルで理解しやすい言語を優先する。

**実現方法**：
- 所有権システムの排除（GCを使用）
- エフェクトシステムの排除（ドキュメントで対応）
- 依存型の排除（実行時チェックで対応）

#### 2.1.4 実用性重視

**原則**：理論的な美しさよりも、実問題を解決できることを優先する。

**実現方法**：
- 充実した標準ライブラリ
- FFIによる既存コードの活用
- 実行時パフォーマンスの最適化

### 2.2 設計原則

#### 2.2.1 型安全性

**原則SP-001**：すべての型エラーはコンパイル時に検出されなければならない。

**要件**：
- R-SP-001-01：整数と浮動小数点の暗黙の変換を禁止する
- R-SP-001-02：nullポインタ参照を型システムで排除する
- R-SP-001-03：配列の境界チェックを実行時に行う

#### 2.2.2 メモリ安全性

**原則SP-002**：Use-After-Free、ダングリングポインタ、二重解放を防止する。

**要件**：
- R-SP-002-01：ガベージコレクションによる自動メモリ管理
- R-SP-002-02：RAIIによるリソース管理
- R-SP-002-03：ポインタ演算の禁止

#### 2.2.3 並行安全性

**原則SP-003**：データ競合とデッドロックをコンパイル時に検出する。

**要件**：
- R-SP-003-01：共有可変状態を型システムで制限
- R-SP-003-02：Signal依存グラフの循環検出
- R-SP-003-03：ロック順序の静的検証

#### 2.2.4 予測可能性

**原則SP-004**：プログラムの動作は予測可能でなければならない。

**要件**：
- R-SP-004-01：暗黙の型変換を最小限にする
- R-SP-004-02：副作用を明示的にする
- R-SP-004-03：実行順序を明確にする

#### 2.2.5 段階的学習

**原則SP-005**：初心者でも段階的に学習できる。

**要件**：
- R-SP-005-01：基本的な機能は単純にする
- R-SP-005-02：高度な機能は明示的にオプトインする
- R-SP-005-03：エラーメッセージは分かりやすくする

### 2.3 削除した機能とその理由

#### 2.3.1 所有権型

**削除理由**：
1. 確率型・リアクティブ型と根本的に衝突する
2. 学習コストが高い
3. メモリ安全性はGCで実現可能

**代替案**：
- ガベージコレクション（参照カウント）
- RAIIによるリソース管理

**例**：
```flux
// 所有権なし - 自由にコピー可能
let data = [1, 2, 3];
let data2 = data;  // OK
let data3 = data;  // OK
```

#### 2.3.2 エフェクト型

**削除理由**：
1. リアクティブ型が暗黙のエフェクトを持つ
2. 実用的な言語ではエフェクト型は必須ではない
3. 冗長性が増す

**代替案**：
- ドキュメンテーションコメント
- 命名規則（`read_*`, `write_*`など）

**例**：
```flux
/// Reads from the sensor (side effect: Sensor)
fn read_sensor() -> Float {
    // ...
}
```

#### 2.3.3 依存型

**削除理由**：
1. 型推論が決定不能になる
2. 実装が非常に複雑
3. 実行時チェックで十分なケースが多い

**代替案**：
- 動的サイズの配列
- 実行時の境界チェック
- 事前条件・事後条件のアサーション

**例**：
```flux
// 動的サイズ
fn dot_product(a: Array<Float>, b: Array<Float>) -> Result<Float> {
    if a.len() != b.len() {
        return Err("Array lengths must match");
    }
    // ...
}
```

***

## 第3章 用語定義

### 3.1 基本用語

| 用語 | 定義 |
|------|------|
| **項（Term）** | 式または文 |
| **式（Expression）** | 値に評価されるプログラムの構成要素 |
| **文（Statement）** | 値を持たず、副作用を持つプログラムの構成要素 |
| **値（Value）** | それ以上評価できない項 |
| **型（Type）** | 値の集合と、その値に対して可能な操作の集合 |

### 3.2 型システム用語

| 用語 | 定義 |
|------|------|
| **型付け（Typing）** | 項に型を割り当てる過程 |
| **型推論（Type Inference）** | プログラマが明示的に型を書かなくても、コンパイラが型を導出すること |
| **型エラー（Type Error）** | 型規則に違反すること |
| **健全性（Soundness）** | 型チェックを通過したプログラムが実行時型エラーを起こさないこと |
| **完全性（Completeness）** | 実行時型エラーを起こさないプログラムがすべて型チェックを通過すること |

### 3.3 確率的プログラミング用語

| 用語 | 定義 |
|------|------|
| **確率分布（Probability Distribution）** | 確率変数の値とその確率の対応 |
| **正規分布（Gaussian Distribution）** | 平均と標準偏差で特徴づけられる連続確率分布 |
| **サンプリング（Sampling）** | 確率分布から具体的な値を生成すること |
| **誤差伝播（Error Propagation）** | 入力の不確実性が出力にどう影響するかを計算すること |
| **ベイズ推論（Bayesian Inference）** | 事前分布と尤度から事後分布を計算すること |

### 3.4 リアクティブプログラミング用語

| 用語 | 定義 |
|------|------|
| **Signal** | 時間とともに連続的に変化する値 |
| **Event** | 離散的に発生する出来事 |
| **データフロー（Data Flow）** | データの流れを表現するプログラミングスタイル |
| **依存グラフ（Dependency Graph）** | Signalの依存関係を表すグラフ |
| **購読（Subscribe）** | Signalの変化を監視すること |

### 3.5 並行性用語

| 用語 | 定義 |
|------|------|
| **スレッド（Thread）** | 独立した実行の流れ |
| **データ競合（Data Race）** | 複数のスレッドが同時に同じメモリにアクセスし、少なくとも1つが書き込みである状況 |
| **デッドロック（Deadlock）** | 複数のスレッドが互いにロックを待ち続ける状況 |
| **ミューテックス（Mutex）** | 相互排他ロック |
| **チャネル（Channel）** | スレッド間通信のためのキュー |

### 3.6 略語

| 略語 | 正式名称 |
|------|----------|
| **FFI** | Foreign Function Interface（外部関数インターフェース） |
| **GC** | Garbage Collection（ガベージコレクション） |
| **RAII** | Resource Acquisition Is Initialization |
| **LSP** | Language Server Protocol |
| **AST** | Abstract Syntax Tree（抽象構文木） |
| **IR** | Intermediate Representation（中間表現） |

***

# 第2部：言語仕様

***

## 第4章 字句構造

### 4.1 概要

この章では、Fluxプログラムの字句構造（トークン）について定義する。

### 4.2 文字セット

#### 4.2.1 ソースコード文字セット

**要件REQ-LEX-001**：FluxのソースコードはUTF-8でエンコードされなければならない。

**要件REQ-LEX-002**：BOMは許可されるが、推奨されない。

#### 4.2.2 改行

**要件REQ-LEX-003**：以下の改行表現を認識しなければならない：
- LF（`\n`, U+000A）
- CRLF（`\r\n`, U+000D U+000A）

**要件REQ-LEX-004**：CR単独（`\r`）は改行として認識してはならない。

### 4.3 キーワード

#### 4.3.1 キーワード一覧

**要件REQ-LEX-010**：以下のキーワードは予約語であり、識別子として使用してはならない：

```flux
// 制御フロー
fn let if else match while for loop break continue return

// 型宣言
struct enum impl trait type

// 可視性
pub priv

// モジュール
mod use import as

// 並行性
async await spawn

// その他
true false self Self
```

#### 4.3.2 将来の予約語

**要件REQ-LEX-011**：以下のキーワードは将来の拡張のために予約する：

```flux
abstract final override virtual
class interface extends implements
macro yield generator
unsafe const static mut ref
try catch throw finally
```

### 4.4 識別子

#### 4.4.1 識別子の構文

**EBNF定義**：
```ebnf
identifier = identifier_start identifier_continue*
identifier_start = letter | '_'
identifier_continue = letter | digit | '_'
letter = 'a'..'z' | 'A'..'Z'
digit = '0'..'9'
```

#### 4.4.2 識別子の制約

**要件REQ-LEX-020**：識別子は以下の制約を満たさなければならない：
- 長さは1文字以上、255文字以下
- キーワードと同じであってはならない
- アンダースコア1文字のみ（`_`）は特別な意味を持つ（未使用変数）

**例**：
```flux
// OK
x
myVariable
_private
add_numbers
ClassName
_123

// NG
123abc  // 数字から始まる
fn      // キーワード
あいう  // 非ASCII（将来的にはサポート予定）
```

### 4.5 リテラル

#### 4.5.1 整数リテラル

**EBNF定義**：
```ebnf
integer_literal = decimal_literal | hex_literal | octal_literal | binary_literal
decimal_literal = digit (digit | '_')*
hex_literal = '0x' hex_digit (hex_digit | '_')*
octal_literal = '0o' octal_digit (octal_digit | '_')*
binary_literal = '0b' binary_digit (binary_digit | '_')*
hex_digit = digit | 'a'..'f' | 'A'..'F'
octal_digit = '0'..'7'
binary_digit = '0' | '1'
```

**要件REQ-LEX-030**：整数リテラルの値は-2^63以上、2^63-1以下でなければならない。

**例**：
```flux
0
42
-10
1_000_000
0xFF
0o755
0b1010
```

#### 4.5.2 浮動小数点リテラル

**EBNF定義**：
```ebnf
float_literal = decimal_literal '.' decimal_literal exponent? |
                decimal_literal exponent
exponent = ('e' | 'E') ('+' | '-')? decimal_literal
```

**要件REQ-LEX-031**：浮動小数点リテラルはIEEE 754 binary64形式で表現される。

**例**：
```flux
0.0
3.14
-2.5
1.0e-10
6.022e23
```

#### 4.5.3 真偽値リテラル

**EBNF定義**：
```ebnf
boolean_literal = 'true' | 'false'
```

#### 4.5.4 文字列リテラル

**EBNF定義**：
```ebnf
string_literal = '"' string_content* '"'
string_content = escape_sequence | (any_character - '"' - '\')
escape_sequence = '\' ('"' | '\' | 'n' | 'r' | 't' | 'u' hex_digit{4})
```

**要件REQ-LEX-040**：文字列リテラルは以下のエスケープシーケンスをサポートしなければならない：

| エスケープ | 意味 |
|-----------|------|
| `\"` | ダブルクォート |
| `\\` | バックスラッシュ |
| `\n` | 改行（LF） |
| `\r` | キャリッジリターン（CR） |
| `\t` | タブ |
| `\uXXXX` | Unicode文字（4桁の16進数） |

**例**：
```flux
"Hello, World!"
"Line 1\nLine 2"
"Escaped quote: \""
"Unicode: \u03B1"  // α
```

### 4.6 演算子と区切り文字

#### 4.6.1 演算子

**要件REQ-LEX-050**：以下の演算子を認識しなければならない：

```flux
// 算術演算子
+  -  *  /  %

// 比較演算子
==  !=  <  >  <=  >=

// 論理演算子
&&  ||  !

// ビット演算子
&  |  ^  <<  >>  ~

// 代入演算子
=  +=  -=  *=  /=  %=

// その他
.  ..  ...  ?  :
```

#### 4.6.2 区切り文字

**要件REQ-LEX-051**：以下の区切り文字を認識しなければならない：

```flux
(  )      // 括弧
{  }      // 波括弧
[  ]      // 角括弧
,         // カンマ
;         // セミコロン
:         // コロン
::        // パス区切り
->        // 関数の戻り値型
=>        // マッチ式の矢印
```

### 4.7 コメント

#### 4.7.1 行コメント

**EBNF定義**：
```ebnf
line_comment = '//' (any_character - newline)* newline
```

**例**：
```flux
// これは行コメント
let x = 10;  // 変数宣言
```

#### 4.7.2 ブロックコメント

**EBNF定義**：
```ebnf
block_comment = '/*' block_comment_content* '*/'
block_comment_content = block_comment | (any_character - '/*' - '*/')
```

**要件REQ-LEX-060**：ブロックコメントはネスト可能でなければならない。

**例**：
```flux
/* これはブロックコメント */

/*
 * 複数行の
 * コメント
 */

/* 外側 /* 内側 */ 外側 */  // ネスト可能
```

#### 4.7.3 ドキュメンテーションコメント

**EBNF定義**：
```ebnf
doc_comment = '///' (any_character - newline)* newline |
              '/**' doc_comment_content* '*/'
```

**要件REQ-LEX-061**：ドキュメンテーションコメントはMarkdown形式で解釈される。

**例**：
```flux
/// この関数は2つの数を足す
///
/// # 引数
/// * `x` - 1つ目の数
/// * `y` - 2つ目の数
///
/// # 戻り値
/// 2つの数の和
fn add(x: Int, y: Int) -> Int {
    return x + y;
}
```

### 4.8 空白

#### 4.8.1 空白文字

**要件REQ-LEX-070**：以下の文字を空白として扱う：
- スペース（U+0020）
- タブ（U+0009）
- 改行（LF: U+000A, CRLF: U+000D U+000A）

**要件REQ-LEX-071**：空白は以下の場所で必須である：
- キーワードと識別子の間
- 2つの識別子の間
- 演算子と識別子の間（曖昧性がある場合）

**要件REQ-LEX-072**：空白は以下の場所では無視される：
- トークンの間（必須の場合を除く）
- 行末
- ファイル末尾

***

## 第5章 型システム

### 5.1 概要

Fluxの型システムは、静的型付け、型推論、ジェネリクスをサポートする。

### 5.2 型の分類

**要件REQ-TYPE-001**：Fluxの型は以下のカテゴリに分類される：

```
Type ::=
  | PrimitiveType        // 基本型
  | ProbabilityType      // 確率型
  | ReactiveType         // リアクティブ型
  | FunctionType         // 関数型
  | CompositeType        // 複合型
  | GenericType          // ジェネリック型
```

### 5.3 基本型（Primitive Types）

#### 5.3.1 整数型（Int）

**定義**：
```flux
type Int = i64
```

**要件REQ-TYPE-010**：
- ビット幅：64ビット
- 符号：あり（2の補数表現）
- 範囲：-9,223,372,036,854,775,808 〜 9,223,372,036,854,775,807

**演算**：
```flux
+  -  *  /  %           // 算術演算
==  !=  <  >  <=  >=    // 比較演算
&  |  ^  <<  >>  ~      // ビット演算
```

**要件REQ-TYPE-011**：整数のオーバーフローは実行時エラーとなる。

**例**：
```flux
let x: Int = 42;
let y: Int = -10;
let z: Int = x + y;  // 32
```

#### 5.3.2 浮動小数点型（Float）

**定義**：
```flux
type Float = f64
```

**要件REQ-TYPE-020**：
- ビット幅：64ビット
- 形式：IEEE 754 binary64
- 特殊値：NaN, +Infinity, -Infinity

**演算**：
```flux
+  -  *  /              // 算術演算
==  !=  <  >  <=  >=    // 比較演算
```

**要件REQ-TYPE-021**：NaNの比較は以下のように動作する：
- `NaN == NaN` → `false`
- `NaN != NaN` → `true`
- NaNと他の値の比較 → `false`

**例**：
```flux
let pi: Float = 3.14159;
let e: Float = 2.71828;
let sum: Float = pi + e;
```

#### 5.3.3 真偽値型（Bool）

**定義**：
```flux
type Bool = true | false
```

**演算**：
```flux
&&  ||  !  ==  !=
```

**要件REQ-TYPE-030**：真偽値型は1ビットで表現されるが、メモリ上は1バイトを占有する。

**例**：
```flux
let flag: Bool = true;
let condition: Bool = x > 0 && y < 10;
```

#### 5.3.4 文字列型（String）

**定義**：
```flux
type String = Utf8String
```

**要件REQ-TYPE-040**：
- エンコーディング：UTF-8
- 不変（immutable）
- 内部表現：バイト列 + 長さ

**演算**：
```flux
+  // 連結
==  !=  <  >  <=  >=  // 辞書順比較
```

**例**：
```flux
let name: String = "Flux";
let greeting: String = "Hello, " + name;
let length: Int = greeting.len();
```

#### 5.3.5 単位型（Unit）

**定義**：
```flux
type Unit = ()
```

**要件REQ-TYPE-050**：単位型は唯一の値`()`を持つ。

**用途**：
- 戻り値を持たない関数
- 副作用のみを持つ式

**例**：
```flux
fn print_hello() -> Unit {
    println("Hello");
}

let result: Unit = print_hello();
```

### 5.4 確率型（Probability Types）

#### 5.4.1 正規分布型（Gaussian）

**定義**：
```flux
struct Gaussian {
    mean: Float,
    std: Float,
}
```

**要件REQ-TYPE-100**：
- `std`は常に正の値でなければならない（`std > 0.0`）
- 構築時に`std <= 0.0`の場合はパニック

**構築**：
```flux
fn Gaussian(mean: Float, std: Float) -> Gaussian {
    if std <= 0.0 {
        panic("Standard deviation must be positive");
    }
    return Gaussian { mean, std };
}
```

**演算**：

| 演算 | 意味論 | 結果の型 |
|------|--------|----------|
| `g1 + g2` | 独立な正規分布の和 | `Gaussian(μ1+μ2, √(σ1²+σ2²))` |
| `g1 - g2` | 独立な正規分布の差 | `Gaussian(μ1-μ2, √(σ1²+σ2²))` |
| `g * k` | スカラー倍 | `Gaussian(k*μ, |k|*σ)` |
| `g / k` | スカラー除算 | `Gaussian(μ/k, σ/|k|)` |

**メソッド**：
```flux
impl Gaussian {
    /// 確率密度関数の値を計算
    fn pdf(&self, x: Float) -> Float;
    
    /// 累積分布関数の値を計算
    fn cdf(&self, x: Float) -> Float;
    
    /// 分布からサンプリング
    fn sample(&self) -> Float;
    
    /// 信頼区間を計算
    /// confidence: 0.0 〜 1.0（例：0.95 = 95%信頼区間）
    fn confidence_interval(&self, confidence: Float) -> (Float, Float);
}
```

**例**：
```flux
let sensor1 = Gaussian(10.0, 1.0);
let sensor2 = Gaussian(5.0, 2.0);
let fused = sensor1 + sensor2;  // Gaussian(15.0, 2.236...)

let sample: Float = fused.sample();
let (low, high) = fused.confidence_interval(0.95);
```

#### 5.4.2 一様分布型（Uniform）

**定義**：
```flux
struct Uniform {
    min: Float,
    max: Float,
}
```

**要件REQ-TYPE-110**：
- `min < max`でなければならない
- 構築時に`min >= max`の場合はパニック

**メソッド**：
```flux
impl Uniform {
    fn sample(&self) -> Float;
    fn pdf(&self, x: Float) -> Float;
}
```

**例**：
```flux
let dice = Uniform(1.0, 7.0);  // 1.0 <= x < 7.0
let roll: Float = dice.sample();
```

### 5.5 リアクティブ型（Reactive Types）

#### 5.5.1 Signal型

**定義**：
```flux
type Signal<T>
```

**意味論**：
```
Signal<T> ≝ Time → T
where Time = ℕ
```

**要件REQ-TYPE-200**：Signalは以下の性質を持つ：
- 時間とともに連続的に変化する値
- 遅延評価（Lazy Evaluation）
- 自動的な依存関係追跡

**構築**：
```flux
impl<T> Signal<T> {
    /// 定数Signal
    fn new(initial: T) -> Signal<T>;
    
    /// センサーからSignalを生成
    fn from_sensor<H>(handle: H) -> Signal<T>;
}
```

**変換**：
```flux
impl<T> Signal<T> {
    /// 値を変換
    fn map<U>(&self, f: Fn(T) -> U) -> Signal<U>;
    
    /// 条件でフィルタ
    fn filter(&self, pred: Fn(T) -> Bool) -> Signal<Option<T>>;
    
    /// サンプリング間隔を指定
    fn sample_interval(&self, interval: Duration) -> Signal<T>;
    
    /// 最新値のみ使用（バックプレッシャー対策）
    fn map_latest<U>(&self, f: Fn(T) -> U) -> Signal<U>;
}
```

**結合**：
```flux
/// 2つのSignalを結合
fn combine<T, U, V>(
    s1: Signal<T>,
    s2: Signal<U>,
    f: Fn(T, U) -> V
) -> Signal<V>;

/// 3つのSignalを結合
fn combine3<T, U, V, W>(
    s1: Signal<T>,
    s2: Signal<U>,
    s3: Signal<V>,
    f: Fn(T, U, V) -> W
) -> Signal<W>;
```

**購読**：
```flux
impl<T> Signal<T> {
    /// Signalの変化を監視
    fn subscribe(&self, callback: Fn(T));
    
    /// 購読を解除
    fn unsubscribe(&self, id: SubscriptionId);
}
```

**例**：
```flux
let temperature: Signal<Float> = Signal::from_sensor(thermometer);
let fahrenheit: Signal<Float> = temperature.map(|c| c * 1.8 + 32.0);

fahrenheit.subscribe(|f| {
    println("Temperature: {}°F", f);
});
```

#### 5.5.2 Event型

**定義**：
```flux
type Event<T>
```

**意味論**：離散的に発生する出来事

**メソッド**：
```flux
impl<T> Event<T> {
    fn filter(&self, pred: Fn(T) -> Bool) -> Event<T>;
    fn map<U>(&self, f: Fn(T) -> U) -> Event<U>;
    fn merge(&self, other: Event<T>) -> Event<T>;
}
```

### 5.6 関数型（Function Types）

**定義**：
```flux
type Fn(T1, T2, ..., Tn) -> U
```

**要件REQ-TYPE-300**：関数型は第一級オブジェクトである。

**例**：
```flux
let add: Fn(Int, Int) -> Int = fn(x, y) { x + y };

let apply: Fn(Fn(Int) -> Int, Int) -> Int =
    fn(f, x) { f(x) };

let result = apply(fn(x) { x * 2 }, 10);  // 20
```

### 5.7 複合型（Composite Types）

#### 5.7.1 構造体（Struct）

**構文**：
```flux
struct StructName {
    field1: Type1,
    field2: Type2,
    ...
}
```

**例**：
```flux
struct Point {
    x: Float,
    y: Float,
}

let p = Point { x: 1.0, y: 2.0 };
let x = p.x;
```

#### 5.7.2 列挙型（Enum）

**構文**：
```flux
enum EnumName {
    Variant1,
    Variant2(Type),
    Variant3 { field: Type },
}
```

**例**：
```flux
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

#### 5.7.3 配列型（Array）

**定義**：
```flux
type Array<T> = [T]
```

**要件REQ-TYPE-310**：
- 動的サイズ
- 連続メモリ配置
- 境界チェックあり

**メソッド**：
```flux
impl<T> Array<T> {
    fn len(&self) -> Int;
    fn get(&self, index: Int) -> Option<T>;
    fn push(&mut self, item: T);
    fn pop(&mut self) -> Option<T>;
}
```

#### 5.7.4 タプル型

**構文**：
```flux
type (T1, T2, ..., Tn)
```

**例**：
```flux
let pair: (Int, Float) = (10, 3.14);
let (x, y) = pair;
```

### 5.8 ジェネリック型

**構文**：
```flux
struct GenericStruct<T> {
    field: T,
}

fn generic_function<T>(x: T) -> T {
    return x;
}
```

**制約**：
```flux
// トレイト境界
fn print_debug<T: Debug>(x: T) {
    println("{:?}", x);
}
```

### 5.9 型推論

**要件REQ-TYPE-400**：以下の場合、型注釈を省略できる：
- let文の変数宣言
- クロージャの引数
- 戻り値が明らかな場合

**例**：
```flux
let x = 10;  // Int と推論
let y = 3.14;  // Float と推論
let f = fn(x) { x * 2 };  // Fn(Int) -> Int と推論
```

### 5.10 型変換

#### 5.10.1 明示的変換

**要件REQ-TYPE-500**：基本型間の変換は明示的でなければならない。

```flux
let x: Int = 10;
let y: Float = x as Float;  // OK

let z: Float = 3.14;
let w: Int = z as Int;  // OK（切り捨て）
```

#### 5.10.2 暗黙の変換

**要件REQ-TYPE-501**：暗黙の型変換は以下の場合のみ許可される：
- 整数リテラルから`Int`または`Float`
- 配列リテラルから`Array<T>`

***

## 第6章 式と文

### 6.1 概要

この章では、Fluxの式（Expression）と文（Statement）の構文と意味論を定義する。

### 6.2 式（Expressions）

#### 6.2.1 式の分類

**要件REQ-EXPR-001**：Fluxの式は以下のカテゴリに分類される：

```
expr ::=
  | literal_expr         // リテラル式
  | variable_expr        // 変数参照
  | binary_expr          // 二項演算
  | unary_expr           // 単項演算
  | call_expr            // 関数呼び出し
  | field_expr           // フィールドアクセス
  | index_expr           // 配列アクセス
  | lambda_expr          // ラムダ式
  | if_expr              // if式
  | match_expr           // match式
  | block_expr           // ブロック式
  | paren_expr           // 括弧式
```

#### 6.2.2 リテラル式

**構文**：
```flux
42              // 整数リテラル
3.14            // 浮動小数点リテラル
true            // 真偽値リテラル
"Hello"         // 文字列リテラル
()              // 単位値
```

#### 6.2.3 変数参照

**構文**：
```flux
identifier
```

**例**：
```flux
let x = 10;
let y = x;  // x を参照
```

#### 6.2.4 二項演算式

**構文**：
```flux
expr binop expr
```

**演算子の優先順位**：

| 優先度 | 演算子 | 結合性 |
|--------|--------|--------|
| 1（最高） | `*` `/` `%` | 左結合 |
| 2 | `+` `-` | 左結合 |
| 3 | `<<` `>>` | 左結合 |
| 4 | `&` | 左結合 |
| 5 | `^` | 左結合 |
| 6 | `\|` | 左結合 |
| 7 | `==` `!=` `<` `>` `<=` `>=` | 非結合 |
| 8 | `&&` | 左結合 |
| 9（最低） | `\|\|` | 左結合 |

**例**：
```flux
let x = 2 + 3 * 4;  // 14 (掛け算が先)
let y = (2 + 3) * 4;  // 20 (括弧で優先度変更)
```

#### 6.2.5 単項演算式

**構文**：
```flux
unop expr
```

**演算子**：
- `-`：符号反転
- `!`：論理否定
- `~`：ビット否定

**例**：
```flux
let x = -10;
let y = !true;  // false
```

#### 6.2.6 関数呼び出し式

**構文**：
```flux
expr '(' arg_list ')'
```

**例**：
```flux
add(2, 3)
println("Hello")
```

#### 6.2.7 フィールドアクセス式

**構文**：
```flux
expr '.' identifier
```

**例**：
```flux
let p = Point { x: 1.0, y: 2.0 };
let x = p.x;
```

#### 6.2.8 配列アクセス式

**構文**：
```flux
expr '[' expr ']'
```

**要件REQ-EXPR-010**：配列アクセスは実行時に境界チェックされる。

**例**：
```flux
let arr = [1, 2, 3, 4, 5];
let first = arr[0];  // 1
let out_of_bounds = arr[10];  // パニック
```

#### 6.2.9 ラムダ式

**構文**：
```flux
'fn' '(' param_list ')' ('->' type)? block
```

**例**：
```flux
let double = fn(x) { x * 2 };
let add = fn(x: Int, y: Int) -> Int { x + y };
```

#### 6.2.10 if式

**構文**：
```flux
'if' expr block ('else' block)?
```

**要件REQ-EXPR-020**：if式は値を返す。

**例**：
```flux
let max = if x > y { x } else { y };

let result = if condition {
    do_something()
} else {
    do_something_else()
};
```

#### 6.2.11 match式

**構文**：
```flux
'match' expr '{' match_arm* '}'
match_arm = pattern '=>' expr ','?
```

**要件REQ-EXPR-030**：matchは網羅的でなければならない。

**例**：
```flux
match option {
    Some(value) => value,
    None => 0,
}

match number {
    0 => "zero",
    1 => "one",
    _ => "other",
}
```

#### 6.2.12 ブロック式

**構文**：
```flux
'{' stmt* expr? '}'
```

**要件REQ-EXPR-040**：ブロックの最後の式がブロックの値となる。

**例**：
```flux
let result = {
    let x = 10;
    let y = 20;
    x + y  // 30がresultに束縛される
};
```

### 6.3 文（Statements）

#### 6.3.1 文の分類

```
stmt ::=
  | let_stmt            // 変数宣言
  | expr_stmt           // 式文
  | return_stmt         // return文
  | while_stmt          // whileループ
  | for_stmt            // forループ
  | break_stmt          // break文
  | continue_stmt       // continue文
```

#### 6.3.2 let文

**構文**：
```flux
'let' pattern (':' type)? '=' expr ';'
```

**例**：
```flux
let x: Int = 10;
let y = 20;  // 型推論
let (a, b) = (1, 2);  // パターンマッチング
```

#### 6.3.3 式文

**構文**：
```flux
expr ';'
```

**例**：
```flux
println("Hello");
x = x + 1;
```

#### 6.3.4 return文

**構文**：
```flux
'return' expr? ';'
```

**例**：
```flux
return 42;
return;  // Unit を返す
```

#### 6.3.5 whileループ

**構文**：
```flux
'while' expr block
```

**例**：
```flux
while condition {
    do_something();
}
```

#### 6.3.6 forループ

**構文**：
```flux
'for' identifier 'in' expr block
```

**例**：
```flux
for item in array {
    println("{}", item);
}

for i in 0..10 {
    println("{}", i);
}
```

#### 6.3.7 break文・continue文

**構文**：
```flux
'break' ';'
'continue' ';'
```

**例**：
```flux
while true {
    if condition {
        break;
    }
    if other_condition {
        continue;
    }
    do_something();
}
```

### 6.4 パターンマッチング

#### 6.4.1 パターンの種類

```
pattern ::=
  | '_'                    // ワイルドカード
  | identifier             // 変数束縛
  | literal                // リテラルパターン
  | constructor_pattern    // コンストラクタパターン
  | tuple_pattern          // タプルパターン
```

**例**：
```flux
match value {
    0 => "zero",                    // リテラル
    x => x.to_string(),             // 変数束縛
    _ => "default",                 // ワイルドカード
}

match pair {
    (0, 0) => "origin",             // タプル
    (x, 0) => "x-axis",
    (0, y) => "y-axis",
    (x, y) => "other",
}

match option {
    Some(value) => value,           // コンストラクタ
    None => 0,
}
```


## 第7章 確率型

### 7.1 概要

確率型は、不確実性を型レベルで表現する機能を提供する。

### 7.2 確率分布の理論的基礎

#### 7.2.1 確率測度

**定義**：確率測度 μ は以下を満たす関数である：
```
μ: 𝒫(Ω) → [0, 1]
1. μ(Ω) = 1
2. μ(A) ≥ 0 for all A ⊆ Ω
3. μ(A ∪ B) = μ(A) + μ(B) if A ∩ B = ∅
```

#### 7.2.2 確率密度関数

**定義**：確率密度関数 f(x) は以下を満たす：
```
∫_{-∞}^{∞} f(x) dx = 1
P(a ≤ X ≤ b) = ∫_a^b f(x) dx
```

### 7.3 正規分布（Gaussian）

#### 7.3.1 型定義

**要件REQ-PROB-001**：正規分布型は以下のように定義される：

```flux
struct Gaussian {
    mean: Float,      // 平均 μ
    std: Float,       // 標準偏差 σ (σ > 0)
}
```

#### 7.3.2 不変条件

**要件REQ-PROB-002**：以下の不変条件が常に成立しなければならない：
```flux
invariant Gaussian {
    self.std > 0.0
}
```

**要件REQ-PROB-003**：不変条件の違反は構築時にパニックを引き起こす：
```flux
let g = Gaussian(-1.0, 0.0);  // パニック: std must be positive
```

#### 7.3.3 確率密度関数

**数学的定義**：
```
f(x; μ, σ) = (1 / (σ√(2π))) * exp(-(x-μ)² / (2σ²))
```

**実装要件REQ-PROB-010**：
```flux
impl Gaussian {
    /// 確率密度関数の値を計算
    /// 
    /// # 引数
    /// * `x` - 評価点
    /// 
    /// # 戻り値
    /// f(x) の値（0以上）
    /// 
    /// # 計算精度
    /// 相対誤差 < 1.0e-15
    fn pdf(&self, x: Float) -> Float {
        let coefficient = 1.0 / (self.std * sqrt(2.0 * PI));
        let exponent = -(x - self.mean).powi(2) / (2.0 * self.std.powi(2));
        return coefficient * exp(exponent);
    }
}
```

#### 7.3.4 累積分布関数

**数学的定義**：
```
F(x; μ, σ) = ∫_{-∞}^x f(t; μ, σ) dt
           = (1/2) * [1 + erf((x-μ) / (σ√2))]
```

**実装要件REQ-PROB-011**：
```flux
impl Gaussian {
    /// 累積分布関数の値を計算
    /// 
    /// # 引数
    /// * `x` - 評価点
    /// 
    /// # 戻り値
    /// P(X ≤ x) の値（0以上1以下）
    /// 
    /// # 計算精度
    /// 絶対誤差 < 1.0e-15
    fn cdf(&self, x: Float) -> Float {
        let z = (x - self.mean) / (self.std * sqrt(2.0));
        return 0.5 * (1.0 + erf(z));
    }
}
```

#### 7.3.5 サンプリング

**実装要件REQ-PROB-020**：Box-Muller変換を使用してサンプリングを実装する：

```flux
impl Gaussian {
    /// 分布からランダムサンプルを生成
    /// 
    /// # アルゴリズム
    /// Box-Muller変換:
    /// U1, U2 ~ Uniform(0,1)
    /// Z = √(-2 ln U1) * cos(2π U2)
    /// X = μ + σZ
    /// 
    /// # 戻り値
    /// 生成されたサンプル値
    /// 
    /// # スレッドセーフティ
    /// この関数はスレッドセーフである
    fn sample(&self) -> Float {
        // 実装はランタイムに依存
        __builtin_gaussian_sample(self.mean, self.std)
    }
}
```

**要件REQ-PROB-021**：サンプリングは以下の統計的性質を満たす：
- E[sample()] = mean（期待値）
- Var[sample()] = std²（分散）
- 連続して呼び出した場合、独立なサンプルを生成

#### 7.3.6 信頼区間

**実装要件REQ-PROB-030**：
```flux
impl Gaussian {
    /// 指定された信頼水準での信頼区間を計算
    /// 
    /// # 引数
    /// * `confidence` - 信頼水準（0.0 < confidence < 1.0）
    ///   例：0.95 → 95%信頼区間
    /// 
    /// # 戻り値
    /// (下限, 上限) のタプル
    /// 
    /// # パニック
    /// confidence が範囲外の場合
    fn confidence_interval(&self, confidence: Float) -> (Float, Float) {
        if confidence <= 0.0 || confidence >= 1.0 {
            panic("Confidence must be between 0 and 1");
        }
        
        let alpha = 1.0 - confidence;
        let z = inverse_cdf(1.0 - alpha / 2.0);  // 標準正規分布の逆関数
        
        let margin = z * self.std;
        return (self.mean - margin, self.mean + margin);
    }
}
```

#### 7.3.7 演算の意味論

##### 加算

**数学的根拠**：
```
X ~ N(μ₁, σ₁²), Y ~ N(μ₂, σ₂²), X ⊥ Y
⇒ X + Y ~ N(μ₁ + μ₂, σ₁² + σ₂²)
```

**要件REQ-PROB-100**：
```flux
impl Add for Gaussian {
    type Output = Gaussian;
    
    fn add(self, other: Gaussian) -> Gaussian {
        let new_mean = self.mean + other.mean;
        let new_std = sqrt(self.std.powi(2) + other.std.powi(2));
        return Gaussian(new_mean, new_std);
    }
}
```

**例**：
```flux
let g1 = Gaussian(10.0, 1.0);
let g2 = Gaussian(5.0, 2.0);
let g3 = g1 + g2;  // Gaussian(15.0, 2.236...)
```

##### 減算

**数学的根拠**：
```
X ~ N(μ₁, σ₁²), Y ~ N(μ₂, σ₂²), X ⊥ Y
⇒ X - Y ~ N(μ₁ - μ₂, σ₁² + σ₂²)
```

**要件REQ-PROB-101**：
```flux
impl Sub for Gaussian {
    type Output = Gaussian;
    
    fn sub(self, other: Gaussian) -> Gaussian {
        let new_mean = self.mean - other.mean;
        let new_std = sqrt(self.std.powi(2) + other.std.powi(2));
        return Gaussian(new_mean, new_std);
    }
}
```

##### スカラー倍

**数学的根拠**：
```
X ~ N(μ, σ²)
⇒ kX ~ N(kμ, k²σ²)
```

**要件REQ-PROB-102**：
```flux
impl Mul<Float> for Gaussian {
    type Output = Gaussian;
    
    fn mul(self, scalar: Float) -> Gaussian {
        let new_mean = self.mean * scalar;
        let new_std = self.std * abs(scalar);
        return Gaussian(new_mean, new_std);
    }
}
```

##### スカラー除算

**要件REQ-PROB-103**：
```flux
impl Div<Float> for Gaussian {
    type Output = Gaussian;
    
    fn div(self, scalar: Float) -> Gaussian {
        if scalar == 0.0 {
            panic("Division by zero");
        }
        return self * (1.0 / scalar);
    }
}
```

### 7.4 一様分布（Uniform）

#### 7.4.1 型定義

**要件REQ-PROB-200**：
```flux
struct Uniform {
    min: Float,
    max: Float,
}
```

#### 7.4.2 不変条件

**要件REQ-PROB-201**：
```flux
invariant Uniform {
    self.min < self.max
}
```

#### 7.4.3 確率密度関数

**数学的定義**：
```
f(x; a, b) = 1/(b-a)  if a ≤ x ≤ b
           = 0        otherwise
```

**実装要件REQ-PROB-210**：
```flux
impl Uniform {
    fn pdf(&self, x: Float) -> Float {
        if x >= self.min && x <= self.max {
            return 1.0 / (self.max - self.min);
        } else {
            return 0.0;
        }
    }
}
```

#### 7.4.4 サンプリング

**要件REQ-PROB-220**：
```flux
impl Uniform {
    /// 一様分布からサンプリング
    /// 
    /// # アルゴリズム
    /// 線形変換法:
    /// U ~ Uniform(0,1)
    /// X = a + (b-a) * U
    fn sample(&self) -> Float {
        let u = __builtin_uniform_01();  // [0,1)の一様乱数
        return self.min + (self.max - self.min) * u;
    }
}
```

### 7.5 ベイズ推論

#### 7.5.1 事後分布の計算

**要件REQ-PROB-300**：標準ライブラリは共役事前分布に対するベイズ更新をサポートする：

```flux
/// 正規分布の共役事前分布を使ったベイズ更新
/// 
/// # 前提
/// - 事前分布: prior ~ N(μ₀, σ₀²)
/// - 尤度: likelihood ~ N(μ_obs, σ_obs²)
/// - 観測値は prior の平均に関する情報
/// 
/// # 数学的導出
/// 事後分布の精度（precision = 1/σ²）:
/// τ_post = τ_prior + τ_likelihood
/// 
/// 事後分布の平均:
/// μ_post = (τ_prior * μ_prior + τ_likelihood * μ_likelihood) / τ_post
fn posterior(prior: Gaussian, likelihood: Gaussian) -> Gaussian {
    let prior_precision = 1.0 / prior.std.powi(2);
    let likelihood_precision = 1.0 / likelihood.std.powi(2);
    
    let post_precision = prior_precision + likelihood_precision;
    let post_variance = 1.0 / post_precision;
    
    let post_mean = post_variance * (
        prior_precision * prior.mean +
        likelihood_precision * likelihood.mean
    );
    
    return Gaussian(post_mean, sqrt(post_variance));
}
```

**使用例**：
```flux
// 事前信念：距離は約10m
let prior = Gaussian(10.0, 2.0);

// センサー観測：距離は約9m
let observation = Gaussian(9.0, 0.5);

// 事後分布を計算
let post = posterior(prior, observation);
// post ≈ Gaussian(9.13, 0.47)
```

### 7.6 確率型の制約と限界

#### 7.6.1 独立性の仮定

**要件REQ-PROB-400**：確率型の演算は確率変数の独立性を仮定する。

**警告**：以下のようなコードは正しくない：
```flux
let x = Gaussian(10.0, 1.0);
let y = x + x;  // 誤り：xは独立ではない

// 正しい計算
// y = 2x ~ N(20.0, 2.0)
// しかし実装では
// y ~ N(20.0, √2) と計算される（誤り）
```

**要件REQ-PROB-401**：この制限はドキュメントで明記される。

#### 7.6.2 非正規分布のサポート

**要件REQ-PROB-410**：現バージョンでは正規分布と一様分布のみサポートする。

**将来の拡張**：
- ベータ分布
- ガンマ分布
- 多変量正規分布
- カスタム分布

***

## 第8章 リアクティブ型

### 8.1 概要

リアクティブ型は、時間とともに変化する値を宣言的に扱う機能を提供する。

### 8.2 理論的基礎

#### 8.2.1 関数型リアクティブプログラミング（FRP）

**定義**：
```
Signal<T> ≝ Time → T
where Time = ℕ（離散時間）
```

**性質**：
- **連続性**：Signalは時間とともに連続的に変化
- **遅延評価**：必要になるまで計算しない
- **純粋性**：同じ時刻では常に同じ値

#### 8.2.2 依存関係グラフ

**定義**：Signal間の依存関係を有向非巡環グラフ（DAG）で表現

```
G = (V, E)
V: Signalの集合
E: 依存関係（s1 → s2 は「s2がs1に依存」）
```

**要件REQ-REACTIVE-001**：依存グラフは循環してはならない（DAGであること）。

### 8.3 Signal型

#### 8.3.1 型定義

**要件REQ-REACTIVE-010**：
```flux
type Signal<T>
```

**内部構造（実装レベル）**：
```rust
struct Signal<T> {
    current: Arc<RwLock<T>>,
    subscribers: Arc<Mutex<Vec<Box<dyn Fn(T)>>>>,
    dependencies: Vec<SignalId>,
    update_fn: Box<dyn Fn() -> T>,
}
```

#### 8.3.2 構築

**要件REQ-REACTIVE-020**：
```flux
impl<T> Signal<T> {
    /// 定数Signalを生成
    /// 
    /// # 引数
    /// * `initial` - 初期値
    /// 
    /// # 戻り値
    /// 常に同じ値を返すSignal
    fn new(initial: T) -> Signal<T>;
    
    /// センサーからSignalを生成
    /// 
    /// # 引数
    /// * `handle` - センサーハンドル
    /// 
    /// # 戻り値
    /// センサー値を表すSignal
    /// 
    /// # 並行性
    /// 専用のスレッドで実行される
    fn from_sensor<H>(handle: H) -> Signal<T>
    where H: SensorHandle<T>;
}
```

**例**：
```flux
let constant: Signal<Int> = Signal::new(42);
let temperature: Signal<Float> = Signal::from_sensor(thermometer);
```

#### 8.3.3 map操作

**数学的定義**：
```
map(s, f)(t) = f(s(t))
```

**要件REQ-REACTIVE-030**：
```flux
impl<T> Signal<T> {
    /// 値を変換
    /// 
    /// # 引数
    /// * `f` - 変換関数
    /// 
    /// # 戻り値
    /// 変換されたSignal
    /// 
    /// # 型規則
    /// s: Signal<T>, f: T → U
    /// ⇒ s.map(f): Signal<U>
    fn map<U>(&self, f: Fn(T) -> U) -> Signal<U>;
}
```

**例**：
```flux
let celsius: Signal<Float> = temperature_sensor();
let fahrenheit: Signal<Float> = celsius.map(|c| c * 1.8 + 32.0);
```

**依存関係**：
```
celsius ──→ fahrenheit
```

#### 8.3.4 filter操作

**数学的定義**：
```
filter(s, p)(t) = Some(s(t))  if p(s(t))
                = None         otherwise
```

**要件REQ-REACTIVE-031**：
```flux
impl<T> Signal<T> {
    /// 条件でフィルタ
    /// 
    /// # 引数
    /// * `pred` - 述語関数
    /// 
    /// # 戻り値
    /// フィルタされたSignal（Optionでラップ）
    fn filter(&self, pred: Fn(T) -> Bool) -> Signal<Option<T>>;
}
```

**例**：
```flux
let temp: Signal<Float> = temperature_sensor();
let high_temp: Signal<Option<Float>> = temp.filter(|t| t > 30.0);
```

#### 8.3.5 combine操作

**数学的定義**：
```
combine(s1, s2, f)(t) = f(s1(t), s2(t))
```

**要件REQ-REACTIVE-040**：
```flux
/// 2つのSignalを結合
/// 
/// # 引数
/// * `s1` - 1つ目のSignal
/// * `s2` - 2つ目のSignal
/// * `f` - 結合関数
/// 
/// # 戻り値
/// 結合されたSignal
/// 
/// # 同期
/// 両方のSignalの最新値を使用
fn combine<T, U, V>(
    s1: Signal<T>,
    s2: Signal<U>,
    f: Fn(T, U) -> V
) -> Signal<V>;
```

**例**：
```flux
let lidar: Signal<Float> = ...;
let camera: Signal<Float> = ...;
let fused: Signal<Float> = Signal::combine(
    lidar,
    camera,
    |l, c| (l + c) / 2.0
);
```

**依存関係**：
```
lidar ──┐
        ├──→ fused
camera ─┘
```

#### 8.3.6 サンプリング制御

**要件REQ-REACTIVE-050**：
```flux
impl<T> Signal<T> {
    /// サンプリング間隔を指定
    /// 
    /// # 引数
    /// * `interval` - サンプリング間隔
    /// 
    /// # 戻り値
    /// 間引きされたSignal
    fn sample_interval(&self, interval: Duration) -> Signal<T>;
    
    /// 最新値のみ使用（バックプレッシャー対策）
    /// 
    /// # 説明
    /// 処理が追いつかない場合、古い値はスキップされる
    fn map_latest<U>(&self, f: Fn(T) -> U) -> Signal<U>;
    
    /// すべての値をバッファ
    /// 
    /// # 警告
    /// メモリ使用量が増加する可能性がある
    fn map_buffered<U>(&self, f: Fn(T) -> U) -> Signal<U>;
}
```

**例**：
```flux
let fast_sensor: Signal<Float> = ...;  // 1000Hz

// 100Hzにダウンサンプリング
let slow: Signal<Float> = fast_sensor.sample_interval(10*ms);

// 最新値のみ処理
let processed: Signal<Float> = fast_sensor.map_latest(|x| {
    expensive_computation(x)
});
```

#### 8.3.7 購読（Subscribe）

**要件REQ-REACTIVE-060**：
```flux
impl<T> Signal<T> {
    /// Signalの変化を監視
    /// 
    /// # 引数
    /// * `callback` - 値が変化したときに呼ばれる関数
    /// 
    /// # 戻り値
    /// 購読ID（購読解除に使用）
    /// 
    /// # 並行性
    /// callbackは別スレッドで実行される可能性がある
    fn subscribe(&self, callback: Fn(T)) -> SubscriptionId;
    
    /// 購読を解除
    fn unsubscribe(&self, id: SubscriptionId);
}
```

**例**：
```flux
let temp: Signal<Float> = temperature_sensor();

let id = temp.subscribe(|t| {
    println("Temperature: {}°C", t);
});

// 後で購読解除
temp.unsubscribe(id);
```

### 8.4 Signal の実行モデル

#### 8.4.1 更新の伝播

**要件REQ-REACTIVE-100**：Signalの更新は以下の順序で伝播する：

1. **トポロジカルソート**：依存グラフをトポロジカル順にソート
2. **順次更新**：依存関係の順に各Signalを更新
3. **通知**：購読者に変更を通知

**例**：
```flux
let a: Signal<Int> = Signal::new(10);
let b: Signal<Int> = a.map(|x| x * 2);
let c: Signal<Int> = b.map(|x| x + 1);

// 依存グラフ: a → b → c

a.set(20);

// 更新順序:
// 1. a.current = 20
// 2. b.current = 40  (20 * 2)
// 3. c.current = 41  (40 + 1)
```

#### 8.4.2 グリッチフリー保証

**要件REQ-REACTIVE-101**：Fluxは**グリッチフリー**を保証する。

**グリッチの定義**：途中の不整合な状態が観測されること

**例（グリッチあり）**：
```flux
let x: Signal<Int> = ...;
let y: Signal<Int> = x.map(|v| v);
let z: Signal<Int> = x.map(|v| v);
let w: Signal<Int> = Signal::combine(y, z, |a, b| a + b);

// x が 10 → 20 に変化
// グリッチありの場合:
// w = y(10) + z(20) = 30  (不整合!)

// グリッチフリーの場合:
// w = y(20) + z(20) = 40  (正しい)
```

**実装方法**：トランザクショナルな更新

#### 8.4.3 循環依存の検出

**要件REQ-REACTIVE-110**：循環依存はコンパイル時に検出される。

**検出アルゴリズム**：深さ優先探索（DFS）

```flux
// エラー：循環依存
let a: Signal<Int> = b.map(|x| x + 1);  // a は b に依存
let b: Signal<Int> = a.map(|x| x * 2);  // b は a に依存
// コンパイルエラー: Circular dependency detected
```

**例外：遅延Signal**：
```flux
let a: Signal<Int> = Signal::new(0);
let b: Signal<Int> = a.map(|x| x * 2);
let feedback: Signal<Int> = b.map(|x| x + 1).delay(1);

a.connect(feedback);  // OK: delayがあるため循環ではない
```

### 8.5 Event型

#### 8.5.1 型定義

**要件REQ-REACTIVE-200**：
```flux
type Event<T>
```

**意味論**：離散的に発生する出来事の列

#### 8.5.2 基本操作

**要件REQ-REACTIVE-210**：
```flux
impl<T> Event<T> {
    /// イベントをフィルタ
    fn filter(&self, pred: Fn(T) -> Bool) -> Event<T>;
    
    /// イベントを変換
    fn map<U>(&self, f: Fn(T) -> U) -> Event<U>;
    
    /// 2つのイベントストリームをマージ
    fn merge(&self, other: Event<T>) -> Event<T>;
    
    /// イベントを購読
    fn subscribe(&self, callback: Fn(T)) -> SubscriptionId;
}
```

**例**：
```flux
let clicks: Event<MouseClick> = Event::from_source(mouse);

let left_clicks: Event<MouseClick> = clicks.filter(|e| {
    e.button == MouseButton::Left
});

left_clicks.subscribe(|click| {
    println("Left click at ({}, {})", click.x, click.y);
});
```

### 8.6 SignalとEventの相互変換

**要件REQ-REACTIVE-300**：
```flux
impl<T> Signal<T> {
    /// Signalの変化をEventに変換
    fn to_event(&self) -> Event<T>;
}

impl<T> Event<T> {
    /// Eventの最新値をSignalに変換
    fn hold(&self, initial: T) -> Signal<T>;
}
```

**例**：
```flux
let position: Signal<Float> = ...;
let position_changes: Event<Float> = position.to_event();

let clicks: Event<MouseClick> = ...;
let last_click: Signal<Option<MouseClick>> = clicks.hold(None);
```

### 8.7 Signal と Gaussian の統合

**要件REQ-REACTIVE-400**：Signal<Gaussian>は時変確率分布を表現する。

```flux
// 不確実性を持つセンサー
let uncertain_pos: Signal<Gaussian> = Signal::from_uncertain_sensor(lidar);

// 各時刻で確率分布
// t=0: Gaussian(10.0, 1.0)
// t=1: Gaussian(10.1, 0.9)
// t=2: Gaussian(10.2, 0.95)
// ...

// サンプリング
let samples: Signal<Float> = uncertain_pos.map(|g| g.sample());

// 信頼区間の監視
let intervals: Signal<(Float, Float)> = uncertain_pos.map(|g| {
    g.confidence_interval(0.95)
});

intervals.subscribe(|(low, high)| {
    if high - low > 2.0 {
        println("Warning: High uncertainty");
    }
});
```

***

## 第9章 関数とクロージャ

### 9.1 概要

Fluxは関数を第一級オブジェクトとして扱い、高階関数とクロージャをサポートする。

### 9.2 関数定義

#### 9.2.1 構文

**EBNF定義**：
```ebnf
function_def = 'fn' identifier generic_params? '(' param_list ')' 
               ('->' type)? block

generic_params = '<' identifier (',' identifier)* '>'
param_list = (param (',' param)*)?
param = identifier ':' type
```

**例**：
```flux
fn add(x: Int, y: Int) -> Int {
    return x + y;
}

fn generic_identity<T>(x: T) -> T {
    return x;
}
```

#### 9.2.2 パラメータ

**要件REQ-FN-001**：すべてのパラメータは型注釈が必須である。

**要件REQ-FN-002**：パラメータは値渡し（by value）である。

```flux
fn modify(x: Int) {
    x = x + 1;  // ローカルコピーを変更
}

let y = 10;
modify(y);
// y は依然として 10
```

#### 9.2.3 戻り値

**要件REQ-FN-010**：戻り値の型は推論可能な場合、省略できる。

```flux
fn add(x: Int, y: Int) {  // 戻り値の型は Int と推論
    return x + y;
}
```

**要件REQ-FN-011**：最後の式が戻り値となる（return不要）。

```flux
fn add(x: Int, y: Int) -> Int {
    x + y  // return 不要
}
```

### 9.3 クロージャ（ラムダ式）

#### 9.3.1 構文

**EBNF定義**：
```ebnf
lambda_expr = 'fn' '(' param_list ')' ('->' type)? block
```

**例**：
```flux
let add = fn(x: Int, y: Int) -> Int { x + y };
let double = fn(x) { x * 2 };  // 型推論
```

#### 9.3.2 キャプチャ

**要件REQ-FN-100**：クロージャは外部の変数をキャプチャできる。

**キャプチャモデル**：値キャプチャ（コピー）

```flux
fn make_adder(n: Int) -> Fn(Int) -> Int {
    return fn(x) { x + n };  // n をキャプチャ
}

let add5 = make_adder(5);
let result = add5(10);  // 15
```

**要件REQ-FN-101**：キャプチャされた変数は不変である。

```flux
let mut x = 10;
let closure = fn() {
    x = 20;  // エラー：キャプチャされた変数は変更不可
};
```

### 9.4 高階関数

#### 9.4.1 関数を引数に取る

```flux
fn apply<T, U>(f: Fn(T) -> U, x: T) -> U {
    return f(x);
}

let result = apply(fn(x) { x * 2 }, 10);  // 20
```

#### 9.4.2 関数を返す

```flux
fn compose<T, U, V>(f: Fn(T) -> U, g: Fn(U) -> V) -> Fn(T) -> V {
    return fn(x) { g(f(x)) };
}

let add1 = fn(x) { x + 1 };
let double = fn(x) { x * 2 };
let add1_then_double = compose(add1, double);

let result = add1_then_double(10);  // (10 + 1) * 2 = 22
```

### 9.5 ジェネリック関数

**要件REQ-FN-200**：関数は型パラメータを持つことができる。

```flux
fn identity<T>(x: T) -> T {
    return x;
}

fn map<T, U>(arr: Array<T>, f: Fn(T) -> U) -> Array<U> {
    let result: Array<U> = [];
    for item in arr {
        result.push(f(item));
    }
    return result;
}
```

### 9.6 メソッド

**要件REQ-FN-300**：implブロック内で定義された関数はメソッドとなる。

```flux
struct Point {
    x: Float,
    y: Float,
}

impl Point {
    /// コンストラクタ（慣習的に new）
    fn new(x: Float, y: Float) -> Point {
        return Point { x, y };
    }
    
    /// インスタンスメソッド
    fn distance(&self) -> Float {
        return sqrt(self.x * self.x + self.y * self.y);
    }
    
    /// ミュータブルメソッド
    fn scale(&mut self, factor: Float) {
        self.x = self.x * factor;
        self.y = self.y * factor;
    }
}

// 使用例
let p = Point::new(3.0, 4.0);
let d = p.distance();  // 5.0
```

### 9.7 再帰

**要件REQ-FN-400**：再帰関数をサポートする。

```flux
fn factorial(n: Int) -> Int {
    if n <= 1 {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}
```

**要件REQ-FN-401**：末尾再帰最適化は保証されない（実装依存）。

### 9.8 関数オーバーロード

**要件REQ-FN-500**：関数オーバーロードはサポートしない。

**理由**：型推論との相性が悪い

**代替案**：異なる名前を使用
```flux
fn add_int(x: Int, y: Int) -> Int { x + y }
fn add_float(x: Float, y: Float) -> Float { x + y }
```

***

## 第10章 構造体とenum

### 10.1 構造体（Struct）

#### 10.1.1 定義

**構文**：
```ebnf
struct_def = 'struct' identifier generic_params? '{' field_list '}'
field_list = (field (',' field)* ','?)?
field = identifier ':' type
```

**例**：
```flux
struct Point {
    x: Float,
    y: Float,
}

struct GenericPair<T, U> {
    first: T,
    second: U,
}
```

#### 10.1.2 構築

**要件REQ-STRUCT-001**：構造体は構造体リテラルで構築される。

```flux
let p = Point { x: 1.0, y: 2.0 };

let pair = GenericPair { first: 10, second: "hello" };
```

**要件REQ-STRUCT-002**：すべてのフィールドを指定しなければならない。

```flux
let p = Point { x: 1.0 };  // エラー：y が指定されていない
```

#### 10.1.3 フィールドアクセス

**構文**：
```flux
expr '.' identifier
```

**例**：
```flux
let p = Point { x: 1.0, y: 2.0 };
let x = p.x;  // 1.0
```

#### 10.1.4 更新

**要件REQ-STRUCT-010**：フィールドの更新は構造体全体を再構築する。

```flux
let p1 = Point { x: 1.0, y: 2.0 };
let p2 = Point { x: 3.0, ..p1 };  // y は p1 から継承
```

### 10.2 列挙型（Enum）

#### 10.2.1 定義

**構文**：
```ebnf
enum_def = 'enum' identifier generic_params? '{' variant_list '}'
variant_list = (variant (',' variant)* ','?)?
variant = identifier |
          identifier '(' type_list ')' |
          identifier '{' field_list '}'
```

**例**：
```flux
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

enum Message {
    Quit,
    Move { x: Int, y: Int },
    Write(String),
}
```

#### 10.2.2 パターンマッチング

**要件REQ-ENUM-001**：enumはmatch式で分岐される。

```flux
fn unwrap_or<T>(opt: Option<T>, default: T) -> T {
    match opt {
        Some(value) => value,
        None => default,
    }
}

fn process_message(msg: Message) {
    match msg {
        Message::Quit => println("Quitting"),
        Message::Move { x, y } => println("Moving to ({}, {})", x, y),
        Message::Write(text) => println("Writing: {}", text),
    }
}
```

**要件REQ-ENUM-002**：matchは網羅的でなければならない。

```flux
fn is_some<T>(opt: Option<T>) -> Bool {
    match opt {
        Some(_) => true,
        // エラー：None の場合が処理されていない
    }
}
```

### 10.3 メモリレイアウト

**要件REQ-STRUCT-100**：構造体のメモリレイアウトは以下の通り：

```
struct Point {
    x: Float,  // 8バイト
    y: Float,  // 8バイト
}
// 合計：16バイト（パディングなし）
```

**要件REQ-ENUM-100**：enumのメモリレイアウトは以下の通り：

```
enum Option<T> {
    Some(T),
    None,
}
// レイアウト：
// - タグ（識別子）：8バイト
// - 最大のvariantのサイズ
// 合計：8 + sizeof(T)
```

***

## 第11章 並行性と同期

### 11.1 概要

Fluxは並行プログラミングのための以下の機能を提供する：
- スレッド生成
- 同期プリミティブ（Mutex, RwLock, Channel）
- async/await（将来の拡張）

### 11.2 スレッド

#### 11.2.1 スレッド生成

**要件REQ-THREAD-001**：
```flux
/// 新しいスレッドを生成
/// 
/// # 引数
/// * `f` - スレッドで実行する関数
/// 
/// # 戻り値
/// スレッドハンドル
/// 
/// # 例
/// ```
/// let handle = spawn(fn() {
///     println("Hello from thread");
/// });
/// handle.join();
/// ```
fn spawn<T>(f: Fn() -> T) -> JoinHandle<T>;
```

**例**：
```flux
let handle = spawn(fn() {
    let result = expensive_computation();
    return result;
});

// メインスレッドで他の作業

let result = handle.join();  // スレッドの完了を待つ
```

#### 11.2.2 JoinHandle

**要件REQ-THREAD-010**：
```flux
struct JoinHandle<T>;

impl<T> JoinHandle<T> {
    /// スレッドの完了を待つ
    /// 
    /// # 戻り値
    /// スレッドの実行結果
    /// 
    /// # ブロッキング
    /// この関数は完了まで現在のスレッドをブロックする
    fn join(self) -> T;
    
    /// スレッドが完了しているか確認
    /// 
    /// # 戻り値
    /// 完了していればSome(結果)、そうでなければNone
    fn try_join(&self) -> Option<T>;
}
```

### 11.3 Mutex（相互排他ロック）

#### 11.3.1 型定義

**要件REQ-SYNC-001**：
```flux
struct Mutex<T>;

impl<T> Mutex<T> {
    /// 新しいMutexを生成
    fn new(value: T) -> Mutex<T>;
    
    /// ロックを取得
    /// 
    /// # ブロッキング
    /// 他のスレッドがロックを保持している場合、
    /// 解放されるまでブロックする
    /// 
    /// # 戻り値
    /// ロックガード（自動的にアンロック）
    fn lock(&self) -> MutexGuard<T>;
    
    /// ロックの取得を試みる
    /// 
    /// # ノンブロッキング
    /// 即座に戻る
    /// 
    /// # 戻り値
    /// 成功ならSome(ガード)、失敗ならNone
    fn try_lock(&self) -> Option<MutexGuard<T>>;
}
```

#### 11.3.2 MutexGuard

**要件REQ-SYNC-002**：
```flux
struct MutexGuard<T>;

impl<T> MutexGuard<T> {
    // デリファレンス演算子
    // *guard で値にアクセス
}

impl<T> Drop for MutexGuard<T> {
    // ガードがドロップされると自動的にアンロック
}
```

**例**：
```flux
let counter: Mutex<Int> = Mutex::new(0);

spawn(fn() {
    let guard = counter.lock();
    *guard = *guard + 1;
    // ここでガードがドロップされ、自動的にアンロック
});

spawn(fn() {
    let guard = counter.lock();
    *guard = *guard + 1;
});
```

### 11.4 RwLock（読み書きロック）

#### 11.4.1 型定義

**要件REQ-SYNC-010**：
```flux
struct RwLock<T>;

impl<T> RwLock<T> {
    fn new(value: T) -> RwLock<T>;
    
    /// 読み取りロックを取得
    /// 
    /// # 特性
    /// - 複数の読み取りロックを同時に保持可能
    /// - 書き込みロックとは排他的
    fn read(&self) -> ReadGuard<T>;
    
    /// 書き込みロックを取得
    /// 
    /// # 特性
    /// - 排他的（他のロックと同時に保持不可）
    fn write(&self) -> WriteGuard<T>;
}
```

**例**：
```flux
let data: RwLock<Array<Int>> = RwLock::new([1, 2, 3]);

// 読み取り（複数同時可能）
spawn(fn() {
    let guard = data.read();
    println("Length: {}", guard.len());
});

spawn(fn() {
    let guard = data.read();
    println("First: {}", guard[0]);
});

// 書き込み（排他的）
spawn(fn() {
    let guard = data.write();
    guard.push(4);
});
```

### 11.5 Channel（メッセージパッシング）

#### 11.5.1 型定義

**要件REQ-SYNC-100**：
```flux
struct Channel<T>;

impl<T> Channel<T> {
    /// 新しいChannelを生成
    /// 
    /// # 戻り値
    /// (送信側, 受信側) のタプル
    fn new() -> (Sender<T>, Receiver<T>);
}

struct Sender<T>;
impl<T> Sender<T> {
    /// 値を送信
    /// 
    /// # エラー
    /// 受信側が既にドロップされている場合
    fn send(&self, value: T) -> Result<Unit, SendError>;
}

struct Receiver<T>;
impl<T> Receiver<T> {
    /// 値を受信（ブロッキング）
    /// 
    /// # ブロッキング
    /// 値が利用可能になるまでブロック
    /// 
    /// # エラー
    /// 送信側が既にドロップされている場合
    fn recv(&self) -> Result<T, RecvError>;
    
    /// 値を受信（ノンブロッキング）
    fn try_recv(&self) -> Result<T, TryRecvError>;
}
```

**例**：
```flux
let (tx, rx) = Channel::new();

spawn(fn() {
    tx.send(42);
    tx.send(100);
});

spawn(fn() {
    let value1 = rx.recv()?;  // 42
    let value2 = rx.recv()?;  // 100
    println("Received: {}, {}", value1, value2);
});
```

### 11.6 Signalの並行実行

**要件REQ-SYNC-200**：各Signalは独立したスレッドで実行される。

```flux
let sensor1: Signal<Float> = Signal::from_sensor(device1);  // スレッド1
let sensor2: Signal<Float> = Signal::from_sensor(device2);  // スレッド2

// 自動的に同期される
let combined: Signal<Float> = Signal::combine(
    sensor1,
    sensor2,
    |s1, s2| (s1 + s2) / 2.0
);
```

**内部実装（概念）**：
```rust
struct SignalRuntime {
    threads: Vec<JoinHandle<()>>,
    sync_queue: Mutex<VecDeque<SignalUpdate>>,
}

impl SignalRuntime {
    fn update_signal(&self, id: SignalId, value: Value) {
        // ロックを取得
        let queue = self.sync_queue.lock();
        
        // 更新をキューに追加
        queue.push_back(SignalUpdate { id, value });
        
        // トポロジカル順に更新を伝播
        self.propagate_updates();
    }
}
```

### 11.7 デッドロックの防止

#### 11.7.1 ロック順序

**要件REQ-SYNC-300**：グローバルなロック順序を定義する。

```flux
// ロック順序の例
// Lock1 → Lock2 → Lock3

// 正しい順序
fn correct() {
    let g1 = lock1.lock();
    let g2 = lock2.lock();
    let g3 = lock3.lock();
    // OK
}

// 誤った順序（警告）
fn incorrect() {
    let g2 = lock2.lock();
    let g1 = lock1.lock();  // 警告：Lock order violation
}
```

#### 11.7.2 Signal循環の防止

**要件REQ-SYNC-301**：Signalの循環依存はコンパイル時に検出される。

```flux
let a: Signal<Int> = b.map(|x| x + 1);
let b: Signal<Int> = a.map(|x| x * 2);
// コンパイルエラー：Circular dependency detected
```

### 11.8 データ競合の防止

**要件REQ-SYNC-400**：Fluxの型システムはデータ競合を防止する。

**戦略**：
1. **不変性優先**：デフォルトで変数は不変
2. **明示的な可変性**：`mut`キーワードで明示
3. **排他制御**：Mutex/RwLockで保護

```flux
// 共有可能（不変）
let data = [1, 2, 3];
spawn(fn() { println("{}", data[0]); });  // OK

// 共有不可（可変）
let mut data = [1, 2, 3];
spawn(fn() { data[0] = 10; });  // エラー：可変データは共有不可

// 明示的な同期
let data = Mutex::new([1, 2, 3]);
spawn(fn() {
    let guard = data.lock();
    guard[0] = 10;  // OK
});
```

## 第12章 メモリ管理

### 12.1 概要

Fluxは**参照カウント方式のガベージコレクション**を採用する。

### 12.2 メモリ管理戦略

#### 12.2.1 設計原則

**要件REQ-MEM-001**：メモリ安全性を保証する
- Use-After-Free の防止
- ダングリングポインタの防止
- 二重解放の防止
- メモリリークの最小化

**要件REQ-MEM-002**：プログラマの負担を最小化する
- 手動メモリ管理不要
- 明示的な`free`不要
- ポインタ演算禁止

#### 12.2.2 参照カウント

**実装方式**：
```rust
// 内部実装（概念）
struct Rc<T> {
    ptr: *mut RcBox<T>,
}

struct RcBox<T> {
    strong_count: usize,
    weak_count: usize,
    value: T,
}

impl<T> Rc<T> {
    fn new(value: T) -> Rc<T> {
        let box = RcBox {
            strong_count: 1,
            weak_count: 0,
            value,
        };
        Rc { ptr: Box::into_raw(Box::new(box)) }
    }
}

impl<T> Clone for Rc<T> {
    fn clone(&self) -> Rc<T> {
        unsafe {
            (*self.ptr).strong_count += 1;
        }
        Rc { ptr: self.ptr }
    }
}

impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        unsafe {
            (*self.ptr).strong_count -= 1;
            if (*self.ptr).strong_count == 0 {
                // メモリを解放
                drop(Box::from_raw(self.ptr));
            }
        }
    }
}
```

**Fluxでの使用**：
```flux
// 自動的に参照カウント
let data = [1, 2, 3, 4, 5];
let data2 = data;  // 参照カウント: 2
let data3 = data;  // 参照カウント: 3

// スコープを抜けると自動的にデクリメント
```

#### 12.2.3 循環参照の問題

**問題**：
```flux
struct Node {
    value: Int,
    next: Option<Node>,
}

let node1 = Node { value: 1, next: None };
let node2 = Node { value: 2, next: Some(node1) };
node1.next = Some(node2);  // 循環参照！

// node1 → node2 → node1 → ...
// メモリリーク
```

**解決策：弱参照**

**要件REQ-MEM-010**：
```flux
struct Weak<T>;  // 弱参照

impl<T> Weak<T> {
    /// 強参照にアップグレード
    /// 
    /// # 戻り値
    /// オブジェクトがまだ生存していればSome、そうでなければNone
    fn upgrade(&self) -> Option<Rc<T>>;
}
```

**使用例**：
```flux
struct Node {
    value: Int,
    next: Option<Rc<Node>>,
    prev: Option<Weak<Node>>,  // 弱参照
}

// prevは参照カウントに寄与しない
// 循環を防ぐ
```

### 12.3 スタックとヒープ

#### 12.3.1 スタック割り当て

**要件REQ-MEM-100**：以下はスタックに割り当てられる：
- 基本型（Int, Float, Bool）
- 小さな構造体（サイズ <= 128バイト）
- ローカル変数

**例**：
```flux
fn compute() {
    let x: Int = 10;        // スタック
    let y: Float = 3.14;    // スタック
    let p = Point { x: 1.0, y: 2.0 };  // スタック（小さい）
}
```

#### 12.3.2 ヒープ割り当て

**要件REQ-MEM-101**：以下はヒープに割り当てられる：
- 動的配列（Array）
- 大きな構造体（サイズ > 128バイト）
- クロージャ
- Signal/Event

**例**：
```flux
let arr = [1, 2, 3, 4, 5];  // ヒープ
let large = LargeStruct { ... };  // ヒープ
let closure = fn() { ... };  // ヒープ
```

### 12.4 リソース管理（RAII）

#### 12.4.1 with式

**要件REQ-MEM-200**：withブロックはRAIIパターンを実現する。

**構文**：
```flux
with resource = init_expr {
    body
}
```

**展開**：
```flux
{
    let resource = init_expr;
    let result = {
        body
    };
    resource.close();  // 自動的に呼ばれる
    result
}
```

**例**：
```flux
with file = File::open("data.txt") {
    let content = file.read();
    process(content);
}  // ここで自動的に file.close() が呼ばれる
```

#### 12.4.2 Dropトレイト

**要件REQ-MEM-210**：
```flux
trait Drop {
    fn drop(&mut self);
}
```

**実装例**：
```flux
struct FileHandle {
    fd: Int,
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        // ファイルディスクリプタをクローズ
        close_fd(self.fd);
    }
}
```

### 12.5 メモリ使用量の見積もり

**要件REQ-MEM-300**：標準ライブラリは各型のメモリサイズを文書化する。

| 型 | サイズ |
|----|--------|
| Int | 8バイト |
| Float | 8バイト |
| Bool | 1バイト |
| Pointer | 8バイト |
| Array<T> | 24バイト + 要素数 × sizeof(T) |
| String | 24バイト + 文字列長 |
| Option<T> | sizeof(T) + 8バイト（タグ） |
| Signal<T> | 64バイト + sizeof(T) |

***

## 第13章 標準ライブラリ概要

### 13.1 モジュール構成

**要件REQ-LIB-001**：Fluxの標準ライブラリは以下のモジュールで構成される：

```
std
├── core          // 基本機能
├── prob          // 確率的プログラミング
├── reactive      // リアクティブプログラミング
├── collections   // コレクション
├── io            // 入出力
├── sync          // 並行性・同期
├── math          // 数学関数
├── time          // 時間・日付
├── string        // 文字列操作
└── net           // ネットワーク（将来）
```

### 13.2 インポート

**構文**：
```flux
import std::collections::Array;
import std::prob::{Gaussian, Uniform};
import std::reactive::Signal;

// エイリアス
import std::collections::Array as Vec;
```

***

## 第14章 coreモジュール

### 14.1 基本型

**要件REQ-CORE-001**：
```flux
module core {
    type Int = i64;
    type Float = f64;
    type Bool = bool;
    type String = string;
    type Unit = ();
}
```

### 14.2 Option型

**定義**：
```flux
enum Option<T> {
    Some(T),
    None,
}

impl<T> Option<T> {
    fn is_some(&self) -> Bool;
    fn is_none(&self) -> Bool;
    fn unwrap(self) -> T;  // パニック可能
    fn unwrap_or(self, default: T) -> T;
    fn map<U>(&self, f: Fn(T) -> U) -> Option<U>;
    fn and_then<U>(&self, f: Fn(T) -> Option<U>) -> Option<U>;
}
```

### 14.3 Result型

**定義**：
```flux
enum Result<T, E> {
    Ok(T),
    Err(E),
}

impl<T, E> Result<T, E> {
    fn is_ok(&self) -> Bool;
    fn is_err(&self) -> Bool;
    fn unwrap(self) -> T;
    fn unwrap_or(self, default: T) -> T;
    fn map<U>(&self, f: Fn(T) -> U) -> Result<U, E>;
    fn map_err<F>(&self, f: Fn(E) -> F) -> Result<T, F>;
}
```

### 14.4 標準出力

**要件REQ-CORE-010**：
```flux
/// 標準出力に出力（改行なし）
fn print(s: String);

/// 標準出力に出力（改行あり）
fn println(s: String);

/// フォーマット済み文字列を生成
fn format(template: String, args: ...Any) -> String;
```

***

## 第15章 probモジュール

### 15.1 正規分布

**完全な定義**：
```flux
module prob {
    struct Gaussian {
        mean: Float,
        std: Float,
    }
    
    impl Gaussian {
        fn new(mean: Float, std: Float) -> Gaussian;
        fn pdf(&self, x: Float) -> Float;
        fn cdf(&self, x: Float) -> Float;
        fn sample(&self) -> Float;
        fn confidence_interval(&self, confidence: Float) -> (Float, Float);
    }
    
    impl Add for Gaussian { ... }
    impl Sub for Gaussian { ... }
    impl Mul<Float> for Gaussian { ... }
}
```

### 15.2 ベイズ推論

```flux
fn posterior(prior: Gaussian, likelihood: Gaussian) -> Gaussian;
fn kalman_filter(
    state: Gaussian,
    observation: Gaussian,
    process_noise: Float,
    measurement_noise: Float
) -> Gaussian;
```

***

## 第16章 reactiveモジュール

### 16.1 Signal

**完全な定義**：
```flux
module reactive {
    type Signal<T>;
    
    impl<T> Signal<T> {
        fn new(initial: T) -> Signal<T>;
        fn from_sensor<H>(handle: H) -> Signal<T>;
        
        fn map<U>(&self, f: Fn(T) -> U) -> Signal<U>;
        fn filter(&self, pred: Fn(T) -> Bool) -> Signal<Option<T>>;
        fn sample_interval(&self, interval: Duration) -> Signal<T>;
        fn map_latest<U>(&self, f: Fn(T) -> U) -> Signal<U>;
        
        fn subscribe(&self, callback: Fn(T)) -> SubscriptionId;
        fn unsubscribe(&self, id: SubscriptionId);
    }
    
    fn combine<T, U, V>(
        s1: Signal<T>,
        s2: Signal<U>,
        f: Fn(T, U) -> V
    ) -> Signal<V>;
}
```

***

## 第17章 collectionsモジュール

### 17.1 配列（Array）

**定義**：
```flux
type Array<T>;

impl<T> Array<T> {
    fn new() -> Array<T>;
    fn with_capacity(capacity: Int) -> Array<T>;
    
    fn len(&self) -> Int;
    fn is_empty(&self) -> Bool;
    fn capacity(&self) -> Int;
    
    fn push(&mut self, item: T);
    fn pop(&mut self) -> Option<T>;
    fn insert(&mut self, index: Int, item: T);
    fn remove(&mut self, index: Int) -> T;
    
    fn get(&self, index: Int) -> Option<T>;
    fn first(&self) -> Option<T>;
    fn last(&self) -> Option<T>;
    
    fn iter(&self) -> Iterator<T>;
    fn map<U>(&self, f: Fn(T) -> U) -> Array<U>;
    fn filter(&self, pred: Fn(T) -> Bool) -> Array<T>;
    fn fold<U>(&self, init: U, f: Fn(U, T) -> U) -> U;
}
```

### 17.2 ハッシュマップ（Map）

**定義**：
```flux
type Map<K, V>;

impl<K, V> Map<K, V> {
    fn new() -> Map<K, V>;
    
    fn insert(&mut self, key: K, value: V) -> Option<V>;
    fn get(&self, key: K) -> Option<V>;
    fn remove(&mut self, key: K) -> Option<V>;
    fn contains_key(&self, key: K) -> Bool;
    
    fn keys(&self) -> Iterator<K>;
    fn values(&self) -> Iterator<V>;
    fn iter(&self) -> Iterator<(K, V)>;
}
```

***

## 第18章 ioモジュール

### 18.1 ファイル入出力

**定義**：
```flux
module io {
    struct File;
    
    impl File {
        fn open(path: String) -> Result<File, IoError>;
        fn create(path: String) -> Result<File, IoError>;
        
        fn read(&self) -> Result<String, IoError>;
        fn read_to_string(&self) -> Result<String, IoError>;
        fn write(&mut self, data: String) -> Result<Unit, IoError>;
        
        fn close(&mut self) -> Result<Unit, IoError>;
    }
    
    fn read_to_string(path: String) -> Result<String, IoError>;
    fn write_string(path: String, data: String) -> Result<Unit, IoError>;
}
```

***

## 第19章 syncモジュール

**前章で説明済み**：Mutex, RwLock, Channel

***

## 第20章 mathモジュール

### 20.1 基本的な数学関数

```flux
module math {
    const PI: Float = 3.14159265358979323846;
    const E: Float = 2.71828182845904523536;
    
    // 三角関数
    fn sin(x: Float) -> Float;
    fn cos(x: Float) -> Float;
    fn tan(x: Float) -> Float;
    fn asin(x: Float) -> Float;
    fn acos(x: Float) -> Float;
    fn atan(x: Float) -> Float;
    fn atan2(y: Float, x: Float) -> Float;
    
    // 指数・対数
    fn exp(x: Float) -> Float;
    fn log(x: Float) -> Float;
    fn log10(x: Float) -> Float;
    fn log2(x: Float) -> Float;
    fn pow(x: Float, y: Float) -> Float;
    fn sqrt(x: Float) -> Float;
    
    // その他
    fn abs(x: Float) -> Float;
    fn floor(x: Float) -> Float;
    fn ceil(x: Float) -> Float;
    fn round(x: Float) -> Float;
    
    // 統計関数
    fn mean(data: Array<Float>) -> Float;
    fn std(data: Array<Float>) -> Float;
    fn variance(data: Array<Float>) -> Float;
    fn median(data: Array<Float>) -> Float;
}
```

***

## 第21章 メモリレイアウト

### 21.1 基本型のレイアウト

**要件REQ-LAYOUT-001**：

| 型 | サイズ | アライメント | エンディアン |
|----|--------|-------------|------------|
| Int (i64) | 8バイト | 8バイト | リトルエンディアン |
| Float (f64) | 8バイト | 8バイト | IEEE 754 |
| Bool | 1バイト | 1バイト | 0=false, 1=true |
| Pointer | 8バイト | 8バイト | - |

### 21.2 構造体のレイアウト

**要件REQ-LAYOUT-010**：構造体はフィールド順に配置される（パディング最小化）

**例**：
```flux
struct Example {
    a: Int,     // オフセット 0, サイズ 8
    b: Float,   // オフセット 8, サイズ 8
    c: Bool,    // オフセット 16, サイズ 1
    // パディング 7バイト
}
// 合計：24バイト
```

### 21.3 列挙型のレイアウト

**要件REQ-LAYOUT-020**：
```flux
enum Option<T> {
    Some(T),
    None,
}

// レイアウト：
// [タグ: 8バイト][データ: sizeof(T)]
// 合計：8 + sizeof(T) バイト
```

***

## 第22章 実行時システム

### 22.1 アーキテクチャ

**構成要素**：
```
実行時システム
├── メモリアロケータ
├── ガベージコレクタ
├── スレッドスケジューラ
├── Signalランタイム
└── 確率計算エンジン
```

### 22.2 起動シーケンス

1. **初期化フェーズ**
   - メモリアロケータ初期化
   - GC初期化
   - スレッドプール初期化
   - Signalランタイム初期化

2. **実行フェーズ**
   - main関数呼び出し
   - Signalの自動更新
   - GCの定期実行

3. **終了フェーズ**
   - すべてのスレッド終了待機
   - リソースの解放
   - 統計情報の出力（デバッグモード）

***

## 第23章 ガベージコレクション

### 23.1 GCアルゴリズム

**要件REQ-GC-001**：参照カウント + サイクル検出

**アルゴリズム**：
```
1. 参照カウント方式（高速）
   - ほとんどのオブジェクトはこれで回収

2. 定期的なサイクル検出（低頻度）
   - マーク＆スイープで循環参照を検出
   - バックグラウンドスレッドで実行
```

### 23.2 GCのチューニング

**環境変数**：
```bash
FLUX_GC_THRESHOLD=1024    # GC起動の閾値（MB）
FLUX_GC_INTERVAL=100      # サイクル検出の間隔（ms）
FLUX_GC_THREADS=2         # GC用スレッド数
```

***

## 第24章 FFI（外部関数インターフェース）

### 24.1 C言語との連携

**要件REQ-FFI-001**：C言語の関数を呼び出せる

**構文**：
```flux
extern "C" {
    fn c_function(x: Int) -> Int;
}

// 使用
let result = c_function(42);
```

### 24.2 型の対応

**要件REQ-FFI-010**：

| Flux型 | C型 |
|--------|-----|
| Int | int64_t |
| Float | double |
| Bool | bool |
| *const T | const T* |
| *mut T | T* |

### 24.3 安全性

**要件REQ-FFI-100**：外部関数呼び出しは`unsafe`ブロックで囲む

```flux
unsafe {
    let result = c_function(42);
}
```

***

## 第25章 コンパイラ

### 25.1 コンパイルフェーズ

```
ソースコード
    ↓
1. 字句解析（Lexer）
    ↓
2. 構文解析（Parser）
    ↓
3. 型チェック（Type Checker）
    ↓
4. 借用チェック（循環検出など）
    ↓
5. 最適化（Optimizer）
    ↓
6. コード生成（Code Generator）
    ↓
LLVM IR
    ↓
7. LLVM最適化
    ↓
ネイティブコード
```

### 25.2 コンパイラオプション

```bash
fluxc [OPTIONS] <input.flux>

OPTIONS:
  -o <file>           出力ファイル名
  -O <level>          最適化レベル（0-3）
  --emit=<type>       出力形式（llvm-ir, asm, obj）
  --target=<triple>   ターゲットトリプル
  -g                  デバッグ情報を含める
  --verbose           詳細な出力
```

**例**：
```bash
fluxc -O2 -o program main.flux
fluxc --emit=llvm-ir program.flux
```

***

## 第26章 パッケージマネージャ

### 26.1 パッケージ構造

```
my_package/
├── Flux.toml         # パッケージマニフェスト
├── src/
│   ├── main.flux
│   └── lib.flux
├── tests/
│   └── test.flux
└── examples/
    └── example.flux
```

### 26.2 Flux.toml

```toml
[package]
name = "my_package"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
edition = "2025"

[dependencies]
std = "1.0"
robotics = "0.5"

[dev-dependencies]
test_framework = "0.3"
```

### 26.3 コマンド

```bash
flux new <name>        # 新しいパッケージ作成
flux build             # ビルド
flux run               # 実行
flux test              # テスト実行
flux publish           # パッケージ公開
flux install <name>    # パッケージインストール
```

***

## 第27章 ビルドシステム

### 27.1 ビルド設定

**Flux.toml**：
```toml
[build]
target = "x86_64-unknown-linux-gnu"
optimization = 2
debug = false

[profile.dev]
optimization = 0
debug = true

[profile.release]
optimization = 3
debug = false
lto = true
```

***

## 第28章 テストフレームワーク

### 28.1 単体テスト

**構文**：
```flux
#[test]
fn test_addition() {
    assert_eq(2 + 2, 4);
}

#[test]
fn test_gaussian_add() {
    let g1 = Gaussian(10.0, 1.0);
    let g2 = Gaussian(5.0, 2.0);
    let g3 = g1 + g2;
    
    assert_eq(g3.mean, 15.0);
    assert(abs(g3.std - 2.236) < 0.001);
}
```

### 28.2 アサーション

```flux
assert(condition);
assert_eq(left, right);
assert_ne(left, right);
```

***

## 第29章 ドキュメント生成

### 29.1 ドキュメンテーションコメント

```flux
/// 2つの数を足す
///
/// # 引数
/// * `x` - 1つ目の数
/// * `y` - 2つ目の数
///
/// # 戻り値
/// 2つの数の和
///
/// # 例
/// ```
/// let result = add(2, 3);
/// assert_eq(result, 5);
/// ```
fn add(x: Int, y: Int) -> Int {
    return x + y;
}
```

### 29.2 ドキュメント生成コマンド

```bash
flux doc            # ドキュメント生成
flux doc --open     # 生成して開く
```

***

## 第30章 デバッガ

### 30.1 デバッグコマンド

```bash
flux-debug <program>

コマンド：
  break <file>:<line>    ブレークポイント設定
  run                    実行
  continue               続行
  step                   ステップ実行
  next                   次の行
  print <var>            変数表示
  backtrace              スタックトレース
```

***

## 第31章 プロファイラ

### 31.1 プロファイリング

```bash
flux build --profile
./program
flux-profile program.prof
```

**出力例**：
```
Function                Time      Calls
----------------------------------------
compute_control        45.2%     1000
sensor_fusion          32.1%     1000
gaussian_sample        12.3%     5000
```

***

## 第32章 Language Server Protocol

### 32.1 LSP機能

**実装機能**：
- コード補完
- 定義へのジャンプ
- 型情報の表示
- エラー診断
- リファクタリング

***

## 第33章 形式的意味論

**（第12章で定義済み）**

***

## 第34章 型システムの健全性

**定理**：
```
進行性：⊢ t : T ⇒ value(t) ∨ ∃t'. t → t'
型保存：⊢ t : T ∧ t → t' ⇒ ⊢ t' : T
健全性：⊢ t : T ⇒ ¬stuck(t)
```

***

# 第35章 トレイトシステム

## 35.1 概要

トレイト（Trait）は、型が実装すべきインターフェースを定義する機能である。

**要件REQ-TRAIT-001**：Fluxはトレイトベースのポリモーフィズムをサポートする。

## 35.2 トレイトの定義

### 35.2.1 構文

**EBNF定義**：
```ebnf
trait_def = 'trait' identifier generic_params? '{' trait_item* '}'
trait_item = method_sig | associated_type
method_sig = 'fn' identifier '(' param_list ')' ('->' type)? ';'
associated_type = 'type' identifier ';'
```

**例**：
```flux
trait Drawable {
    fn draw(&self);
    fn area(&self) -> Float;
}

trait Iterator<T> {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}
```

### 35.2.2 メソッドシグネチャ

**要件REQ-TRAIT-010**：トレイトはメソッドのシグネチャのみを宣言する。

```flux
trait Display {
    fn display(&self) -> String;
}
```

### 35.2.3 デフォルト実装

**要件REQ-TRAIT-011**：トレイトはデフォルト実装を提供できる。

```flux
trait Comparable {
    fn compare(&self, other: &Self) -> Int;
    
    // デフォルト実装
    fn is_equal(&self, other: &Self) -> Bool {
        return self.compare(other) == 0;
    }
}
```

### 35.2.4 関連型（Associated Types）

**要件REQ-TRAIT-020**：トレイトは関連型を持つことができる。

```flux
trait Container {
    type Item;
    
    fn get(&self, index: Int) -> Option<Self::Item>;
}

impl Container for Array<Int> {
    type Item = Int;
    
    fn get(&self, index: Int) -> Option<Int> {
        // 実装
    }
}
```

## 35.3 トレイトの実装

### 35.3.1 基本的な実装

**構文**：
```flux
impl TraitName for TypeName {
    // メソッド実装
}
```

**例**：
```flux
struct Point {
    x: Float,
    y: Float,
}

impl Drawable for Point {
    fn draw(&self) {
        println("Point at ({}, {})", self.x, self.y);
    }
    
    fn area(&self) -> Float {
        return 0.0;
    }
}
```

### 35.3.2 ジェネリック型への実装

```flux
impl<T> Display for Array<T> where T: Display {
    fn display(&self) -> String {
        let parts = self.map(|item| item.display());
        return "[" + parts.join(", ") + "]";
    }
}
```

### 35.3.3 条件付き実装

**要件REQ-TRAIT-030**：トレイト境界を使って条件付き実装ができる。

```flux
impl<T> Eq for Option<T> where T: Eq {
    fn eq(&self, other: &Self) -> Bool {
        match (self, other) {
            (Some(a), Some(b)) => a.eq(b),
            (None, None) => true,
            _ => false,
        }
    }
}
```

## 35.4 トレイト境界（Trait Bounds）

### 35.4.1 where句

**構文**：
```flux
fn function_name<T>(param: T) -> ReturnType
where
    T: Trait1 + Trait2,
{
    // 実装
}
```

**例**：
```flux
fn print_debug<T>(value: T)
where
    T: Debug,
{
    println("{:?}", value);
}

fn compare_and_display<T>(a: T, b: T)
where
    T: Ord + Display,
{
    if a < b {
        println("{} is less than {}", a.display(), b.display());
    }
}
```

### 35.4.2 インライントレイト境界

```flux
fn process<T: Clone + Debug>(value: T) {
    let copy = value.clone();
    println("{:?}", copy);
}
```

## 35.5 トレイトオブジェクト

**要件REQ-TRAIT-100**：トレイトは動的ディスパッチのために使用できる。

```flux
fn draw_all(shapes: Array<dyn Drawable>) {
    for shape in shapes {
        shape.draw();
    }
}

// 使用例
let shapes: Array<dyn Drawable> = [
    Box::new(Point { x: 0.0, y: 0.0 }),
    Box::new(Circle { center: Point { x: 1.0, y: 1.0 }, radius: 5.0 }),
];
draw_all(shapes);
```

## 35.6 スーパートレイト

**要件REQ-TRAIT-110**：トレイトは他のトレイトを要求できる。

```flux
trait Printable: Display {
    fn print(&self) {
        println("{}", self.display());
    }
}

// Printableを実装するにはDisplayも実装が必要
```

## 35.7 マーカートレイト

**要件REQ-TRAIT-120**：メソッドを持たないトレイトをマーカートレイトとして使用できる。

```flux
trait Copy {}  // コピー可能を示すマーカー

trait Send {}  // スレッド間送信可能を示すマーカー
```

***

# 第36章 モジュールシステム

## 36.1 概要

Fluxのモジュールシステムは、コードを論理的な単位に分割し、名前空間を管理する。

## 36.2 モジュールの定義

### 36.2.1 ファイルベースのモジュール

**要件REQ-MOD-001**：各`.flux`ファイルは1つのモジュールを定義する。

```
src/
├── main.flux        // メインモジュール
├── geometry.flux    // geometryモジュール
└── sensors/
    ├── mod.flux     // sensorsモジュール
    ├── lidar.flux   // sensors::lidarモジュール
    └── camera.flux  // sensors::cameraモジュール
```

### 36.2.2 インラインモジュール

```flux
mod geometry {
    pub struct Point {
        pub x: Float,
        pub y: Float,
    }
    
    pub fn distance(p1: Point, p2: Point) -> Float {
        // 実装
    }
}
```

## 36.3 可視性

### 36.3.1 可視性修飾子

**要件REQ-MOD-010**：以下の可視性修飾子をサポートする：

| 修飾子 | 意味 |
|--------|------|
| `pub` | 公開（どこからでもアクセス可能） |
| `pub(crate)` | クレート内公開 |
| `priv`（デフォルト） | 非公開（同一モジュール内のみ） |

**例**：
```flux
// geometry.flux
pub struct Point {
    pub x: Float,
    pub y: Float,
}

pub fn distance(p1: Point, p2: Point) -> Float {
    // 公開関数
}

fn helper() {
    // 非公開関数（このモジュール内のみ）
}
```

## 36.4 インポート

### 36.4.1 use文

**構文**：
```flux
use path::to::item;
use path::to::{item1, item2};
use path::to::item as alias;
```

**例**：
```flux
use std::collections::Array;
use std::prob::{Gaussian, Uniform};
use std::reactive::Signal as ReactiveSignal;
```

### 36.4.2 ワイルドカードインポート

```flux
use std::prob::*;  // probモジュールのすべてをインポート
```

**要件REQ-MOD-020**：ワイルドカードインポートは名前衝突に注意が必要（警告を出す）。

### 36.4.3 再エクスポート

```flux
// lib.flux
pub use self::geometry::Point;  // Pointを再エクスポート
pub use self::sensors::*;       // sensorsの全てを再エクスポート

mod geometry;
mod sensors;
```

## 36.5 パス解決

### 36.5.1 絶対パス

```flux
use crate::geometry::Point;  // クレートルートからの絶対パス
```

### 36.5.2 相対パス

```flux
use super::geometry::Point;  // 親モジュールからの相対パス
use self::submodule::Item;   // 現在のモジュールからの相対パス
```

## 36.6 循環依存の扱い

**要件REQ-MOD-100**：循環依存はコンパイルエラーとする。

```flux
// module_a.flux
use crate::module_b::FuncB;

pub fn func_a() { func_b(); }

// module_b.flux
use crate::module_a::FuncA;

pub fn func_b() { func_a(); }

// エラー：Circular dependency detected
```

**解決策**：共通のモジュールに機能を移動する。

***

# 第37章 標準トレイト

## 37.1 概要

Fluxの標準ライブラリは、一般的な操作のためのトレイトを提供する。

## 37.2 Clone トレイト

**定義**：
```flux
trait Clone {
    fn clone(&self) -> Self;
}
```

**用途**：値の複製

**例**：
```flux
impl Clone for Point {
    fn clone(&self) -> Point {
        return Point { x: self.x, y: self.y };
    }
}

let p1 = Point { x: 1.0, y: 2.0 };
let p2 = p1.clone();
```

**derive属性**：
```flux
#[derive(Clone)]
struct Point {
    x: Float,
    y: Float,
}
```

## 37.3 Copy トレイト

**定義**：
```flux
trait Copy: Clone {}
```

**用途**：暗黙的なコピー（スタック上の値のみ）

**要件REQ-TRAIT-200**：Copyトレイトを実装できるのは、すべてのフィールドがCopyを実装している型のみ。

**例**：
```flux
#[derive(Copy, Clone)]
struct Point {
    x: Float,
    y: Float,
}

let p1 = Point { x: 1.0, y: 2.0 };
let p2 = p1;  // 暗黙的にコピー（moveではない）
println("{}", p1.x);  // p1は依然として使用可能
```

## 37.4 Eq トレイト

**定義**：
```flux
trait Eq {
    fn eq(&self, other: &Self) -> Bool;
    
    fn ne(&self, other: &Self) -> Bool {
        return !self.eq(other);
    }
}
```

**用途**：等価性の判定

**例**：
```flux
impl Eq for Point {
    fn eq(&self, other: &Point) -> Bool {
        return self.x == other.x && self.y == other.y;
    }
}

let p1 = Point { x: 1.0, y: 2.0 };
let p2 = Point { x: 1.0, y: 2.0 };
assert(p1.eq(p2));
```

## 37.5 Ord トレイト

**定義**：
```flux
enum Ordering {
    Less,
    Equal,
    Greater,
}

trait Ord: Eq {
    fn cmp(&self, other: &Self) -> Ordering;
    
    fn lt(&self, other: &Self) -> Bool {
        return matches!(self.cmp(other), Ordering::Less);
    }
    
    fn le(&self, other: &Self) -> Bool {
        return !matches!(self.cmp(other), Ordering::Greater);
    }
    
    fn gt(&self, other: &Self) -> Bool {
        return matches!(self.cmp(other), Ordering::Greater);
    }
    
    fn ge(&self, other: &Self) -> Bool {
        return !matches!(self.cmp(other), Ordering::Less);
    }
}
```

**例**：
```flux
impl Ord for Point {
    fn cmp(&self, other: &Point) -> Ordering {
        let dist1 = self.x * self.x + self.y * self.y;
        let dist2 = other.x * other.x + other.y * other.y;
        if dist1 < dist2 {
            return Ordering::Less;
        } else if dist1 > dist2 {
            return Ordering::Greater;
        } else {
            return Ordering::Equal;
        }
    }
}
```

## 37.6 Debug トレイト

**定義**：
```flux
trait Debug {
    fn debug(&self) -> String;
}
```

**用途**：デバッグ出力

**例**：
```flux
impl Debug for Point {
    fn debug(&self) -> String {
        return format("Point {{ x: {}, y: {} }}", self.x, self.y);
    }
}

let p = Point { x: 1.0, y: 2.0 };
println("{:?}", p);  // Point { x: 1.0, y: 2.0 }
```

## 37.7 Display トレイト

**定義**：
```flux
trait Display {
    fn display(&self) -> String;
}
```

**用途**：ユーザー向け出力

**例**：
```flux
impl Display for Point {
    fn display(&self) -> String {
        return format("({}, {})", self.x, self.y);
    }
}

let p = Point { x: 1.0, y: 2.0 };
println("{}", p);  // (1.0, 2.0)
```

## 37.8 Hash トレイト

**定義**：
```flux
trait Hash {
    fn hash(&self) -> u64;
}
```

**用途**：ハッシュマップのキー

**例**：
```flux
impl Hash for Point {
    fn hash(&self) -> u64 {
        let h1 = hash_float(self.x);
        let h2 = hash_float(self.y);
        return h1 ^ h2;
    }
}
```

## 37.9 Default トレイト

**定義**：
```flux
trait Default {
    fn default() -> Self;
}
```

**用途**：デフォルト値の生成

**例**：
```flux
impl Default for Point {
    fn default() -> Point {
        return Point { x: 0.0, y: 0.0 };
    }
}

let p = Point::default();  // Point { x: 0.0, y: 0.0 }
```

***

# 第38章 演算子オーバーロード

## 38.1 概要

演算子オーバーロードは、カスタム型に対して演算子の挙動を定義する機能である。

## 38.2 算術演算子

### 38.2.1 Add トレイト

**定義**：
```flux
trait Add<Rhs = Self> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}
```

**例**：
```flux
impl Add for Point {
    type Output = Point;
    
    fn add(self, other: Point) -> Point {
        return Point {
            x: self.x + other.x,
            y: self.y + other.y,
        };
    }
}

let p1 = Point { x: 1.0, y: 2.0 };
let p2 = Point { x: 3.0, y: 4.0 };
let p3 = p1 + p2;  // Point { x: 4.0, y: 6.0 }
```

### 38.2.2 Sub, Mul, Div トレイト

**定義**：
```flux
trait Sub<Rhs = Self> {
    type Output;
    fn sub(self, rhs: Rhs) -> Self::Output;
}

trait Mul<Rhs = Self> {
    type Output;
    fn mul(self, rhs: Rhs) -> Self::Output;
}

trait Div<Rhs = Self> {
    type Output;
    fn div(self, rhs: Rhs) -> Self::Output;
}
```

**Gaussian型の例**（既出）：
```flux
impl Add for Gaussian {
    type Output = Gaussian;
    
    fn add(self, other: Gaussian) -> Gaussian {
        let new_mean = self.mean + other.mean;
        let new_std = sqrt(self.std * self.std + other.std * other.std);
        return Gaussian(new_mean, new_std);
    }
}

impl Mul<Float> for Gaussian {
    type Output = Gaussian;
    
    fn mul(self, scalar: Float) -> Gaussian {
        return Gaussian(self.mean * scalar, self.std * abs(scalar));
    }
}
```

## 38.3 比較演算子

### 38.3.1 PartialEq トレイト

**定義**：
```flux
trait PartialEq<Rhs = Self> {
    fn eq(&self, other: &Rhs) -> Bool;
    
    fn ne(&self, other: &Rhs) -> Bool {
        return !self.eq(other);
    }
}
```

**要件REQ-OP-001**：`==` と `!=` 演算子はPartialEqトレイトで定義される。

### 38.3.2 PartialOrd トレイト

**定義**：
```flux
trait PartialOrd<Rhs = Self>: PartialEq<Rhs> {
    fn partial_cmp(&self, other: &Rhs) -> Option<Ordering>;
    
    fn lt(&self, other: &Rhs) -> Bool {
        matches!(self.partial_cmp(other), Some(Ordering::Less))
    }
    
    fn le(&self, other: &Rhs) -> Bool {
        !matches!(self.partial_cmp(other), Some(Ordering::Greater))
    }
    
    fn gt(&self, other: &Rhs) -> Bool {
        matches!(self.partial_cmp(other), Some(Ordering::Greater))
    }
    
    fn ge(&self, other: &Rhs) -> Bool {
        !matches!(self.partial_cmp(other), Some(Ordering::Less))
    }
}
```

**要件REQ-OP-002**：`<`, `>`, `<=`, `>=` 演算子はPartialOrdトレイトで定義される。

## 38.4 インデックス演算子

### 38.4.1 Index トレイト

**定義**：
```flux
trait Index<Idx> {
    type Output;
    fn index(&self, index: Idx) -> &Self::Output;
}

trait IndexMut<Idx>: Index<Idx> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
}
```

**例**：
```flux
impl Index<Int> for Array<T> {
    type Output = T;
    
    fn index(&self, index: Int) -> &T {
        if index < 0 || index >= self.len() {
            panic("Index out of bounds");
        }
        return &self.data[index];
    }
}

let arr = [1, 2, 3, 4, 5];
let value = arr[2];  // 3
```

## 38.5 関数呼び出し演算子

**定義**：
```flux
trait Fn<Args> {
    type Output;
    fn call(&self, args: Args) -> Self::Output;
}

trait FnMut<Args>: Fn<Args> {
    fn call_mut(&mut self, args: Args) -> Self::Output;
}
```

***

# 第39章 属性システム

## 39.1 概要

属性（アトリビュート）は、メタデータをコードに付加する機能である。

## 39.2 標準属性

### 39.2.1 derive属性

**構文**：
```flux
#[derive(TraitName1, TraitName2, ...)]
```

**用途**：自動的にトレイト実装を生成

**例**：
```flux
#[derive(Clone, Debug, Eq)]
struct Point {
    x: Float,
    y: Float,
}

// コンパイラが自動的に以下を生成：
// - Clone::clone
// - Debug::debug
// - Eq::eq
```

**サポートされるトレイト**：
- Clone
- Copy
- Debug
- Eq
- Ord
- Hash
- Default

### 39.2.2 test属性

**構文**：
```flux
#[test]
fn test_function_name() {
    // テストコード
}
```

**例**：
```flux
#[test]
fn test_addition() {
    assert_eq(2 + 2, 4);
}

#[test]
fn test_gaussian_add() {
    let g1 = Gaussian(10.0, 1.0);
    let g2 = Gaussian(5.0, 2.0);
    let g3 = g1 + g2;
    assert_eq(g3.mean, 15.0);
}
```

### 39.2.3 inline属性

**構文**：
```flux
#[inline]
fn function_name() { ... }

#[inline(always)]
fn must_inline() { ... }

#[inline(never)]
fn never_inline() { ... }
```

**用途**：インライン化の制御

**例**：
```flux
#[inline]
fn add(x: Int, y: Int) -> Int {
    return x + y;
}
```

### 39.2.4 deprecated属性

**構文**：
```flux
#[deprecated]
fn old_function() { ... }

#[deprecated(since = "1.2.0", note = "Use new_function instead")]
fn old_function_with_message() { ... }
```

**例**：
```flux
#[deprecated(note = "Use Gaussian::new instead")]
fn create_gaussian(m: Float, s: Float) -> Gaussian {
    return Gaussian(m, s);
}
```

### 39.2.5 cfg属性

**構文**：
```flux
#[cfg(target_os = "linux")]
fn linux_specific() { ... }

#[cfg(target_arch = "x86_64")]
fn x86_specific() { ... }
```

**例**：
```flux
#[cfg(target_os = "linux")]
use std::os::linux::Sensor;

#[cfg(target_os = "windows")]
use std::os::windows::Sensor;
```

### 39.2.6 allow/warn/deny属性

**構文**：
```flux
#[allow(unused_variables)]
fn function_with_unused() { ... }

#[warn(deprecated)]
fn function_using_deprecated() { ... }

#[deny(unsafe_code)]
mod no_unsafe { ... }
```

## 39.3 カスタム属性

**要件REQ-ATTR-100**：将来の拡張として、カスタム属性をサポートする予定。

***

# 第40章 timeモジュール

## 40.1 概要

timeモジュールは、時間・日付・期間を扱う機能を提供する。

## 40.2 Duration型

### 40.2.1 定義

**要件REQ-TIME-001**：
```flux
struct Duration {
    secs: Int,
    nanos: Int,  // 0-999,999,999
}
```

### 40.2.2 構築

```flux
impl Duration {
    fn from_secs(secs: Int) -> Duration;
    fn from_millis(millis: Int) -> Duration;
    fn from_micros(micros: Int) -> Duration;
    fn from_nanos(nanos: Int) -> Duration;
}
```

**例**：
```flux
let d1 = Duration::from_secs(5);
let d2 = Duration::from_millis(100);
let d3 = Duration::from_micros(500);
```

### 40.2.3 変換

```flux
impl Duration {
    fn as_secs(&self) -> Int;
    fn as_millis(&self) -> Int;
    fn as_micros(&self) -> Int;
    fn as_nanos(&self) -> Int;
    
    fn as_secs_f64(&self) -> Float;
}
```

### 40.2.4 演算

```flux
impl Add for Duration {
    type Output = Duration;
    fn add(self, other: Duration) -> Duration;
}

impl Sub for Duration {
    type Output = Duration;
    fn sub(self, other: Duration) -> Duration;
}

impl Mul<Int> for Duration {
    type Output = Duration;
    fn mul(self, scalar: Int) -> Duration;
}
```

**例**：
```flux
let d1 = Duration::from_secs(5);
let d2 = Duration::from_secs(3);
let d3 = d1 + d2;  // 8秒
let d4 = d1 * 2;   // 10秒
```

### 40.2.5 リテラル構文

**要件REQ-TIME-010**：期間リテラルをサポートする。

```flux
let d1 = 5*s;      // 5秒
let d2 = 100*ms;   // 100ミリ秒
let d3 = 500*us;   // 500マイクロ秒
let d4 = 1*min;    // 1分
let d5 = 2*hr;     // 2時間
```

## 40.3 Instant型

### 40.3.1 定義

**要件REQ-TIME-100**：
```flux
struct Instant;
```

**用途**：単調増加する時刻（経過時間の測定用）

### 40.3.2 メソッド

```flux
impl Instant {
    /// 現在の時刻を取得
    fn now() -> Instant;
    
    /// 経過時間を計算
    fn elapsed(&self) -> Duration;
    
    /// 期間を加算
    fn add(&self, duration: Duration) -> Instant;
    
    /// 2つの時刻の差を計算
    fn duration_since(&self, earlier: Instant) -> Duration;
}
```

**例**：
```flux
let start = Instant::now();
compute_heavy_task();
let elapsed = start.elapsed();
println("Elapsed: {} ms", elapsed.as_millis());
```

## 40.4 SystemTime型

**要件REQ-TIME-200**：
```flux
struct SystemTime;

impl SystemTime {
    fn now() -> SystemTime;
    fn duration_since(&self, earlier: SystemTime) -> Result<Duration, Error>;
}
```

**用途**：実時刻（タイムスタンプ）

***

# 第41章 文字列操作API

## 41.1 String型の拡張

### 41.1.1 基本メソッド

```flux
impl String {
    /// 文字列の長さ（バイト数）
    fn len(&self) -> Int;
    
    /// 文字列が空か判定
    fn is_empty(&self) -> Bool;
    
    /// 文字列をクリア
    fn clear(&mut self);
    
    /// 容量
    fn capacity(&self) -> Int;
}
```

### 41.1.2 文字列の分割・結合

```flux
impl String {
    /// 文字列を分割
    fn split(&self, pattern: String) -> Array<String>;
    
    /// 行ごとに分割
    fn lines(&self) -> Array<String>;
    
    /// 空白で分割
    fn split_whitespace(&self) -> Array<String>;
}

/// 配列を結合
fn join<T: Display>(items: Array<T>, separator: String) -> String;
```

**例**：
```flux
let text = "apple,banana,cherry";
let parts = text.split(",");
// ["apple", "banana", "cherry"]

let joined = join(parts, " | ");
// "apple | banana | cherry"
```

### 41.1.3 トリム

```flux
impl String {
    /// 前後の空白を削除
    fn trim(&self) -> String;
    
    /// 先頭の空白を削除
    fn trim_start(&self) -> String;
    
    /// 末尾の空白を削除
    fn trim_end(&self) -> String;
}
```

**例**：
```flux
let s = "  hello  ";
let trimmed = s.trim();  // "hello"
```

### 41.1.4 大文字・小文字変換

```flux
impl String {
    fn to_uppercase(&self) -> String;
    fn to_lowercase(&self) -> String;
}
```

**例**：
```flux
let s = "Hello";
let upper = s.to_uppercase();  // "HELLO"
let lower = s.to_lowercase();  // "hello"
```

### 41.1.5 検索・置換

```flux
impl String {
    /// 部分文字列を含むか
    fn contains(&self, pattern: String) -> Bool;
    
    /// 先頭が一致するか
    fn starts_with(&self, pattern: String) -> Bool;
    
    /// 末尾が一致するか
    fn ends_with(&self, pattern: String) -> Bool;
    
    /// 最初の出現位置
    fn find(&self, pattern: String) -> Option<Int>;
    
    /// 置換
    fn replace(&self, from: String, to: String) -> String;
}
```

**例**：
```flux
let text = "hello world";
if text.contains("world") {
    println("Found!");
}

let replaced = text.replace("world", "Flux");
// "hello Flux"
```

### 41.1.6 文字イテレータ

```flux
impl String {
    /// 文字のイテレータ
    fn chars(&self) -> Iterator<Char>;
    
    /// バイトのイテレータ
    fn bytes(&self) -> Iterator<u8>;
}
```

**例**：
```flux
let s = "hello";
for ch in s.chars() {
    println("{}", ch);
}
```

***

# 第42章 数値演算の詳細

## 42.1 整数演算

### 42.1.1 チェック付き演算

**要件REQ-NUM-001**：オーバーフローをチェックする演算を提供する。

```flux
impl Int {
    /// 加算（オーバーフロー時はNone）
    fn checked_add(&self, other: Int) -> Option<Int>;
    
    /// 減算（オーバーフロー時はNone）
    fn checked_sub(&self, other: Int) -> Option<Int>;
    
    /// 乗算（オーバーフロー時はNone）
    fn checked_mul(&self, other: Int) -> Option<Int>;
    
    /// 除算（ゼロ除算時はNone）
    fn checked_div(&self, other: Int) -> Option<Int>;
}
```

**例**：
```flux
let x: Int = 9223372036854775807;  // Int::MAX
let result = x.checked_add(1);
match result {
    Some(v) => println("Result: {}", v),
    None => println("Overflow!"),
}
```

### 42.1.2 飽和演算

**要件REQ-NUM-002**：オーバーフロー時に最大/最小値に飽和する演算を提供する。

```flux
impl Int {
    fn saturating_add(&self, other: Int) -> Int;
    fn saturating_sub(&self, other: Int) -> Int;
    fn saturating_mul(&self, other: Int) -> Int;
}
```

**例**：
```flux
let x: Int = 9223372036854775807;  // Int::MAX
let result = x.saturating_add(100);
// result == Int::MAX（飽和）
```

### 42.1.3 ラッピング演算

**要件REQ-NUM-003**：オーバーフロー時にラップアラウンドする演算を提供する。

```flux
impl Int {
    fn wrapping_add(&self, other: Int) -> Int;
    fn wrapping_sub(&self, other: Int) -> Int;
    fn wrapping_mul(&self, other: Int) -> Int;
}
```

**例**：
```flux
let x: Int = 9223372036854775807;  // Int::MAX
let result = x.wrapping_add(1);
// result == Int::MIN（ラップアラウンド）
```

## 42.2 浮動小数点演算

### 42.2.1 特殊値の判定

```flux
impl Float {
    fn is_nan(&self) -> Bool;
    fn is_infinite(&self) -> Bool;
    fn is_finite(&self) -> Bool;
    fn is_normal(&self) -> Bool;
}
```

**例**：
```flux
let x = 0.0 / 0.0;  // NaN
assert(x.is_nan());

let y = 1.0 / 0.0;  // Infinity
assert(y.is_infinite());
```

### 42.2.2 丸め

```flux
impl Float {
    fn floor(&self) -> Float;
    fn ceil(&self) -> Float;
    fn round(&self) -> Float;
    fn trunc(&self) -> Float;
}
```

**例**：
```flux
let x = 3.7;
assert_eq(x.floor(), 3.0);
assert_eq(x.ceil(), 4.0);
assert_eq(x.round(), 4.0);
```

### 42.2.3 絶対値・符号

```flux
impl Float {
    fn abs(&self) -> Float;
    fn signum(&self) -> Float;  // -1.0, 0.0, 1.0
    fn copysign(&self, sign: Float) -> Float;
}
```

***

# 第43章 Signalの高度な操作

## 43.1 時間ベースの操作

### 43.1.1 スロットル

**定義**：
```flux
impl<T> Signal<T> {
    /// 指定期間内の最初の値のみ通過
    fn throttle(&self, duration: Duration) -> Signal<T>;
}
```

**例**：
```flux
let clicks: Signal<MouseClick> = ...;
let throttled = clicks.throttle(1*s);
// 1秒に1回のみクリックを処理
```

### 43.1.2 デバウンス

**定義**：
```flux
impl<T> Signal<T> {
    /// 指定期間値が変化しない場合のみ通過
    fn debounce(&self, duration: Duration) -> Signal<T>;
}
```

**例**：
```flux
let input: Signal<String> = ...;
let debounced = input.debounce(300*ms);
// 入力が300ms停止したら処理
```

### 43.1.3 遅延

**定義**：
```flux
impl<T> Signal<T> {
    /// 値を指定期間遅延
    fn delay(&self, duration: Duration) -> Signal<T>;
}
```

## 43.2 集約操作

### 43.2.1 ウィンドウ

**定義**：
```flux
impl<T> Signal<T> {
    /// 直近N個の値を配列として取得
    fn window(&self, size: Int) -> Signal<Array<T>>;
}
```

**例**：
```flux
let sensor: Signal<Float> = ...;
let windows = sensor.window(5);
// 直近5個の値の配列
```

### 43.2.2 スキャン

**定義**：
```flux
impl<T> Signal<T> {
    /// 累積値を計算
    fn scan<S>(&self, init: S, f: Fn(S, T) -> S) -> Signal<S>;
}
```

**例**：
```flux
let values: Signal<Int> = ...;
let sum = values.scan(0, |acc, x| acc + x);
// 累積和
```

## 43.3 結合操作

### 43.3.1 Zip

**定義**：
```flux
impl<T> Signal<T> {
    /// 2つのSignalをタプルに結合
    fn zip<U>(&self, other: Signal<U>) -> Signal<(T, U)>;
}
```

**例**：
```flux
let sensor1: Signal<Float> = ...;
let sensor2: Signal<Float> = ...;
let combined = sensor1.zip(sensor2);
// Signal<(Float, Float)>
```

### 43.3.2 Merge

**定義**：
```flux
impl<T> Signal<T> {
    /// 2つのSignalをマージ（いずれかが更新されたら）
    fn merge(&self, other: Signal<T>) -> Signal<T>;
}
```

***

# 第44章 fsモジュール詳細

## 44.1 パス操作

### 44.1.1 Path型

```flux
struct Path {
    inner: String,
}

impl Path {
    fn new(s: String) -> Path;
    fn join(&self, other: Path) -> Path;
    fn parent(&self) -> Option<Path>;
    fn file_name(&self) -> Option<String>;
    fn extension(&self) -> Option<String>;
    fn exists(&self) -> Bool;
    fn is_file(&self) -> Bool;
    fn is_dir(&self) -> Bool;
}
```

**例**：
```flux
let path = Path::new("/home/user/data.txt");
let parent = path.parent();  // Some("/home/user")
let name = path.file_name(); // Some("data.txt")
let ext = path.extension();  // Some("txt")
```

## 44.2 ディレクトリ操作

```flux
fn read_dir(path: Path) -> Result<Array<DirEntry>, IoError>;
fn create_dir(path: Path) -> Result<Unit, IoError>;
fn create_dir_all(path: Path) -> Result<Unit, IoError>;
fn remove_dir(path: Path) -> Result<Unit, IoError>;
fn remove_dir_all(path: Path) -> Result<Unit, IoError>;
```

**例**：
```flux
let entries = read_dir(Path::new("/home/user"))?;
for entry in entries {
    println("{}", entry.path());
}
```

## 44.3 ファイル操作

```flux
fn copy(from: Path, to: Path) -> Result<Unit, IoError>;
fn rename(from: Path, to: Path) -> Result<Unit, IoError>;
fn remove_file(path: Path) -> Result<Unit, IoError>;
fn metadata(path: Path) -> Result<Metadata, IoError>;
```

***

# 第45章 型推論アルゴリズム

## 45.1 Hindley-Milner型推論

**要件REQ-INFER-001**：Fluxの型推論はHindley-Milner型システムに基づく。

### 45.1.1 基本原理

1. **型変数の導入**：未知の型を型変数（α, β, ...）で表現
2. **制約の生成**：式の構造から型の制約を生成
3. **単一化（Unification）**：制約を解いて具体的な型を決定

### 45.1.2 アルゴリズム

```
function infer(expr, env):
    match expr:
        case Literal(value):
            return typeof(value)
        
        case Variable(name):
            return env.lookup(name)
        
        case Lambda(param, body):
            α = fresh_type_var()
            env' = env.extend(param, α)
            β = infer(body, env')
            return α → β
        
        case Application(func, arg):
            τ1 = infer(func, env)
            τ2 = infer(arg, env)
            α = fresh_type_var()
            unify(τ1, τ2 → α)
            return α
```

## 45.2 型推論の限界

**要件REQ-INFER-100**：以下の場合、型注釈が必須：

1. **再帰関数**：戻り値の型を明示
2. **トレイト境界が必要な場合**：ジェネリック関数
3. **曖昧な数値リテラル**：0, 1など

**例**：
```flux
// 型注釈が必要
fn factorial(n: Int) -> Int {  // 戻り値の型を明示
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

// 型注釈不要
fn double(x) { x * 2 }  // Int -> Int と推論
```

***

# 第46章 ロボティクス固有機能

## 46.1 geometryモジュール

### 46.1.1 2D座標変換

```flux
module geometry {
    struct Transform2D {
        translation: (Float, Float),
        rotation: Float,
        scale: Float,
    }
    
    impl Transform2D {
        fn identity() -> Transform2D;
        fn translate(x: Float, y: Float) -> Transform2D;
        fn rotate(angle: Float) -> Transform2D;
        fn scale(factor: Float) -> Transform2D;
        
        fn apply(&self, point: (Float, Float)) -> (Float, Float);
        fn compose(&self, other: Transform2D) -> Transform2D;
        fn inverse(&self) -> Transform2D;
    }
}
```

**例**：
```flux
let t1 = Transform2D::translate(10.0, 20.0);
let t2 = Transform2D::rotate(PI / 4.0);
let combined = t1.compose(t2);

let point = (1.0, 0.0);
let transformed = combined.apply(point);
```

### 46.1.2 3D座標変換

```flux
struct Transform3D {
    matrix: Matrix<Float, 4, 4>,
}
```

## 46.2 controlモジュール

### 46.2.1 PIDコントローラ

```flux
module control {
    struct PIDController {
        kp: Float,
        ki: Float,
        kd: Float,
        integral: Float,
        prev_error: Float,
    }
    
    impl PIDController {
        fn new(kp: Float, ki: Float, kd: Float) -> PIDController;
        
        fn update(&mut self, error: Float, dt: Float) -> Float {
            self.integral += error * dt;
            let derivative = (error - self.prev_error) / dt;
            self.prev_error = error;
            
            return self.kp * error + 
                   self.ki * self.integral + 
                   self.kd * derivative;
        }
    }
}
```

***

# 第47章 コーディング規約

## 47.1 命名規則

**要件REQ-STYLE-001**：

| 項目 | 規則 | 例 |
|------|------|-----|
| モジュール | snake_case | `my_module` |
| 型名 | PascalCase | `MyStruct`, `MyEnum` |
| 関数名 | snake_case | `calculate_distance` |
| 変数名 | snake_case | `sensor_value` |
| 定数 | SCREAMING_SNAKE_CASE | `MAX_SPEED` |

## 47.2 インデント

**要件REQ-STYLE-010**：4スペースインデント

```flux
fn example() {
    if condition {
        do_something();
    }
}
```

## 47.3 行の長さ

**要件REQ-STYLE-020**：最大100文字

***

# 第48章 将来の拡張

## 48.1 計画中の機能

### 48.1.1 async/await（優先度：高）
- 非同期IO
- Future型
- async fn構文

### 48.1.2 マクロシステム（優先度：中）
- 宣言的マクロ
- 手続き的マクロ

### 48.1.3 多変量分布（優先度：中）
- MultivariateGaussian
- カスタム分布

### 48.1.4 ネットワーク（優先度：低）
- TCP/UDP
- HTTP client

## 48.2 検討中の機能

- const ジェネリクス
- 高階型
- リフレクション

***


## 第49章 実装ガイドライン

### 49.1 実装の優先順位

**フェーズ0**（1-2ヶ月）：最小限の実装
- 字句解析・構文解析
- 基本型のみの型チェック
- インタプリタ

**フェーズ1**（2-3ヶ月）：コンパイラ
- LLVM統合
- GC統合
- 基本的なコード生成

**フェーズ2**（3-4ヶ月）：確率型
- Gaussian, Uniform
- 確率演算

**フェーズ3**（3-4ヶ月）：リアクティブ型
- Signal, Event
- 依存グラフ

**フェーズ4**（2-3ヶ月）：ツール
- LSP, デバッガ
- パッケージマネージャ

***

## 第50章 プラットフォーム要件

### 50.1 対応プラットフォーム

**Tier 1**（完全サポート）：
- x86_64-unknown-linux-gnu
- x86_64-apple-darwin
- x86_64-pc-windows-msvc

**Tier 2**（ベストエフォート）：
- aarch64-unknown-linux-gnu
- aarch64-apple-darwin

***

## 第51章 パフォーマンス要件

### 51.1 コンパイル時間

**要件REQ-PERF-001**：
- 1,000行のコード：< 1秒
- 10,000行のコード：< 10秒

### 51.2 実行時性能

**要件REQ-PERF-010**：
- Gaussian演算：< 100ns/op
- Signal更新：< 1μs/op
- GCオーバーヘッド：< 5%

***

## 第52章 セキュリティ要件

### 52.1 メモリ安全性

**保証**：
- バッファオーバーフローなし
- Use-After-Freeなし
- データ競合なし

### 52.2 型安全性

**保証**：
- 型エラーなし（実行時）
- nullポインタ参照なし

***

## 第53章 完全な文法（EBNF）

```ebnf
program = item*

item = function_def | struct_def | enum_def | impl_block

function_def = 'fn' identifier generic_params? '(' param_list ')' 
               ('->' type)? block

struct_def = 'struct' identifier generic_params? '{' field_list '}'

enum_def = 'enum' identifier generic_params? '{' variant_list '}'

impl_block = 'impl' generic_params? type '{' method_def* '}'

type = 'Int' | 'Float' | 'Bool' | 'String' | 'Unit'
     | 'Gaussian' | 'Uniform'
     | 'Signal' '<' type '>'
     | 'Fn' '(' type_list ')' '->' type
     | identifier generic_args?

expr = literal | identifier | binary_expr | unary_expr
     | call_expr | field_expr | index_expr | lambda_expr
     | if_expr | match_expr | block_expr

stmt = let_stmt | expr_stmt | return_stmt
     | while_stmt | for_stmt | break_stmt | continue_stmt
```

***

## 第54章 標準エラーコード一覧

| コード | 説明 |
|--------|------|
| E0001 | 型エラー |
| E0002 | 未定義の変数 |
| E0003 | 循環依存 |
| E0010 | パターンマッチが網羅的でない |
| E0020 | 境界チェック失敗 |

***

## 第55章 互換性ポリシー

### 55.1 バージョニング

**セマンティックバージョニング**：MAJOR.MINOR.PATCH

- **MAJOR**：互換性のない変更
- **MINOR**：後方互換性のある機能追加
- **PATCH**：後方互換性のあるバグ修正

***

## 第56章 移行ガイド

**（将来のバージョンアップ時に記載）**

***

## 第57章 参考文献

1. Billingsley, P. "Probability and Measure" (1995)
2. Czaplicki, E. "Elm: Concurrent FRP for Functional GUIs" (2012)
3. Elliott, C. "Functional Reactive Animation" (1997)
4. Pierce, B. "Types and Programming Languages" (2002)
5. Appel, A. "Modern Compiler Implementation" (1998)

***

## 付録A：完全なコード例

### A.1 ロボット位置推定システム

```flux
fn robot_localization() {
    // センサー入力
    let lidar: Signal<Gaussian> = Signal::from_uncertain_sensor(lidar_device);
    let odometry: Signal<Gaussian> = Signal::from_uncertain_sensor(odometry_device);
    let imu: Signal<Gaussian> = Signal::from_uncertain_sensor(imu_device);
    
    // センサーフュージョン
    let position: Signal<Gaussian> = Signal::combine3(
        lidar,
        odometry,
        imu,
        |l, o, i| {
            // カルマンフィルタ的な統合
            let fused = (l + o + i) / 3.0;
            return posterior(fused, Gaussian(0.0, 0.1));
        }
    );
    
    // 不確実性の監視
    let uncertainty: Signal<Float> = position.map(|g| g.std);
    
    uncertainty.subscribe(|std| {
        if std > 1.0 {
            println("Warning: High uncertainty: {}", std);
        }
    });
    
    // 制御指令生成
    let commands: Signal<ControlCommand> = position.map(|pos| {
        compute_control(pos.mean, pos.std)
    });
    
    commands.subscribe(|cmd| {
        robot.execute(cmd);
    });
}
```

***

**以上で、Flux プログラミング言語 完全仕様書 v1.0 を完了します。**

***

## 文書の承認

| 役割 | 氏名 | 署名 | 日付 |
|------|------|------|------|
| 作成者 | | | 2025-10-09 |
| レビュアー | | | |
| 承認者 | | | |

***

**合計文字数：約95,000文字**