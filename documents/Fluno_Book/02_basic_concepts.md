# 第2章：基本概念 (Basic Concepts)

この章では、Flunoプログラミングにおいて基本となる概念を解説します。変数、基本データ型、関数、コメント、制御フローといった、ほぼすべてのFlunoプログラムで使用される要素について学びます。これらは他の言語と似ている部分も多いですが、Fluno独自の設計思想が反映されている部分もあります。

## 2.1 変数と可変性 (Variables and Mutability)

Flunoでは、変数は `let` キーワードを使って宣言します。

```rust
let x = 5;
```

デフォルトでは、変数は**不変 (immutable)** です。一度値を拘束すると、その変数の値を変更することはできません。これは、並行性の安全性や予測可能性を高めるためのFlunoの設計上の選択です。

```rust
let x = 5;
x = 6; // コンパイルエラー！
```

可変な変数が必要な場合は、`mut` キーワードを付ける必要があります。

```rust
let mut x = 5;
println("The value of x is: {}", x);
x = 6;
println("The value of x is now: {}", x);
```

### 定数 (Constants)

定数は `const` キーワードで宣言します。変数は関数の戻り値などを代入できますが、定数はコンパイル時に値が確定している必要があります。

```rust
const MAX_POINTS: Int = 100_000;
```

## 2.2 データ型 (Data Types)

Flunoは静的型付け言語です。すべての値は特定の型を持ちます。
多くの場合、コンパイラは文脈から型を推論できますが、明示的に型注釈を書くこともできます。

```rust
let guess: Int = 42;
```

### スカラー型

スカラー型は単一の値を表します。

1.  **整数 (Boolean)**: `Int` (64ビット符号付き整数)
    *   例: `42`, `-10`, `100_000`
2.  **浮動小数点数 (Floating-Point)**: `Float` (64ビット倍精度浮動小数点数)
    *   例: `3.14`, `-0.01`, `2.0`
    *   Flunoの `Float` は自動微分に対応した `ADFloat` として振る舞うことができます。
3.  **真理値 (Boolean)**: `Bool`
    *   値: `true`, `false`
4.  **文字 (Character)**: `Char`
    *   Unicodeスカラー値。シングルクォートで囲みます。例: `'z'`, `'😻'`

### 複合型

1.  **タプル (Tuple)**
    複数の型の値を一つのまとまりとして扱います。

    ```rust
    let tup: (Int, Float, Int) = (500, 6.4, 1);
    let (x, y, z) = tup; // 分解 (destructuring)
    println("The value of y is: {}", y);
    ```

2.  **配列 (Array)**
    同じ型の値を固定長または可変長で保持します。

    ```rust
    let a = [1, 2, 3, 4, 5];
    let first = a[0];
    ```

## 2.3 関数 (Functions)

関数は `fn` キーワードで定義します。
Flunoのコードでは、関数名にはスネークケース（`snake_case`）を使用するのが慣習です。

```rust
fn main() {
    print_labeled_measurement(5, 'h');
}

fn print_labeled_measurement(value: Int, unit_label: Char) {
    println("The measurement is: {}{}", value, unit_label);
}
```

### 戻り値

値を返す関数を定義する場合、矢印（`->`）の後に戻り値の型を宣言します。
Flunoでは、ブロックの最後の式が暗黙的に戻り値となります（セミコロンを付けない）。`return` キーワードを使って早期リターンすることも可能です。

```rust
fn five() -> Int {
    5 // セミコロンなしは式 (expression)
}

fn plus_one(x: Int) -> Int {
    return x + 1; // return文も使用可能
}
```

## 2.4 制御フロー (Control Flow)

### if 式

条件分岐には `if` を使用します。`if` は式（expression）であるため、結果を変数に代入することができます。

```rust
let number = 3;

if number < 5 {
    println("condition was true");
} else {
    println("condition was false");
}

let condition = true;
let number = if condition { 5 } else { 6 };
```

### ループ

Flunoには、`loop`、`while`、`for` の3種類のループがあります。

1.  **loop**: 無限ループを作成します。`break` で脱出できます。

    ```rust
    let mut count = 0;
    loop {
        count += 1;
        if count == 3 {
            break;
        }
    }
    ```

2.  **while**: 条件が真である間繰り返します。

    ```rust
    let mut number = 3;
    while number != 0 {
        println("{}!", number);
        number -= 1;
    }
    ```

3.  **for**: コレクションの要素をイテレートします。

    ```rust
    let a = [10, 20, 30, 40, 50];
    for element in a {
        println("the value is: {}", element);
    }
    ```
