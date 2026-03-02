# 第4章：構造体と列挙型 (Structs and Enums)

この章では、Flunoのカスタムデータ型を定義する方法について学びます。構造体（Structs）を使うことで関連するデータをまとめ、列挙型（Enums）を使うことで値が取り得るパターンを表現できます。

## 4.1 構造体の定義とインスタンス化

構造体は、フィールドに名前を付けてデータをグループ化する仕組みです。

```rust
struct User {
    username: String,
    email: String,
    sign_in_count: Int,
    active: Bool,
}
```

### インスタンスの生成

定義した構造体を使用するには、各フィールドに具体的な値を指定してインスタンスを作成します。

```rust
let user1 = User {
    email: "someone@example.com",
    username: "someusername123",
    active: true,
    sign_in_count: 1,
};
```

フィールドへのアクセスにはドット記法を使います。

```rust
println("User email: {}", user1.email);
```

### メソッド構文

構造体に関連付けられた関数（メソッド）を定義するには、`impl` ブロックを使用します。

```rust
struct Rectangle {
    width: Int,
    height: Int,
}

impl Rectangle {
    fn area(self) -> Int {
        return self.width * self.height;
    }

    fn can_hold(self, other: Rectangle) -> Bool {
        return self.width > other.width && self.height > other.height;
    }
}
```

メソッドの呼び出し：

```rust
let rect1 = Rectangle { width: 30, height: 50 };
println("The area of the rectangle is {} square pixels.", rect1.area());
```

## 4.2 列挙型 (Enums)

列挙型は、変数が取り得る「いくつかの異なるバリアント」の一つであることを表現します。

```rust
enum IpAddrKind {
    V4,
    V6,
}

let four = IpAddrKind::V4;
let six = IpAddrKind::V6;
```

### データを保持する列挙型

FlunoのEnumは、各バリアントにデータを紐付けることができます。これは非常に強力な機能です。

```rust
enum Message {
    Quit,
    Move { x: Int, y: Int },
    Write(String),
    ChangeColor(Int, Int, Int),
}
```

## 4.3 match 制御フロー演算子

`match` 式を使用すると、値のパターンに応じてコードを分岐させることができます。これは C言語やJavaの `switch` に似ていますが、より強力で、すべてのケースを網羅しているかコンパイラがチェックしてくれます。

```rust
fn process_message(msg: Message) {
    match msg {
        Message::Quit => {
            println("The Quit variant has no data to destructure.");
        },
        Message::Move { x, y } => {
            println("Move in the x direction {} and in the y direction {}", x, y);
        },
        Message::Write(text) => {
            println("Text message: {}", text);
        },
        Message::ChangeColor(r, g, b) => {
            println("Change the color to red {}, green {}, and blue {}", r, g, b);
        },
    }
}
```

### Option型

Flunoには `null` がありません。その代わり、値が存在するかどうかを表す `Option<T>` 列挙型が標準ライブラリで定義されています。

```rust
enum Option<T> {
    Some(T),
    None,
}
```

`Option` を使うことで、値が存在しない可能性を明示的にハンドリングすることを強制され、安全性マが高まります。

```rust
let some_number = Option::Some(5);
let absent_number: Option<Int> = Option::None;

match some_number {
    Option::Some(i) => println("Got a number: {}", i),
    Option::None => println("Got nothing!"),
}
```
