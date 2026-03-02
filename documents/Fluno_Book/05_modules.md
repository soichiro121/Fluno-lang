# 第5章：モジュールシステム (Modules)

プログラムが大規模になるにつれて、コードを整理・分割することが重要になります。Flunoのモジュールシステムは、コードの可視性（public/private）を制御し、名前空間を管理するためのツールを提供します。

## 5.1 モジュールの定義

`mod` キーワードを使ってモジュールを定義します。モジュールの中に、関数、構造体、列挙型、あるいは他のモジュールを入れることができます。

```rust
// lib.fln
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
        fn seat_at_table() {}
    }

    mod serving {
        fn take_order() {}
        fn serve_order() {}
        fn take_payment() {}
    }
}
```

## 5.2 パスによるアイテムの参照

モジュール内のアイテムを参照するには、パスを使います。パスには絶対パスと相対パスの2種類があります。

*   **絶対パス**: クレートのルート（`crate`）から始まるパス。
*   **相対パス**: 現在のモジュール（`self`）や親モジュール（`super`）から始まるパス。

```rust
pub fn eat_at_restaurant() {
    // 絶対パス
    crate::front_of_house::hosting::add_to_waitlist();

    // 相対パス
    front_of_house::hosting::add_to_waitlist();
}
```

### プライバシー境界 (Privacy Boundary)

デフォルトでは、モジュール内のアイテムはすべて **プライベート (private)** です。親モジュールからは、子モジュールのプライベートなアイテムにはアクセスできません。
外部からアクセス可能にするには、`pub` キーワードを付けて **パブリック (public)** にする必要があります。

上記の例で `hosting` モジュールと `add_to_waitlist` 関数に `pub` が付いているのはそのためです。

## 5.3 use キーワード

`use` キーワードを使うと、パスをスコープに持ち込むことができ、あたかもそのアイテムがローカルに定義されているかのように短く記述できます。

```rust
use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
    hosting::add_to_waitlist();
    hosting::add_to_waitlist();
}
```

### 外部パッケージの使用

標準ライブラリや外部パッケージを使用する場合も `use` を使います。

```rust
use std::collections::HashMap;

fn main() {
    let mut map = HashMap::new();
    map.insert(1, 2);
}
```

これで、Flunoを使った大規模なアプリケーション開発の準備が整いました。
次章からは、いよいよFlunoの最も特徴的な機能である「確率的プログラミング」の世界に入っていきます。
