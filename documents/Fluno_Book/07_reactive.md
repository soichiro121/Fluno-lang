# 第7章：リアクティブプログラミング (Reactive Programming/FRP)

ロボティクスやGUIアプリでは、時間とともに変化する値（センサー入力、マウス座標など）を扱う必要があります。Flunoは **Functional Reactive Programming (FRP)** の概念を取り入れ、これらの値を **Signal** (シグナル) として扱います。

## 7.1 Signal型

`Signal<T>` は、「時間とともに変化する型 `T` の値」を表します。通常の変数とは異なり、Signalの値は時間の経過とともに自動的に更新されます。

```rust
use std::reactive::Signal;

// 時変値の作成（通常はライブラリやセンサーから提供される）
let sensor_input: Signal<Float> = get_sensor_stream();
```

## 7.2 依存関係と自動更新

Signalの真価は、他のSignalから新しいSignalを作るときに発揮されます。`map` や演算子を使うことで、値の依存関係を宣言的に記述できます。

```rust
// センサー値（摂氏）
let celsius: Signal<Float> = ...;

// 華氏に変換。celsiusが変化すると自動的にfahrenheitも変化する
let fahrenheit: Signal<Float> = celsius.map(|c| c * 1.8 + 32.0);
```

このように記述しておくと、`celsius` の値が更新されるたびに、自動的に `fahrenheit` の再計算が行われます。プログラマが手動でイベントリスナーを登録したり、コールバック地獄に陥ったりする必要はありません。

## 7.3 データフローグラフ

Flunoのランタイムは、Signal間の依存関係を**データフローグラフ**として管理しています。

1.  入力（Source Signal）が変化する。
2.  そのSignalに依存している計算ノード（Computed Signal）がダーティ（更新必要）としてマークされる。
3.  必要に応じて再計算が実行され、値が伝播する（Push型またはPull型）。

### 高階Signal (Higher-Order Signals)

Signalの中にSignalが入っているような構造（`Signal<Signal<T>>`）も扱うことができ、動的に変化するシステム構成などを表現するのに役立ちます。

## 7.4 使用例：ロボットの障害物回避

```rust
fn avoid_obstacle(distance_sensor: Signal<Float>) -> Signal<String> {
    // 距離が1.0未満なら "STOP"、そうでなければ "GO"
    let command = distance_sensor.map(|d| {
        if d < 1.0 {
            "STOP"
        } else {
            "GO"
        }
    });
    
    return command;
}
```

この関数は一度呼ばれるだけですが、戻り値の `Signal<String>` は、`distance_sensor` の値が変わるたびに永遠に（そのSignalが生存している限り）値を更新し続けます。これがリアクティブプログラミングの「宣言的」な強みです。
