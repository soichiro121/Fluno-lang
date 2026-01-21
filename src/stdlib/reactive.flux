// リアクティブ拡張: Signal型
// 注意: 実際のロジックはVM組み込み(Builtin)で処理されます

// Signalコンストラクタ
// 内部的に Builtin("Signal_new") を呼ぶなどの実装を想定
fn Signal::new(initial_value) {
    // Native implementation hook
    __builtin_signal_new(initial_value)
}

// 2つのシグナルを結合
fn Signal::combine(s1, s2, combiner_func) {
    __builtin_signal_combine(s1, s2, combiner_func)
}

// ※ map / filter はメソッド構文 s.map(...) で呼び出されるため、
//    ここでの関数定義ではなくVMの eval_field_access で処理されます。
