// src/compiler/prelude.rs

pub const PRELUDE_CODE: &str = r#"
// ==========================================
// Fluno Runtime Prelude
// ==========================================
use std::ops::{Add, Sub, Mul, Div};
use std::rc::Rc;
use std::cell::RefCell;

// --- Helper Functions ---
pub fn println(s: impl std::fmt::Display) { println!("{}", s); }

// --- Gaussian Distribution ---
#[derive(Debug, Clone, Copy)]
pub struct Gaussian {
    pub mean: f64,
    pub std: f64,
}

pub fn Gaussian(mean: f64, std: f64) -> Gaussian {
    Gaussian { mean, std }
}

impl Add for Gaussian {
    type Output = Gaussian;
    fn add(self, other: Gaussian) -> Gaussian {
        let new_mean = self.mean + other.mean;
        let new_std = (self.std.powi(2) + other.std.powi(2)).sqrt();
        Gaussian { mean: new_mean, std: new_std }
    }
}

impl Sub for Gaussian {
    type Output = Gaussian;
    fn sub(self, other: Gaussian) -> Gaussian {
        let new_mean = self.mean - other.mean;
        let new_std = (self.std.powi(2) + other.std.powi(2)).sqrt();
        Gaussian { mean: new_mean, std: new_std }
    }
}

impl Mul<f64> for Gaussian {
    type Output = Gaussian;
    fn mul(self, scalar: f64) -> Gaussian {
        Gaussian { 
            mean: self.mean * scalar, 
            std: self.std * scalar.abs() 
        }
    }
}

impl Mul<Gaussian> for f64 {
    type Output = Gaussian;
    fn mul(self, g: Gaussian) -> Gaussian {
        g * self
    }
}

// --- Reactive Types (Signal) ---
// 簡易的なPull型Signal: 値が必要になった瞬間に計算する関数をラップ
#[derive(Clone)]
pub struct Signal<T> {
    compute: Rc<dyn Fn() -> T>,
}

impl<T: Clone + 'static> Signal<T> {
    pub fn new(val: T) -> Self {
        Signal { compute: Rc::new(move || val.clone()) }
    }

    pub fn get(&self) -> T {
        (self.compute)()
    }

    pub fn map<U: Clone + 'static, F: Fn(T) -> U + 'static>(&self, f: F) -> Signal<U> {
        let prev = self.compute.clone();
        Signal {
            compute: Rc::new(move || f(prev()))
        }
    }
    
    // combine は型推論が複雑になるため、今回は簡易実装に留めるか、ヘルパー関数で対応
}

// FlunoのSignalコンストラクタ
pub fn Signal_new<T: Clone + 'static>(val: T) -> Signal<T> {
    Signal::new(val)
}

// --- Option / Result Helpers if needed ---
// Rust標準のOption/Resultをそのまま使うため、特別な定義は不要

// ==========================================
"#;
