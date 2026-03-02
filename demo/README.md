# Fluno Demos

This directory contains demonstration programs showcasing the capabilities of the Fluno programming language, particularly its Probabilistic Programming (PPL) features.

## How to Run

You can run these demos using the `fluno` command provided by the `cargo run` wrapper.

```bash
# General Syntax
cargo run --bin fluno -- run demo/<filename>.fln
```

## Available Demos

### 1. Sensor Fusion (`sensor_fusion.fln`)
Simulates a sensor fusion problem (like GPS + Accelerometer) using a Kalman Filter-like approach implemented with PPL.
- **Concept**: State Estimation, Reactive Signal Processing
- **Run**: 
  ```bash
  cargo run --bin fluno -- run demo/sensor_fusion.fln
  ```

### 2. Bayesian Coin Toss (`bayesian_coin.fln`)
Infers the bias of a coin from a sequence of coin flips.
- **Concept**: Parameter Estimation, Beta-Bernoulli Model
- **Run**:
  ```bash
  cargo run --bin fluno -- run demo/bayesian_coin.fln
  ```

### 3. Bayesian Linear Regression (`linear_regression.fln`)
Fits a linear model ($y = ax + b$) to noisy data, estimating the posterior distributions of the slope $a$, intercept $b$, and noise $\sigma$.
- **Concept**: Regression, Multi-parameter Inference
- **Run**: 
  ```bash
  cargo run --bin fluno -- run demo/linear_regression.fln
  ```
