// src/vm/distributions.rs

use crate::ad::types::ADFloat;
use rand::Rng;
use std::f64::consts::PI;
use crate::vm::value::Value;
use crate::vm::{RuntimeResult, RuntimeError};

pub trait DistributionTrait {
    fn sample(&self, rng: &mut dyn rand::RngCore) -> RuntimeResult<f64>;
    
    fn sample_ad(&self, rng: &mut dyn rand::RngCore) -> RuntimeResult<ADFloat> {
        Ok(ADFloat::Concrete(self.sample(rng)?))
    }
    
    fn log_pdf(&self, x: &ADFloat) -> ADFloat;
}

#[derive(Debug, Clone)]
pub struct Gaussian {
    pub mean: ADFloat,
    pub std: ADFloat,
}

impl DistributionTrait for Gaussian {
    fn sample(&self, rng: &mut dyn rand::RngCore) -> RuntimeResult<f64> {
        use rand_distr::{Normal, Distribution};
        let m = self.mean.value();
        let s = self.std.value();
        let n = Normal::new(m, s).map_err(|e| RuntimeError::DistributionError { 
            message: format!("Invalid Gaussian parameters: mean={}, std={}: {}", m, s, e) 
        })?;
        Ok(n.sample(rng))
    }
    
    fn sample_ad(&self, rng: &mut dyn rand::RngCore) -> RuntimeResult<ADFloat> {
        use rand_distr::StandardNormal;
        if self.std.value() <= 0.0 {
            return Err(RuntimeError::DistributionError { 
                message: format!("Standard deviation must be positive, got {}", self.std.value()) 
            });
        }
        let eps: f64 = rng.sample(StandardNormal);
        Ok(self.mean.clone() + self.std.clone() * ADFloat::Concrete(eps))
    }

    fn log_pdf(&self, x: &ADFloat) -> ADFloat {
        let two_pi = 2.0 * PI;
        let term1 = ADFloat::Concrete(-0.5 * two_pi.ln());
        let term2 = self.std.clone().ln(); 
        let diff = x.clone() - self.mean.clone();
        let z = diff / self.std.clone();
        let term3 = ADFloat::Concrete(0.5) * z.clone() * z;
         
        term1 - term2 - term3
    }
}

impl std::ops::Add for Gaussian {
    type Output = Gaussian;
    fn add(self, other: Gaussian) -> Gaussian {
        let new_mean = self.mean + other.mean;
        let var1 = self.std.clone() * self.std;
        let var2 = other.std.clone() * other.std;
        let new_std = (var1 + var2).sqrt();
        Gaussian { mean: new_mean, std: new_std }
    }
}

impl std::ops::Sub for Gaussian {
    type Output = Gaussian;
    fn sub(self, other: Gaussian) -> Gaussian {
        let new_mean = self.mean - other.mean;
        let var1 = self.std.clone() * self.std;
        let var2 = other.std.clone() * other.std;
        let new_std = (var1 + var2).sqrt();
        Gaussian { mean: new_mean, std: new_std }
    }
}

impl std::ops::Mul<ADFloat> for Gaussian {
    type Output = Gaussian;
    fn mul(self, scalar: ADFloat) -> Gaussian {
        Gaussian { 
            mean: self.mean * scalar.clone(), 
            std: self.std * scalar.abs() 
        }
    }
}

impl std::ops::Mul<f64> for Gaussian {
    type Output = Gaussian;
    fn mul(self, scalar: f64) -> Gaussian {
        self * ADFloat::Concrete(scalar)
    }
}

#[derive(Debug, Clone)]
pub struct Uniform {
    pub min: ADFloat,
    pub max: ADFloat,
}

impl DistributionTrait for Uniform {
    fn sample(&self, rng: &mut dyn rand::RngCore) -> RuntimeResult<f64> {
        use rand::Rng;
        if self.min.value() >= self.max.value() {
            return Err(RuntimeError::DistributionError { 
                message: format!("Invalid Uniform parameters: min={}, max={}", self.min.value(), self.max.value()) 
            });
        }
        Ok(rng.gen_range(self.min.value()..self.max.value()))
    }
    
    fn sample_ad(&self, rng: &mut dyn rand::RngCore) -> RuntimeResult<ADFloat> {
        use rand::Rng;
        if self.min.value() >= self.max.value() {
            return Err(RuntimeError::DistributionError { 
                message: format!("Invalid Uniform parameters: min={}, max={}", self.min.value(), self.max.value()) 
            });
        }
        let eps: f64 = rng.gen_range(0.0..1.0);
        Ok(self.min.clone() + (self.max.clone() - self.min.clone()) * ADFloat::Concrete(eps))
    }

    fn log_pdf(&self, x: &ADFloat) -> ADFloat {
        let val = x.value();
        if val >= self.min.value() && val <= self.max.value() {
            let diff = self.max.clone() - self.min.clone();
            ADFloat::Concrete(-1.0) * diff.ln()
        } else {
            ADFloat::Concrete(f64::NEG_INFINITY)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Bernoulli {
    pub p: ADFloat,
}

impl DistributionTrait for Bernoulli {
    fn sample(&self, rng: &mut dyn rand::RngCore) -> RuntimeResult<f64> {
        use rand_distr::{Bernoulli, Distribution};
        let p_val = self.p.value();
        let b = Bernoulli::new(p_val).map_err(|e| RuntimeError::DistributionError {
            message: format!("Invalid Bernoulli parameter p={}: {}", p_val, e)
        })?;
        if b.sample(rng) { Ok(1.0) } else { Ok(0.0) }
    }
    
    fn log_pdf(&self, x: &ADFloat) -> ADFloat {
        let p = self.p.clone();
        let one = ADFloat::Concrete(1.0);
        
        x.clone() * p.clone().ln() + (one.clone() - x.clone()) * (one - p).ln()
    }
}

#[derive(Debug, Clone)]
pub struct Beta {
    pub alpha: ADFloat,
    pub beta: ADFloat,
}

impl DistributionTrait for Beta {
    fn sample(&self, rng: &mut dyn rand::RngCore) -> RuntimeResult<f64> {
        use rand_distr::{Beta, Distribution};
        let b = Beta::new(self.alpha.value(), self.beta.value()).map_err(|e| RuntimeError::DistributionError {
            message: format!("Invalid Beta parameters alpha={}, beta={}: {}", self.alpha.value(), self.beta.value(), e)
        })?;
        Ok(b.sample(rng))
    }
    
    fn sample_ad(&self, rng: &mut dyn rand::RngCore) -> RuntimeResult<ADFloat> {
        use rand_distr::{Beta, Distribution};
        let a_val = self.alpha.value();
        let b_val = self.beta.value();
        
        let b_dist = Beta::new(a_val, b_val).map_err(|e| RuntimeError::DistributionError {
             message: format!("Invalid Beta parameters for AD alpha={}, beta={}: {}", a_val, b_val, e)
        })?; 
        let z = b_dist.sample(rng);
        
        Ok(self.alpha.clone().beta_sample(self.beta.clone(), z))
    }
    
    fn log_pdf(&self, x: &ADFloat) -> ADFloat {
        let a = self.alpha.clone();
        let b = self.beta.clone();
        
        let term1 = (a.clone() + b.clone()).lgamma() - a.lgamma() - b.lgamma();
        
        let one = ADFloat::Concrete(1.0);
        let a_minus_1 = self.alpha.clone() - one.clone();
        let b_minus_1 = self.beta.clone() - one.clone();
        
        term1 + a_minus_1 * x.clone().ln() + b_minus_1 * (one - x.clone()).ln()
    }
}
pub fn get_distribution(val: &Value) -> Option<Box<dyn DistributionTrait>> {
    match val {
        Value::Gaussian { mean, std } => Some(Box::new(Gaussian { mean: mean.clone(), std: std.clone() })),
        Value::Uniform { min, max } => Some(Box::new(Uniform { min: min.clone(), max: max.clone() })),
        Value::Bernoulli { p } => Some(Box::new(Bernoulli { p: p.clone() })),
        Value::Beta { alpha, beta } => Some(Box::new(Beta { alpha: alpha.clone(), beta: beta.clone() })),
        _ => None,
    }
}
