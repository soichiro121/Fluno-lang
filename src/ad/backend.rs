// src/ad/backend.rs

use std::fmt::Debug;

pub trait TensorStorage: Clone + Debug + Send + Sync {
    fn shape(&self) -> &[usize];
    fn ndim(&self) -> usize { self.shape().len() }
    fn size(&self) -> usize { self.shape().iter().product() }
    fn get(&self, index: &[usize]) -> f64;
    fn sum(&self) -> f64;
    fn into_vec(self) -> Vec<f64>;
}

pub trait TensorBackend: Clone + Default + Debug {
    type Storage: TensorStorage;

    fn from_elem(shape: &[usize], value: f64) -> Self::Storage;
    fn zeros(shape: &[usize]) -> Self::Storage { Self::from_elem(shape, 0.0) }
    fn ones(shape: &[usize]) -> Self::Storage { Self::from_elem(shape, 1.0) }
    fn from_vec(shape: &[usize], data: Vec<f64>) -> Self::Storage;

    fn add(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn sub(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn mul(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn div(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn matmul(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    
    fn neg(a: &Self::Storage) -> Self::Storage;
    fn exp(a: &Self::Storage) -> Self::Storage;
    fn log(a: &Self::Storage) -> Self::Storage;
    fn sqrt(a: &Self::Storage) -> Self::Storage;
    fn sin(a: &Self::Storage) -> Self::Storage;
    fn cos(a: &Self::Storage) -> Self::Storage;
    
    fn broadcast_add(a: &Self::Storage, scalar: f64) -> Self::Storage;
    fn broadcast_mul(a: &Self::Storage, scalar: f64) -> Self::Storage;
}
