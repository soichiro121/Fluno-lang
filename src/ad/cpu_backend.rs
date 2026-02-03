// src/ad/cpu_backend.rs

use ndarray::{ArrayD, IxDyn, Ix2};
use crate::ad::backend::{TensorBackend, TensorStorage};
use std::ops::Deref;

#[derive(Clone, Debug, PartialEq)]
pub struct NdarrayStorage(pub ArrayD<f64>);

impl Deref for NdarrayStorage {
    type Target = ArrayD<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl TensorStorage for NdarrayStorage {
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }
    
    fn get(&self, index: &[usize]) -> f64 {
        self.0[IxDyn(index)]
    }
    
    fn sum(&self) -> f64 {
        self.0.sum()
    }
    
    fn into_vec(self) -> Vec<f64> {
        self.0.into_raw_vec_and_offset().0
    }
}

impl From<ArrayD<f64>> for NdarrayStorage {
    fn from(arr: ArrayD<f64>) -> Self {
        NdarrayStorage(arr)
    }
}

impl From<NdarrayStorage> for ArrayD<f64> {
    fn from(s: NdarrayStorage) -> Self {
        s.0
    }
}

#[derive(Clone, Debug, Default)]
pub struct CpuBackend;

impl TensorBackend for CpuBackend {
    type Storage = NdarrayStorage;

    fn from_elem(shape: &[usize], value: f64) -> Self::Storage {
        NdarrayStorage(ArrayD::from_elem(IxDyn(shape), value))
    }
    
    fn from_vec(shape: &[usize], data: Vec<f64>) -> Self::Storage {
        NdarrayStorage(ArrayD::from_shape_vec(IxDyn(shape), data).expect("Shape mismatch"))
    }

    fn add(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        NdarrayStorage(&a.0 + &b.0)
    }

    fn sub(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        NdarrayStorage(&a.0 - &b.0)
    }

    fn mul(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        NdarrayStorage(&a.0 * &b.0)
    }

    fn div(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        NdarrayStorage(&a.0 / &b.0)
    }

    fn matmul(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        let l = a.0.view().into_dimensionality::<Ix2>().expect("matmul requires 2D");
        let r = b.0.view().into_dimensionality::<Ix2>().expect("matmul requires 2D");
        NdarrayStorage(l.dot(&r).into_dyn())
    }

    fn neg(a: &Self::Storage) -> Self::Storage {
        NdarrayStorage(-&a.0)
    }

    fn exp(a: &Self::Storage) -> Self::Storage {
        NdarrayStorage(a.0.mapv(f64::exp))
    }

    fn log(a: &Self::Storage) -> Self::Storage {
        NdarrayStorage(a.0.mapv(f64::ln))
    }

    fn sqrt(a: &Self::Storage) -> Self::Storage {
        NdarrayStorage(a.0.mapv(f64::sqrt))
    }

    fn sin(a: &Self::Storage) -> Self::Storage {
        NdarrayStorage(a.0.mapv(f64::sin))
    }

    fn cos(a: &Self::Storage) -> Self::Storage {
        NdarrayStorage(a.0.mapv(f64::cos))
    }

    fn broadcast_add(a: &Self::Storage, scalar: f64) -> Self::Storage {
        NdarrayStorage(&a.0 + scalar)
    }

    fn broadcast_mul(a: &Self::Storage, scalar: f64) -> Self::Storage {
        NdarrayStorage(&a.0 * scalar)
    }
}

unsafe impl Send for NdarrayStorage {}
unsafe impl Sync for NdarrayStorage {}
