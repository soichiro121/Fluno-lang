// benches/tensor_bench_v2.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fluno::ad::{create_tape, with_tape};
use fluno::ad::tensor::ADTensor;
use fluno::ad::optimizer::optimize_graph;
use ndarray::ArrayD;

fn bench_eager(c: &mut Criterion) {
    c.bench_function("tensor_eager", |b| {
        b.iter(|| {
            let tape_id = create_tape();
            let arr = ArrayD::from_elem(ndarray::IxDyn(&[100, 100]), 2.0);
            
            let a = ADTensor::new_input(arr.clone(), tape_id);
            let b_val = ADTensor::new_input(arr.clone(), tape_id);
            let temp = (a * b_val).eval(); 
            
            let res = ADTensor::new_input(temp, tape_id);
            let c_val = ADTensor::new_input(arr.clone(), tape_id);
            let final_res = (res + c_val).eval();
            
            black_box(final_res);
        })
    });
}

fn bench_lazy_opt(c: &mut Criterion) {
    c.bench_function("tensor_lazy_opt", |b| {
        b.iter(|| {
            let tape_id = create_tape();
            let arr = ArrayD::from_elem(ndarray::IxDyn(&[100, 100]), 2.0);
            let a = ADTensor::new_input(arr.clone(), tape_id);
            let b_val = ADTensor::new_input(arr.clone(), tape_id);
            let c_val = ADTensor::new_input(arr.clone(), tape_id);
            
            let d = black_box((a * b_val) + c_val);
            
            with_tape(tape_id, |tape| {
                optimize_graph(tape);
            });
            
            let _ = d.eval();
        })
    });
}

fn bench_cse(c: &mut Criterion) {
    c.bench_function("tensor_lazy_cse", |b| {
        b.iter(|| {
            let tape_id = create_tape();
            let arr = ArrayD::from_elem(ndarray::IxDyn(&[100, 100]), 2.0);
            
            let a = ADTensor::new_input(arr.clone(), tape_id);
            let b_val = ADTensor::new_input(arr.clone(), tape_id);
            
            let x = a.clone() * b_val.clone();
            let y = a * b_val;
            
            let z = black_box(x + y);
            
            with_tape(tape_id, |tape| {
                optimize_graph(tape);
            });
            
            let _ = z.eval();
        })
    });
}

criterion_group!(benches, bench_eager, bench_lazy_opt, bench_cse);
criterion_main!(benches);
