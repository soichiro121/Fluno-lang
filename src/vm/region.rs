use crate::vm::value::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;

pub type ContainerHandle = usize;

pub enum ContainerData {
    Array(Vec<Value>),
    Dict(HashMap<String, Value>),
}

const GC_INITIAL_THRESHOLD: usize = 256;
const GC_GROWTH_FACTOR: usize = 2;

pub struct ContainerStore {
    containers: HashMap<ContainerHandle, RefCell<ContainerData>>,
    next_handle: ContainerHandle,
    alloc_count_since_gc: usize,
    gc_threshold: usize,
}

impl ContainerStore {
    pub fn new() -> Self {
        Self {
            containers: HashMap::new(),
            next_handle: 1,
            alloc_count_since_gc: 0,
            gc_threshold: GC_INITIAL_THRESHOLD,
        }
    }

    pub fn alloc_array(&mut self, data: Vec<Value>) -> ContainerHandle {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.containers
            .insert(handle, RefCell::new(ContainerData::Array(data)));
        self.alloc_count_since_gc += 1;
        handle
    }

    pub fn alloc_dict(&mut self, data: HashMap<String, Value>) -> ContainerHandle {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.containers
            .insert(handle, RefCell::new(ContainerData::Dict(data)));
        self.alloc_count_since_gc += 1;
        handle
    }

    pub fn get(&self, handle: ContainerHandle) -> Option<&RefCell<ContainerData>> {
        self.containers.get(&handle)
    }

    pub fn remove(&mut self, handle: ContainerHandle) {
        self.containers.remove(&handle);
    }

    pub fn contains(&self, handle: ContainerHandle) -> bool {
        self.containers.contains_key(&handle)
    }

    pub fn should_gc(&self) -> bool {
        self.alloc_count_since_gc >= self.gc_threshold
    }

    /// Mark-and-Sweep: 到達可能なハンドル集合を受け取り、それ以外を解放する
    pub fn sweep(&mut self, reachable: &HashSet<ContainerHandle>) {
        self.containers
            .retain(|handle, _| reachable.contains(handle));
        self.gc_threshold = (self.containers.len() * GC_GROWTH_FACTOR).max(GC_INITIAL_THRESHOLD);
        self.alloc_count_since_gc = 0;
    }

    /// コンテナの中身から子ハンドルを収集し、再帰的にマークを広げる
    pub fn trace_from(&self, roots: &[ContainerHandle]) -> HashSet<ContainerHandle> {
        let mut reachable = HashSet::new();
        let mut worklist: Vec<ContainerHandle> = roots.to_vec();

        while let Some(handle) = worklist.pop() {
            if !reachable.insert(handle) {
                continue; // 既にマーク済み
            }
            if let Some(cell) = self.containers.get(&handle) {
                let borrow = cell.borrow();
                match &*borrow {
                    ContainerData::Array(arr) => {
                        for val in arr {
                            collect_handles_from_value(val, &mut worklist);
                        }
                    }
                    ContainerData::Dict(dict) => {
                        for val in dict.values() {
                            collect_handles_from_value(val, &mut worklist);
                        }
                    }
                }
            }
        }
        reachable
    }

    pub fn handle_count(&self) -> usize {
        self.containers.len()
    }
}

// --- Helper methods for safe access ---

impl ContainerStore {
    pub fn with_array<F, R>(&self, handle: ContainerHandle, f: F) -> Option<R>
    where
        F: FnOnce(&Vec<Value>) -> R,
    {
        self.containers.get(&handle).and_then(|cell| {
            let borrow = cell.borrow();
            match &*borrow {
                ContainerData::Array(arr) => Some(f(arr)),
                _ => None,
            }
        })
    }

    pub fn with_array_mut<F, R>(&self, handle: ContainerHandle, f: F) -> Option<R>
    where
        F: FnOnce(&mut Vec<Value>) -> R,
    {
        self.containers.get(&handle).and_then(|cell| {
            let mut borrow = cell.borrow_mut();
            match &mut *borrow {
                ContainerData::Array(arr) => Some(f(arr)),
                _ => None,
            }
        })
    }

    pub fn with_dict<F, R>(&self, handle: ContainerHandle, f: F) -> Option<R>
    where
        F: FnOnce(&HashMap<String, Value>) -> R,
    {
        self.containers.get(&handle).and_then(|cell| {
            let borrow = cell.borrow();
            match &*borrow {
                ContainerData::Dict(dict) => Some(f(dict)),
                _ => None,
            }
        })
    }

    pub fn with_dict_mut<F, R>(&self, handle: ContainerHandle, f: F) -> Option<R>
    where
        F: FnOnce(&mut HashMap<String, Value>) -> R,
    {
        self.containers.get(&handle).and_then(|cell| {
            let mut borrow = cell.borrow_mut();
            match &mut *borrow {
                ContainerData::Dict(dict) => Some(f(dict)),
                _ => None,
            }
        })
    }

    pub fn clone_array(&self, handle: ContainerHandle) -> Option<Vec<Value>> {
        self.with_array(handle, |arr| arr.clone())
    }

    pub fn clone_dict(&self, handle: ContainerHandle) -> Option<HashMap<String, Value>> {
        self.with_dict(handle, |dict| dict.clone())
    }
}

/// Value から ContainerHandle を再帰的に抽出するユーティリティ
fn collect_handles_from_value(val: &Value, out: &mut Vec<ContainerHandle>) {
    match val {
        Value::Array(h) | Value::Map(h) => {
            out.push(*h);
        }
        Value::Struct { fields: h, .. } => {
            out.push(*h);
        }
        Value::Tuple(elems) => {
            for v in elems.iter() {
                collect_handles_from_value(v, out);
            }
        }
        Value::Enum { fields, .. } => {
            for v in fields.iter() {
                collect_handles_from_value(v, out);
            }
        }
        Value::Some(b) => {
            collect_handles_from_value(b, out);
        }
        _ => {}
    }
}

/// VM のルートから直接的なハンドルを抽出する公開ユーティリティ
pub fn extract_handles(val: &Value) -> Vec<ContainerHandle> {
    let mut handles = Vec::new();
    collect_handles_from_value(val, &mut handles);
    handles
}
