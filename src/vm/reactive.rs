// src/vm/reactive.rs

use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;
use crate::vm::value::Value;


#[derive(Debug, Clone, Copy, PartialEq)] // Copy を追加
pub enum NodeKind {
    Signal,
    Event,
}


// 依存グラフのノード（1つのSignalに対応）
#[derive(Debug, Clone)]
pub struct SignalNode {
    pub id: usize,
    pub current_value: Value,
    pub kind: NodeKind, 
    // このSignalが依存している親SignalのID一覧
    pub dependencies: Vec<usize>,
    // このSignalに依存している子SignalのID一覧
    pub dependents: Vec<usize>,
    // トポロジカルランク（ルート=0。更新順序の決定に使用）
    pub rank: usize,
    // 値を再計算するための関数（ルートSignalの場合はNone）
    // Value::Function を保持する
    pub update_fn: Option<Value>,
}


impl SignalNode {
    // 修正: dependencies 引数を追加して、依存関係を保存する
    fn new(id: usize, value: Value, rank: usize, update_fn: Option<Value>, dependencies: Vec<usize>, kind: NodeKind) -> Self {
        Self {
            id,
            current_value: value,
            kind,
            dependencies,
            dependents: Vec::new(),
            rank,
            update_fn,
        }
    }
}

// 更新順序を管理するためのヘルパー構造体
#[derive(Copy, Clone, Eq, PartialEq)]
struct UpdateItem {
    id: usize,
    rank: usize,
}

// BinaryHeapは最大ヒープなので、ランクが低い（親に近い）順に取り出すために順序を逆にする
impl Ord for UpdateItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.rank.cmp(&self.rank)
    }
}

impl PartialOrd for UpdateItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// リアクティブシステムの状態管理
#[derive(Debug)]
pub struct ReactiveRuntime {
    nodes: HashMap<usize, SignalNode>,
    next_id: usize,
}

impl ReactiveRuntime {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    // ルートSignal（ソース）を作成
    pub fn create_root(&mut self, initial_value: Value, kind: NodeKind) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let node = SignalNode::new(id, initial_value, 0, None, Vec::new(), kind);
        self.nodes.insert(id, node);
        id
    }

    // 計算Signal（map/combine等）を作成
    pub fn create_computed(&mut self, dependencies: Vec<usize>, initial_value: Value, update_fn: Value, kind: NodeKind) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let mut max_rank = 0;
        for dep_id in &dependencies {
            if let Some(node) = self.nodes.get(dep_id) {
                if node.rank > max_rank {
                    max_rank = node.rank;
                }
            }
        }
        let rank = max_rank + 1;

        let deps_clone = dependencies.clone();
        let node = SignalNode::new(id, initial_value, rank, Some(update_fn), deps_clone, kind);
        self.nodes.insert(id, node);

        for dep_id in dependencies {
            if let Some(node) = self.nodes.get_mut(&dep_id) {
                node.dependents.push(id);
            }
        }
        id
    }
    
    
    pub fn get_node_kind(&self, id: usize) -> Option<NodeKind> {
        self.nodes.get(&id).map(|n| n.kind)
    }

    // Signalの値を取得
    pub fn get_value(&self, id: usize) -> Option<Value> {
        self.nodes.get(&id).map(|n| n.current_value.clone())
    }
    
    // Signalの値を更新（ルートSignal用）
    pub fn set_value(&mut self, id: usize, value: Value) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.current_value = value;
        }
    }

    // 指定したSignalが変更された場合に更新が必要なSignalのIDリストを
    // 正しい順序（トポロジカルソート順）で返す
    pub fn get_propagation_order(&self, start_id: usize) -> Vec<usize> {
        let mut queue = BinaryHeap::new();
        let mut visited = HashSet::new();
        let mut order = Vec::new();

        // 起点となるノードの子孫をキューに追加
        if let Some(node) = self.nodes.get(&start_id) {
            for &dep_id in &node.dependents {
                self.enqueue_if_needed(dep_id, &mut queue, &mut visited);
            }
        }

        while let Some(item) = queue.pop() {
            order.push(item.id);
            
            // このノードが更新されたら、その子孫も更新対象に追加
            if let Some(node) = self.nodes.get(&item.id) {
                for &child_id in &node.dependents {
                    self.enqueue_if_needed(child_id, &mut queue, &mut visited);
                }
            }
        }
        
        order
    }
    
    fn enqueue_if_needed(&self, id: usize, queue: &mut BinaryHeap<UpdateItem>, visited: &mut HashSet<usize>) {
        if !visited.contains(&id) {
            if let Some(node) = self.nodes.get(&id) {
                visited.insert(id);
                queue.push(UpdateItem { id, rank: node.rank });
            }
        }
    }
    
    pub fn get_node(&self, id: usize) -> Option<&SignalNode> {
        self.nodes.get(&id)
    }
}
