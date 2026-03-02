use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use rand::Rng;

// ========== Signal型 ===========
type Callback<T> = Arc<Mutex<dyn FnMut(&T) + Send>>;
static mut SIGNAL_ID_COUNTER: u64 = 1;

#[derive(Debug, Clone)]
pub struct Signal<T: Clone + Send + 'static> {
    value: Arc<Mutex<T>>,
    subscribers: Arc<Mutex<HashMap<u64, Callback<T>>>>,
    dependencies: Arc<Mutex<HashSet<u64>>>,
    id: u64,
    next_sub_id: Arc<Mutex<u64>>,
    history: Arc<Mutex<Vec<T>>>,           // 履歴保持
}

impl<T: Clone + Send + 'static> Signal<T> {
    pub fn new(initial: T) -> Self {
        let id = unsafe { let v = SIGNAL_ID_COUNTER; SIGNAL_ID_COUNTER += 1; v };
        Signal {
            value: Arc::new(Mutex::new(initial.clone())),
            subscribers: Arc::new(Mutex::new(HashMap::new())),
            dependencies: Arc::new(Mutex::new(HashSet::new())),
            next_sub_id: Arc::new(Mutex::new(1)),
            history: Arc::new(Mutex::new(vec![initial])),
            id,
        }
    }
    pub fn get(&self) -> T {
        self.value.lock().unwrap().clone()
    }
    pub fn set(&self, new_val: T) {
        {
            let mut h = self.history.lock().unwrap();
            h.push(new_val.clone());
        }
        let mut v = self.value.lock().unwrap();
        *v = new_val.clone();
        drop(v);
        for cb in self.subscribers.lock().unwrap().values_mut() {
            cb.lock().unwrap()(&new_val);
        }
    }
    pub fn subscribe<F>(&self, mut callback: F) -> u64 where F: FnMut(&T) + Send + 'static {
        let mut sub_id = self.next_sub_id.lock().unwrap();
        let id = *sub_id; *sub_id += 1; drop(sub_id);
        self.subscribers.lock().unwrap().insert(id, Arc::new(Mutex::new(callback)));
        id
    }
    // async購読例（実際はtrue async環境で拡張可、ここではthreadで模擬）
    pub fn subscribe_async<F>(&self, mut callback: F) -> u64 where F: FnMut(&T) + Send + 'static {
        let id = self.subscribe(move |v| {
            let v = v.clone();
            thread::spawn(move || {
                callback(&v); // 疑似非同期
            });
        });
        id
    }
    pub fn unsubscribe(&self, id: u64) {
        self.subscribers.lock().unwrap().remove(&id);
    }
    pub fn map<U, F>(&self, f: F) -> Signal<U> where U: Clone + Send + 'static, F: Fn(&T) -> U + Send + 'static {
        let s = Signal::new(f(&self.get()));
        s.dependencies.lock().unwrap().insert(self.id);
        let myself = self.clone();
        myself.subscribe(move |v| { s.set(f(v)); });
        s
    }
    // window: 過去n個の値を見る
    pub fn window(&self, size: usize) -> Signal<Vec<T>> {
        let s = Signal::new(vec![self.get()]);
        s.dependencies.lock().unwrap().insert(self.id);
        let myself = self.clone();
        myself.subscribe(move |v| {
            let mut hist = myself.history.lock().unwrap();
            let len = hist.len();
            let win = if len < size { hist.clone() }
                      else { hist[len-size..].to_vec() };
            s.set(win);
        });
        s
    }
    // scan: アキュムレータで値を畳み込む
    pub fn scan<U, F>(&self, initial: U, f: F) -> Signal<U> where U: Clone + Send + 'static, F: Fn(U, &T) -> U + Send + 'static {
        let s = Signal::new(f(initial.clone(), &self.get()));
        s.dependencies.lock().unwrap().insert(self.id);
        let mut state = Arc::new(Mutex::new(initial));
        let myself = self.clone();
        myself.subscribe(move |v| {
            let mut st = state.lock().unwrap();
            let new = f(st.clone(), v);
            *st = new.clone();
            s.set(new);
        });
        s
    }
    // flatten: Vec<Signal<T>>を1つSignal<T>に
    pub fn flatten(signals: Vec<Signal<T>>) -> Signal<T> {
        let first_val = signals[0].get();
        let s = Signal::new(first_val.clone());
        for sig in signals {
            sig.subscribe({
                let s_c = s.clone();
                move |v| { s_c.set(v.clone()); }
            });
            s.dependencies.lock().unwrap().insert(sig.id);
        }
        s
    }
    // combine_latest: 複数のSignal<T>の最新値で合成
    pub fn combine_latest<U, F>(signals: Vec<Signal<T>>, comb: F, initial: U) -> Signal<U>
    where U: Clone + Send + 'static, F: Fn(&[T]) -> U + Send + 'static {
        let s = Signal::new(initial.clone());
        let val_vec = Arc::new(Mutex::new(signals.iter().map(|sig| sig.get()).collect::<Vec<T>>()));
        for (i, sig) in signals.into_iter().enumerate() {
            let s_c = s.clone();
            let val_vec = Arc::clone(&val_vec);
            sig.subscribe(move |v| {
                let mut vals = val_vec.lock().unwrap();
                vals[i] = v.clone();
                s_c.set(comb(&vals));
            });
            s.dependencies.lock().unwrap().insert(sig.id);
        }
        s
    }
    // aggregate: 全履歴に対し集計関数
    pub fn aggregate<U, F>(&self, f: F) -> Signal<U> where U: Clone + Send + 'static, F: Fn(&[T]) -> U + Send + 'static {
        let s = Signal::new(f(&self.history.lock().unwrap()));
        let myself = self.clone();
        myself.subscribe(move |_| {
            let v = myself.history.lock().unwrap();
            s.set(f(&v));
        });
        s
    }
    // 履歴取得
    pub fn history(&self) -> Vec<T> {
        self.history.lock().unwrap().clone()
    }
    // 依存・循環
    pub fn dependencies(&self) -> HashSet<u64> { self.dependencies.lock().unwrap().clone() }
    pub fn detect_cycle(&self, dag: &HashMap<u64, HashSet<u64>>) -> Result<(), SignalError> {
        fn dfs(id: u64, dag: &HashMap<u64, HashSet<u64>>, seen: &mut HashSet<u64>, stack: &mut HashSet<u64>) -> bool {
            if stack.contains(&id) { return true; }
            if seen.contains(&id) { return false; }
            seen.insert(id); stack.insert(id);
            if let Some(deps) = dag.get(&id) { for &d in deps { if dfs(d, dag, seen, stack) { return true; } } }
            stack.remove(&id); false
        }
        let mut seen = HashSet::new(); let mut stack = HashSet::new();
        if dfs(self.id, dag, &mut seen, &mut stack) {
            Err(SignalError::CycleDetected(self.id))
        } else { Ok(()) }
    }
}

// ========== SignalError ===========
#[derive(Debug)]
pub enum SignalError {
    CycleDetected(u64),
    Other(String)
}

// ========== Event型 ===========
#[derive(Debug, Clone)]
pub struct Event<T: Clone + Send + 'static> {
    subscribers: Arc<Mutex<HashMap<u64, Callback<T>>>>,
    next_sub_id: Arc<Mutex<u64>>,
    history: Arc<Mutex<Vec<T>>>,
}

impl<T: Clone + Send + 'static> Event<T> {
    pub fn new() -> Self {
        Event {
            subscribers: Arc::new(Mutex::new(HashMap::new())),
            next_sub_id: Arc::new(Mutex::new(1)),
            history: Arc::new(Mutex::new(vec![])),
        }
    }
    pub fn emit(&self, value: T) {
        self.history.lock().unwrap().push(value.clone());
        for cb in self.subscribers.lock().unwrap().values_mut() {
            cb.lock().unwrap()(&value);
        }
    }
    pub fn subscribe<F>(&self, mut callback: F) -> u64 where F: FnMut(&T) + Send + 'static {
        let mut sub_id = self.next_sub_id.lock().unwrap();
        let id = *sub_id; *sub_id += 1; drop(sub_id);
        self.subscribers.lock().unwrap().insert(id, Arc::new(Mutex::new(callback)));
        id
    }
    pub fn subscribe_async<F>(&self, mut callback: F) -> u64 where F: FnMut(&T) + Send + 'static {
        let id = self.subscribe(move |v| {
            let v = v.clone();
            thread::spawn(move || { callback(&v); });
        });
        id
    }
    pub fn unsubscribe(&self, id: u64) {
        self.subscribers.lock().unwrap().remove(&id);
    }
    // window: 過去n件イベント
    pub fn window(&self, size: usize) -> Event<Vec<T>> {
        let ev = Event::new();
        let self_c = self.clone();
        self.subscribe(move |v| {
            let hist = self_c.history.lock().unwrap();
            let len = hist.len();
            let win = if len < size { hist.clone() } else { hist[len-size..].to_vec() };
            ev.emit(win);
        });
        ev
    }
    // aggregate: 全履歴に対し集計
    pub fn aggregate<U, F>(&self, f: F) -> Event<U> where U: Clone + Send + 'static, F: Fn(&[T]) -> U + Send + 'static {
        let ev = Event::new();
        let self_c = self.clone();
        self.subscribe(move |_| {
            let hist = self_c.history.lock().unwrap();
            ev.emit(f(&hist));
        });
        ev
    }
    // flatten: Vec<Event<T>>を1つEvent<T>に
    pub fn flatten(events: Vec<Event<T>>) -> Event<T> {
        let ev = Event::new();
        for e in events {
            let ev_c = ev.clone();
            e.subscribe(move |v| { ev_c.emit(v.clone()); });
        }
        ev
    }
    // combine_latest: 複数Eventの最新値Vecで発火
    pub fn combine_latest(events: Vec<Event<T>>) -> Event<Vec<T>> {
        let ev = Event::new();
        let val_vec = Arc::new(Mutex::new(events.iter().map(|e| e.history.lock().unwrap().last().cloned().unwrap_or_else(|| panic!())).collect::<Vec<T>>()));
        for (i, e) in events.into_iter().enumerate() {
            let ev_c = ev.clone();
            let val_vec = Arc::clone(&val_vec);
            e.subscribe(move |v| {
                let mut vals = val_vec.lock().unwrap();
                vals[i] = v.clone();
                ev_c.emit(vals.clone());
            });
        }
        ev
    }
    pub fn history(&self) -> Vec<T> {
        self.history.lock().unwrap().clone()
    }
}

// ========== テスト ===============
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_signal_scan() {
        let s = Signal::new(1);
        let sc = s.scan(0, |acc, x| acc + x);
        let mut vals = Vec::new();
        sc.subscribe(|v| vals.push(*v));
        s.set(2); s.set(3); s.set(4);
        assert_eq!(vals, vec![1,3,6,10]);
    }

    #[test]
    fn test_signal_window() {
        let s = Signal::new(10);
        let win = s.window(3);
        let mut vals = Vec::new();
        win.subscribe(|v| vals.push(v.clone()));
        s.set(11); s.set(12); s.set(13);
        assert_eq!(vals.last().unwrap(), &vec![11,12,13]);
    }

    #[test]
    fn test_signal_flatten() {
        let s1 = Signal::new(1);
        let s2 = Signal::new(2);
        let fl = Signal::flatten(vec![s1.clone(), s2.clone()]);
        let mut vals = Vec::new();
        fl.subscribe(|v| vals.push(*v));
        s2.set(7);
        assert_eq!(vals.contains(&7), true);
    }

    #[test]
    fn test_signal_combine_latest() {
        let s1 = Signal::new(1);
        let s2 = Signal::new(2);
        let s3 = Signal::combine_latest(vec![s1.clone(), s2.clone()], |v| v.iter().sum(), 3);
        let mut vals = Vec::new();
        s3.subscribe(|v| vals.push(*v));
        s1.set(3); s2.set(5);
        assert_eq!(vals.last(), Some(&8));
    }

    #[test]
    fn test_signal_aggregate_history() {
        let s = Signal::new(1);
        let agg = s.aggregate(|h| h.iter().sum::<i32>());
        let mut vals = Vec::new();
        agg.subscribe(|v| vals.push(*v));
        s.set(2); s.set(3);
        assert_eq!(vals.last().unwrap(), &6);
    }

    #[test]
    fn test_signal_async_subscribe() {
        let s = Signal::new(5);
        let flag = Arc::new(AtomicUsize::new(0));
        let flag2 = flag.clone();
        s.subscribe_async(move |v| {
            flag2.store(*v as usize, Ordering::SeqCst);
        });
        s.set(123);
        thread::sleep(Duration::from_millis(10));
        assert_eq!(flag.load(Ordering::SeqCst), 123);
    }

    #[test]
    fn test_signal_cycle_detection_and_error() {
        let a = Signal::new(1);
        let b = a.map(|x| x+1);
        let mut dag = HashMap::new();
        dag.insert(a.id, a.dependencies());
        dag.insert(b.id, b.dependencies());
        assert!(b.detect_cycle(&dag) == Ok(()));
        dag.get_mut(&a.id).unwrap().insert(b.id);
        assert!(matches!(a.detect_cycle(&dag), Err(SignalError::CycleDetected(_))));
    }

    #[test]
    fn test_event_window_aggregate_flatten() {
        let e = Event::new();
        let win = e.window(2);
        let agg = e.aggregate(|h| h.iter().sum::<i32>());
        let mut winvals = Vec::new();
        let mut aggvals = Vec::new();
        win.subscribe(|v| winvals.push(v.clone()));
        agg.subscribe(|v| aggvals.push(*v));
        e.emit(10); e.emit(20);
        assert_eq!(winvals.last().unwrap(), &vec![10,20]);
        assert_eq!(aggvals.last().unwrap(), &30);
        let e2 = Event::new();
        let f = Event::flatten(vec![e.clone(), e2.clone()]);
        let mut fvals = Vec::new();
        f.subscribe(|v| fvals.push(*v));
        e2.emit(99);
        assert_eq!(fvals.last().unwrap(), &99);
    }

    #[test]
    fn test_event_combine_latest() {
        let e1 = Event::new(); let e2 = Event::new();
        let cl = Event::combine_latest(vec![e1.clone(), e2.clone()]);
        let mut vals = Vec::new();
        cl.subscribe(|v| vals.push(v.clone()));
        e1.emit(100); e2.emit(200);
        assert_eq!(vals.last().unwrap(), &vec![100,200]);
    }

    #[test]
    fn test_event_async_subscribe() {
        let e = Event::new();
        let flag = Arc::new(AtomicUsize::new(0));
        let flag2 = flag.clone();
        e.subscribe_async(move |v| { flag2.store(*v as usize, Ordering::SeqCst); });
        e.emit(41);
        thread::sleep(Duration::from_millis(10));
        assert_eq!(flag.load(Ordering::SeqCst), 41);
    }
}

