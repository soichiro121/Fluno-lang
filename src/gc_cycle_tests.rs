#[cfg(test)]
mod tests {
    use crate::gc::{Rc, Weak};
    use std::cell::RefCell;

    // 循環参照テスト用のノード構造体
    struct Node {
        value: i32,
        next: Option<Rc<RefCell<Node>>>,
        prev: Option<Weak<RefCell<Node>>>, // 循環を防ぐための弱参照
    }

    impl Node {
        fn new(value: i32) -> Rc<RefCell<Self>> {
            Rc::new(RefCell::new(Node {
                value,
                next: None,
                prev: None,
            }))
        }
    }

    #[test]
    fn test_weak_upgrade() {
        let strong = Rc::new(100);
        let weak = Rc::downgrade(&strong);

        // strongが存在する間はupgradeできる
        assert!(weak.upgrade().is_some());
        assert_eq!(*weak.upgrade().unwrap(), 100);
        
        assert_eq!(Rc::strong_count(&strong), 1);
        assert_eq!(Rc::weak_count(&strong), 1);

        drop(strong);

        // strongが消えたらupgradeできない
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_cycle_prevention() {
        // ノード作成
        let node1 = Node::new(1);
        let node2 = Node::new(2);

        // 相互参照を作成
        // node1.next -> node2 (Strong)
        node1.borrow_mut().next = Some(node2.clone());
        
        // node2.prev -> node1 (Weak)
        // ここでStrongを使うと drop 後も count=1 でリークするが、Weakなら安全
        node2.borrow_mut().prev = Some(Rc::downgrade(&node1));

        // 参照カウント確認
        // node1: strong=1 (変数node1のみ), nextからは参照されていない
        // node2: strong=2 (変数node2 + node1.next)
        assert_eq!(Rc::strong_count(&node1), 1);
        assert_eq!(Rc::strong_count(&node2), 2);
        
        // node1 の weak count は node2.prev からの1つ
        assert_eq!(Rc::weak_count(&node1), 1);

        // スコープを抜けるとドロップされる
        // node1 drop -> count=0 -> node1破棄 -> node1.next破棄 -> node2のcount--
        // node2 drop -> count=1 -> 0 -> node2破棄
    }

    #[test]
    fn test_memory_leak_check() {
        // メモリリークしていないことを Rc::weak_count 等で間接的に確認するのは難しいが、
        // 循環参照を作って drop した後に weak ref が無効になっていることで確認する。

        let mut weak_observer: Option<Weak<RefCell<Node>>> = None;

        {
            let node1 = Node::new(10);
            let node2 = Node::new(20);

            // 循環: node1 -> node2 -> node1 (prevをWeakにすることで切断)
            node1.borrow_mut().next = Some(node2.clone());
            node2.borrow_mut().prev = Some(Rc::downgrade(&node1));

            // 外部から node1 を監視
            weak_observer = Some(Rc::downgrade(&node1));
            
            assert!(weak_observer.as_ref().unwrap().upgrade().is_some());
        } // ここで node1, node2 はスコープ外へ

        // もし node2.prev も Strong だったら、相互参照で count が残り、データは解放されない。
        // Weak を使っているので、ここで解放されているはず。
        assert!(weak_observer.unwrap().upgrade().is_none(), "Memory leaked! Node1 is still alive.");
    }
}
