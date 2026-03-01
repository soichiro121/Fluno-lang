#[cfg(test)]
mod tests {
    use crate::gc::{Rc, Weak};
    use std::cell::RefCell;

    struct Node {
        value: i32,
        next: Option<Rc<RefCell<Node>>>,
        prev: Option<Weak<RefCell<Node>>>,
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

        assert!(weak.upgrade().is_some());
        assert_eq!(*weak.upgrade().unwrap(), 100);

        assert_eq!(Rc::strong_count(&strong), 1);
        assert_eq!(Rc::weak_count(&strong), 1);

        drop(strong);

        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_cycle_prevention() {
        let node1 = Node::new(1);
        let node2 = Node::new(2);

        node1.borrow_mut().next = Some(node2.clone());

        node2.borrow_mut().prev = Some(Rc::downgrade(&node1));

        assert_eq!(Rc::strong_count(&node1), 1);
        assert_eq!(Rc::strong_count(&node2), 2);

        assert_eq!(Rc::weak_count(&node1), 1);
    }

    #[test]
    fn test_memory_leak_check() {
        let mut weak_observer: Option<Weak<RefCell<Node>>> = None;

        {
            let node1 = Node::new(10);
            let node2 = Node::new(20);

            node1.borrow_mut().next = Some(node2.clone());
            node2.borrow_mut().prev = Some(Rc::downgrade(&node1));

            weak_observer = Some(Rc::downgrade(&node1));

            assert!(weak_observer.as_ref().unwrap().upgrade().is_some());
        }

        assert!(
            weak_observer.unwrap().upgrade().is_none(),
            "Memory leaked! Node1 is still alive."
        );
    }
}
