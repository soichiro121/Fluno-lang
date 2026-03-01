#[cfg(test)]
mod tests {
    use crate::gc::Rc;

    #[test]
    fn test_rc_basic() {
        let v = Rc::new(10);
        assert_eq!(*v, 10);
        assert_eq!(Rc::strong_count(&v), 1);
    }

    #[test]
    fn test_rc_clone() {
        let v1 = Rc::new(10);
        let v2 = v1.clone();

        assert_eq!(*v1, 10);
        assert_eq!(*v2, 10);
        assert_eq!(Rc::strong_count(&v1), 2);
        assert_eq!(Rc::strong_count(&v2), 2);
    }

    #[test]
    fn test_rc_drop() {
        let v1 = Rc::new(10);
        {
            let _v2 = v1.clone();
            assert_eq!(Rc::strong_count(&v1), 2);
        }

        assert_eq!(Rc::strong_count(&v1), 1);
    }

    #[test]
    fn test_rc_string() {
        let s1 = Rc::new("hello".to_string());
        let s2 = s1.clone();
        assert_eq!(*s1, "hello");
        assert_eq!(Rc::strong_count(&s1), 2);
    }

    #[test]
    fn test_rc_struct_display() {
        let s = Rc::new("world".to_string());
        let msg = format!("Hello {}", s);
        assert_eq!(msg, "Hello world");
    }
}
