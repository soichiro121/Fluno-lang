use crate::gc::weak::Weak;
use std::alloc::{alloc, dealloc, Layout};
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::ptr::NonNull;

const MAX_REFCOUNT: usize = usize::MAX / 2;

pub(crate) struct RcBox<T> {
    pub(crate) strong_count: Cell<usize>,
    pub(crate) weak_count: Cell<usize>,
    pub(crate) value: ManuallyDrop<T>,
}

impl<T> RcBox<T> {
    #[inline]
    pub(crate) fn inc_strong(&self) {
        let count = self.strong_count.get();
        if count > MAX_REFCOUNT {
            panic!("Rc strong count overflow");
        }
        self.strong_count.set(count + 1);
    }

    #[inline]
    pub(crate) fn dec_strong(&self) -> usize {
        let count = self.strong_count.get();
        debug_assert!(count > 0, "Rc strong count underflow");
        let new_count = count - 1;
        self.strong_count.set(new_count);
        new_count
    }

    #[inline]
    pub(crate) fn inc_weak(&self) {
        let count = self.weak_count.get();
        if count > MAX_REFCOUNT {
            panic!("Rc weak count overflow");
        }
        self.weak_count.set(count + 1);
    }

    #[inline]
    pub(crate) fn dec_weak(&self) -> usize {
        let count = self.weak_count.get();
        debug_assert!(count > 0, "Rc weak count underflow");
        let new_count = count - 1;
        self.weak_count.set(new_count);
        new_count
    }
}

pub struct Rc<T> {
    pub(crate) ptr: NonNull<RcBox<T>>,
}

impl<T> Rc<T> {
    pub fn new(value: T) -> Self {
        let layout = Layout::new::<RcBox<T>>();
        let ptr = unsafe { alloc(layout) as *mut RcBox<T> };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        unsafe {
            std::ptr::write(
                ptr,
                RcBox {
                    strong_count: Cell::new(1),
                    weak_count: Cell::new(0),
                    value: ManuallyDrop::new(value),
                },
            );

            Rc {
                ptr: NonNull::new_unchecked(ptr),
            }
        }
    }

    #[inline]
    fn inner(&self) -> &RcBox<T> {
        unsafe { self.ptr.as_ref() }
    }

    pub fn downgrade(this: &Self) -> Weak<T> {
        this.inner().inc_weak();
        Weak { ptr: this.ptr }
    }

    pub fn strong_count(this: &Self) -> usize {
        this.inner().strong_count.get()
    }

    pub fn weak_count(this: &Self) -> usize {
        this.inner().weak_count.get()
    }

    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr == other.ptr
    }
}

impl<T> Clone for Rc<T> {
    fn clone(&self) -> Self {
        self.inner().inc_strong();
        Rc { ptr: self.ptr }
    }
}

impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        let inner = self.inner();
        let new_strong = inner.dec_strong();

        if new_strong == 0 {
            let weak_count = inner.weak_count.get();

            unsafe {
                ManuallyDrop::drop(&mut self.ptr.as_mut().value);
            }

            if weak_count == 0 {
                unsafe {
                    let layout = Layout::new::<RcBox<T>>();
                    dealloc(self.ptr.as_ptr() as *mut u8, layout);
                }
            }
        }
    }
}

impl<T> Deref for Rc<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &self.ptr.as_ref().value }
    }
}

impl<T> AsRef<T> for Rc<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T: fmt::Display> fmt::Display for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: fmt::Debug> fmt::Debug for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: PartialEq> PartialEq for Rc<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.ptr == other.ptr {
            return true;
        }
        **self == **other
    }
}

impl<T: PartialOrd> PartialOrd for Rc<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: Eq> Eq for Rc<T> {}

impl<T: Ord> Ord for Rc<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}
