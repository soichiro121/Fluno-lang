use std::ptr::NonNull;
use std::alloc::{dealloc, Layout};
use crate::gc::rc::{Rc, RcBox};

#[derive(Debug, PartialEq)]
pub struct Weak<T> {
    pub(crate) ptr: NonNull<RcBox<T>>,
}


impl<T> Weak<T> {
    #[inline]
    fn inner(&self) -> &RcBox<T> {
        unsafe { self.ptr.as_ref() }
    }

    pub fn upgrade(&self) -> Option<Rc<T>> {
        let inner = self.inner();
        if inner.strong_count.get() > 0 {
            inner.inc_strong();
            Some(Rc { ptr: self.ptr })
        } else {
            None
        }
    }

    pub fn strong_count(&self) -> usize {
        self.inner().strong_count.get()
    }

    pub fn weak_count(&self) -> usize {
        self.inner().weak_count.get()
    }
}

impl<T> Clone for Weak<T> {
    fn clone(&self) -> Self {
        self.inner().inc_weak();
        Weak { ptr: self.ptr }
    }
}

impl<T> Drop for Weak<T> {
    fn drop(&mut self) {
        let inner = self.inner();
        
        if inner.dec_weak() == 0 && inner.strong_count.get() == 0 {
            unsafe {
                let layout = Layout::new::<RcBox<T>>();
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

