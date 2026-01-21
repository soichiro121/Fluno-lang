use std::ptr::NonNull;
use std::alloc::{alloc, dealloc, Layout};
use std::ops::Deref;
use std::fmt;
use std::cmp::Ordering;
use std::mem::ManuallyDrop;
use crate::gc::weak::Weak;

pub(crate) struct RcBox<T> {
    pub(crate) strong_count: usize,
    pub(crate) weak_count: usize,
    pub(crate) value: ManuallyDrop<T>,
}

pub struct Rc<T> {
    pub(crate) ptr: NonNull<RcBox<T>>,
}

impl<T> Rc<T> {
    pub fn new(value: T) -> Self {
        unsafe {
            let layout = Layout::new::<RcBox<T>>();
            let ptr = alloc(layout) as *mut RcBox<T>;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            
            std::ptr::write(ptr, RcBox {
                strong_count: 1,
                weak_count: 0,
                value: ManuallyDrop::new(value),
            });

            Rc {
                ptr: NonNull::new_unchecked(ptr),
            }
        }
    }

    pub fn downgrade(this: &Self) -> Weak<T> {
        unsafe {
            let mut ptr = this.ptr;
            ptr.as_mut().weak_count += 1;
            Weak { ptr: this.ptr }
        }
    }

    pub fn strong_count(this: &Self) -> usize {
        unsafe { this.ptr.as_ref().strong_count }
    }

    pub fn weak_count(this: &Self) -> usize {
        unsafe { this.ptr.as_ref().weak_count }
    }
}

impl<T> Clone for Rc<T> {
    fn clone(&self) -> Self {
        unsafe {
            let mut ptr = self.ptr;
            ptr.as_mut().strong_count += 1;
            Rc { ptr: self.ptr }
        }
    }
}

impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        unsafe {
            let mut ptr = self.ptr;
            let box_ptr = ptr.as_mut();
            
            box_ptr.strong_count -= 1;

            if box_ptr.strong_count == 0 {
                // 強参照がなくなったら、まずデータを破棄する
                ManuallyDrop::drop(&mut box_ptr.value);

                // 弱参照もなければ、管理領域(RcBox)自体を解放する
                if box_ptr.weak_count == 0 {
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
        if self.ptr == other.ptr { return true; }
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
