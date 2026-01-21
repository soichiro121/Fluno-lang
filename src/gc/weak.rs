use std::ptr::NonNull;
use std::alloc::{dealloc, Layout};
use crate::gc::rc::{Rc, RcBox};

#[derive(Debug, PartialEq)]
pub struct Weak<T> {
    pub(crate) ptr: NonNull<RcBox<T>>,
}


impl<T> Weak<T> {
    // データが既に破棄されている(strong_count == 0)場合は None を返す
    pub fn upgrade(&self) -> Option<Rc<T>> {
        unsafe {
            let box_ptr = self.ptr.as_ref();
            if box_ptr.strong_count > 0 {
                let mut ptr = self.ptr;
                ptr.as_mut().strong_count += 1;
                Some(Rc { ptr: self.ptr })
            } else {
                None
            }
        }
    }

    pub fn strong_count(&self) -> usize {
        unsafe { self.ptr.as_ref().strong_count }
    }

    pub fn weak_count(&self) -> usize {
        unsafe { self.ptr.as_ref().weak_count }
    }
}

impl<T> Clone for Weak<T> {
    fn clone(&self) -> Self {
        unsafe {
            let mut ptr = self.ptr;
            ptr.as_mut().weak_count += 1;
            Weak { ptr: self.ptr }
        }
    }
}

impl<T> Drop for Weak<T> {
    fn drop(&mut self) {
        unsafe {
            let mut ptr = self.ptr;
            let box_ptr = ptr.as_mut();
            
            box_ptr.weak_count -= 1;

            // 強参照が既になく(データ破棄済み)、かつ弱参照もこれが最後なら
            // 管理領域(RcBox)を解放する
            if box_ptr.strong_count == 0 && box_ptr.weak_count == 0 {
                let layout = Layout::new::<RcBox<T>>();
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}
