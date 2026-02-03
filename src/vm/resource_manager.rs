// src/vm/resource_manager.rs
use std::any::Any;
use std::collections::HashMap;

pub struct ResourceManager {
    resources: HashMap<usize, Box<dyn Any>>,
    next_id: usize,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn register<T: Any + 'static>(&mut self, resource: T) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.resources.insert(id, Box::new(resource));
        id
    }

    pub fn get<T: Any + 'static>(&self, id: usize) -> Option<&T> {
        self.resources.get(&id)?.downcast_ref::<T>()
    }
    pub fn get_mut<T: Any + 'static>(&mut self, id: usize) -> Option<&mut T> {
        self.resources.get_mut(&id)?.downcast_mut::<T>()
    }

    pub fn remove(&mut self, id: usize) -> Option<Box<dyn Any>> {
        self.resources.remove(&id)
    }
}
