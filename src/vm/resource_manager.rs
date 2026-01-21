use std::any::Any;
use std::collections::HashMap;

// resource_manager.rs
// Manages external resources (Rust objects) that are passed to Flux as opaque handles.

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

    // Register a new resource and return its handle ID.
    pub fn register<T: Any + 'static>(&mut self, resource: T) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.resources.insert(id, Box::new(resource));
        id
    }

    // Get a reference to a resource by its handle ID.
    pub fn get<T: Any + 'static>(&self, id: usize) -> Option<&T> {
        self.resources.get(&id)?.downcast_ref::<T>()
    }

    // Get a mutable reference to a resource by its handle ID.
    pub fn get_mut<T: Any + 'static>(&mut self, id: usize) -> Option<&mut T> {
        self.resources.get_mut(&id)?.downcast_mut::<T>()
    }

    // Remove a resource by its handle ID.
    pub fn remove(&mut self, id: usize) -> Option<Box<dyn Any>> {
        self.resources.remove(&id)
    }
}
