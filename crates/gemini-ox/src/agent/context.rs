use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub struct RunContext<T: Clone + Send + Sync + 'static> {
    pub state: T,
}

impl<T> RunContext<T>
where
    T: Clone + Send + Sync + 'static,
{
    pub fn new(state: T) -> RunContext<T> {
        RunContext { state }
    }
}

impl<T: Clone + Send + Sync + 'static> From<T> for RunContext<T> {
    fn from(state: T) -> Self {
        RunContext::new(state)
    }
}

// impl RunContext {
//     pub fn new() -> Self {
//         RunContext::default()
//     }
//     pub fn insert<T: Any + Send + Sync>(&self, value: T) -> Option<T> {
//         let key = TypeId::of::<T>();
//         self.deps
//             .lock()
//             .insert(key, Box::new(value))
//             .and_then(|old_boxed_value| old_boxed_value.downcast::<T>().ok())
//             .map(|boxed_t| *boxed_t)
//     }
//     pub fn get<T: Any + Send + Sync>(&self) -> Option<MappedMutexGuard<'_, T>> {
//         let key = TypeId::of::<T>();
//         let guard = self.deps.lock();
//         MutexGuard::try_map(guard, |map| {
//             // Get a mutable reference to the Box<dyn Any> if the key exists.
//             map.get_mut(&key)
//                 // Then, try to downcast the mutable reference inside the Box to &mut T.
//                 .and_then(|boxed_value| boxed_value.downcast_mut::<T>())
//         })
//         .ok()
//     }
// }
