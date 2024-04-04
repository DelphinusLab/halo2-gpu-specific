use std::{
    fmt::Debug,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

#[derive(Debug)]
pub struct Parallel<T: Debug>(Arc<Mutex<T>>);

unsafe impl<T: Debug> Sync for Parallel<T> {}

impl<T: Debug> Parallel<T> {
    pub(crate) fn new(v: T) -> Self {
        Parallel(Arc::new(Mutex::new(v)))
    }

    pub(crate) fn into_inner(self) -> T {
        Arc::try_unwrap(self.0).unwrap().into_inner().unwrap()
    }

    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, T>, PoisonError<MutexGuard<'_, T>>> {
        self.0.lock()
    }
}
