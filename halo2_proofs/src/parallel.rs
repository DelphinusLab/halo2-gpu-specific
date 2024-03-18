use std::{
    fmt::Debug,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

#[derive(Debug)]
pub struct Parallel<T: Debug>(Arc<Mutex<T>>);

// derive failed
impl<T: Debug> Clone for Parallel<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

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
