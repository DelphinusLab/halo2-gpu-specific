use std::{
    fmt::Debug,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

#[derive(Clone, Debug)]
pub(crate) struct Parallel<T: Debug>(pub(crate) Arc<Mutex<T>>);

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
