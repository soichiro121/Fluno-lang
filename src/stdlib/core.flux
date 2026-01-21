// ============================
// stdlib/core.flux
// ============================

enum Result<T, E> {
    Ok(T),
    Err(E),
}

impl<T, E> Result<T, E> {
    fn is_ok(&self) -> Bool {
        match self {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    fn is_err(&self) -> Bool {
        match self {
            Ok(_) => false,
            Err(_) => true,
        }
    }

    fn unwrap(self) -> T {
        match self {
            Ok(value) => value,
            Err(error) => panic("called unwrap on Err: " + error),
        }
    }

    fn unwrap_or(self, default: T) -> T {
        match self {
            Ok(value) => value,
            Err(_) => default,
        }
    }

    fn map<U>(self, f: Fn(T) -> U) -> Result<U, E> {
        match self {
            Ok(value) => Ok(f(value)),
            Err(e) => Err(e),
        }
    }

    fn map_err<F>(self, f: Fn(E) -> F) -> Result<T, F> {
        match self {
            Ok(value) => Ok(value),
            Err(error) => Err(f(error)),
        }
    }
}
