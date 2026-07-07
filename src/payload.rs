//! Per-vector metadata and the filters that query it.
//!
//! A real vector search rarely wants "the nearest vectors" in the abstract — it
//! wants "the nearest vectors *that are in English*", or "*published after
//! 2020*". So each vector can carry a [`Payload`] of typed fields, and a
//! [`Filter`] restricts a search to the vectors whose payload matches.

use std::collections::BTreeMap;

/// A typed metadata value attached to a vector.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A UTF-8 string.
    Str(String),
    /// A signed integer.
    Int(i64),
    /// A 64-bit float.
    Float(f64),
    /// A boolean.
    Bool(bool),
}

impl Value {
    /// Interpret numeric values (`Int`/`Float`) as `f64`, for range comparisons.
    /// Non-numeric values return `None`.
    fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }
}

/// An ordered set of `key → value` fields attached to a stored vector.
pub type Payload = BTreeMap<String, Value>;

/// A predicate over a [`Payload`], evaluated per candidate during search.
///
/// A vector with no payload matches only the structural combinators when they
/// bottom out favourably (e.g. `Not` of a non-matching leaf); any field
/// predicate against an absent payload is `false`.
#[derive(Debug, Clone)]
pub enum Filter {
    /// The field exists and equals this value (exact, type-sensitive).
    Eq(String, Value),
    /// The field exists, is numeric, and is strictly greater than this bound.
    Gt(String, f64),
    /// The field exists, is numeric, and is strictly less than this bound.
    Lt(String, f64),
    /// Every sub-filter matches (an empty list matches everything).
    And(Vec<Filter>),
    /// At least one sub-filter matches (an empty list matches nothing).
    Or(Vec<Filter>),
    /// The sub-filter does not match.
    Not(Box<Filter>),
}

impl Filter {
    /// Evaluate this filter against an optional payload.
    pub fn matches(&self, payload: Option<&Payload>) -> bool {
        match self {
            Filter::Eq(key, value) => payload.and_then(|p| p.get(key)).is_some_and(|v| v == value),
            Filter::Gt(key, bound) => payload
                .and_then(|p| p.get(key))
                .and_then(Value::as_f64)
                .is_some_and(|x| x > *bound),
            Filter::Lt(key, bound) => payload
                .and_then(|p| p.get(key))
                .and_then(Value::as_f64)
                .is_some_and(|x| x < *bound),
            Filter::And(subs) => subs.iter().all(|f| f.matches(payload)),
            Filter::Or(subs) => subs.iter().any(|f| f.matches(payload)),
            Filter::Not(sub) => !sub.matches(payload),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload() -> Payload {
        let mut p = Payload::new();
        p.insert("lang".into(), Value::Str("en".into()));
        p.insert("year".into(), Value::Int(2021));
        p.insert("public".into(), Value::Bool(true));
        p
    }

    #[test]
    fn eq_is_type_sensitive() {
        let p = payload();
        assert!(Filter::Eq("lang".into(), Value::Str("en".into())).matches(Some(&p)));
        assert!(!Filter::Eq("lang".into(), Value::Str("fr".into())).matches(Some(&p)));
        // Int field is not equal to a Float of the same magnitude.
        assert!(!Filter::Eq("year".into(), Value::Float(2021.0)).matches(Some(&p)));
    }

    #[test]
    fn numeric_range_matches() {
        let p = payload();
        assert!(Filter::Gt("year".into(), 2020.0).matches(Some(&p)));
        assert!(!Filter::Gt("year".into(), 2021.0).matches(Some(&p)));
        assert!(Filter::Lt("year".into(), 2022.0).matches(Some(&p)));
    }

    #[test]
    fn combinators_compose() {
        let p = payload();
        let f = Filter::And(vec![
            Filter::Eq("lang".into(), Value::Str("en".into())),
            Filter::Gt("year".into(), 2000.0),
        ]);
        assert!(f.matches(Some(&p)));

        let f = Filter::Or(vec![
            Filter::Eq("lang".into(), Value::Str("fr".into())),
            Filter::Eq("public".into(), Value::Bool(true)),
        ]);
        assert!(f.matches(Some(&p)));

        assert!(
            Filter::Not(Box::new(Filter::Eq("lang".into(), Value::Str("fr".into()))))
                .matches(Some(&p))
        );
    }

    #[test]
    fn absent_payload_fails_field_predicates() {
        let f = Filter::Eq("lang".into(), Value::Str("en".into()));
        assert!(!f.matches(None));
        // But `Not` of a non-match is still true.
        assert!(Filter::Not(Box::new(f)).matches(None));
    }
}
