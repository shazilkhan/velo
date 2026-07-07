//! An HTTP server exposing an in-memory [`HnswIndex`] over a small JSON API.
//!
//! This lives behind the `server` cargo feature so the core library stays
//! dependency-free — only turning the server on pulls in axum, tokio, and serde.
//! The router is built by [`app`], which the `server` binary wraps in a tokio
//! runtime and which the tests drive directly.
//!
//! # Endpoints
//!
//! | Method & path   | Body                              | Purpose                    |
//! | --------------- | --------------------------------- | -------------------------- |
//! | `GET  /health`  | —                                 | liveness probe             |
//! | `GET  /stats`   | —                                 | count / dim / metric       |
//! | `POST /vectors` | `{id, vector, payload?}`          | insert a vector            |
//! | `POST /search`  | `{vector, k?, filter?}`           | (filtered) nearest search  |
//! | `POST /save`    | `{path}`                          | snapshot index to disk     |
//! | `POST /load`    | `{path}`                          | replace index from disk    |

use std::sync::{Arc, RwLock};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::{Filter, HnswIndex, Payload, Value, VectorIndex};

type Db = Arc<RwLock<HnswIndex>>;
type ApiError = (StatusCode, String);

/// Build the router serving `index`, which it takes ownership of and shares
/// behind a read/write lock.
pub fn app(index: HnswIndex) -> Router {
    let db: Db = Arc::new(RwLock::new(index));
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/stats", get(stats))
        .route("/vectors", post(add_vector))
        .route("/search", post(search))
        .route("/save", post(save))
        .route("/load", post(load))
        .with_state(db)
}

#[derive(Serialize)]
struct StatsResponse {
    count: usize,
    dim: usize,
    metric: String,
}

async fn stats(State(db): State<Db>) -> Json<StatsResponse> {
    let index = db.read().expect("lock poisoned");
    Json(StatsResponse {
        count: index.len(),
        dim: index.dim(),
        metric: format!("{:?}", index.metric()),
    })
}

#[derive(Deserialize)]
struct AddRequest {
    id: u64,
    vector: Vec<f32>,
    #[serde(default)]
    payload: Option<serde_json::Map<String, serde_json::Value>>,
}

async fn add_vector(
    State(db): State<Db>,
    Json(req): Json<AddRequest>,
) -> Result<Json<StatsResponse>, ApiError> {
    let mut index = db.write().expect("lock poisoned");
    if req.vector.len() != index.dim() {
        return Err(bad_request(format!(
            "expected {}-dim vector, got {}",
            index.dim(),
            req.vector.len()
        )));
    }
    match req.payload {
        Some(map) => {
            let payload = json_to_payload(map)?;
            index.add_with_payload(req.id, &req.vector, payload);
        }
        None => index.add(req.id, &req.vector),
    }
    Ok(Json(StatsResponse {
        count: index.len(),
        dim: index.dim(),
        metric: format!("{:?}", index.metric()),
    }))
}

#[derive(Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filter: Option<FilterDto>,
}

fn default_k() -> usize {
    10
}

#[derive(Serialize)]
struct Hit {
    id: u64,
    distance: f32,
}

async fn search(
    State(db): State<Db>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Vec<Hit>>, ApiError> {
    let index = db.read().expect("lock poisoned");
    if req.vector.len() != index.dim() {
        return Err(bad_request(format!(
            "expected {}-dim vector, got {}",
            index.dim(),
            req.vector.len()
        )));
    }
    let hits = match req.filter {
        Some(dto) => {
            let filter = dto.into_filter()?;
            index.search_filtered(&req.vector, req.k, &filter)
        }
        None => index.search(&req.vector, req.k),
    };
    Ok(Json(
        hits.into_iter()
            .map(|r| Hit {
                id: r.id,
                distance: r.distance,
            })
            .collect(),
    ))
}

#[derive(Deserialize)]
struct PathRequest {
    path: String,
}

async fn save(State(db): State<Db>, Json(req): Json<PathRequest>) -> Result<StatusCode, ApiError> {
    db.read()
        .expect("lock poisoned")
        .save(&req.path)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn load(State(db): State<Db>, Json(req): Json<PathRequest>) -> Result<StatusCode, ApiError> {
    let loaded = HnswIndex::load(&req.path).map_err(|e| bad_request(e.to_string()))?;
    *db.write().expect("lock poisoned") = loaded;
    Ok(StatusCode::NO_CONTENT)
}

/// JSON shape of a [`Filter`], externally tagged: `{"eq": {"key": "...",
/// "value": ...}}`, `{"gt": {"key": "...", "value": 3}}`, `{"and": [ ... ]}`,
/// `{"not": { ... }}`, and so on.
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum FilterDto {
    Eq {
        key: String,
        value: serde_json::Value,
    },
    Gt {
        key: String,
        value: f64,
    },
    Lt {
        key: String,
        value: f64,
    },
    And(Vec<FilterDto>),
    Or(Vec<FilterDto>),
    Not(Box<FilterDto>),
}

impl FilterDto {
    fn into_filter(self) -> Result<Filter, ApiError> {
        Ok(match self {
            FilterDto::Eq { key, value } => Filter::Eq(key, json_to_value(value)?),
            FilterDto::Gt { key, value } => Filter::Gt(key, value),
            FilterDto::Lt { key, value } => Filter::Lt(key, value),
            FilterDto::And(subs) => Filter::And(
                subs.into_iter()
                    .map(FilterDto::into_filter)
                    .collect::<Result<_, _>>()?,
            ),
            FilterDto::Or(subs) => Filter::Or(
                subs.into_iter()
                    .map(FilterDto::into_filter)
                    .collect::<Result<_, _>>()?,
            ),
            FilterDto::Not(sub) => Filter::Not(Box::new(sub.into_filter()?)),
        })
    }
}

fn json_to_payload(map: serde_json::Map<String, serde_json::Value>) -> Result<Payload, ApiError> {
    let mut payload = Payload::new();
    for (key, value) in map {
        payload.insert(key, json_to_value(value)?);
    }
    Ok(payload)
}

fn json_to_value(value: serde_json::Value) -> Result<Value, ApiError> {
    match value {
        serde_json::Value::String(s) => Ok(Value::Str(s)),
        serde_json::Value::Bool(b) => Ok(Value::Bool(b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::Int(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::Float(f))
            } else {
                Err(bad_request(format!("unsupported number {n}")))
            }
        }
        other => Err(bad_request(format!(
            "payload values must be string, number, or bool; got {other}"
        ))),
    }
}

fn bad_request(message: String) -> ApiError {
    (StatusCode::BAD_REQUEST, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_numbers_map_to_int_or_float() {
        assert_eq!(json_to_value(serde_json::json!(5)).unwrap(), Value::Int(5));
        assert_eq!(
            json_to_value(serde_json::json!(2.5)).unwrap(),
            Value::Float(2.5)
        );
        assert_eq!(
            json_to_value(serde_json::json!("hi")).unwrap(),
            Value::Str("hi".into())
        );
        assert_eq!(
            json_to_value(serde_json::json!(true)).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn filter_dto_parses_and_converts() {
        let dto: FilterDto = serde_json::from_value(serde_json::json!({
            "and": [
                {"eq": {"key": "lang", "value": "en"}},
                {"gt": {"key": "year", "value": 2020}}
            ]
        }))
        .unwrap();
        let filter = dto.into_filter().unwrap();
        match filter {
            Filter::And(subs) => assert_eq!(subs.len(), 2),
            other => panic!("expected And, got {other:?}"),
        }
    }
}
