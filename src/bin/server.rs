//! velo HTTP server binary.
//!
//! A thin tokio `main` around [`velo::server::app`]. Configuration comes from
//! the environment:
//!
//! * `VELO_DIM`  — vector dimensionality (default `128`)
//! * `VELO_ADDR` — bind address (default `0.0.0.0:8080`)
//!
//! ```text
//! cargo run --features server --bin server
//! ```

use velo::{HnswIndex, Metric};

#[tokio::main]
async fn main() {
    let dim: usize = std::env::var("VELO_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let addr = std::env::var("VELO_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());

    let index = HnswIndex::new(dim, Metric::Cosine);
    let app = velo::server::app(index);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {addr}: {e}"));
    println!("velo server listening on http://{addr} (dim = {dim}, metric = cosine)");
    axum::serve(listener, app).await.expect("server error");
}
