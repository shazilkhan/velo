//! End-to-end example: content-based search over text with velo.
//!
//! ```text
//! cargo run --example semantic_search
//! ```
//!
//! To keep the example dependency-free, the "embedding" here is a simple hashed
//! bag of character trigrams — enough to demonstrate content-based retrieval end
//! to end. In a real system you would replace [`embed`] with a proper embedding
//! model (OpenAI, a local transformer, ...) and *nothing else in this file would
//! change*: that is the point of a vector database.

use velo::{HnswIndex, Metric, VectorIndex};

const DIM: usize = 256;

fn main() {
    let corpus = [
        "Rust is a systems programming language focused on safety and speed",
        "Python is a high level language popular for data science and scripting",
        "The borrow checker enforces memory safety without a garbage collector",
        "Goroutines make concurrent programming in Go lightweight",
        "Cats are small carnivorous mammals often kept as house pets",
        "Dogs are loyal domesticated animals descended from wolves",
        "The Andromeda galaxy is the nearest large galaxy to the Milky Way",
        "A black hole is a region of spacetime where gravity is extreme",
        "Sourdough bread is leavened with a fermented flour and water starter",
        "Espresso is brewed by forcing hot water through finely ground coffee",
        "Vector databases index embeddings for approximate nearest neighbor search",
        "HNSW builds a layered graph for fast approximate nearest neighbor queries",
    ];

    let mut index = HnswIndex::new(DIM, Metric::Cosine);
    for (id, text) in corpus.iter().enumerate() {
        index.add(id as u64, &embed(text));
    }

    for query in [
        "a fast language for low level systems work",
        "animals people keep in their homes",
        "searching embeddings quickly",
    ] {
        println!("\nquery: {query:?}");
        for hit in index.search(&embed(query), 3) {
            println!("  {:.3}  {}", hit.distance, corpus[hit.id as usize]);
        }
    }
}

/// A dependency-free stand-in embedding: a hashed bag of character trigrams.
/// Texts that share substrings land near each other under cosine distance.
fn embed(text: &str) -> Vec<f32> {
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut v = vec![0.0f32; DIM];
    for window in chars.windows(3) {
        // FNV-1a hash of the trigram, bucketed into DIM dimensions.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for &c in window {
            hash ^= c as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        v[(hash as usize) % DIM] += 1.0;
    }
    v
}
