# ---- build stage ----
FROM rust:1.82-slim AS build
WORKDIR /app

# Copy the manifest and sources, then build only the server binary.
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY benches ./benches
RUN cargo build --release --features server --bin server

# ---- runtime stage ----
FROM debian:bookworm-slim
COPY --from=build /app/target/release/server /usr/local/bin/velo-server
EXPOSE 8080
ENV VELO_ADDR=0.0.0.0:8080 \
    VELO_DIM=128
ENTRYPOINT ["velo-server"]
