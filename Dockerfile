FROM rust:latest AS builder
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN cargo build --release -p karta-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/karta-server /usr/local/bin/
ENV KARTA_HOST=0.0.0.0
ENV KARTA_PORT=8080
EXPOSE 8080
RUN useradd -r -s /bin/false karta && mkdir -p /data && chown karta:karta /data
USER karta
WORKDIR /data
CMD ["karta-server"]
