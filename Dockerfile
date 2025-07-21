# Build stage
FROM rust:1.85-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy everything
COPY . .

# Build application
RUN cargo build --release --bin kairei-api

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1001 kairei

# Copy binary from builder
COPY --from=builder /app/target/release/kairei-api /usr/local/bin/kairei-api

# Create necessary directories
RUN mkdir -p /app/models /app/loras /app/datasets && \
    chown -R kairei:kairei /app

# Copy migrations
COPY --from=builder /app/migrations /app/migrations
RUN chown -R kairei:kairei /app/migrations

# Switch to app user
USER kairei
WORKDIR /app

# Expose port
EXPOSE 8080

# Run the API server
CMD ["kairei-api"]