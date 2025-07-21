# Kairei API

RESTful API server for the Kairei-v2 AgentCulture Framework.

## Features

- 🚀 High-performance API built with Axum
- 🔐 JWT authentication with Auth0 support
- 📚 Auto-generated OpenAPI documentation
- 🛡️ CORS support
- 📊 Comprehensive logging and tracing

## Quick Start

### Development (without authentication)

```bash
# Use development config
cargo run -p kairei-api -- -c config.development.example.json

# Or disable auth via environment variable
KAIREI_AUTH_ENABLED=false cargo run -p kairei-api
```

### Production (with authentication)

```bash
# Copy and configure the example
cp config.example.json config.json
# Edit config.json with your Auth0 settings

# Run with config
cargo run -p kairei-api -- -c config.json
```

## Configuration

The API can be configured through:

1. **Configuration file** (recommended for production)
2. **Environment variables**
3. **Command-line arguments**

Priority: CLI args > Environment variables > Config file > Defaults

### Configuration File

See `config.example.json` for a complete example.

```json
{
  "auth": {
    "enabled": true,
    "auth0_domain": "your-tenant.auth0.com",
    "auth0_audience": "https://api.kairei.dev"
  }
}
```

### Environment Variables

- `KAIREI_HOST`: Server host (default: "127.0.0.1")
- `KAIREI_PORT`: Server port (default: 3000)
- `KAIREI_LOG_LEVEL`: Log level (default: "info")
- `KAIREI_AUTH_ENABLED`: Enable/disable auth (default: true)
- `RUST_LOG`: Fine-grained log control

### Command Line Arguments

```bash
kairei-api [OPTIONS]

Options:
  -H, --host <HOST>              Host address [env: KAIREI_HOST=] [default: 127.0.0.1]
  -p, --port <PORT>              Port to listen on [env: KAIREI_PORT=] [default: 3000]
  -l, --log-level <LOG_LEVEL>    Log level [env: KAIREI_LOG_LEVEL=] [default: info]
  -c, --config-file <CONFIG>     Config file path [env: KAIREI_CONFIG=]
      --disable-swagger          Disable Swagger UI [env: KAIREI_DISABLE_SWAGGER=]
  -h, --help                     Print help
  -V, --version                  Print version
```

## API Documentation

When the server is running, visit:
- Swagger UI: http://localhost:3000/swagger-ui
- OpenAPI JSON: http://localhost:3000/api-docs/openapi.json

## Authentication

The API uses JWT authentication with Auth0. When enabled (default), all API endpoints require a valid JWT token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### Disabling Authentication for Development

**Warning**: Only disable authentication for local development!

```bash
# Option 1: Use development config
cargo run -p kairei-api -- -c config.development.example.json

# Option 2: Environment variable
KAIREI_AUTH_ENABLED=false cargo run -p kairei-api

# Option 3: Create a development config file
echo '{"auth": {"enabled": false}}' > config.dev.json
cargo run -p kairei-api -- -c config.dev.json
```

## Security Notice

⚠️ **Authentication is enabled by default for security**. Only disable it for local development environments. Never disable authentication in production!

## Development

### Running Tests

```bash
cargo test -p kairei-api
```

### Debug Logging

```bash
# Enable debug logs
cargo run -p kairei-api -- --log-level debug

# Or with environment variable
RUST_LOG=debug cargo run -p kairei-api
```