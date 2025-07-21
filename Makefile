.PHONY: all build check test fmt lint clean dev release db-up db-down db-migrate test-integration test-all db-clean db-logs db-connect

# Default task
all: fmt lint test

# Build
build:
	cargo build

# Check (fast)
check:
	cargo check

# Run tests
test: db-up db-migrate
	cargo test --quiet -- --test-threads=1 --nocapture
	@echo "✅ All tests passed"

# Format
fmt:
	cargo fmt
	cargo clippy --fix --allow-dirty
	cargo clippy -- -D warnings
	cargo fmt
	@if [ -d "training" ]; then \
		echo "Formatting training directory..."; \
		cd training && poetry run black . && poetry run ruff check --fix . || true; \
	fi

# Lint (frequently used!)
lint: fmt

# Clean
clean:
	cargo clean

# Development (format -> lint -> build)
dev: fmt
	cargo clippy --allow-dirty && cargo build

# Release build
release:
	cargo build --release

# All checks (for CI)
ci: fmt
	cargo clippy -- -D warnings
	cargo test

# Generate documentation
doc:
	cargo doc --open

# Update dependencies
update:
	cargo update

# Watch mode (requires cargo-watch)
watch:
	cargo watch -x check -x test -x clippy

# ========== Python Evaluation Suite ==========

# Setup evaluation environment
eval-setup:
	cd evaluation && poetry install

# Run evaluation tests
eval-test:
	cd evaluation && poetry run pytest -v

# Format evaluation code
eval-fmt:
	cd evaluation && poetry run black . && poetry run isort .

# Quick evaluation test
eval-quick:
	cd evaluation && poetry run pytest -v -k "Hello"

# All evaluation tasks
eval: eval-fmt eval-test

# ========== LoRA Commands ==========

# Convert PEFT to candle-lora format
lora-convert:
	cargo run --bin kairei -- lora convert ./training/test_lora_output/final_model -o ./test_converted.safetensors

# Test LoRA chat with TinyLlama
lora-chat:
	cargo run --bin kairei -- chat --candle --model-type llama2 --lora test_rust_trained --base-model models/tinyllama/model.safetensors --tokenizer models/tokenizer.json --once --message "What is LoRA fine-tuning?" --max-tokens 100

# Test without LoRA (base model only)
base-chat:
	cargo run --bin kairei -- chat --candle --model-type llama2 --base-model models/tinyllama/model.safetensors --tokenizer models/tokenizer.json --once --message "What is LoRA fine-tuning?"

# ========== Database & Integration Tests ==========

# Start Docker Compose services
db-up:
	docker-compose up -d postgres
	@echo "⏳ Waiting for PostgreSQL to be ready..."
	@sleep 5
	@docker-compose ps

# Stop Docker Compose services
db-down:
	docker-compose down

# Run database migrations
db-migrate:
	cd crates/kairei && sqlx migrate run --source ../../migrations

# Run integration tests with database
test-integration: db-up db-migrate
	cargo test -p kairei --test integration_tests -- --test-threads=1 --nocapture
	@echo "✅ Integration tests completed"

# Clean database (remove volumes)
db-clean:
	docker-compose down -v
	@echo "🧹 Database volumes cleaned"

# Show database logs
db-logs:
	docker-compose logs -f postgres

# Connect to database with psql
db-connect:
	docker-compose exec postgres psql -U kairei_user -d kairei_dev

# ========== Local Development ==========

# Run API server locally
api-run: db-up
	cargo run -p kairei-api --bin kairei-api
	
# Run API server with watch mode (requires cargo-watch)
api-watch: db-up
	cargo watch -x run -p kairei-api --bin kairei-api

# Build API server
api-build:
	cargo build --bin kairei-api --release

# Start development environment (DB + pgAdmin)
dev-up:
	docker-compose up -d
	@echo "✅ Development services started"
	@echo "PostgreSQL: localhost:5432"
	@echo "pgAdmin: http://localhost:5050"
	@echo ""
	@echo "Run 'make api-run' to start the API server locally"

# Stop development environment
dev-down:
	docker-compose down

# View development logs
dev-logs:
	docker-compose logs -f

# Clean everything (including volumes)
dev-clean:
	docker-compose down -v
	@echo "🧹 Development environment cleaned"
