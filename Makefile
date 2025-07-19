.PHONY: all build check test fmt lint clean dev release

# Default task
all: fmt lint test

# Build
build:
	cargo build

# Check (fast)
check:
	cargo check

# Run tests
test:
	cargo test

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