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
