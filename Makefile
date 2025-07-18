.PHONY: all build check test fmt lint clean dev release

# デフォルトタスク
all: fmt lint test

# ビルド
build:
	cargo build

# チェック（高速）
check:
	cargo check

# テスト実行
test:
	cargo test

# フォーマット
fmt:
	cargo fmt
	cargo clippy --fix --allow-dirty
	cargo clippy -- -D warnings
	cargo fmt

# リント（よく使うやつ！）
lint: fmt

# クリーン
clean:
	cargo clean

# 開発用（フォーマット → リント → ビルド）
dev: fmt
	cargo clippy --allow-dirty && cargo build

# リリースビルド
release:
	cargo build --release

# すべてのチェック（CI用）
ci: fmt
	cargo clippy -- -D warnings
	cargo test

# ドキュメント生成
doc:
	cargo doc --open

# 依存関係の更新
update:
	cargo update

# 監視モード（cargo-watchが必要）
watch:
	cargo watch -x check -x test -x clippy