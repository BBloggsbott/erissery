.PHONY: format check build install

format:
	cargo fmt --all

check:
	cargo clippy --workspace --all-targets -- -D warnings

build:
	cargo build

install:
# Should ideally run cargo install, but keeping it as build till a stable and satisfactory version is ready.
	cargo build --release
