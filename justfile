default:
    just --list

# Run an optimized version of the specified binary.
run-rel NAME:
   RUSTFLAGS="-C target-cpu=native" cargo.exe run --bin {{NAME}} --release

# Run a debug version of the specified binary.
run NAME:
   cargo.exe run --bin {{NAME}}

# Run benchmarks.
bench:
   cargo bench

# Perform linting and formatting.
lint:
   cargo.exe fmt && cargo.exe clippy --fix --allow-staged --allow-dirty 
