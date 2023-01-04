export LIBTORCH := "/home/ben/.cache/pypoetry/virtualenvs/paint-rl-models-v-oY54ye-py3.10/lib/python3.10/site-packages/torch"
export LD_LIBRARY_PATH := "/home/ben/.cache/pypoetry/virtualenvs/paint-rl-models-v-oY54ye-py3.10/lib/python3.10/site-packages/torch/lib"
export LIBTORCH_CXX11_ABI := "0"
export PYO3_PYTHON := "python3.10"
export RUST_BACKTRACE := "1"

default:
    just --list

# Run an optimized version of the specified binary.
@run-rel NAME:
   RUSTFLAGS="-C target-cpu=native" cargo run --bin {{NAME}} --release

# Run a debug version of the specified binary.
@run NAME:
   cargo run --bin {{NAME}}

# Run benchmarks.
@bench:
   cargo bench

# Perform linting and formatting.
@lint:
   cargo fmt && cargo clippy --fix --allow-staged --allow-dirty 
