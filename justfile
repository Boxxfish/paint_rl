export LIBTORCH := "/home/ben/.cache/pypoetry/virtualenvs/paint-rl-models-v-oY54ye-py3.10/lib/python3.10/site-packages/torch"
export LD_LIBRARY_PATH := "/home/ben/.cache/pypoetry/virtualenvs/paint-rl-models-v-oY54ye-py3.10/lib/python3.10/site-packages/torch/lib"
export LIBTORCH_CXX11_ABI := "0"
export PYO3_PYTHON := "python3.10"
export RUST_BACKTRACE := "1"

default:
    just --list

# Run an optimized version of the specified binary.
@run-rel NAME *ARGS:
   RUSTFLAGS="-C target-cpu=native" cargo run --bin {{NAME}} --release -- {{ARGS}}

# Run a debug version of the specified binary.
@run NAME *ARGS:
   cargo run --bin {{NAME}} -- {{ARGS}}

# Run benchmarks.
@bench TARGET:
   cargo bench -- {{TARGET}}

# Run tuner.
@tune NAME OBJECTIVE TRIALS *ARGS:
   RUSTFLAGS="-C target-cpu=native" python python/tune.py --name {{NAME}} --objective {{OBJECTIVE}} --trials {{TRIALS}} --wandb-project 'paint-rl' -- {{ARGS}}

# Perform linting and formatting.
@lint:
   cargo fmt && cargo clippy --fix --allow-staged --allow-dirty 
