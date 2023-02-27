set dotenv-load := true

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
