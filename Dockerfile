FROM ubuntu

ENV RUSTFLAGS="-C target-cpu=native"
ENV EXPERIMENT_NAME="copy_stroke"
# ENV LIBTORCH="/home/ben/.cache/pypoetry/virtualenvs/paint-rl-models-v-oY54ye-py3.10/lib/python3.10/site-packages/torch"
# ENV LD_LIBRARY_PATH="/home/ben/.cache/pypoetry/virtualenvs/paint-rl-models-v-oY54ye-py3.10/lib/python3.10/site-packages/torch/lib"
ENV LIBTORCH_CXX11_ABI="0"
ENV PYO3_PYTHON="python3.10"

COPY . /root/paint_rl
WORKDIR /root/paint_rl

RUN apt update && apt upgrade -y
RUN apt install curl python3.10 -y

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
RUN . "$HOME/.cargo/env"

# Install Python
RUN curl -sSL https://install.python-poetry.org | python3.10 - && export PATH="/root/.local/bin:$PATH"

# Install Poetry environment
RUN poetry install

# Build binaries
RUN cargo build --release