# PaintGym
aka my attempts to use AI to quickly add lineart, base colors, and rendering to sketches.

First pass:
- [x] Add ability to control size of the image through config.
- [x] Get human rendering mode working.
- [x] Add ability to stroke a line on screen with a start and end point.
- [x] Create agent with random policy for testing.
- [x] Get benchmarking set up with realistic training times.
- [x] Add real time plotting functionality.

Second pass (Get agent working on environment):
- [x] Get model exporting working.
- [x] Set up model in copy stroke environment for single env.
- [x] Upgrade to batch mode.
- [x] Check if cuda makes a difference in training speed.
- [x] Add CLI parsing for hyperparameters and fix environments not reloading correctly.
- [x] Add monitor.
- [x] Hyperparameter search integration.
- [x] Create debug environments (constant obs, constant reward, etc).
- [x] Switch to PPO.
- [x] Add benchmark for rollout buffer.
- [x] Add proper brush (thickness) support.
- [x] Anneal standard deviation of brush stroke positions over time.

Third pass (Get cloud training working):
- [x] Integrate cloud logging service for monitor.
- [x] Integrate cloud logging service for Optuna.
- [x] Add categorical distribution.
- [x] Add normal distribution.
- [x] Check PPO implementation with cartpole.
- [ ] Create docker container for repo.
- [ ] Set up linting in CI (clippy, rustfmt, black, mypy).
- [ ] Set up docker container building and uploading on `main`.
- [ ] Run docker container using EC2 spot instance.
- [ ] Get logging working in cloud.
- [ ] Create CLI for deploying workloads in cloud.
- [ ] Add price estimation for workloads.
- [ ] Get checkpointing and restoring working using S3 (or equivalent), adding the option in the CLI to auto continue if terminated early.
- [ ] Add ability to request different instances (like compute or memory optimized).

Third pass (Get copy stroke working)
- [ ] As a test, make sure the agent can learn to stroke a simple square.
- [ ] Iterate on random stroke 64x64 sized image with brush thickness of 4 until model gives satisfactory results.
- [ ] Add colored brush support.
- [ ] Iterate on environment until model gives satisfactory results.
- [ ] Add bezier curve support.
- [ ] Iterate on environment until model gives satisfactory results.
- [ ] Add brush radius support.
- [ ] Iterate on environment until model gives satisfactory results.
- [ ] Add opacity support.
- [ ] Iterate on environment until model gives satisfactory results.