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
- [ ] Add proper brush (thickness) support.
- [ ] Iterate on 128x128 sized image with brush thickness of 4 until model gives satisfactory results.
- [ ] Add colored brush support support.
- [ ] Iterate on environment until model gives satisfactory results.