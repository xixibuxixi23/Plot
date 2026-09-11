# Inference pipelines

Pipelines compose trained modules with committed resident state and sparse world
memory. They own inference order and state transitions, while neural networks
remain in `plot.models` and shared state contracts remain at the `plot` package
root.
