# Environment requirements

The base install is declared in `pyproject.toml`. Files here pin specialized
machine profiles that need additional CUDA or training dependencies.

- `m3-b200.txt`: tested M3 profile for Linux x86-64 and B200-class systems.

Use the packaged relocatable `.venv` for the fastest matching-machine setup, or
create a clean environment and install one of these profiles when the target
CUDA stack differs.
