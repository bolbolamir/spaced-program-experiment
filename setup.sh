#!/bin/zsh
set -eu
cd "$(dirname "$0")"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python -m ipykernel install --prefix .venv --name minilm-weight-space --display-name 'MiniLM research (.venv)'
