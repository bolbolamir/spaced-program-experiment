#!/bin/zsh
set -eu
cd "$(dirname "$0")"
exec .venv/bin/jupyter lab minilm_weight_space_research.ipynb
