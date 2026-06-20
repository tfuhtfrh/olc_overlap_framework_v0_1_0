#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/Pytnon/olc_overlap_framework_v0_1_0/olc_overlap_framework
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.11

if [ -d .venv ]; then
  backup=".venv_py314_backup"
  if [ -e "$backup" ]; then
    backup=".venv_py314_backup_$(/bin/date +%Y%m%d_%H%M%S)"
  fi
  mv .venv "$backup"
  echo "backed_up_to=$backup"
fi

uv venv --python 3.11 .venv
.venv/bin/python -V
uv pip install --python .venv/bin/python -e ".[qubo,sqa]" pytest

.venv/bin/python - <<'PY'
import sys
from importlib.metadata import version
print("python", sys.version)
import parasail
import dimod
import dwave.samplers
import openjij
import olc_pipeline
print("parasail ok")
print("dimod", dimod.__version__)
print("openjij", version("openjij"))
print("olc_pipeline", olc_pipeline.__version__)
PY
