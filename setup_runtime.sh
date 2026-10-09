#!/usr/bin/env bash
set -euo pipefail
POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$POLICY_ROOT"
export UV_PROJECT_ENVIRONMENT="$POLICY_ROOT/.venv"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$POLICY_ROOT/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-8}"
export GIT_LFS_SKIP_SMUDGE=1
unset VIRTUAL_ENV
if [[ -f runtime-wheel-manifest.json && -d runtime_wheels ]]; then
    if [[ ! -x .venv/bin/python ]]; then
        uv venv --python 3.11 .venv
    fi
    export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-8}"
    .venv/bin/python scripts/install_runtime_wheels.py
else
    uv sync --frozen --no-dev --python 3.11
fi
uv pip check --python "$POLICY_ROOT/.venv/bin/python"
uv pip freeze --python "$POLICY_ROOT/.venv/bin/python" > "$POLICY_ROOT/runtime-packages.txt"
# This tokenizer is included with the policy so evaluation need not download it.
"$POLICY_ROOT/.venv/bin/python" - <<'PYTOKENIZER'
from pathlib import Path
import hashlib, urllib.request
path=Path("runtime_assets/openpi/big_vision/paligemma_tokenizer.model")
expected="8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
if not path.exists():
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".partial")
    urllib.request.urlretrieve("https://storage.googleapis.com/big_vision/paligemma_tokenizer.model",temporary)
    if hashlib.sha256(temporary.read_bytes()).hexdigest()!=expected:
        raise RuntimeError("Tokenizer checksum mismatch")
    temporary.replace(path)
assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, "Tokenizer checksum mismatch"
print("Tokenizer SHA256 verified")
PYTOKENIZER
