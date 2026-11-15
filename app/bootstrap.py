"""Builds the local models on first use when `make build` was skipped.

The jury may run `make predict` or `make serve` right after cloning. Without the rule picker, the verdict model
and the knowledge base the service still answers, but a little worse (on validation 9 of 1000 categories and
37 drafts differ), so the missing models are built once from data/handout (~2-3 minutes, fully local).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ("rulepick.pkl", "verdict.pkl", "kb.pkl")


def ensure_models() -> None:
    missing = [name for name in REQUIRED if not (ROOT / "models" / name).exists()]
    if not missing:
        return
    if not (ROOT / "data" / "handout" / "tickets_train.jsonl.gz").exists():
        print(f"models missing ({', '.join(missing)}) and no data/handout/tickets_train.jsonl.gz: "
              "running with reduced quality, see README (make build)", file=sys.stderr)
        return
    print(f"models missing ({', '.join(missing)}): building them once, as `make build` does (~2-3 min)...",
          file=sys.stderr)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_models.py")], check=True, cwd=ROOT)
