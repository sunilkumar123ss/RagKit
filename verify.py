#!/usr/bin/env python3
"""Run application checks in the documented development environment."""
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
required = ("pytest", "fastapi", "httpx", "pypdf", "qdrant_client", "mcp", "langchain_core", "openai", "anthropic")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if sys.version_info < (3, 10) or missing:
    print("Full application verification requires Python 3.10+ and the documented development dependencies.")
    print("python -m pip install -e '.[web,providers,mcp,langchain,dev]' 'qdrant-client>=1.15,<2'")
    print("The original offline reference remains runnable with: python main.py")
    raise SystemExit(2)

checks = [
    ("application tests", ["-m", "pytest", "-q", "-m", "not live_models"]),
    ("installed CLI help", ["-m", "ragkit", "--help"]),
    ("original reference demo", ["main.py"]),
]
failed = False
for name, args in checks:
    result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=180)
    print(("PASS " if result.returncode == 0 else "FAIL ") + name)
    if result.returncode:
        print(result.stdout + result.stderr)
        failed = True
    elif name == "application tests":
        print(result.stdout.strip())
print("Real neural models are checked separately with RAGKIT_RUN_LIVE_MODELS=1 and pytest -m live_models.")
raise SystemExit(1 if failed else 0)
