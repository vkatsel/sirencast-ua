#!/usr/bin/env python3
"""
Root wrapper for .agents/log_ai_usage.py
Ensures execution succeeds whether invoked from repo root or .agents directory.
"""
import sys
from pathlib import Path
import runpy

target = Path(__file__).parent / ".agents" / "log_ai_usage.py"
runpy.run_path(str(target), run_name="__main__")
