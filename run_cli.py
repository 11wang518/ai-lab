#!/usr/bin/env python
"""命令行入口：python run_cli.py --provider ollama --model qwen2.5:1.5b"""

from app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
