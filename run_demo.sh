#!/usr/bin/env bash
# Two terminals. This is terminal 1.
set -e
python3 refit.py --demo
echo
echo "Now open http://localhost:8000  — it will say 'waiting for a feature vector'."
echo "In a SECOND terminal run:"
echo "    python3 push.py --replay examples/session.jsonl --interval 2"
echo
python3 serve.py
