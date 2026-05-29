#!/usr/bin/env python3
import glob
import json
import os
import sys


root = sys.argv[1] if len(sys.argv) > 1 else "eval_outputs"

for path in sorted(glob.glob(os.path.join(root, "**", "output.jsonl"), recursive=True)):
    run_dir = os.path.dirname(path)
    run = os.path.relpath(run_dir, root)
    print(f"=== {run} ===")

    for line in open(path):
        d = json.loads(line)
        m = d["metrics"]["accumulated_token_usage"]
        p = m.get("prompt_tokens", 0) / 1000
        c = m.get("completion_tokens", 0) / 1000
        print(
            f"{d['instance_id']:>14}  events={len(d['history']):>3}  in={p:>7.1f}k  out={c:>5.1f}k"
        )

    report = os.path.join(run_dir, "output.report.json")
    if os.path.exists(report):
        with open(report) as f:
            print(json.dumps(json.load(f), indent=4))
    print()
