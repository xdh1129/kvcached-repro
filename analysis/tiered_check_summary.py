"""Summarize results/tiered_check/<config>/ into one table (markdown on stdout and summary.md).

Per config: whether every checksum matched after restore (and the control
detected its zeroed KV), the bytes on each path, the restore time split per path,
and how many restore bytes passed through host DRAM.
"""

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tiered_check")
ORDER = ["dram+dram", "ssd+ssd", "gds+gds", "dram+ssd", "dram+gds", "dram+ssd:seq", "dram+gds:seq"]
GB = 1e9


def main():
    configs = [c for c in ORDER if os.path.isdir(os.path.join(ROOT, c))]
    configs += sorted(c for c in os.listdir(ROOT) if c not in configs and os.path.isdir(os.path.join(ROOT, c)))
    lines = ["| config (weights+KV) | checksums after restore | control caught zeroed KV | weights+state GB | KV GB "
             "| weights path s | KV path s | restore s (map + loads) | through host DRAM GB |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in configs:
        path = os.path.join(ROOT, c, "tiered.jsonl")
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path)]
        res = [r for r in recs if r["event"] == "resume"]
        restore = next((r for r in res if not r["debug_skip_kv_load"]), None)
        control = next((r for r in res if r["debug_skip_kv_load"]), None)
        if restore is None:
            continue
        v = restore["verify"]
        ok = all(v[k] == 0 for k in v if k.endswith("_mismatch"))
        caught = control is not None and control["verify"]["kv_mismatch"] == control["verify"]["kv_regions"] > 0
        w = restore["weights_end_s"] - restore["weights_start_s"]
        kv = restore.get("kv_end_s", 0) - restore.get("kv_start_s", 0)
        # end of the last load; total_s of older logs also counted the checksums
        done = max(restore["weights_end_s"], restore.get("kv_end_s", 0))
        host = restore["weights_host_bytes"] + restore.get("kv_host_bytes", 0)
        lines.append(f"| {c} | {'all match' if ok else 'MISMATCH ' + json.dumps(v)} | {'yes' if caught else 'NO'} "
                     f"| {restore['weights_bytes'] / GB:.2f} | {restore.get('kv_bytes', 0) / GB:.2f} "
                     f"| {w:.3f} | {kv:.3f} | {done:.3f} | {host / GB:.2f} |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(ROOT, "summary.md"), "w") as f:
        f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
