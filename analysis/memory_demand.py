"""Lower-bound GPU memory demand of Prism's "8 models on 2 GPUs" experiment.

Workload and placement come straight from the Prism repo (third_party/prism-research):
  trace      benchmark/multi-model/real_trace.pkl
  mapping    trace.py generate_e2e_benchmark_reqs: trace ranks -> model_1..model_8
  placement  experiments/experiments_1/model_configs/swap_8_2gpus.json
  time scale 0.2 (experiments_1/benchmark_e2e.sh); load scaled by replicating requests

Demand at time t on a GPU, assuming an ideal server (no queueing, every request
starts decoding on arrival at its dedicated-GPU TPOT):
  weights  bf16 weights of every model on the GPU with >= 1 in-flight request
           ("on-demand", i.e. perfect zero-cost swapping), or of all models on
           the GPU ("static", every model resident)
  KV       sum over in-flight requests of (prompt + generated tokens), rounded up
           to 16-token blocks, times the model's KV bytes per token
Real servers queue, so requests live longer and hold more KV: this is a lower bound.
Activations, CUDA graphs and allocator slack are excluded.

usage: python analysis/memory_demand.py [--replications 1] [--capacity-gib 95.6]
"""

import argparse
import csv
import json
import math
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRISM = os.path.join(ROOT, "third_party", "prism-research")
GIB = 1024**3
BLOCK_TOKENS = 16

# trace.py generate_e2e_benchmark_reqs: trace adapter rank -> model_{i+1}
SELECTED_RANKS = [2, 14, 3, 10, 5, 19, 23, 24]
# p95 TPOT (ms) measured by Prism on dedicated GPUs, same function.
PRISM_TPOT_MS = {
    "model_1": 11.46289839, "model_2": 9.607242716, "model_3": 6.097079518,
    "model_4": 11.23110303, "model_5": 11.08433425, "model_6": 8.795845509,
    "model_7": 6.114510298, "model_8": 5.484097324,
}


class _Request:
    pass


class _TraceUnpickler(pickle.Unpickler):
    """Only materializes the plain Request records; refuses any other class."""

    def find_class(self, module, name):
        if name == "Request":
            return _Request
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def load_trace(path, time_scale):
    with open(path, "rb") as f:
        _, reqs = _TraceUnpickler(f).load()
    rank_to_model = {r: f"model_{i + 1}" for i, r in enumerate(SELECTED_RANKS)}
    out = {}
    for r in reqs:
        model = rank_to_model.get(int(r.adapter_dir.split("-")[-1]))
        if model:
            out.setdefault(model, []).append((r.req_time * time_scale, r.prompt_len, r.output_len))
    return {m: np.array(v, dtype=float) for m, v in out.items()}


# trace.py generate_e2e_benchmark_reqs_18m: model_{i+1} -> rank; several models share a rank.
SELECTED_RANKS_18M = [5, 10, 4, 17, 2, 10, 23, 2, 10, 14, 19, 14, 18, 14, 21, 2, 6, 19]


def load_trace_18m(path, time_scale):
    """Prism's 18-model request set. Models sharing a rank get that rank's requests,
    the k-th of them offset by 10*k s (after time scaling): earlier for ranks 14 and 19,
    later for ranks 2 and 10. Arrivals can be negative (Prism sends those at once)."""
    with open(path, "rb") as f:
        _, reqs = _TraceUnpickler(f).load()
    rank_models = {}
    for i, r in enumerate(SELECTED_RANKS_18M):
        rank_models.setdefault(r, []).append(f"model_{i + 1}")
    out = {}
    for r in reqs:
        rank = int(r.adapter_dir.split("-")[-1])
        for k, model in enumerate(rank_models.get(rank, [])):
            shift = -10 * k if rank in (14, 19) else 10 * k if rank in (2, 10) else 0
            out.setdefault(model, []).append((r.req_time * time_scale + shift, r.prompt_len, r.output_len))
    return {m: np.array(v, dtype=float) for m, v in out.items()}


def model_spec(repo, cache):
    """Weight bytes (bf16) and KV bytes per token, from the HF config and safetensors metadata."""
    if repo not in cache:
        from huggingface_hub import HfApi, hf_hub_download

        with open(hf_hub_download(repo, "config.json")) as f:
            cfg = json.load(f)
        params = HfApi().model_info(repo, expand=["safetensors"]).safetensors.total
        kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        cache[repo] = {
            "params": params,
            "weight_bytes": params * 2,
            "kv_bytes_per_token": cfg["num_hidden_layers"] * kv_heads * head_dim * 2 * 2,
        }
    return cache[repo]


def demand(trace, placement, specs, replication, dt):
    """Per-GPU demand timelines (bytes) for one replication factor."""
    t_end = max(
        float(np.max(a[:, 0] + a[:, 2] * PRISM_TPOT_MS[m] / 1000)) for m, a in trace.items()
    )
    t = np.arange(0.0, t_end + dt, dt)
    gpus = sorted({g for p in placement.values() for g in p["gpus"]})
    out = {g: {"t": t, "weights_ondemand": np.zeros_like(t), "weights_static": np.zeros_like(t),
               "kv": np.zeros_like(t), "active_models": np.zeros_like(t)} for g in gpus}

    for model, p in placement.items():
        spec = specs[p["repo"]]
        reqs = trace.get(model, np.zeros((0, 3)))
        tpot = PRISM_TPOT_MS[model] / 1000
        kv = np.zeros_like(t)
        busy = np.zeros_like(t, dtype=bool)
        for arrival, prompt, output in reqs:
            i0 = int(math.ceil(arrival / dt))
            i1 = int(math.ceil((arrival + output * tpot) / dt))
            if i1 <= i0:
                continue
            generated = np.minimum((t[i0:i1] - arrival) / tpot, output)
            tokens = np.ceil((prompt + generated) / BLOCK_TOKENS) * BLOCK_TOKENS
            kv[i0:i1] += tokens * spec["kv_bytes_per_token"]
            busy[i0:i1] = True
        for g in p["gpus"]:
            d = out[g]
            d["kv"] += kv * replication
            d["weights_ondemand"] += busy * spec["weight_bytes"]
            d["weights_static"] += spec["weight_bytes"]
            d["active_models"] += busy
    return out


def summarize(timelines, capacity, replication):
    rows = []
    for g, d in timelines.items():
        row = {"replication": replication, "gpu": g}
        for policy in ("ondemand", "static"):
            total = d[f"weights_{policy}"] + d["kv"]
            over = total > capacity
            row[f"{policy}_peak_gib"] = round(total.max() / GIB, 2)
            row[f"{policy}_pct_time_over"] = round(100 * over.mean(), 1)
            row[f"{policy}_max_excess_gib"] = round(max(0.0, (total.max() - capacity) / GIB), 2)
        row["peak_kv_gib"] = round(d["kv"].max() / GIB, 2)
        row["static_weights_gib"] = round(d["weights_static"][0] / GIB, 2)
        row["max_active_models"] = int(d["active_models"].max())
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=os.path.join(PRISM, "benchmark/multi-model/real_trace.pkl"))
    ap.add_argument("--placement", default=os.path.join(
        PRISM, "experiments/experiments_1/model_configs/swap_8_2gpus.json"))
    ap.add_argument("--time-scale", type=float, default=0.2)
    ap.add_argument("--replications", type=int, nargs="+", default=[1])
    # RTX PRO 6000 Blackwell (our GPUs) as reported by nvidia-smi: 97887 MiB.
    ap.add_argument("--capacity-gib", type=float, default=97887 / 1024)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "memory_demand"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.placement) as f:
        placement = {m["model_name"]: {"repo": m["model_path"],
                                       "gpus": [g for p in m["init_placements"] for g in p["gpu_ids"]]}
                     for m in json.load(f)}

    spec_path = os.path.join(args.out, "model_specs.json")
    specs = json.load(open(spec_path)) if os.path.exists(spec_path) else {}
    for p in placement.values():
        model_spec(p["repo"], specs)
    with open(spec_path, "w") as f:
        json.dump(specs, f, indent=2)

    trace = load_trace(args.trace, args.time_scale)
    capacity = args.capacity_gib * GIB

    print("model    repo                                   gpu  reqs  weights_GiB  KV_KiB/token")
    for m, p in placement.items():
        s = specs[p["repo"]]
        print(f"{m:8} {p['repo']:38} {p['gpus']}  {len(trace.get(m, [])):4d}  "
              f"{s['weight_bytes'] / GIB:11.1f}  {s['kv_bytes_per_token'] / 1024:12.0f}")

    rows = []
    for rep in args.replications:
        timelines = demand(trace, placement, specs, rep, args.dt)
        rows += summarize(timelines, capacity, rep)
        for g, d in timelines.items():
            with open(os.path.join(args.out, f"timeline_gpu{g}_rep{rep}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "weights_ondemand_gib", "weights_static_gib", "kv_gib", "active_models"])
                for i in range(len(d["t"])):
                    w.writerow([f"{d['t'][i]:.2f}", f"{d['weights_ondemand'][i] / GIB:.3f}",
                                f"{d['weights_static'][i] / GIB:.3f}", f"{d['kv'][i] / GIB:.3f}",
                                int(d["active_models"][i])])

    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\ncapacity {args.capacity_gib:.1f} GiB per GPU")
    hdr = ["replication", "gpu", "static_weights_gib", "peak_kv_gib", "ondemand_peak_gib",
           "ondemand_pct_time_over", "static_peak_gib", "static_pct_time_over", "max_active_models"]
    print("  ".join(hdr))
    for r in rows:
        print("  ".join(str(r[h]).rjust(len(h)) for h in hdr))


if __name__ == "__main__":
    main()
