# kvcached-repro

在我們的機器（8× NVIDIA RTX PRO 6000 Blackwell，每張 96 GB）上重現 [kvcached](https://github.com/ovg-project/kvcached) 和 [Prism](https://arxiv.org/abs/2505.04021) 的 multi-LLM serving workload，量測 VRAM 壓力。目標是確認 multi-LLM serving 時 VRAM 會不會不夠，作為 weight swapping 和 KV offload 研究的 motivation。

## 實驗：Prism §7.2「8 個模型、2 張 GPU」，預設設定

| 項目 | 設定 |
|---|---|
| Workload | Prism repo 的 `real_trace.pkl`。這是 Chatbot Arena 對話紀錄的抽樣（S-LoRA／VTC 整理），只保留到達時間、目標模型、prompt 長度和輸出長度。trace 編號對應到 model_1–8，共 754 個 request，時間 ×0.2 後約 120 秒 |
| 模型與擺放 | Prism repo 的 `swap_8_2gpus.json`：GPU0 放 3× Mistral-Nemo-12B 和 Llama-3.1-8B；GPU1 放 2× Mistral-7B 和 2× Llama-3.2-3B |
| 對應到我們的 GPU | Prism 的 GPU0 對應我們的 GPU 1，GPU1 對應 GPU 2，容量都是 95.6 GiB |
| Engine | vLLM 0.30.0 + kvcached v0.1.6 |

### 結果

| | 不做 swap | DRAM weight swapping |
|---|---|---|
| 結果資料夾 | `results/prism_8m2g_kvcached_default/` | `results/prism_8m2g_swap_dram_default/` |
| GPU 1 的 4 個模型 | 第 4 個模型 OOM，GPU 用到 95.4 / 95.6 GiB | 全部啟動，閒置的模型睡眠，各佔 1.8–2.5 GiB |
| request | model_4 的 262 個 request（35%）無法服務 | 754 / 754 成功 |
| swap | — | 9 次：8 次冷啟動喚醒、1 次換出。單次 0.16–0.52 秒，總共約 3.3 秒 |
| GPU 1 尖峰用量 | 95.4 GiB（OOM） | 87.6 GiB |

- 每個模型除了權重，還要 2.3–3.5 GiB 的固定開銷，包括 CUDA context、activation 和 CUDA graph。
- KV cache 需求小，單一模型最多 4.5 GiB。
- DRAM swap 版本中，GPU 1 的 TPOT 中位數是 59–68 ms，約為單獨執行的 5–6 倍。原因是 3 個大模型同時 decode、互搶運算資源，不是 swap 造成的。

### 解讀時的限制

- **trace 不同**：論文 evaluation 用的 Hyperbolic 和 Arena-Chat trace 沒有公開，所以只能比趨勢，不能比絕對數字。
- **硬體不同**：論文用 H100 80 GB，我們是 96 GB，而且沒有 NVLink。
- **擺放不平均**：VRAM 不夠主要是 repo 的初始擺放不平均造成的，GPU0 拿到 83 GiB 權重和 80% 的流量。如果把模型平均擺放，兩張 96 GB 都放得下。
- **swap 策略是我們自己寫的**（`scripts/swap_proxy.py`：on-demand、LRU、drain），搬移機制則是 vLLM 的 sleep mode（level 1）。這屬於 time-sharing baseline，不是完整的 Prism。

## 結果資料夾

| 資料夾 | 內容 |
|---|---|
| `results/memory_demand/` | 用 trace 算出的記憶體需求下界（時間 ×0.2）|
| `results/memory_demand_timescale1/` | 同上，時間不縮放 |
| `results/prism_8m2g_kvcached_default/` | 實測，不做 swap |
| `results/prism_8m2g_swap_dram_default/` | 實測，DRAM weight swapping |
| `results/test_*`、`results/smoke_*` | **只是測試流程用，不能當結果引用**。`test_*` 用的是 Qwen3-8B 和人為設定；`smoke_*` 是最早的 kvcached 合成 workload，用的是 kvcached main，不是 v0.1.6 |

每次實測的資料夾裡有：

| 檔案 | 內容 |
|---|---|
| `summary.md`、`summary.json`、`report.png` | 總結表格和圖 |
| `config.txt` | 版本、設定和每個階段的時間戳 |
| `instances.tsv` | 每個模型的 GPU 和 port |
| `startup.tsv` | 每個模型啟動花的時間和記憶體 |
| `vram.csv` | 每 0.25 秒一筆：每張 GPU、每個模型的記憶體、KV 用量、排隊數、搶佔次數 |
| `requests.jsonl` | 每個 request 的 TTFT、TPOT、E2E |
| `swaps.jsonl`、`proxy_requests.jsonl` | 只有 swap 版本才有：每次 swap 的紀錄，以及每個 request 等 swap 的時間 |
| `server_*.log` | 每個 vLLM instance 的 log |

## 重跑

```bash
bash scripts/setup_env.sh                              # .venv：vLLM 0.30.0 + kvcached v0.1.6（CUDA 13.0 toolchain）
.venv/bin/python scripts/download_prism_models.py      # 下載 4 個模型的 HF safetensors

.venv/bin/python analysis/memory_demand.py && .venv/bin/python analysis/plot_memory_demand.py
bash scripts/run_prism_trace.sh kvcached prism_8m2g_kvcached_default             # 不做 swap
SWAP=1 bash scripts/run_prism_trace.sh kvcached prism_8m2g_swap_dram_default     # DRAM swap
```

GPU 預設用 1 和 2，可以用 `PRISM_GPUS="3 4"` 更改。GPU 0 通常被其他服務佔用。

## 目錄

| 路徑 | 內容 |
|---|---|
| `third_party/kvcached` | submodule，鎖在 v0.1.6 |
| `third_party/prism-research` | submodule，鎖在 `595ec1f` |
| `scripts/` | 啟動、監控、trace 重播、swap proxy |
| `analysis/` | 記憶體需求計算和報告產生 |
| `configs/experiment.env` | kvcached 三模型合成 workload 的設定 |
