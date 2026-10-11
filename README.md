# kvcached-repro

在我們的機器（8× NVIDIA RTX PRO 6000 Blackwell，每張 96 GB）上重現 [kvcached](https://github.com/ovg-project/kvcached) 和 [Prism](https://arxiv.org/abs/2505.04021) 的 multi-LLM serving workload，量測 VRAM 壓力。目標是確認 multi-LLM serving 時 VRAM 會不會不夠，作為 weight swapping 和 KV offload 研究的 motivation。

## 對照圖

![Prism 8-model comparison](results/comparison/prism_8m2g_comparison.png)

第 1 列是 GPU 1，第 2 列是 GPU 2；左邊不做 swap，右邊做 DRAM weight swapping。同一個模型在每格都用同一個顏色。

- **(a) GPU 1，不做 swap**：4 個模型都載入了，但 GPU 只剩 3.3 GiB，低於 kvcached 預設的 5% 安全保留空間（4.8 GiB）。model_4 配不到任何 KV，它的 262 個 request 一直排隊、從來沒被處理。model_1 也幾乎沒有 KV 可用：被搶佔 398 次，TTFT 中位數 94 秒（做 swap 時是 0.17 秒）。
- **(b) GPU 1，DRAM swap**：閒置的模型把權重放在 host DRAM，有 request 時才換回 GPU（▲），空間不夠時換出（▽）。4 個模型都能服務。
- **(c)(d) GPU 2**：4 個較小的模型共 49 GiB，兩種做法都放得下，沒有壓力。
- **(e) GPU 1 啟動過程（不做 swap）**：模型依序載入，第 4 個載入後只剩 3.3 GiB。第一次跑同樣的實驗時，第 4 個模型在這一步直接 OOM（`results/prism_8m2g_kvcached_default/`）。
- **(f) 服務的 request（兩邊都是實測）**：不做 swap 是 492 / 754，model_4 的 262 個全部失敗；做 DRAM swap 是 754 / 754。
- **(g) swap 成本**：從 DRAM 喚醒一個模型需要 0.15–0.52 秒，大致和模型大小成正比。整段 trace 共 9 次 swap、約 3.3 秒。

圖由 `analysis/plot_comparison.py` 產生。

## 實驗：Prism §7.2「8 個模型、2 張 GPU」，預設設定

| 項目 | 設定 |
|---|---|
| Workload | Prism repo 的 `real_trace.pkl`：Chatbot Arena 使用者對話紀錄的抽樣，只保留到達時間、目標模型、prompt 長度和輸出長度。trace 編號對應到 model_1–8，共 754 個 request，時間 ×0.2 後約 120 秒 |
| 模型與擺放 | Prism repo 的 `swap_8_2gpus.json`：GPU0 放 3× Mistral-Nemo-12B 和 Llama-3.1-8B；GPU1 放 2× Mistral-7B 和 2× Llama-3.2-3B |
| 對應到我們的 GPU | Prism 的 GPU0 對應我們的 GPU 1，GPU1 對應 GPU 2，容量都是 95.6 GiB |
| Engine | vLLM 0.30.0 + kvcached v0.1.6（`KVCACHED_GPU_UTILIZATION=0.95`，也就是保留 5% 安全空間） |

### Trace 的來源
- **論文 evaluation 用了兩份 trace**：Hyperbolic 和 Arena-Chat。
- **Hyperbolic** 是商業資料，沒有公開。
- **Arena-Chat**：論文 [arXiv v2](https://arxiv.org/abs/2505.04021v2)（2025 年 5 月）的 §7 寫的是「Arena-Chat [48]」，而 [48] 是 VTC 論文（Sheng et al., *Fairness in Serving Large Language Models*, OSDI'24）。Prism repo 的 `real_trace.pkl` 和 VTC artifact 公開的 `real_trace.pkl` 逐 byte 相同，repo 的實驗腳本也都使用它。所以這份 trace **很可能就是論文 evaluation 的 Arena-Chat**。
- **不能 100% 確定的地方**：論文和 repo 都沒有明寫檔名。另外論文描述的 Arena-Chat 有 84 個模型、11 天，這個檔案只有 27 個模型、壓縮成 600 秒，比較可能是從完整紀錄抽樣出來的版本。
- 論文 v3（2026 年 6 月）把這個引用改成 Chatbot Arena 平台本身。

### 結果

| | 不做 swap | DRAM weight swapping |
|---|---|---|
| 結果資料夾 | `results/prism_8m2g_noswap_default/` | `results/prism_8m2g_swap_dram_default/` |
| GPU 1 的 4 個模型 | 都載入了，但只剩 3.3 GiB，低於 5% 安全保留空間 | 全部啟動，閒置的模型睡眠，各佔 1.8–2.5 GiB |
| request | 492 / 754：model_4 的 262 個 request 配不到 KV，全部失敗 | 754 / 754 成功 |
| model_1（最熱門） | 被搶佔 398 次，TTFT 中位數 94 秒 | 沒有被搶佔，TTFT 中位數 0.17 秒 |
| swap | — | 9 次：8 次冷啟動喚醒、1 次換出。單次 0.15–0.52 秒，總共約 3.3 秒 |
| GPU 1 trace 期間尖峰 | 93.7 GiB | 87.6 GiB |
| GPU 2 trace 期間尖峰 | 49.4 GiB | 49.2 GiB |

- **第一次跑「不做 swap」**（`results/prism_8m2g_kvcached_default/`）：第 4 個模型在啟動的 profiling 階段就 OOM 了。兩次都是 GPU 1 裝不下 4 個模型的權重加上 KV，差別只是記憶體耗盡發生在啟動時還是服務時。
- **第二次的停止方式**：model_4 的 request 卡住後，重播在 trace 開始 2,874 秒後被手動停止。這 262 個 request 在 `requests.jsonl` 裡標記為 `added_after_stop: true`。
- **每個模型的固定開銷**：除了權重，還要 2.3–3.5 GiB，包括 CUDA context、activation 和 CUDA graph。
- **DRAM swap 時 GPU 1 的 TPOT**：中位數 59–68 ms，約為單獨執行的 5–6 倍。原因是 3 個大模型同時 decode、互搶運算資源，不是 swap 造成的。

### 解讀時的限制

- **trace**：論文兩份 trace 中只有 Arena-Chat 有公開，而且很可能就是我們用的這份。Hyperbolic 沒有公開。
- **硬體**：論文用 H100 80 GB，我們是 96 GB，而且沒有 NVLink。
- **擺放**：VRAM 不夠主要是 repo 的初始擺放不平均造成的，GPU0 拿到 83 GiB 權重和 80% 的流量。如果把模型平均擺放，兩張 96 GB 都放得下。
- **swap 策略**：`scripts/swap_proxy.py` 是我們自己寫的（on-demand、LRU、drain），搬移機制是 vLLM 的 sleep mode（level 1）。這屬於 time-sharing baseline，不是完整的 Prism。
- **swap 次數少的原因**：這個 trace 雖然常常交錯，但 GPU 1 同時有 request 的模型最多 3 個，剛好放得下。所以 swap 不頻繁，DRAM swap 的代價也很小。

## 結果資料夾

| 資料夾 | 內容 |
|---|---|
| `results/comparison/` | 對照圖 |
| `results/memory_demand/` | 用 trace 算出的記憶體需求下界（時間 ×0.2）|
| `results/memory_demand_timescale1/` | 同上，時間不縮放 |
| `results/prism_8m2g_noswap_default/` | 實測，不做 swap（完整版：啟動失敗的模型會跳過，然後照樣重播 trace） |
| `results/prism_8m2g_kvcached_default/` | 實測，不做 swap（第一次：第 4 個模型啟動時 OOM，之後就停止了） |
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
CONTINUE_ON_FAIL=1 bash scripts/run_prism_trace.sh kvcached prism_8m2g_noswap_default   # 不做 swap
SWAP=1 bash scripts/run_prism_trace.sh kvcached prism_8m2g_swap_dram_default           # DRAM swap
.venv/bin/python analysis/plot_comparison.py                                            # 對照圖
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
