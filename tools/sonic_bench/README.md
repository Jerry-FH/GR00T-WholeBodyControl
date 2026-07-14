# sonic_bench — release vs low_latency 自動化評估

在 MuJoCo sim2sim 中全自動比較 `policy/release` 與 `policy/low_latency` 兩套 ONNX policy：
每個 trial 都是「全新 headless 模擬器 + 全新容器內部署程序」，腳本注入按鍵（`]` 啟動 →
程式化落地（等同 `9`）→ `n` 導航 → `t` 播放），錄製 ZMQ `g1_debug` 串流與 CSV log，
離線計算延遲 / 追蹤誤差 / 平滑度 / 成功率，最後產出配對統計對比報告。

## 需求

- `g1-deploy-dev` 容器在跑（host network、`gear_sonic_deploy` bind mount 到 `/workspace/g1_deploy`，binary 已編譯）
- host 上有 `.venv_sim`（zmq / msgpack / numpy / scipy）
- 兩套 policy 已下載到 `gear_sonic_deploy/policy/{release,low_latency}/`

## 用法（repo 根目錄）

```bash
# 煙霧測試：1 動作 × 1 trial × 兩變體（約 5–10 分鐘）
.venv_sim/bin/python tools/sonic_bench/run_benchmark.py \
    --variants release,low_latency \
    --motions dance_in_da_party_001__A464 --trials 1

# 完整跑：13 動作 × 3 trials × 兩變體（約 1–2 小時）
.venv_sim/bin/python tools/sonic_bench/run_benchmark.py --motions all --trials 3

# 產出報告
.venv_sim/bin/python tools/sonic_bench/analyze.py \
    --results-root gear_sonic_deploy/logs/sonic_bench
```

報告在 `gear_sonic_deploy/logs/sonic_bench/summary.md`（另有 `trial_metrics.json` 原始數據、
裝了 matplotlib 才會出 `summary_plots.png`）。

## 指標

| 指標 | 意義 | 來源 |
|---|---|---|
| Policy p50/p95 (µs) | 部署堆疊內的 policy 推理延遲 | deploy stdout `Loop timing` 行 |
| Obs→Motor (µs) | 觀測到馬達指令的端到端延遲 | 同上 |
| Loop overruns | 控制迴圈超過 22ms 的次數（50Hz+10%） | `q.csv` 單調時間戳 |
| RMSE legs/waist/arms (rad) | 關節追蹤誤差（測量 vs 目標，僅動作播放區間） | `g1_debug` 串流 × `motion_playing.csv` |
| Base ori err (deg) | base 姿態 geodesic 誤差 | 同上 |
| Δaction RMS | 動作平滑度（越小越平滑） | `last_action` |
| Success rate | 動作完整播完且無跌倒 | stdout 標記 + sim 端跌倒計數 |

注意：`freq_test.txt` 的隔離推理延遲量的是 **CPU EP**（freq_test 的 session options
在建立 session 後才套用，未生效），僅供 ONNX 圖複雜度參考；正式延遲以迴圈內 Policy µs 為準。

## 檔案

- `common.py` — 常數、stdout 標記 regex、關節分組、容器工具
- `sim_server.py` — headless MuJoCo sim + ZMQ REP 控制（`drop`/`state`/`reset`/`quit`；`--onscreen` 可看畫面除錯）
- `deploy_client.py` — 容器內部署程序管理 + stdin 按鍵注入 + stdout 事件解析
- `record_debug_stream.py` — `g1_debug` msgpack → `stream.npz`
- `run_benchmark.py` — 編排器（`--skip-warmup`、`--skip-freqtest`、`--policy-precision` 可調）
- `analyze.py` — 指標 + 配對 t-test/Wilcoxon → `summary.md`

## Trial 生命週期與失敗語意

`trial_meta.json` 的 `status`：`success` / `fell` / `landing_failed` / `timeout` /
`wrong_motion`（事後用 `motion_name.csv` 驗證，會自動重試一次）/ `deploy_died` / `error`。
每變體會先跑一次 `_warmup`（丟棄，容忍 TensorRT 引擎重建最長 12 分鐘）。
teardown 一律在容器內 `pkill` 部署程序（只殺 host 端 `docker exec` 會留孤兒）。
