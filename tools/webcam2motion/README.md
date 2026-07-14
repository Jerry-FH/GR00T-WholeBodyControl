# webcam2motion — 相機 → SMPL → SONIC 即時動作串流

> 系統層級總覽（架構、介面、數據流、需求、限制、未來計畫）見 [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)。

把人體動作（webcam / 影片 / SMPL 資料）轉成 GEAR-SONIC 的 ZMQ protocol v3 串流，
讓 G1 追蹤人的動作。SONIC policy 的 SMPL encoder 內部處理 human→robot retargeting，
這裡只負責產生 `smpl_joints/smpl_pose/body_quat_w/joint_pos(wrists)` 串流。

## 架構

```
webcam / video / pkl
   │  (pose estimator: GVHMR)
   ▼
SMPL params (body_pose 63 + global_orient 3)     ← betas 被丟棄（FK 用標準骨架）
   │  smpl_adapter.py: y-up→z-up, FK, root-local joints, 手腕映射
   ▼
publisher.py: 5 幀滑動視窗 @50Hz, pack_pose_message v3 (HEADER 1280)
   │  ZMQ PUB tcp://*:5556, topic "pose"
   ▼
g1_deploy_onnx_ref --input-type zmq  (low_latency policy)
   │  DDS (domain 0, lo)
   ▼
MuJoCo sim（之後：實機）
```

## 里程碑狀態

- [x] M0.a 合成動作 → ZMQ → sim（`test_m0.py`；主頻驗證 0.268Hz≈0.25Hz、零摔倒）
- [x] M0.b sample SMPL 走路 pkl 回放（`test_m0.py --pkl ...`；30s 零摔倒、手臂擺動 0.303 rad）
- [x] M1 離線影片 → GVHMR → 回放（tennis.mp4：手臂 0.564 rad、30s 零摔倒。
      注意：**不要跑 GVHMR 的 demo.py**（render 需 SMPL pkl+chumpy），用 `estimators/m1_infer.py`，
      推論流程見下方「M1 離線影片管線」）
- [x] M2 即時串流管線（`stream_webcam_zmq.py` + `estimators/gvhmr_runner.py`：
      GVHMR 滑動視窗 causal 估計 7.3fps@137ms（W=32、no-flip、no-postproc——精度不減，MAE 5.1°）；
      假相機端到端 `test_m2.py` PASS（40s 零摔倒）。**真人 webcam 實測待做**：`./run_live.sh --camera 0`）
- [x] M3 一鍵啟動（`docker/run.sh --build` 建 image；`run_live.sh` 起串流；`test_m0/m2.py` 全自動驗證）
- [ ] M4 RealSense / 位移跟隨（研究向）

## 使用

### M0 自動測試（無相機）

```bash
# 需要 g1-deploy-dev 容器在跑（test 會自動 docker start）
.venv_sim/bin/python tools/webcam2motion/test_m0.py                 # 合成動作
.venv_sim/bin/python tools/webcam2motion/test_m0.py \
    --pkl sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl   # 走路資料
```

sample 資料：`python download_from_hf.py --sample`（~4MB）。

### 手動串流（看 MuJoCo 畫面）

```bash
# T1: sim（有畫面）
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py

# T2: deploy —— 必須在 g1-deploy-dev 容器內跑（deploy.sh 會 just build，
#     build cache 是容器路徑 /workspace/g1_deploy，host 上跑會 CMake 報錯）
docker exec -it g1-deploy-dev bash -c 'cd /workspace/g1_deploy && \
  ./deploy.sh --cp policy/low_latency/model \
      --obs-config policy/low_latency/observation_config.yaml \
      --input-type zmq sim'
#    然後 ']' 開始 → sim 按 '9' 放下 → ENTER 切串流（顯示 ZMQ STREAMING MODE: ENABLED）

# T3: 串流（host）
.venv_sim/bin/python tools/webcam2motion/replay_smpl_zmq.py --synthetic --loop
# 或回放影片動作：
.venv_sim/bin/python tools/webcam2motion/replay_smpl_zmq.py \
    --pkl tools/webcam2motion/outputs/tennis/tennis.pkl --src-fps 30 --loop
```

### M1 離線影片管線（webcam2motion 容器）

```bash
# 1) GVHMR 推論（跳過渲染的 driver；~50s/10s 影片，核心推論僅 0.3s）
docker run --rm --gpus all \
  -v ~/GR00T-WholeBodyControl:/workspace/gr00t-wbc:rw \
  -v ~/GR00T-WholeBodyControl/tools/webcam2motion/checkpoints:/opt/GVHMR/inputs/checkpoints:ro \
  -w /opt/GVHMR webcam2motion \
  python /workspace/gr00t-wbc/tools/webcam2motion/estimators/m1_infer.py \
    --video <影片路徑，容器內> -s \
    --output_root /workspace/gr00t-wbc/tools/webcam2motion/outputs

# 2) 轉 pkl
docker run --rm -v ~/GR00T-WholeBodyControl:/workspace/gr00t-wbc:rw -w /workspace/gr00t-wbc \
  webcam2motion python tools/webcam2motion/estimators/gvhmr_offline.py \
    tools/webcam2motion/outputs/<名稱>/hmr4d_results.pt \
    tools/webcam2motion/outputs/<名稱>/<名稱>.pkl

# 3) 回放驗證（host；GVHMR 一律當 30fps，60fps 影片想要原速用 --src-fps 60）
.venv_sim/bin/python tools/webcam2motion/test_m0.py \
    --pkl tools/webcam2motion/outputs/<名稱>/<名稱>.pkl --src-fps 30
```

### Docker（M1+，GVHMR）

```bash
tools/webcam2motion/docker/run.sh --build     # 首次建置
tools/webcam2motion/docker/run.sh             # 進容器
```

模型檔（**需自行註冊下載，不可 commit / 打包進 image**）：
- SMPL-X：https://smpl-x.is.tue.mpg.de → 放 `~/smpl_models/`（bind-mount 到 `/models/smpl`）
- GVHMR checkpoints → `tools/webcam2motion/checkpoints/`（bind-mount 到 `/opt/GVHMR/inputs/checkpoints`）

### M2 即時串流（webcam 或假相機）

```bash
# 全自動端到端測試（headless sim，tennis.mp4 假相機）
.venv_sim/bin/python tools/webcam2motion/test_m2.py

# 真人實測（三終端）：T1 sim、T2 deploy（同上「手動串流」），然後：
tools/webcam2motion/run_live.sh --camera 0            # 筆電 webcam
tools/webcam2motion/run_live.sh --video docs/example_video/tennis.mp4  # 假相機
# deploy 端按 ENTER 開串流。站 ~3m、全身入鏡、橫式。
# 偵測丟失會自動停播（機器人持姿），回到鏡頭前自動恢復。
```

效能（RTX 5080 Laptop、tennis.mp4 基準，fp16 前端＋估計器獨立執行緒後）：
**est 13.3fps（間隔 77ms）、publish 穩定 50Hz**（單幀分解：yolo 1–14ms + kp2d 42ms +
feat 19ms + head 19ms；因果 vs 離線 MAE 5.2°）。
glass-to-motion 估算：smooth 模式 ~270ms、predict 模式 ~190ms（原版 ~390ms）。

### 預覽視窗（骨架疊加＋延遲 HUD）

容器內畫好 bbox＋COCO17 骨架＋各階段耗時 HUD，經 ZMQ :5559 送出 JPEG；host 開視窗：

```bash
.venv_sim/bin/python tools/webcam2motion/preview_viewer.py     # q 離開
```

`run_live.sh` 已預設開 `--preview`。HUD 顯示 `est_age`（畫面從擷取到完成估計的延遲）、
各階段 ms、est/publish fps——延遲來源一目了然。

### 手掌朝向＋RH56DFQ 手指追蹤（`--no-hands` 可關）

COCO17 只有腕「點」、單眼 GVHMR 的腕旋轉幾乎停在中性（掌心朝內）——所以掌心朝上/下/內/外
由 **MediaPipe HandLandmarker** 補：在 ViTPose 腕點旁裁手部 ROI（縮至 224px、每 2 個估計跑一次），
21 個手部關鍵點 → 掌面法向量 → 前臂系腕滾轉，覆蓋 SMPL 那路的 roll（pitch/yaw 仍用 SMPL）。
防護（重要，沒有會摔）：偵測閃爍時淡入 0.4s/淡出 0.6s、腕關節每 tick 增量鉗制 0.06 rad
（`hand_tracker.WristBlender`）。

**Inspire RH56DFQ 手指指令**：同一組關鍵點經角度鏈映射為 6 DOF
（順序＝SDK 暫存器序 `[小指, 無名, 中指, 食指, 拇指彎, 拇指旋]`，0=張開、1=握緊/對掌），
隨 pose 串流以 `left/right_hand_rh56 (6,)` 欄位發布（deploy 忽略；未來 RH56 driver
訂閱同一 topic 取用，硬體端只需做 0..1 → 0-1000 的比例＋方向校正）。

**預覽視窗**（同視窗）：影像上疊手部關鍵點（紫）＋roll 讀數；左右下角各一塊
**手部模擬面板**——正規化掌面視角的手骨架（與相機角度無關）＋RH56 六軸指令條
（`L R M I Tb Tr`），即時看到手部控制器會下的指令。

### 延遲模式（`--latency-mode`，run_live.sh 可透傳）

| 模式 | 行為 | 延遲 |
|---|---|---|
| `smooth`（預設） | 在最近兩次估計間插值，50Hz 平滑 | +1 估計間隔（~87ms） |
| `hold` | 直接持有最新估計（階梯狀） | 最低 |
| `predict` | 等速外插（激烈動作可能過衝） | 最低且平滑 |

## 檔案

| 檔案 | 用途 |
|---|---|
| `publisher.py` | 50Hz 5 幀滑動視窗 v3 發布（用 repo 的 `pack_pose_message`，HEADER=1280） |
| `smpl_adapter.py` | SMPL→串流幀：y-up→z-up、FK（betas 無關）、root-local joints、手腕 euler 映射＋限幅 |
| `replay_smpl_zmq.py` | 合成動作 / pkl 回放器 |
| `test_m0.py` | 全自動 end-to-end 測試（借用 sonic_bench 的 sim server 與 stdin 注入） |
| `docker/` | GVHMR 推論容器（CUDA 12.8, torch cu128, sm_120） |

## 已知細節（坑）

- `pose_estimation_server_onboard_test.py` 的 packer 是舊版（header 1024 ≠ C++ 要的 1280），不要抄。
- 欄位名用 `body_quat_w`（跟 pico server 一致；C++ 兩種名字都收）。
- SMPL 模式（protocol v2/v3 → encode mode 2）下，`g1_debug` 的 `body_q_target` 只反映串流的
  `joint_pos`（幾乎全零），動作意圖走 encoder——驗證追蹤要看 `body_q_measured`。
- `decompose_rotation_aa` 對零旋轉除以零 → adapter 有 epsilon 防護。
- 手腕 euler 合成在極端姿勢會超過 G1 限位 → adapter 限幅 roll ±1.9 / pitch·yaw ±1.6。
- 串流暫停：直接停止發布（機器人持姿）；恢復時 frame_index 跳號會觸發乾淨的 catch-up reset。
- **frame_index 用 epoch 起始**（`publisher.py`）：streamer 重啟後 index 保持前進，不會因為
  倒退 index 被 deploy 端丟棄（症狀：重跑 run_live.sh 後機器人不動）。注意 C++ 端 cast int32，
  所以用 `time.time() % 1e7`。
- **主體鎖定**（`gvhmr_runner._select_subject`）：YOLO 選框用「與上一框 IoU ≥ 0.2」黏住同一人
  ＋框 EMA 平滑，路人走過不搶框（tennis.mp4 驗證：全片框中心最大跳距 27.8px、MAE 2.78°）。
  主體離開畫面 5 幀後 reset，重新以最高信心值鎖定。
- 容器 python stdout 進 pipe 會 block-buffer：自動化腳本要用 `python -u`。
