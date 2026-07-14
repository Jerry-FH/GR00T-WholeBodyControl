# webcam2motion — System Overview

> 單眼相機人體動作 → SMPL → GEAR-SONIC 全身追蹤的即時遙操作系統。
> 使用文件（指令、實測坑）見 [README.md](README.md)；本文件是系統層級的總覽。

---

## 1. 目的（Purpose）

用一般 RGB 相機（筆電 webcam / 影片檔，未來可擴充 RealSense）捕捉人體動作，
即時轉換為 SMPL 姿態參數，經 GEAR-SONIC 官方 ZMQ 串流介面驅動 Unitree G1
做全身動作跟隨——**human→robot retargeting 由 SONIC policy 內建的 SMPL encoder
完成，本系統不做顯式 IK**。MuJoCo sim2sim 驗證先行，與實機共用同一條 motion 路徑。

定位：官方 PICO VR teleop 的「免 VR 硬體」替代輸入源；也可離線把影片轉為動作回放。

## 2. 系統架構（Architecture）

```
┌────────────────────────── webcam2motion 容器 (GPU) ──────────────────────────┐
│  CaptureThread          EstimatorThread                    main thread        │
│  /dev/video0 或影片 ──► GVHMRStreamEstimator ──► filters ──► 50Hz 重採樣/發布  │
│   (最新幀, 30fps)        YOLO→ViTPose→HMR2→GVHMR   OneEuro    (smooth/hold/    │
│                          (滑動視窗 causal)         +quat LP    predict 三模式)  │
│                              │                                    │           │
│                              ▼ 預覽疊加(骨架+HUD)                  ▼           │
│                          ZMQ PUB :5559 "preview"       ZMQ PUB :5556 "pose"   │
└──────────────┬───────────────────────────────────────────────┬───────────────┘
        host: preview_viewer.py                     g1-deploy-dev 容器 (C++)
               (cv2 視窗)                    g1_deploy_onnx_ref --input-type zmq
                                             StreamedMotionMerger → SMPL encoder
                                             (mode 2) → SONIC policy → LowCmd
                                                            │ DDS (domain 0, lo)
                                                            ▼
                                              host: MuJoCo sim (.venv_sim)
                                              run_sim_loop.py / sim_server.py
                                              （實機部署時換成 G1 本體）
```

三個執行環境，全部 `--network host`，經 localhost 通訊：

| 環境 | 內容 | GPU |
|---|---|---|
| `webcam2motion` 容器 | 姿態估計＋串流（本專案） | 需要（CUDA 12.8, sm_120） |
| `g1-deploy-dev` 容器 | SONIC deploy binary（官方） | 需要（TensorRT） |
| host `.venv_sim` | MuJoCo sim、測試 harness、預覽視窗 | 不需要 |

## 3. 模型與技術（Models & Techniques）

### 3.1 管線各階段：做什麼、為什麼需要

每個模型解決前一階段「看不到」的東西——理解這條鏈的關鍵是每一階段的**盲區**：

1. **YOLOv8x 人物偵測**（1–14ms，每 2 幀 1 次）
   *做什麼*：整張畫面 → 人物邊界框。
   *為什麼*：後續所有模型都吃「以人為中心的裁切」，不是整張畫面——裁切讓
   ViT 模型解析度集中在人身上，也是多人場景的入口。偵測不需要每幀跑
   （框移動慢），隔幀重用省 ~10ms。
2. **主體鎖定**（~0ms）
   *做什麼*：多個偵測框中，選「與上一幀 IoU ≥ 0.2」的那個＋EMA 平滑。
   *為什麼*：純「信心值最高」會被路人搶框（實測 1 秒的搶框讓 MAE 從 2.78°
   劣化到 5.24°）。IoU 黏性 = 最便宜的單目標追蹤器。
3. **ViTPose-huge 2D 姿態**（~42ms，fp16、關 flip-test）
   *做什麼*：裁切 → COCO17 個 2D 關鍵點（含信心值）。
   *為什麼*：GVHMR 的觀測之一是 2D 關鍵點（強幾何約束——SMPL 投影回去要對
   得上）；同時腕/肘點供手部 ROI 定位、肩髖點供軀幹尺度。**盲區：只有
   「腕點」沒有手，掌心朝向對它不可觀測**。
4. **MediaPipe HandLandmarker 手部**（~30ms/幀，VIDEO 追蹤模式）
   *做什麼*：以 ViTPose 腕/肘點定位手部 ROI → 21 個手部關鍵點（含相對深度）
   → (a) 掌面法向量 → 前臂系腕滾轉；(b) 角度鏈 → RH56DFQ 6 DOF 手指指令。
   *為什麼*：補上第 3 步的盲區——掌心朝向與手指開合只能從手部特寫獲得。
   VIDEO 模式（偵測一次→追蹤）而非 IMAGE 模式（每幀重偵測）是連續性的關鍵。
5. **HMR2.0 ViT 影像特徵**（~19ms，fp16）
   *做什麼*：同一份裁切 → 1024 維外觀特徵向量。
   *為什麼*：2D 關鍵點丟失了「體型輪廓、肢體遮擋、朝向歧義」等外觀線索；
   GVHMR 需要這份特徵消除單眼歧義（例如手臂在身前還是身後）。
6. **GVHMR motion head**（~19ms @W=32，關 postproc）
   *做什麼*：對最近 32 幀的（2D 關鍵點＋影像特徵＋相機資訊）序列做時序推論
   → 每幀 SMPL `body_pose(63)+global_orient(3)`，取末幀。
   *為什麼*：時序模型讓輸出在時間上一致（單幀模型逐幀抖）；「滑動視窗因果化」
   是我們對離線模型的即時化改造，實測與離線全片結果只差 MAE 2.78°。
   關掉的 postproc（IK 精修＋transl 後處理）實測 80ms 換 0 精度——因為
   transl 我們本來就丟棄。
7. **濾波**（~0ms）：OneEuro（速度自適應——慢動作強濾抖、快動作低延遲）
   於 **FK 之前**作用在 63 維 body_pose；四元數低通處理朝向；手部另有
   獨立 OneEuro＋淡入淡出＋增量鉗制（見 3.2）。
8. **座標轉換＋FK**（~1ms，全部重用 repo 現成函式）
   *做什麼*：y-up→z-up → `compute_human_joints` 前向運動學（標準骨架）→
   去 base rotation → root-local 關節位置。
   *為什麼*：SONIC 的 SMPL encoder 吃的是「去除朝向的局部關節位置＋朝向
   四元數」。**betas（體型）被丟棄**——FK 用固定中性骨架，這使單眼的
   深度/尺度誤差無法污染輸出（只有旋轉誤差要緊），是單眼方案成立的核心。
9. **手腕映射**（~0ms）：SMPL 肘旋轉做 twist/swing 分解——twist（肘屈伸）
   留在肘，swing（前臂旋前）併進腕 → euler 合成 → G1 六腕關節＋限幅。
   MediaPipe 的腕滾轉在此覆蓋 SMPL 值（見 3.2 防護）。
10. **SONIC policy**（deploy 端）：SMPL encoder（mode 2）把串流目標編碼為
    latent，decoder 以 50Hz 輸出 29 關節指令並自行維持平衡——
    **human→robot retargeting 就發生在這裡**，本系統不做顯式 IK。

| 元件 | 模型 | 單幀耗時* |
|---|---|---|
| 人物偵測 | YOLOv8x | 1–14ms |
| 2D 姿態 | ViTPose-huge（fp16） | ~42ms |
| 手部（預設） | MediaPipe HandLandmarker（VIDEO 模式，**CPU**） | ~30ms |
| 手部（`--hand-backend wilor`） | WiLoR（wilor-mini，MANO 迴歸，**GPU fp16**） | ~25–40ms |
| 影像特徵 | HMR2.0 ViT（fp16） | ~19ms |
| 動作估計 | GVHMR 滑動視窗（W=32、no-postproc） | ~19ms |

手部後端二選一（`estimators/hand_tracker.py` vs `estimators/wilor_tracker.py`，
同一 `track()` 介面）：MediaPipe 跑 CPU（XNNPACK）、輕但與 sim/deploy 搶核心且
靠幾何推導；WiLoR 直接迴歸 MANO 參數（腕旋轉與指彎更穩、抗模糊/遮擋），跑 GPU
釋放 CPU。WiLoR 沿用我們的 ViTPose 腕部 ROI（`predict_with_bboxes` 跳過其內建
YOLO 偵測器），輸出 21 點順序 = OpenPose = MediaPipe，下游（rh56 映射、
WristBlender、預覽面板）零改動。權重首跑自動下載到
`checkpoints/wilor/`（官方 `MANO_RIGHT.pkl` 已預先放置，不會抓鏡像副本）。

\* RTX 5080 Laptop 實測；估計器整體 ~8–11fps（依手部開關）、發布穩定 50Hz、
因果 vs 離線 MAE 2.78°（tennis.mp4、主體鎖定後）。

### 3.2 手部鏈的穩定性防護（為什麼需要）

手部偵測天生會閃爍（遮擋、握拳、動態模糊）。防護三層，缺一實測會摔
（tennis 迴歸曾因硬切換 26 falls）：
- **VIDEO 追蹤模式**：偵測一次→逐幀追蹤（tennis 偵測率 20/42% → 34/73%）
- **hold-not-fade**（`WristBlender`）：掉偵測時**維持最後手部值** 3 秒再慢淡出
  ——因為 SMPL 的腕值是「錯誤答案」（中性、掌心朝內），回退等於 glitch
- **增量鉗制**：腕關節每 tick ≤0.06 rad（3 rad/s），削掉任何殘餘尖峰

## 4. 子系統與檔案（Packages / Sub-systems）

```
tools/webcam2motion/
├── stream_webcam_zmq.py     即時串流主程式（3 執行緒：擷取/估計/發布）
├── estimators/
│   ├── gvhmr_runner.py      GVHMRStreamEstimator：滑動視窗 causal 即時估計
│   ├── m1_infer.py          離線影片推論 driver（跳過 GVHMR demo 的渲染）
│   ├── gvhmr_offline.py     hmr4d_results.pt → 回放 pkl 轉換
│   └── bench_stream.py      估計器效能/精度基準（vs 離線結果）
├── smpl_adapter.py          SMPL→串流幀（FK、root-local、手腕映射＋限幅）
├── publisher.py             50Hz 5 幀滑動視窗 protocol-v3 發布（epoch frame_index）
├── filters.py               OneEuro / QuatLowpass / DeltaClamp
├── replay_smpl_zmq.py       合成動作、pkl 回放器（M0/M1 驗證）
├── preview_viewer.py        host 端預覽視窗（SUB :5559）
├── hand_viewer/             MuJoCo RH56DFQ 雙手檢視器（SUB :5556 rh56 欄位、
│                            unitree_ros DFQ URDF+meshes、mimic 比例 qpos 展開）
├── test_m0.py / test_m2.py  全自動端到端測試（借用 tools/sonic_bench harness）
├── run_live.sh              一鍵起即時串流容器
├── docker/                  Dockerfile（CUDA 12.8.1 + torch cu128 + GVHMR + pytorch3d CPU）
└── checkpoints/             GVHMR/YOLO/ViTPose/HMR2 權重＋SMPL 模型檔（bind-mount，不進 git）
```

相依的外部子系統：
- `gear_sonic/`（官方 Python）：`pack_pose_message`（唯一正確 packer，HEADER=1280）、
  `process_smpl_joints` 座標鏈、`compute_human_joints` FK、MuJoCo sim
- `gear_sonic_deploy/`（官方 C++）：deploy binary、`StreamedMotionMerger`、SMPL encoder、policy
- `tools/sonic_bench/`（自建）：headless sim server、deploy stdin 控制、g1_debug 錄製與分析

## 5. 介面一覽（Topics / Services 對照）

ZMQ（皆走 localhost, host network）：

| 埠 | 模式 | Topic | 內容 | 發布者 → 訂閱者 |
|---|---|---|---|---|
| 5556 | PUB/SUB | `pose` | protocol v3：`smpl_pose(N,21,3)`+`smpl_joints(N,24,3)`+`body_quat_w(N,4)`+`joint_pos/vel(N,29)`+`frame_index(N)`＋可選 `left/right_hand_joints(7)`（Dex3）、`left/right_hand_rh56(6)`（Inspire，未來 hand driver 訂閱） | publisher.py → deploy binary（未知欄位忽略） |
| 5557 | PUB/SUB | `g1_debug` / `robot_config` | msgpack：measured/target 關節、base quat 等（50Hz） | deploy binary → 測試/分析 |
| 5559 | PUB/SUB | `preview` | JPEG（骨架+延遲 HUD 疊加） | EstimatorThread → preview_viewer.py |
| 5560 | REQ/REP | — | sim 控制 JSON（`drop`/`state`/`reset`/`quit`） | 測試 harness → sim_server.py |

其他介面：
- **DDS**（CycloneDDS/FastRTPS, domain 0, 介面 `lo`）：deploy ↔ sim/實機的 `LowState`/`LowCmd`
- **deploy 鍵盤（stdin）**：`]` 開控制、`ENTER` 切串流模式、`O` 急停、`Q/E` 手動 heading 微調
- 「service」類比：sim REP 通道；「action」類比：deploy 的鍵盤狀態機（無 ROS action）

### 訊息格式（wire format）

`[topic bytes][1280-byte JSON header][little-endian binary fields]`。
header 描述欄位名/dtype/shape。**勿用 `pose_estimation_server_onboard_test.py` 的舊 packer
（header 1024，C++ 端無法解析）**；一律用 `gear_sonic/utils/teleop/zmq/zmq_planner_sender.pack_pose_message`。

## 6. 數據流（Data Flow）

```
相機幀 (BGR, ~30fps)
 → [EstimatorThread] YOLO 選框（IoU 鎖定主體）→ 256×256 裁切
 → ViTPose kp2d(17,3) ─┬─ HMR2 特徵(1024)        ← 共用同一裁切、fp16
 │                     └─ 手部 ROI（腕/肘點定位）→ MediaPipe 21 關鍵點
 │                          → 掌面法向量→腕滾轉 ＋ 角度鏈→RH56 6DOF
 → 環形緩衝（32 幀）→ GVHMR motion head → smpl_params_global 取末幀
 → OneEuro(body_pose) + 四元數低通(orient) + 手部 OneEuro   【估計率 8–11fps】
 → [main] 50Hz tick：兩次估計間插值（smooth）/持有（hold）/外插（predict）
 → smpl_adapter：y-up→z-up、FK→root-local smpl_joints、腕映射→joint_pos[23..28]
 → WristBlender：MediaPipe 腕滾轉覆蓋（淡入/hold 3s/慢淡出＋增量鉗制）
 → publisher：5 幀滑動視窗、epoch frame_index、protocol v3
     ＋ left/right_hand_rh56(6) 欄位 → :5556
 ├→ [deploy] StreamedMotionMerger（去重、catch-up）→ SMPL encoder（mode 2）
 │   → SONIC policy（50Hz，lookahead 4 幀 clamp 到最新）→ LowCmd → DDS
 │   → [sim/實機] 關節執行；g1_debug 回流量測      （rh56 欄位被 deploy 忽略）
 └→ [hand_viewer] 訂閱同一 topic 讀 rh56 → RH56DFQ URDF qpos（mimic 比例展開）
     → MuJoCo 檢視窗（未來：同一訂閱換成 Inspire SDK = 硬體 driver）
```

延遲預算（實測分解）：相機 ~33ms ＋ 估計 77ms ＋ 插值 holdback 0–77ms（依模式）
＋ policy lookahead 80ms ＋ 控制 ≈ **190–270ms glass-to-motion**。

## 7. 輸入 / 輸出（I/O）

**輸入**：UVC 相機（`/dev/video*`）或影片檔（假相機，原生 fps 循環播放）；
單人、全身入鏡效果最佳（~3m、橫式）。

**輸出**：
- `pose` ZMQ 串流（上表）→ 驅動機器人
- `preview` JPEG 串流 → 效果確認/延遲定位
- 離線模式：`outputs/<名稱>/hmr4d_results.pt` 與回放 pkl

**目前追蹤的自由度**（皆經 sim 端到端驗證）：
- 全身姿勢（SMPL 21 關節，MAE 2.78° vs 離線）
- **身體朝向（heading）**：轉身 ±28.6° 指令 → 機器人 yaw 實掃 48.1°（跟隨率 ~84%）
- **六個腕關節**（roll/pitch/yaw × 2）：0.3Hz 腕部翻轉指令 → 實測 0.31Hz、0.285 rad
- **掌心朝向（腕滾轉/pronation）**：COCO17/單眼 SMPL 看不到這軸——由 MediaPipe 手部
  關鍵點的掌面法向量補上（淡入淡出＋增量鉗制防護，偵測不到時回退 SMPL 值）

未追蹤：手指（協定欄位與手部關鍵點都已就緒，缺映射，見未來計畫）、
全域位移（root translation，SONIC v3 SMPL 路徑不吃）。

## 8. 配置（Configuration）

`stream_webcam_zmq.py` 參數（`run_live.sh` 可透傳）：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--camera N` / `--video PATH` | camera 0 | 輸入源 |
| `--latency-mode` | `smooth` | `smooth`（插值）/`hold`（持有）/`predict`（外插） |
| `--preview` / `--preview-port` | off / 5559 | 預覽串流 |
| `--yolo-period` | 2 | 每 N 幀跑一次偵測 |
| `--no-fp16` | — | 關 fp16（除錯用） |
| `--min-cutoff` / `--beta` | 1.0 / 0.1 | OneEuro 參數（調平滑/延遲取捨） |

估計器內部（`gvhmr_runner.py` 建構參數）：`window=32`（視窗）、`min_window=16`（暖機）、
`flip_test=False`、`postproc=False`（重要：開了慢 80ms 且精度不變）。
deploy 端：policy 選 `policy/low_latency/`（lookahead 4 幀；release 是 10 幀更 lag）。

## 9. 需求（Requirements）

- **硬體**：NVIDIA GPU（sm_120/Blackwell 需 CUDA ≥12.8 與 cu128+ wheels；實測 RTX 5080 Laptop 16GB）、UVC 相機
- **Host**：docker + NVIDIA container toolkit；`.venv_sim`（uv, py3.10, torch cu130, mujoco, pyzmq）；TensorRT（`~/TensorRT`，deploy 容器 bind-mount）
- **模型檔（需自行註冊下載，不可 commit/打包）**：
  - SMPL-X：https://smpl-x.is.tue.mpg.de → `checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`（推論必需）
  - SMPL pkl 只有 GVHMR 官方渲染用得到（本系統跳過渲染，不需要）
- **checkpoints**（HF 可自動下載）：GVHMR ckpt、ViTPose-h、YOLOv8x、HMR2
- **記憶體**：建 image 時 pytorch3d 編譯 `MAX_JOBS=4`（開高會 OOM 殺桌面 session）

## 10. 啟動程序（Startup）

```bash
# 0) 首次：建 image（~20 分鐘）
tools/webcam2motion/docker/run.sh --build

# 1) deploy 容器（長駐；--rm 重開機後要重跑）
docker run -d --rm --name g1-deploy-dev --network host --ipc host --gpus all \
  -v ~/GR00T-WholeBodyControl/gear_sonic_deploy:/workspace/g1_deploy:rw \
  -v ~/GR00T-WholeBodyControl/gear_sonic:/workspace/gear_sonic:rw \
  -v ~/TensorRT:/opt/TensorRT:ro \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e ROS_DOMAIN_ID=0 \
  -w /workspace/g1_deploy g1-deploy-dev sleep infinity

# 2) T1: MuJoCo sim（host）
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py

# 3) T2: deploy（容器內；host 跑會 CMake 錯）
docker exec -it g1-deploy-dev bash -c 'cd /workspace/g1_deploy && \
  ./deploy.sh --cp policy/low_latency/model \
      --obs-config policy/low_latency/observation_config.yaml --input-type zmq sim'

# 4) T3: 即時串流（模型載入 ~40s）
tools/webcam2motion/run_live.sh --camera 0

# 5) T4: 預覽（host，可選）
.venv_sim/bin/python tools/webcam2motion/preview_viewer.py

# 5b) T5: MuJoCo 手部檢視器（host，可選；--demo 不需串流可先看）
.venv_sim/bin/python tools/webcam2motion/hand_viewer/hand_viewer.py

# 6) 操作順序：T2 按 ']' → sim 按 '9' 放下 → T2 按 ENTER（串流開）
# 全自動驗證（免手動）：
.venv_sim/bin/python tools/webcam2motion/test_m2.py            # 假相機端到端
.venv_sim/bin/python tools/webcam2motion/test_m0.py --synthetic-motion wrist_heading
```

## 11. 相關專案（Related Projects）

| 專案 | 關係 |
|---|---|
| [GR00T-WholeBodyControl / GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) | 宿主 repo；deploy binary、SMPL encoder policy、ZMQ 協定（`docs/source/tutorials/zmq.md`） |
| [GVHMR](https://github.com/zju3dv/GVHMR) | 姿態估計核心（容器內 `/opt/GVHMR`） |
| NVIDIA GEM / GEM-X | 官方離線 video-to-motion（內用 GVHMR）；本系統是其即時化對應物 |
| `tools/sonic_bench` | 自建 benchmark harness；本系統測試借用其 sim server/deploy 控制/錄製 |
| [GMR](https://github.com/YanjieZe/GMR) | 備案：CPU 即時 retarget → protocol v1（joint 路徑） |
| PICO teleop（`gear_sonic/scripts/pico_manager_thread_server.py`） | 官方 VR 串流參考實作；本系統的協定/座標鏈範本 |

## 12. 已知限制與問題（Known Limitations）

- **無位移跟隨**：SONIC v3 SMPL 路徑只吃朝向＋root-local 關節；人走動只呈現腿部動作，
  機器人原地追蹤。真位移要走 `heading_increment` 或 planner（`zmq_manager`）。
- **延遲 ~190–270ms**：估計器 77ms＋policy lookahead 80ms（固定）＋插值模式。
  `predict` 模式最低但快動作會過衝。
- **單眼歧義**：深度/尺度不影響（FK 用標準骨架），但腕部旋轉、與相機平行的動作
  仍是單眼弱項；側身/遮擋時品質下降（信心門檻會停播持姿）。
- **單人假設**：主體鎖定黏一人；主體離開 5 幀後重鎖畫面中最高信心者。
- 影片假相機在循環邊界會有 ~1s 重置（偵測跳變 → 暖機）。
- policy lookahead clamp（未來幀=最新幀）＝理論上比訓練分布保守，激烈動作跟不緊。
- deploy 容器 `--rm`：重開機後消失，要重建（見啟動程序 1）。
- GVHMR 一律當 30fps 處理；高 fps 輸入的動力學會被略微平滑。
- **DDS domain 0 只能有一個 simulator**：`run_sim_loop.py` / `run_sim_rh56.py` /
  sonic_bench sim_server 都在 domain 0 發布 `rt/lowstate`——同時開兩個，deploy 的
  觀測會被交錯污染，機器人「無緣無故」狂跌（2026-07-14 整晚教訓）。deploy 端
  domain 寫死 0（`g1_deploy_onnx_ref.cpp` `ChannelFactory::Init(0,...)`）無法隔離；
  `test_m0/test_m2` 啟動時會偵測並直接報錯（`assert_dds_clear`）。

## 13. 未來計畫（Future Work）

1. **靈巧手（Inspire RH56DFQ）**：視覺側已完成——MediaPipe 手部關鍵點 → 6 DOF 指令
   （`rh56_from_landmarks`，順序 `[小指,無名,中,食,拇彎,拇旋]`、0..1 正規化）隨 pose 串流
   發布（`left/right_hand_rh56`）、預覽有模擬面板。**缺硬體 driver**：訂閱 :5556 pose topic
   讀 rh56 欄位 → Inspire SDK（Modbus/串列，0-1000 range）＋每指方向/行程校正。
   （Dex3 的 `left/right_hand_joints(7)` passthrough 也保留著。）
2. **RealSense**：metric root translation（位移跟隨的前置）、深度輔助主體鎖定。
3. **位移/走路跟隨**：`heading_increment` 或 planner 模式整合。
4. **更快前端**：ViTPose TensorRT 化或換小模型（目前 42ms 是最大單項）。
5. **實機部署**：sim2sim 已通，同一條 motion 路徑；上實機前建議 30 分鐘無摔倒 soak test。
6. 多人場景指定主體（點擊預覽選人）。
