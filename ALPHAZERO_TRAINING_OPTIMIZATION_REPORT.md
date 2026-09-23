# AlphaZero Training Optimization Report

Ngày audit: 2026-09-23  
Máy đo: NVIDIA GeForce RTX 5060 8 GB, PyTorch 2.11.0+cu128, Windows 10.

## 1. Kết luận ngắn

Training không chậm vì model hoặc tensor vô tình chạy trên CPU. Model và tensor training đều ở CUDA. Bottleneck thực tế là self-play/search bằng Python trên CPU: trong run đã hoàn tất, self-play chiếm **99,274%** thời gian được profile, còn toàn bộ forward/backward/optimizer chỉ chiếm **0,717%**. Vì mạng rất nhỏ và mỗi lần inference chỉ có khoảng 1–2 state, GPU mạnh vẫn phải chờ CPU.

Training kết thúc sau 5 iteration vì runner duyệt một `SCHEDULE` hardcode đúng 5 phần tử. Trong từng iteration, điều kiện reward cũng dừng sớm khi mean reward của 1.000 game gần nhất đạt 0,1 sau hơn 1.000 game. Checkpoint production hiện tại đã có `complete=True`, `current_iteration=5`; code cũ vì vậy trả về ngay khi resume.

Nguyên nhân chất lượng quan trọng hơn tốc độ: implementation này là **policy-only REINFORCE + root prior + random rollout**, không có value head, replay buffer hoặc tree MCTS/PUCT kiểu AlphaZero chuẩn. Vì vậy tên `AlphaZeroAgent` không đồng nghĩa với thuật toán AlphaZero chuẩn. Benchmark hiện tại xác nhận model thắng Random 99/100 nhưng thua RuleBased 78/100, Minimax 55/100 và AlphaBeta 58/100.

Patch giữ nguyên luật Connect Four và interface 8 agent. Nó mở rộng training thay vì rewrite thuật toán: profile DEBUG/QUICK/FULL rõ ràng, continuation sau iteration 5, replay buffer có checkpoint, nhiều optimizer update có kiểm soát, evaluation holdout theo cadence, profiler chi tiết và benchmark trước/sau tái lập được.

## 2. Flow thực tế trước patch

```text
iteration (5 stage hardcode)
  -> chụp frozen opponent từ learner đầu iteration
  -> sinh từng batch 20 self-play game
       learner: sample policy
       opponent: 1 policy prior ở root + N random rollout
  -> gom trajectory của đúng batch hiện tại
  -> 1 FP32 forward + backward + optimizer.step()
  -> cập nhật running mean reward trên tối đa 1.000 training game
  -> dừng iteration nếu mean >= 0,1 sau >1.000 game
     hoặc nếu game >25.000
  -> lưu checkpoint iteration
  -> iteration kế tiếp
```

Đây không có dataset epoch theo nghĩa thông thường. Cột `epochs` cũ trong CSV thực chất là **batch count = optimizer step count**.

## 3. Training thực tế đã chạy bao nhiêu

Nguồn: `results/training_iterations.csv` và checkpoint production hiện tại.

| Iteration | Games | Positions | Batches / optimizer steps | Stop |
|---:|---:|---:|---:|---|
| 0 | 23.280 | 194.047 | 1.164 | running reward |
| 1 | 14.360 | 121.222 | 718 | running reward |
| 2 | 3.200 | 23.102 | 160 | running reward |
| 3 | 6.960 | 45.967 | 348 | running reward |
| 4 | 4.200 | 50.235 | 210 | running reward |
| **Tổng** | **52.000** | **434.573** | **2.600** | schedule hết ở iteration 5 |

- Self-play game/batch: 20.
- Position trung bình/update: 167,14.
- Optimizer update trước patch: 1/batch; tổng 2.600.
- Epoch thực: không áp dụng; dữ liệu không được lặp theo epoch.
- Replay trước patch: không có; mỗi position chỉ tham gia đúng một update.
- `max_steps=50` là số lượt learner tối đa trong một episode, không phải số optimizer step.

## 4. Toàn bộ stopping condition đã tìm thấy

### Runner production cũ

- Số iteration: `range(current_iteration, len(SCHEDULE))`, với `len(SCHEDULE) == 5`.
- Dừng iteration do reward: `running_reward >= 0.1 and episodes > 1000`.
- Safety cap: `episodes > 25000` (có thể chạy tới 25.020 vì kiểm tra sau batch 20).
- `--stop-after-batches`: đường dừng có chủ đích dành cho smoke/resume test.
- Checkpoint `complete=True`: resume cũ trả `already_complete` ngay.
- Không có patience, max wall time, target win rate độc lập hay exception bị catch rồi âm thầm thoát. Worker exception được chuyển về parent và raise `RuntimeError`.
- `max_steps=50` chỉ kết thúc episode không hoàn tất; không dừng toàn run.
- Checkpoint/evaluation không tự dừng ngoài các điều kiện trên.

### Script `train_alphazero.py` cũ

Script tương thích notebook cũng dùng schedule 5 phần tử, reward target 0,1 sau 1.000 game và cap 25.000 game. Validation mode là run ngắn có chủ đích, không phải FULL_TRAIN.

## 5. Bottleneck GPU đã chứng minh

Tổng run 52.000 game:

| Pha | Thời gian | % tổng profile |
|---|---:|---:|
| Self-play/search | 2.968,982 s | **99,274%** |
| Neural training | 21,445 s | **0,717%** |
| Stop/evaluation bookkeeping | 0,278 s | **0,009%** |

Thông lượng tổng: 17,39 game/s và 145,31 position/s. GPU monitor có 3.218 mẫu full-training, utilization trung bình **3,68%**, peak VRAM chỉ **289 MiB**. Theo từng iteration, GPU mean nằm trong khoảng 2,16–4,82%.

Nguyên nhân:

1. Opponent search thực hiện hàng chục/hàng trăm random rollout Python cho mỗi nước đi.
2. Neural network chỉ được gọi một lần ở root, không gọi cho từng MCTS node. Do đó giả thuyết “network(state) cho từng node” **không đúng với code này**.
3. Worker gửi một state rồi block chờ response; tối đa chỉ có số request outstanding bằng số worker. Live inference batch trước đây khoảng 1,6 dù GPU hỗ trợ batch lớn.
4. Model rất nhỏ; một update vài trăm position hoàn tất trong mili-giây. Tăng batch/AMP không thể bù phần CPU search chiếm 99%.
5. Không có process nào copy model CUDA riêng. Kiến trúc hiện tại đúng hướng: nhiều CPU worker -> một inference queue -> một CUDA service.

Không nên coi GPU 100% là mục tiêu. Benchmark mới tăng throughput đáng kể trong khi GPU utilization gần như không đổi.

## 6. MCTS/search audit

`AlphaZeroAgent` không xây tree và không evaluate từng node bằng network. Mỗi action:

1. Chạy policy network đúng một lần trên root.
2. Chọn root move bằng prior + empirical rollout value.
3. Mô phỏng phần còn lại hoàn toàn ngẫu nhiên trên CPU.
4. Ghi kết quả theo root move.

Vì các rollout không cần neural inference, batching “node inference” kiểu AlphaZero chuẩn không áp dụng nếu không thay đổi thuật toán. Patch không đổi logic này. Thay vào đó, nó giữ inference queue cho các root state độc lập từ nhiều game, đo batch thực theo từng model forward (không còn nhầm queue batch trộn learner/opponent với CUDA batch), và đo riêng MCTS CPU time/inference time.

## 7. Replay buffer và training update

Trước patch:

- Buffer chỉ là trajectory của batch 20 game hiện tại.
- Dữ liệu bị bỏ ngay sau một update.
- `training_updates_per_position = 1,0` theo số lượt position được đưa qua loss.

Sau patch:

- FULL_TRAIN giữ tối đa 5.000 game trajectory trên CPU.
- Update đầu luôn dùng batch mới; update thứ hai sample ngẫu nhiên tối đa 20 game từ replay.
- Mặc định FULL_TRAIN có 2 optimizer step/batch, nên tỷ lệ position trained/generated mục tiêu xấp xỉ 2,0 mà không tăng mù quáng.
- Replay buffer, tổng game, tổng position và tổng optimizer step được lưu trong state checkpoint format v2.
- Checkpoint v1 hiện có vẫn load được; global counters được khôi phục từ CSV cũ khi cần.

Không có value loss vì model không có value head. Log `loss` là policy REINFORCE objective; `value_loss` là N/A.

## 8. Patch đã thực hiện

### `connect4_ai/training/run_full_alphazero.py`

- Thêm profile `debug`, `quick_test`, `full_train`.
- FULL_TRAIN mặc định 20 iteration; sau 5 stage gốc sẽ lặp stage cuối `(weight=0,9, rollouts=250)` với opponent snapshot mới mỗi iteration.
- Thêm `max_iterations`, `min_games_per_iteration`, `self_play_games_per_iteration`, `max_total_games`, `max_optimizer_steps`.
- Sửa cap game theo `>=` và batch cuối có thể nhỏ hơn 20 để không vượt cap.
- Cho phép checkpoint `complete=True` ở iteration 5 tiếp tục nếu cấu hình mới yêu cầu nhiều iteration hơn.
- Khôi phục RNG cả khi resume ở ranh giới iteration (code cũ chỉ restore giữa iteration).
- Thêm replay buffer + nhiều update/batch.
- Thay stopping reward lấy trực tiếp từ training batch bằng holdout evaluation định kỳ, cân bằng red/yellow.
- Thêm JSONL profiler, TensorBoard metrics và checkpoint counters.
- Giữ append compatibility với CSV production cũ; run mới nhận schema mở rộng.

### `connect4_ai/training/parallel_self_play.py`

- Đo data prep, H2D, forward, response, actual model batch size.
- Đo riêng forward/backward/optimizer của update bằng CUDA events.
- Giữ một CUDA service duy nhất; không duplicate model trên worker.

### `connect4_ai/training/self_play_worker.py`

- Đo MCTS/random-rollout CPU time, thời gian chờ inference và số request trên từng trajectory.

### `connect4_ai/training/train_alphazero.py`

- Thêm cùng instrumentation cho đường self-play tuần tự.

### `connect4_ai/agents/alphazero.py`

- Thêm `policy_inference_time` và `rollout_time`; không đổi constructor hay `select_action` API.

### Benchmark và test

- Thêm `benchmark/training_pipeline_benchmark.py`.
- Bổ sung test phase instrumentation trong `tests/test_alphazero.py` và `tests/test_alphazero_training.py`.

## 9. Profiler và metrics mới

`results/training_profile.jsonl` ghi từng batch với:

- SELF PLAY wall time và % wall time;
- MCTS CPU aggregate time;
- NN INFERENCE;
- DATA PREP;
- H2D;
- FORWARD;
- BACKWARD;
- OPTIMIZER;
- EVALUATION;
- SAVE dưới dạng checkpoint event;
- GPU memory allocated/reserved;
- game, generated/trained position, optimizer step;
- inference request và average actual model batch;
- games/s, positions/s và optimizer steps/s qua TensorBoard/live monitor.

Lưu ý: MCTS CPU time là tổng CPU-seconds của nhiều worker nên có thể lớn hơn wall time; chỉ ba pha top-level self-play/training/evaluation cộng thành 100% wall time.

## 10. Before/after benchmark sau patch

Command:

```powershell
python benchmark/training_pipeline_benchmark.py --games 40 --rollouts 100 --workers 8 --inference-batch-size 2 --updates-per-batch 2
```

Workload hai phía cùng seed, model, 40 game, batch 20, opponent weight 0,3, 100 rollout và FP32 loss. Nhánh before mô phỏng pipeline cũ: tuần tự + 1 update/batch. Nhánh after: persistent workers + single CUDA service + 2 update/batch có replay. Startup/shutdown worker được tính vào after.

| Metric | Before | After | Speedup |
|---|---:|---:|---:|
| Games/s | 6,60 | 11,94 | **1,81x** |
| Generated positions/s | 56,89 | 90,15 | **1,58x** |
| Trained positions/s | 56,89 | 186,27 | **3,27x** |
| Optimizer steps | 2 | 4 | 2,00x work |
| Actual inference batch | 1,00 | 1,19 | 1,19x |
| Average GPU utilization | 0,92% | 1,00% | gần như không đổi |
| Peak VRAM | 227 MiB | 250 MiB | +23 MiB |

Số position giữa hai nhánh khác nhẹ vì self-play stochastic/multiprocessing cho chuỗi random khác, dù cùng phân phối và seed task. Vì vậy games/s là so sánh end-to-end chính; trained positions/s thể hiện cả throughput lẫn lượng training work tăng thêm.

Artifact: `results/audit_before_after.json` và `results/audit_benchmark_gpu_metrics.csv`.

Kết quả quan trọng: throughput tăng mà GPU vẫn khoảng 1%. Điều này xác nhận không nên tối ưu bằng cách ép GPU utilization lên 100%; random rollout CPU vẫn là giới hạn.

## 11. Evaluation và lý do model chưa vượt heuristic

Benchmark 100 game/opponent hiện tại:

| Opponent | Win | Loss | Draw |
|---|---:|---:|---:|
| Random | 99% | 1% | 0% |
| RuleBasedThink3 | 20% | 78% | 2% |
| DepthMinimax | 40% | 55% | 5% |
| AlphaBeta | 41% | 58% | 1% |
| PositionEval | 19% | 81% | 0% |
| MCTS | 34% | 66% | 0% |
| AlphaGo | 35% | 63% | 2% |

Các nguyên nhân theo mức ảnh hưởng:

- **CRITICAL:** objective chỉ học policy từ reward game thưa, không có value head/value loss.
- **CRITICAL:** trước patch dữ liệu dùng một lần rồi bỏ; chỉ 2.600 update cho 52.000 game.
- **HIGH:** opponent curriculum là snapshot cùng họ model và random rollout, không trực tiếp buộc model học tactical defense mà heuristic/minimax khai thác.
- **HIGH:** điều kiện dừng cũ dùng chính game training, nên noisy/in-sample và đạt 0,1 khá sớm.
- **MEDIUM:** illegal action vẫn được sample rồi phạt -1 theo notebook gốc; cách này tốn sample so với legal mask nhưng patch chưa đổi để giữ semantics.
- **LOW:** GPU thấp không phải nguyên nhân trực tiếp làm policy yếu; nó chủ yếu làm chậm tốc độ sinh data.

Replay + nhiều update + holdout evaluation giải quyết ba điểm đầu ở mức ít xâm lấn. Tuy nhiên không thể bảo đảm đạt sức mạnh AlphaZero chuẩn nếu vẫn giữ policy-only/random-rollout architecture. Bước lớn tiếp theo, chỉ nên làm sau khi đo quality của patch này, là thêm value head + PUCT/tree reuse; đó là thay đổi thuật toán đáng kể và nằm ngoài patch tối thiểu hiện tại.

## 12. FULL_TRAIN đề xuất

Mặc định mới:

| Setting | Value |
|---|---:|
| Iterations | 20 |
| Min self-play/iteration | 5.000 |
| Cap self-play/iteration | 25.000 |
| Batch | 20 game |
| Replay | 5.000 game |
| Updates/batch | 2 |
| Replay sample/update | 20 game |
| Evaluation | 40 holdout game mỗi 50 batch |
| Target mean reward | 0,1 |
| Workers | 8 |
| Inference queue batch cap | 2 |
| Precision | FP32 |

Để tiếp tục lineage hiện tại nhưng đặt safety cap hữu hạn dựa trên số đo đã có:

```powershell
python connect4_ai/training/run_full_alphazero.py --resume --profile full_train `
  --max-total-games 150000 --max-optimizer-steps 12000
```

Checkpoint v1 hiện tại có 52.000 game và 2.600 update; runner sẽ khôi phục các tổng này từ CSV rồi tiếp tục ở iteration 5. Với minimum 5.000 game cho 15 iteration còn lại, kế hoạch tối thiểu thêm khoảng 75.000 game; cap 150.000 tổng và 12.000 update còn đủ headroom nhưng vẫn hữu hạn. Nên benchmark quality ở mỗi checkpoint thay vì mặc định dùng hết cap.

Các profile ngắn:

```powershell
# Smoke rất ngắn
python connect4_ai/training/run_full_alphazero.py --no-resume --profile debug <các output path riêng>

# Validation ngắn
python connect4_ai/training/run_full_alphazero.py --no-resume --profile quick_test <các output path riêng>
```

Không dùng `--no-resume` với đường dẫn production nếu muốn giữ lineage.

## 13. Resume/checkpoint

State v2 lưu:

- learner model và frozen opponent model;
- Adam optimizer state;
- iteration, in-iteration game/batch;
- total game, position, optimizer step;
- replay buffer;
- reward window, history và profiler accumulators;
- Python/NumPy/Torch CPU/Torch CUDA/policy sampler/opponent RNG state;
- config đầy đủ.

Project không có scheduler nên không có scheduler state để lưu. Save dùng temporary file + atomic replace. Smoke đã xác nhận resume từ 20 lên 40 game, giữ replay và optimizer state, sau đó hoàn tất đúng iteration.

## 14. Verification

- `python -m unittest discover -s tests -q`: **91/91 test pass** trong 8,333 s.
- CUDA smoke: 40 game, 2 update, checkpoint v2, replay 40 game, resume thành công.
- Bản sao checkpoint production v1 (`complete=True`, iteration 5) đã tiếp tục đúng ở iteration 5; sau một batch state v2 ghi 52.020 total game, 435.063 total position và 2.602 optimizer step. File production gốc không bị sửa.
- Sample profile: self-play 92,1%, training 7,9%, GPU allocated ~44,7 MiB, reserved 64 MiB.
- Before/after benchmark: 1,81x games/s và 3,27x trained positions/s.
- Production checkpoint/model không bị ghi đè; smoke nằm trong `archive/validation/audit_smoke/`.

## 15. Hạn chế còn lại

- Đây vẫn không phải canonical AlphaZero; chưa có value network, PUCT, full tree, Dirichlet noise hay tree reuse.
- Replay trajectory-level là cải tiến tối thiểu cho REINFORCE hiện tại, không phải `(state, MCTS policy, outcome)` replay của AlphaZero chuẩn.
- GPU utilization sampling phụ thuộc cửa sổ `nvidia-smi`; kernel ngắn có thể bị bỏ lỡ. CUDA event timings là nguồn chính cho các phase nhỏ.
- Evaluation mới so với frozen previous-iteration snapshot. Benchmark heuristic/minimax vẫn cần chạy theo checkpoint milestone để theo dõi sức mạnh tuyệt đối.
- Chưa chạy long FULL_TRAIN sau patch; không đưa ra tuyên bố chất lượng model chưa được đo.
