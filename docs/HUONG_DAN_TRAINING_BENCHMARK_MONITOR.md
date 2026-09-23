# Hướng dẫn sử dụng, training, benchmark và monitor

Tài liệu này dành cho phiên bản đã tổ chức lại của dự án Connect Four. Mọi lệnh bên dưới được chạy từ thư mục gốc:

```text
D:\AlphaGoSimplified-main
```

Không chạy full training hoặc benchmark 700 game nếu bạn chưa chủ động quyết định dành thời gian cho chúng.

## 1. Cấu trúc dự án

```text
connect4_ai/
  agents/       8 agent dùng chung API select_action(env)
  env/          môi trường Connect Four chuẩn, bàn cờ (7, 6)
  models/       các mạng PyTorch
  training/     training AlphaZero và các model phụ trợ
  utils/        CUDA, monitor và tiện ích runtime
benchmark/      benchmark AlphaZero với 7 agent còn lại
checkpoints/
  pretrained/   checkpoint chuyển đổi/đã xác thực từ nguồn sách
  training/     checkpoint của dòng training AlphaZero hiện tại
configs/        cấu hình mặc định của 8 agent
docs/           tài liệu sử dụng
results/        số liệu production hoặc long-run thực tế
runs/           TensorBoard production
tests/          unit/integration test đang hoạt động
archive/        notebook, Keras, báo cáo cũ và artifact smoke test
```

Notebook và utility cũ không bị xóa. Chúng đã được chuyển vào `archive/` và không còn được import bởi runtime.

## 2. Môi trường đã xác thực

Cấu hình hiện đã kiểm tra:

- Windows, Python 3.14.7;
- PyTorch 2.11.0+cu128;
- CUDA runtime của PyTorch 12.8;
- NVIDIA GeForce RTX 5060 8 GB;
- NumPy 2.5.3;
- psutil 7.2.2;
- TensorBoard 2.21.0.

Không cần thay đổi môi trường đã xác thực. Kiểm tra lại CUDA trước khi training:

```powershell
python scripts/verify_pytorch_cuda.py
```

Chạy toàn bộ test:

```powershell
python -m unittest discover -s tests -v
```

Kết quả sau khi tổ chức lại: 90 test đã pass, gồm legal move, fixed-state, CUDA inference, training update, multiprocessing, factory đủ 8 agent và benchmark pipeline.

## 3. Sử dụng 8 agent

Danh mục và cấu hình mặc định nằm trong `configs/agents.json`. Có thể tạo agent qua factory chung:

```python
from connect4_ai import ConnectFourEnv, create_agent

env = ConnectFourEnv(seed=123)
agent = create_agent("AlphaBetaAgent", seed=123, depth=3)
action = agent.select_action(env)
assert action in env.validinputs
env.step(action)
```

Ví dụ tạo từng agent:

```python
random_agent = create_agent("RandomAgent", seed=1)
rule_agent = create_agent("RuleBasedThink3Agent", seed=1)
minimax_agent = create_agent("DepthMinimaxAgent", seed=1, depth=3)
alphabeta_agent = create_agent("AlphaBetaAgent", seed=1, depth=3)
position_agent = create_agent("PositionEvalAgent", seed=1, depth=3)
mcts_agent = create_agent(
    "MCTSAgent", seed=1, num_rollouts=100, exploration_constant=1.4
)
alphago_agent = create_agent(
    "AlphaGoAgent", seed=1, weight=0.75, depth=45, num_rollouts=584
)
alphazero_agent = create_agent(
    "AlphaZeroAgent", seed=1, weight=0.75, num_rollouts=584
)
```

Ba agent dùng mạng nơ-ron mặc định chạy trên CUDA và không âm thầm chuyển sang CPU. Có thể truyền `checkpoint=...`, `policy_checkpoint=...`, `value_checkpoint=...` hoặc `rollout_policy_checkpoint=...` nếu cần thay model.

Checkpoint mặc định:

| Thành phần | File |
|---|---|
| PositionEval | `checkpoints/pretrained/position_eval.pt` |
| AlphaGo fast rollout policy | `checkpoints/pretrained/policy_conn.pt` |
| AlphaGo policy-gradient policy | `checkpoints/pretrained/policy_gradient_conn.pt` |
| AlphaGo value | `checkpoints/pretrained/value_conn.pt` |
| AlphaZero model theo sách | `checkpoints/pretrained/connzero_book_iter_004.pt` |
| AlphaZero state đang training | `checkpoints/training/connzero_training_state.pt` |
| AlphaZero final sau khi hoàn tất | `checkpoints/training/connzero_final_trained.pt` |

Checkpoint `connzero_book_iter_004.pt` là model chuyển đổi từ model đi kèm sách. Nó không phải kết quả của dòng training production đang dở.

## 4. Trạng thái training hiện tại

Long-run đã được dừng an toàn và không tự chạy lại. State hiện có:

- iteration hiện tại: 3;
- trạng thái: đang ở giữa iteration;
- game đã sinh trong iteration 3: 3.800;
- batch đã hoàn thành trong iteration 3: 190;
- iteration 0, 1 và 2 đã có checkpoint riêng;
- chưa có `connzero_final_trained.pt`.

Các file cần giữ:

```text
checkpoints/training/connzero_iter_000.pt
checkpoints/training/connzero_iter_001.pt
checkpoints/training/connzero_iter_002.pt
checkpoints/training/connzero_training_state.pt
results/training_iterations.csv
results/gpu_metrics.csv
results/final_training_config.json
runs/alphazero_full/
```

## 5. Tiếp tục training hiện tại

Lệnh chính xác:

```powershell
python connect4_ai/training/run_full_alphazero.py --resume --state-checkpoint checkpoints/training/connzero_training_state.pt
```

Do các đường dẫn còn lại đã là mặc định, lệnh trên sẽ:

- đọc model, optimizer, opponent snapshot và toàn bộ RNG state;
- tiếp tục iteration 3 tại 3.800 game / 190 batch;
- ghi checkpoint mỗi 10 batch;
- nối tiếp `results/training_iterations.csv` và `results/gpu_metrics.csv`;
- nối tiếp TensorBoard trong `runs/alphazero_full`;
- tạo `checkpoints/training/connzero_final_trained.pt` khi hoàn thành toàn bộ lịch.

Nhấn `Ctrl+C` để dừng. Do state được lưu mỗi 10 batch, phần công việc chưa checkpoint từ lần lưu gần nhất có thể phải chạy lại. Không xóa state checkpoint.

## 6. Bắt đầu một dòng training mới, không ghi đè dòng hiện tại

Dùng các đường dẫn riêng sau:

```powershell
python connect4_ai/training/run_full_alphazero.py --no-resume `
  --output-dir checkpoints/training/new_run `
  --state-checkpoint checkpoints/training/new_run/connzero_training_state.pt `
  --final-checkpoint checkpoints/training/new_run/connzero_final_trained.pt `
  --iterations-output results/new_run_training_iterations.csv `
  --metrics-output results/new_run_gpu_metrics.csv `
  --config-output results/new_run_training_config.json `
  --tensorboard-logdir runs/alphazero_new_run
```

Không chạy `--no-resume` với đường dẫn production hiện tại, vì nó sẽ tạo một dòng model mới và làm mất khả năng tiếp tục đúng lineage cũ ở chính đường dẫn đó.

Để resume dòng `new_run`:

```powershell
python connect4_ai/training/run_full_alphazero.py --resume `
  --output-dir checkpoints/training/new_run `
  --state-checkpoint checkpoints/training/new_run/connzero_training_state.pt `
  --final-checkpoint checkpoints/training/new_run/connzero_final_trained.pt `
  --iterations-output results/new_run_training_iterations.csv `
  --metrics-output results/new_run_gpu_metrics.csv `
  --config-output results/new_run_training_config.json `
  --tensorboard-logdir runs/alphazero_new_run
```

## 7. Cấu hình và hyperparameter training

Lịch training trung thành với notebook Chương 21:

| Iteration | Trọng số đối thủ | Rollout đối thủ |
|---:|---:|---:|
| 0 | 0,05 | 50 |
| 1 | 0,30 | 100 |
| 2 | 0,50 | 150 |
| 3 | 0,70 | 200 |
| 4 | 0,90 | 250 |

Các tham số cố định:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `batch_size` | 20 game | Số trajectory mỗi lần cập nhật |
| `gamma` | 0,95 | Hệ số chiết khấu reward |
| `max_steps` | 50 | Ngưỡng an toàn cho một episode |
| learning rate | 0,00025 | Learning rate Adam |
| Adam epsilon | 1e-7 | Epsilon của optimizer |
| global clip norm | 1,0 | Cắt gradient toàn cục |
| seed | 20260923 | Seed mặc định |
| checkpoint interval | 10 batch | Tương đương tối đa khoảng 200 game giữa hai lần lưu |
| promotion score | 0,1 | Promote khi `(wins-losses)/games` trên cửa sổ tối đa 1.000 game đạt ngưỡng và đã đủ minimum game của profile |
| max episodes | 25.000 | Giới hạn an toàn; mặc định không tự promote candidate dưới ngưỡng |

Cấu hình đã chọn cho máy i5-13500 / RTX 5060:

| Tham số CLI | Giá trị |
|---|---:|
| `--parallel-workers` | 8 |
| `--parallel-inference-batch-size` | 2 |
| `--parallel-min-rollouts` | 100 |
| `--parallel-pin-memory` | bật |
| precision | float32 |
| AMP | tắt |
| `torch.compile` | tắt |

Iteration 0 có 50 rollout nên dùng đường chạy tuần tự. Iteration 1–4 có từ 100 rollout nên dùng CPU worker và một CUDA inference service; không tạo một model GPU cho mỗi worker.

Các lựa chọn CLI hữu ích:

- `--seed N`: đổi seed;
- `--checkpoint-every-batches N`: chu kỳ lưu state;
- `--progress-every-batches N`: chu kỳ in tiến độ;
- `--stop-after-batches N`: chỉ dùng cho smoke test có chủ đích;
- `--no-parallel-pin-memory`: tắt pinned memory khi chẩn đoán;
- `--parallel-workers N`: thử số worker khác, nhưng không nên mặc định dùng cả 20 thread.
- `--promote-on-cap`: chỉ dùng khi chủ động muốn giữ hành vi cũ là chấp nhận candidate ở safety cap; mặc định tắt.

## 8. TensorBoard

Mở TensorBoard cho dòng production:

```powershell
python -m tensorboard.main --logdir runs/alphazero_full --port 6006
```

Sau đó mở:

```text
http://localhost:6006
```

Nếu dùng dòng mới:

```powershell
python -m tensorboard.main --logdir runs/alphazero_new_run --port 6006
```

Ý nghĩa các chart:

| Chart | Ý nghĩa |
|---|---|
| `batch/loss` | Policy-gradient objective của từng batch; có thể âm, không diễn giải như cross-entropy thông thường |
| `batch/running_reward` | Mean reward trên cửa sổ tối đa 1.000 episode, dùng cho điều kiện dừng iteration |
| `batch/mean_episode_reward` | Mean reward của đúng batch hiện tại |
| `batch/policy_steps` | Số state/action được dùng để cập nhật trong batch |
| `batch/gradient_norm_before_clip` | Chuẩn gradient trước khi clip ở 1,0 |
| `train/total_loss`, `train/policy_loss` | Objective policy-only của optimizer; model hiện không có value head |
| `train/learning_rate` | Learning rate hiện tại |
| `selfplay/games_per_sec`, `selfplay/positions_per_sec` | Thông lượng riêng của giai đoạn sinh game |
| `selfplay/avg_game_length` | Số nước hợp lệ trung bình mỗi game |
| `arena/wins`, `arena/losses`, `arena/draws` | W/L/D của evaluation candidate so với frozen previous checkpoint |
| `arena/score_vs_previous` | `(wins-losses)/games`; không thay bằng win rate khi có hòa |
| `arena/as_first/*`, `arena/as_second/*` | Tỷ lệ theo phía candidate đi trước/đi sau |
| `arena/first_move_gap` | Score đi trước trừ score đi sau |
| `arena/promoted` | 1 nếu checkpoint được chấp nhận, 0 nếu bị từ chối |
| `system/replay_buffer_size` | Số trajectory hiện có trong replay buffer |
| `system/inference_batch_size` | Batch forward GPU thực tế trung bình |
| `system/inference_batches_per_sec` | Số forward batch inference mỗi giây self-play |
| `system/training_steps_per_sec` | Số optimizer update mỗi giây training |
| `throughput/games_per_sec` | Thông lượng self-play đầu-cuối tích lũy trong iteration |
| `throughput/training_samples_per_sec` | Số policy step xử lý mỗi giây |
| `duration/self_play_batch_seconds` | Thời gian sinh game của batch |
| `duration/training_batch_seconds` | Thời gian forward/backward/update |
| `duration/evaluation_batch_seconds` | Thời gian cập nhật reward và điều kiện dừng |
| `resource/gpu_util_percent` | GPU utilization ở mẫu monitor gần nhất |
| `resource/gpu_memory_used_mb` | VRAM đang dùng |
| `resource/cpu_util_percent` | CPU toàn hệ thống |
| `resource/ram_used_gb` | RAM toàn hệ thống đang dùng |
| `iteration/*` | Tổng hợp mỗi iteration đã hoàn thành |
| `configuration/json` | Snapshot cấu hình chạy |
| `iteration/checkpoint` | Đường dẫn checkpoint của iteration |

GPU utilization thấp không tự động là lỗi. Self-play, rollout và bookkeeping của thuật toán theo sách chủ yếu là CPU-bound; mục tiêu là thông lượng game đầu-cuối, không phải ép GPU luôn đạt 100%.

## 9. Monitor tài nguyên

Monitor tự động khởi động cùng training và benchmark, lấy mẫu khoảng mỗi giây. Không cần mở thêm process monitor để tạo CSV.

Xem CSV đang tăng theo thời gian trong PowerShell:

```powershell
Get-Content .\results\gpu_metrics.csv -Wait
```

Xem nhanh GPU trực tiếp:

```powershell
nvidia-smi -l 1
```

Các cột của `results/gpu_metrics.csv`:

| Cột | Nội dung |
|---|---|
| `timestamp` | Thời gian ISO-8601 có timezone |
| `phase` | Nguồn và phase, ví dụ `full_training:iteration_3` hoặc `final_benchmark:AlphaGoAgent` |
| `gpu_util_percent` | Phần trăm GPU utilization |
| `gpu_memory_used_mb` | VRAM đã dùng |
| `gpu_power_w` | Công suất GPU |
| `gpu_temperature_c` | Nhiệt độ GPU |
| `cpu_percent` | CPU toàn hệ thống |
| `ram_used_gb` | RAM toàn hệ thống |
| `games_per_sec` | Tốc độ game hiện tại |
| `positions_per_sec` | Tốc độ position hiện tại |
| `training_steps_per_sec` | Tốc độ sample/training step hiện tại |

File được append để giữ lịch sử. Dùng cột `phase` để tách training và benchmark khi phân tích.

## 10. Chạy benchmark cuối cùng

Chỉ chạy sau khi full training đã tạo file:

```text
checkpoints/training/connzero_final_trained.pt
```

Lệnh benchmark chính thức, 100 game cho mỗi matchup:

```powershell
python benchmark/final_benchmark.py `
  --alpha-zero-checkpoint checkpoints/training/connzero_final_trained.pt `
  --games-per-matchup 100
```

Script chạy 7 matchup, tổng cộng 700 game:

- 50 game AlphaZero đi trước và 50 game đi sau trong mỗi matchup;
- seed được kiểm soát;
- mọi action được kiểm tra legal trước khi `env.step`;
- model checkpoint và SHA-256 được ghi vào config;
- nếu bị gián đoạn, chạy lại đúng lệnh để bỏ qua các game đã có trong CSV và tiếp tục phần còn thiếu.

Hyperparameter benchmark mặc định:

| Thành phần | Cấu hình |
|---|---|
| AlphaZero | weight 0,75; 584 rollout |
| DepthMinimax | depth 3 |
| AlphaBeta | depth 3 |
| PositionEval | depth 3 |
| MCTS | 100 rollout; exploration constant 1,4 |
| AlphaGo | weight 0,75; depth 45; 584 rollout; random rollout policy |
| seed | 20260923 |

Script từ chối số game lẻ để đảm bảo phân bổ hai bên bằng nhau. Nó cũng từ chối chạy nếu final checkpoint chưa tồn tại.

## 11. Log benchmark

`results/benchmark_games.csv` có một dòng cho mỗi game:

- `matchup`, `game_id`, `seed`;
- `alpha_zero_side`, `opponent`, `winner`;
- `result_from_alpha_zero_perspective`;
- `move_count`;
- tổng thời gian quyết định của AlphaZero và đối thủ.

`results/benchmark_summary.csv` tổng hợp theo đối thủ:

- số game, thắng, thua, hòa và các tỉ lệ;
- kết quả khi AlphaZero đi trước/đi sau;
- độ dài game trung bình;
- thời gian trung bình mỗi nước của AlphaZero và đối thủ.

`results/final_benchmark_config.json` ghi:

- checkpoint được benchmark và SHA-256;
- seed;
- số game và phân bổ bên đi trước;
- toàn bộ hyperparameter của các agent;
- trạng thái validation cuối run.

Benchmark cũng nối dữ liệu monitor vào `results/gpu_metrics.csv` theo phase `final_benchmark:*`.

## 12. Log training

`results/training_iterations.csv` có một dòng cho mỗi iteration hoàn thành:

- số game và training sample;
- số batch/epoch;
- learning rate và loss trung bình;
- thời gian self-play, update và evaluation;
- GPU utilization trung bình, peak VRAM và CPU trung bình;
- games/second, opponent config, running reward, lý do dừng và checkpoint.

`results/final_training_config.json` là snapshot cấu hình production. Đây là log để audit, không phải file input được tự động đọc để thay CLI.

State checkpoint chứa model, frozen opponent, optimizer, iteration, số game/batch, reward window, accumulator, lịch sử batch và RNG state của Python/NumPy/PyTorch/CUDA. File được ghi theo cơ chế temporary file rồi replace để giảm nguy cơ checkpoint hỏng khi bị gián đoạn.

## 13. Smoke test đã xác nhận

Sau khi tổ chức lại, pipeline đã được kiểm tra ngắn và lưu dưới:

```text
archive/validation/reorg_smoke/
```

Kết quả:

- training batch đầu tạo state ở 20 game / 1 batch;
- resume tiếp tục đúng lên 40 game / 2 batch;
- TensorBoard đọc được toàn bộ scalar/resource tags;
- monitor CSV ghi GPU/VRAM/power/temperature/CPU/RAM;
- benchmark smoke chạy 14 game, đúng 1 game đi trước + 1 game đi sau cho mỗi đối thủ;
- 14/14 game legal, không có invalid move.

Các kết quả smoke này chỉ chứng minh pipeline hoạt động, không được dùng để kết luận sức mạnh agent.

## 14. Quy tắc vận hành an toàn

1. Không xóa `checkpoints/training/connzero_training_state.pt` khi training chưa hoàn tất.
2. Không dùng `--no-resume` trên cùng đường dẫn state production nếu muốn giữ lineage hiện tại.
3. Không benchmark model pretrained của sách rồi gọi đó là kết quả final training mới.
4. Không so sánh win rate từ smoke test 2 game/matchup.
5. Khi thay hyperparameter, dùng đường dẫn output mới để kết quả không bị trộn.
6. Giữ ít nhất state checkpoint gần nhất và checkpoint iteration gần nhất khi sao lưu.
7. Chỉ bắt đầu long-run khi đã kiểm tra nhiệt độ, dung lượng đĩa và TensorBoard/CSV đều đang cập nhật.
