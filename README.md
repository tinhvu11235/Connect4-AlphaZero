# Connect Four AI — 8 agent PyTorch

Dự án đã được tổ chức lại quanh đúng 8 agent:

1. `RandomAgent`
2. `RuleBasedThink3Agent`
3. `DepthMinimaxAgent`
4. `AlphaBetaAgent`
5. `PositionEvalAgent`
6. `MCTSAgent`
7. `AlphaGoAgent`
8. `AlphaZeroAgent`

Mã chạy chính nằm trong `connect4_ai/`; notebook, mã Keras/TensorFlow gốc và kết quả thử nghiệm cũ đã được chuyển vào `archive/` để tham khảo, không còn là dependency lúc chạy.

Xem hướng dẫn đầy đủ bằng tiếng Việt tại [docs/HUONG_DAN_TRAINING_BENCHMARK_MONITOR.md](docs/HUONG_DAN_TRAINING_BENCHMARK_MONITOR.md).

Audit bottleneck, patch replay/profiling và benchmark trước/sau mới nhất nằm tại [ALPHAZERO_TRAINING_OPTIMIZATION_REPORT.md](ALPHAZERO_TRAINING_OPTIMIZATION_REPORT.md).

Kiểm tra nhanh:

```powershell
python -m unittest discover -s tests -v
```

Tiếp tục phiên training đang dở:

```powershell
python connect4_ai/training/run_full_alphazero.py --resume --state-checkpoint checkpoints/training/connzero_training_state.pt
```

Runner có ba profile `debug`, `quick_test`, `full_train`; mặc định là `full_train`. Checkpoint 5-iteration cũ vẫn resume được và sẽ tiếp tục từ iteration 5 theo giới hạn của profile mới.

Lưu ý: lệnh trên là full training và chỉ nên chạy khi bạn chủ động muốn tiếp tục. Không có long-run training nào tự khởi động khi import package hoặc chạy test.
