# AlphaZero training and benchmark pipeline handoff

## Current execution status

Long-running training was stopped at the user's request. No training or benchmark process remains active.

All generated state was preserved. The resumable production state is:

| Field | Preserved value |
|---|---:|
| Current iteration | 3 |
| Games completed in iteration 3 | 3,800 |
| Batches completed in iteration 3 | 190 |
| Rolling reward entries | 1,000 |
| State checkpoint | `checkpoints/training/connzero_training_state.pt` |
| State SHA-256 | `3be28a7576f59acf27f958f2ca1f86129751b6806d5d27c060315598093028d2` |

Completed checkpoints from the actual execution:

| Iteration | Games | Samples | Updates/epochs | Final rolling reward | Stop reason | Checkpoint |
|---:|---:|---:|---:|---:|---|---|
| 0 | 23,280 | 194,047 | 1,164 | 0.103 | Running reward | `checkpoints/training/connzero_iter_000.pt` |
| 1 | 14,360 | 121,222 | 718 | 0.101 | Running reward | `checkpoints/training/connzero_iter_001.pt` |
| 2 | 3,200 | 23,102 | 160 | 0.101 | Running reward | `checkpoints/training/connzero_iter_002.pt` |

Iteration 3 is incomplete and exists only in the resumable state checkpoint. `checkpoints/training/connzero_final_trained.pt` does not exist and no final-strength claim is made. `checkpoints/pretrained/connzero_book_iter_004.pt` is the previously converted book-supplied Keras model, not the output of this interrupted training lineage.

## Manual commands

Run these commands from `D:\AlphaGoSimplified-main`.

### Resume the preserved production run

```powershell
python connect4_ai/training/run_full_alphazero.py --resume --state-checkpoint checkpoints/training/connzero_training_state.pt
```

This resumes iteration 3 at the last saved batch. It does not restart iterations 0–2. The default command below is equivalent because resume is enabled by default:

```powershell
python connect4_ai/training/run_full_alphazero.py
```

### Start a separate run from zero without overwriting preserved data

```powershell
python connect4_ai/training/run_full_alphazero.py --no-resume --state-checkpoint checkpoints/new_run/connzero_training_state.pt --final-checkpoint checkpoints/new_run/connzero_final_trained.pt --output-dir checkpoints/new_run --iterations-output results/new_run_training_iterations.csv --metrics-output results/new_run_gpu_metrics.csv --config-output results/new_run_training_config.json --tensorboard-logdir runs/alphazero_new_run
```

### Open TensorBoard

For the production run:

```powershell
python -m tensorboard.main --logdir runs/alphazero_full --port 6006
```

Then open `http://localhost:6006/`. The preserved long run predates TensorBoard integration, so production events begin when that state is next resumed. The validated smoke events can be inspected immediately with:

```powershell
python -m tensorboard.main --logdir runs/alphazero_smoke --port 6006
```

### Run the final 700-game benchmark after training completes

```powershell
python benchmark/final_benchmark.py --alpha-zero-checkpoint checkpoints/training/connzero_final_trained.pt --games-per-matchup 100
```

The command runs 100 games against each of seven opponents, allocating exactly 50 AlphaZero-first and 50 AlphaZero-second games per matchup. It appends each game immediately and can be run again after interruption; existing games with the same configuration are skipped.

Do not run this command until `checkpoints/training/connzero_final_trained.pt` has been produced by a completed training run.

## Training configuration and hyperparameters

The runner is `connect4_ai/training/run_full_alphazero.py`. It preserves the Chapter 21 training algorithm:

| Setting | Value |
|---|---|
| Seed | 20260923 |
| Device | CUDA only |
| Precision | FP32 |
| AMP / GradScaler | Disabled |
| `torch.compile` | Disabled |
| Game batch | 20 games |
| Learner side | Alternates red/yellow by batch |
| Optimizer | Adam |
| Learning rate | 0.00025 |
| Adam epsilon | 1e-7 |
| Global gradient clipping | 1.0 |
| Discount factor | 0.95 |
| Maximum learner steps per game | 50 |
| Loss | Sum of `-log(policy(action)) * discounted return` |
| Stop window | Latest 1,000 episode rewards |
| Iteration stop | Mean reward >= 0.1 after more than 1,000 games |
| Safety cap | Source condition `episodes > 25000` |
| Checkpoint interval | Every 10 batches / 200 games |

Opponent schedule:

| Iteration | Prior weight | Random rollouts |
|---:|---:|---:|
| 0 | 0.05 | 50 |
| 1 | 0.30 | 100 |
| 2 | 0.50 | 150 |
| 3 | 0.70 | 200 |
| 4 | 0.90 | 250 |

Validated production execution settings:

- iteration 0 remains sequential because multiprocessing is slower below 100 rollouts;
- iterations 1–4 use eight CPU self-play workers;
- one parent process owns the learner and opponent CUDA models;
- inference queue batch size is two;
- pinned host buffers and non-blocking CUDA copies are enabled;
- one vectorized FP32 policy update is performed per source 20-game batch;
- workers are created with Windows `spawn` safety and no worker owns a GPU model.

## Checkpoint and resume behavior

`checkpoints/training/connzero_training_state.pt` is written atomically through a temporary file and replacement. It contains:

- learner and frozen-opponent model states;
- Adam optimizer state;
- current iteration, game count, and batch count;
- latest 1,000 rewards and current iteration history;
- accumulated timing, sample, and loss totals;
- Python, NumPy, Torch CPU, Torch CUDA, policy-sampler, and opponent RNG states;
- full schedule and production configuration.

Completed iteration checkpoints are saved separately as `connzero_iter_000.pt` through `connzero_iter_004.pt`. Completion also writes `connzero_final_trained.pt`. An interruption loses at most the work since the last 200-game checkpoint.

The resume smoke test stopped at 40 games, loaded the saved optimizer/model/RNG state, and continued to 80 games/4 batches. It did not restart from zero.

## TensorBoard charts

TensorBoard writes to `runs/alphazero_full` by default. Resume uses a purge step so any uncheckpointed duplicate events after an interruption are hidden.

### Batch charts

- `batch/loss`: summed REINFORCE loss for the latest 20-game update. Its sign can be positive or negative because returns are signed; it should not be interpreted as cross-entropy.
- `batch/running_reward`: mean AlphaZero-perspective reward over the latest 1,000 games; this is the book's iteration stop signal.
- `batch/mean_episode_reward`: mean result of only the latest 20-game batch.
- `batch/policy_steps`: number of learner decisions used as training samples in the latest batch.
- `batch/gradient_norm_before_clip`: global gradient norm before the source clip limit of 1.0 is applied.

### Throughput charts

- `throughput/games_per_sec`: cumulative games divided by measured self-play, update, and stopping-evaluation time for the current iteration.
- `throughput/training_samples_per_sec`: cumulative learner policy samples divided by the same measured duration.

### Duration charts

- `duration/self_play_batch_seconds`: trajectory generation time for the latest 20 games.
- `duration/training_batch_seconds`: vectorized CUDA forward/backward/update time.
- `duration/evaluation_batch_seconds`: time used to update the reward window and evaluate the source stopping conditions.

### Resource charts

- `resource/gpu_util_percent`: latest approximately one-second NVIDIA utilization sample.
- `resource/gpu_memory_used_mb`: latest total GPU memory used.
- `resource/cpu_util_percent`: latest machine-wide CPU utilization.
- `resource/ram_used_gb`: latest machine-wide used RAM.

### Iteration charts

After each completed iteration, `iteration/*` records games, samples, updates/epochs, mean loss, self-play/training/evaluation durations, GPU mean utilization, GPU peak VRAM, CPU mean utilization, games/s, and final rolling reward. `iteration/checkpoint` records the checkpoint path. `configuration/json` displays the serialized run configuration.

The smoke log was independently inspected with TensorBoard. It contains 14 scalar tags, configuration text, steps 20–80, and no out-of-order scalar steps after resume processing.

## Training logs

### `results/training_iterations.csv`

One row is appended only after an iteration completes. Columns are:

- `iteration`, `games_generated`, `training_samples`, and `epochs` (one optimizer update per epoch/batch);
- `learning_rate` and mean batch `loss`;
- measured `self_play_duration`, `training_duration`, and `evaluation_duration` in seconds;
- `gpu_mean_utilization`, `gpu_peak_vram_mb`, and `cpu_mean_utilization`;
- end-to-end `games_per_sec`;
- opponent weight/rollouts, final running reward, stop reason, and checkpoint path.

The file currently contains the three real completed iterations shown at the start of this report. Iteration 3 has no row because it is incomplete.

### `results/gpu_metrics.csv`

This append-only file contains approximately one row per second with:

```text
timestamp, phase, gpu_util_percent, gpu_memory_used_mb, gpu_power_w,
gpu_temperature_c, cpu_percent, ram_used_gb, games_per_sec,
positions_per_sec, training_steps_per_sec
```

Training phases are named `full_training:iteration_N`. Earlier Gate 7 records remain intact in the same file.

### `results/final_training_config.json`

Records the seed, CUDA requirement, five-iteration schedule, checkpoint cadence, TensorBoard directory, worker/batch configuration, precision decisions, and source notebook.

## Benchmark configuration and logs

The benchmark runner is `benchmark/final_benchmark.py`. Defaults are the book-faithful agent settings:

- AlphaZero: weight 0.75 and 584 random rollouts;
- DepthMinimax and AlphaBeta: depth 3;
- PositionEval: depth 3 with `checkpoints/pretrained/position_eval.pt`;
- MCTS: 100 rollouts and exploration constant 1.4;
- AlphaGo: weight 0.75, depth 45, 584 random rollouts, and the three validated neural checkpoints;
- base seed 20260923;
- 100 games per matchup, split 50/50 by starting side.

### `results/benchmark_games.csv`

Appends every completed game immediately:

```text
matchup, game_id, seed, alpha_zero_side, opponent, winner,
result_from_alpha_zero_perspective, move_count,
alpha_zero_decision_time, opponent_decision_time
```

Decision-time fields are total seconds for that agent in the game. Before every environment step, the selected action is checked against `env.validinputs`; any illegal move aborts rather than being silently recorded.

### `results/benchmark_summary.csv`

Generated atomically from raw games. It includes overall wins/losses/draws and rates, first/second side splits, average game length, and average per-move decision time for each agent.

### `results/final_benchmark_config.json`

Records the benchmark checkpoint path and SHA-256, base seed, complete agent settings, model dependencies, expected side allocation, and final validation status. Reusing output files with a different configuration is rejected. Rerunning the same configuration resumes without duplicating completed games.

The production benchmark CSV/JSON files intentionally do not yet exist because the requested 700-game run was not executed. They are created by the manual benchmark command after final training.

## Short validation evidence

Only short post-stop validation was performed:

- CUDA training smoke: two batches / 40 games;
- resume smoke: two additional batches, continuing from 40 to 80 games;
- TensorBoard inspection: all expected tags and steps present;
- benchmark smoke: 2 games against each of seven agents, 14 games total;
- benchmark allocation: 1 first and 1 second game per matchup;
- benchmark legality: zero invalid moves;
- benchmark rerun: still exactly 14 rows, proving no duplication;
- full unit suite: **103 tests passed in 8.558 seconds**.

Smoke artifacts are clearly separated from production outputs:

- `archive/validation/checkpoints/smoke/connzero_training_state.pt`
- `runs/alphazero_smoke/`
- `results/smoke_training_config.json`
- `results/smoke_gpu_metrics.csv`
- `results/smoke_benchmark_games.csv`
- `results/smoke_benchmark_summary.csv`
- `results/smoke_benchmark_config.json`
- `results/smoke_benchmark_gpu_metrics.csv`

No full training or full benchmark was restarted after the stop instruction.
