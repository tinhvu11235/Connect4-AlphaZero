# Gate 7 — Throughput optimization report

## Selected production configuration

This is the selected configuration for the target Intel i5-13500 / RTX 5060 8 GB system:

| Area | Selected setting |
|---|---|
| Independent non-neural MCTS jobs | 16 persistent processes |
| Neural inference | `torch.inference_mode()`, FP32, batch up to 512 for independent positions |
| AlphaZero self-play below 100 opponent rollouts | Sequential collection; multiprocessing rejected |
| AlphaZero self-play at 100+ opponent rollouts | 8 CPU workers, one parent-owned CUDA model service, inference queue batch 2 |
| Self-play update batch | 20 games, unchanged from the Chapter 21 training schedule |
| Training update | One vectorized FP32 forward/backward pass using the source sum loss |
| Static DataLoader | `num_workers=0`, `pin_memory=True`; no persistent workers |
| Queue inference transfers | Pinned reusable host/device buffers when enabled |
| AMP / GradScaler | Disabled |
| `torch.compile` | Disabled |

Eight self-play workers were selected over the 20-worker microbenchmark winner. In the representative 20-game update workload, 20 workers produced essentially the same sustained throughput as 8 workers (27.95 versus 27.99 games/s in matched earlier runs) while consuming more CPU and RAM and increasing straggler sensitivity. The final repeated 8-worker run reached 31.61 games/s. This is the maximum useful configuration observed under the representative workload, not the configuration with the highest short-case CPU occupancy.

## Scope and semantic safeguards

No agent search, rollout, reward, perspective, move-selection, or training-target semantics were changed. The optimizations alter execution only:

- self-play games remain independent;
- the learner and frozen opponent still use the same source policy distributions;
- random simulation remains uniform;
- one CUDA service owns the neural models, so workers do not create GPU model copies;
- all games in a Chapter 21 update batch use the same learner side, preserving the source alternating-side schedule;
- vectorized training recomputes the unchanged model's policy outputs after collection and applies the same summed REINFORCE loss in one optimizer update;
- parallel tasks receive explicit seeds and are independent of completion order.

Full AlphaZero training was not started.

## Baseline before optimization

The baseline was captured first and saved to `results/performance_baseline.json`. Each phase ran for approximately five seconds with 100 MCTS rollouts, 4 self-play opponent rollouts, and 2 games per training update.

| Metric | Baseline |
|---|---:|
| MCTS rollouts/s | 7,698.98 |
| Policy single-position inference/s | 2,356.21 |
| Value single-position inference/s | 2,717.29 |
| Self-play games/s | 68.18 |
| Self-play policy positions/s | 558.00 |
| Training games/s | 46.47 |
| Training samples/s | 412.26 |
| Average GPU utilization | 7.46% |
| Peak VRAM | 237 MiB |
| Average CPU utilization | 7.64% |
| Peak system RAM in use | 14.43 GiB |

The utilization averages span different serial benchmark phases, so they describe that baseline run rather than a single homogeneous workload.

## Accepted optimizations

| Optimization | Before | After | Result |
|---|---:|---:|---:|
| `torch.inference_mode()` versus grad-enabled single inference | 3,569.57 pos/s | 4,116.69 pos/s | +15.3% |
| Policy batch 512 versus measured baseline single inference | 2,356.21 pos/s | 1,229,808.36 pos/s | 522.0× throughput |
| Value batch 512 versus measured baseline single inference | 2,717.29 pos/s | 1,623,895.97 pos/s | 597.6× throughput |
| Vectorized state preparation | 263,502.61 pos/s | 7,334,480.75 pos/s | 27.8× |
| Pinned/non-blocking static transfer | 909,997.87 pos/s | 990,249.86 pos/s | +8.8% |
| Reused pinned/device transfer buffers | 909,997.87 pos/s | 995,239.72 pos/s | +9.4% |
| Static DataLoader pinning, zero workers | 264,274.82 samples/s | 298,054.70 samples/s | +12.8% |
| 16-process independent MCTS | 7,698.98 rollouts/s | 56,731.35 rollouts/s | 7.37× |
| Vectorized end-to-end training path | 412.26 samples/s | 495.42 samples/s | +20.2% |
| Persistent high-rollout self-play service | 7.77 games/s sequential | 31.61 games/s sustained | 4.07× |

The very large inference batching multipliers compare throughput, not per-position latency. Agents that require one immediate decision continue to use a single position; batching is used only when independent positions are available.

Training kernel throughput continued to rise through the tested synthetic batch sizes (10,036 samples/s at 16 to 476,828 samples/s at 1,024), with only 123.6 MiB peak allocated VRAM at 1,024. The production game batch remains 20 because changing the Chapter 21 update batch would change training behavior.

## Rejected or conditional optimizations

| Candidate | Measurement | Decision |
|---|---|---|
| Detached collection as a standalone workload | 68.18 -> 65.85 games/s | Rejected alone; retained only in the combined training path, which improved 20.2%. |
| Multiprocessing at 20 rollouts | 21.56 -> 11.86 games/s | Rejected; queue/process overhead dominates. |
| Multiprocessing at 50 rollouts | 11.09 -> 9.53 games/s | Rejected. |
| Multiprocessing at 100 rollouts | 7.77 -> 8.77 games/s in the startup-inclusive short test; 31.61 games/s sustained with persistent workers | Accepted only at 100+ rollouts. |
| 20 self-play workers | Same sustained throughput as 8 workers, with higher CPU/RAM and more straggler exposure | Rejected for production. |
| `torch.no_grad()` | 3,947.20 pos/s | Useful, but `inference_mode()` was faster and selected. |
| AMP FP16 | 1,812,196.70 pos/s at batch 1,024; max probability error 0.000633 | Rejected to retain exact FP32 policy probabilities and training behavior. |
| GradScaler | Not applicable after AMP rejection | Disabled. |
| `torch.compile` | Failed because a working Triton installation was unavailable | Rejected; no unstable dependency change was made. |
| DataLoader workers 1/2/4 | Best was 116,340.15 samples/s versus 298,054.70 at zero workers | Rejected for the small in-memory dataset. Persistent workers were therefore also rejected. |
| MCTS with 20 processes | 48,709.01 rollouts/s versus 48,866.57 at 16 in the worker sweep | Rejected due to slight oversubscription loss. |

## Auto-tuning results

The tests covered inference batches 1, 8, 32, 128, 512, 1,024, and 2,048; training kernels 16–1,024; DataLoader workers 0, 1, 2, and 4; MCTS processes 1–20; and self-play workers 1–20 with queue batch sizes 1–32. Batch 512 was the best FP32 inference point at 1,249,219 positions/s in the isolated sweep; larger FP32 batches declined. MCTS peaked at 16 processes. Queue batching remained small because CPU rollouts produce neural requests asynchronously; forcing a large queue batch did not materially improve end-to-end throughput.

## Sustained selected-configuration run

The final selected configuration ran for 180.97 seconds and saved its summary to `results/gate7_sustained.json`.

| Metric | Result |
|---|---:|
| Games completed | 5,720 |
| Policy/training samples | 50,956 |
| Optimizer updates | 286 |
| Games/s | 31.61 |
| Positions/s | 281.57 |
| Training samples/s | 281.57 |
| Average inference queue batch | 1.60 |
| First / middle / last-third games/s | 31.37 / 32.03 / 33.38 |
| Average GPU utilization | 2.75% |
| GPU idle samples | 2.21% |
| Average / peak VRAM | 240.64 / 257 MiB |
| CUDA peak allocated memory | 72.63 MiB |
| Average GPU power | 21.90 W |
| Average GPU temperature | 41.06 °C |
| Average CPU utilization | 31.16% |
| Peak system RAM in use | 14.34 GiB |
| Steady-state RAM growth | -0.006 GiB |
| Steady-state VRAM growth | 20 MiB allocator step, then flat |

The run completed without worker deadlock. Throughput did not fall; the last-third rate exceeded the first-third rate. RAM was flat after process startup. VRAM rose in two allocator steps and then remained flat, leaving about 7.7 GiB of the 8 GiB card unused. Low GPU utilization is expected because root-only search bookkeeping and random rollouts are CPU-bound; attempts to raise GPU occupancy at the expense of end-to-end throughput were rejected.

## Monitoring and artifacts

`benchmark/gate7_monitor.py` samples at approximately one-second intervals and appends the requested timestamp, phase, GPU utilization, GPU memory, power, temperature, CPU, RAM, and live throughput fields to `results/gpu_metrics.csv`.

Generated artifacts:

- `results/performance_baseline.json`
- `results/performance_optimized.json`
- `results/gpu_metrics.csv`
- `results/gpu_utilization.png`
- `results/vram_usage.png`
- `results/throughput.png`
- `results/gate7_autotune.json`
- `results/gate7_mcts_workers.json`
- `results/gate7_parallel_autotune_r100_steady.json`
- `results/gate7_sustained.json`

## Verification

Command:

```text
python -m unittest discover -s tests -v
```

Result: **100 tests passed in 8.55 seconds**. This includes the existing eight-agent correctness suite plus Gate 7 checks for detached trajectory equivalence, vectorized parameter updates, and the Windows-safe single-GPU parallel worker service.

## Known limitations

- The baseline and isolated cases are deliberately short and can vary with desktop background load; accepted production choices were checked against sustained end-to-end behavior.
- The root-only book searches inherently offer little neural batching, so inference batches in live self-play average about 1.6 even with a queue.
- Multiprocessing changes the exact seeded random sequence relative to single-process `torch.multinomial`, although each worker is deterministic for its assigned seed and uses the same categorical policy and uniform-rollout distributions.
- `torch.compile` was not retested after installing Triton because changing the validated environment was explicitly out of scope.
- FP16/AMP was not accepted despite higher raw throughput because it changed policy probabilities and could change stochastic decisions.
- No full AlphaZero training, MCTS redesign, PUCT, persistent search tree, replay buffer, or other algorithm change was performed.
