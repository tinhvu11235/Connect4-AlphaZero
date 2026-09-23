"""One-second GPU, CPU, RAM, and throughput monitoring utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import psutil


CSV_FIELDS = [
    "timestamp",
    "phase",
    "gpu_util_percent",
    "gpu_memory_used_mb",
    "gpu_power_w",
    "gpu_temperature_c",
    "cpu_percent",
    "ram_used_gb",
    "games_per_sec",
    "positions_per_sec",
    "training_steps_per_sec",
]


@dataclass
class LiveStatus:
    phase: str = "idle"
    games_per_sec: float = 0.0
    positions_per_sec: float = 0.0
    training_steps_per_sec: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, phase: str, **rates: float) -> None:
        with self.lock:
            self.phase = phase
            self.games_per_sec = rates.get("games_per_sec", 0.0)
            self.positions_per_sec = rates.get("positions_per_sec", 0.0)
            self.training_steps_per_sec = rates.get(
                "training_steps_per_sec", 0.0
            )

    def snapshot(self) -> dict[str, str | float]:
        with self.lock:
            return {
                "phase": self.phase,
                "games_per_sec": self.games_per_sec,
                "positions_per_sec": self.positions_per_sec,
                "training_steps_per_sec": self.training_steps_per_sec,
            }


def query_gpu() -> tuple[float, float, float, float]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    values = completed.stdout.strip().splitlines()[0].split(",")
    return tuple(float(value.strip()) for value in values)  # type: ignore[return-value]


class PerformanceMonitor:
    def __init__(
        self,
        output: Path,
        status: LiveStatus,
        *,
        interval_seconds: float = 1.0,
        run_label: str,
    ) -> None:
        self.output = output
        self.status = status
        self.interval_seconds = interval_seconds
        self.run_label = run_label
        self.records: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> PerformanceMonitor:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.output.exists() or self.output.stat().st_size == 0
        if write_header:
            with self.output.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=CSV_FIELDS).writeheader()
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 5.0)

    def _run(self) -> None:
        next_sample = time.monotonic()
        while not self._stop.is_set():
            try:
                gpu_util, memory, power, temperature = query_gpu()
            except Exception:
                gpu_util = memory = power = temperature = float("nan")
            virtual_memory = psutil.virtual_memory()
            status = self.status.snapshot()
            row = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "phase": f"{self.run_label}:{status['phase']}",
                "gpu_util_percent": gpu_util,
                "gpu_memory_used_mb": memory,
                "gpu_power_w": power,
                "gpu_temperature_c": temperature,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_used_gb": virtual_memory.used / (1024**3),
                "games_per_sec": status["games_per_sec"],
                "positions_per_sec": status["positions_per_sec"],
                "training_steps_per_sec": status["training_steps_per_sec"],
            }
            self.records.append(row)
            with self.output.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=CSV_FIELDS).writerow(row)
            next_sample += self.interval_seconds
            self._stop.wait(max(0.0, next_sample - time.monotonic()))

    def summary(self) -> dict[str, Any]:
        def finite_values(key: str) -> list[float]:
            return [
                float(record[key])
                for record in self.records
                if float(record[key]) == float(record[key])
            ]

        gpu = finite_values("gpu_util_percent")
        vram = finite_values("gpu_memory_used_mb")
        cpu = finite_values("cpu_percent")
        ram = finite_values("ram_used_gb")
        return {
            "samples": len(self.records),
            "interval_seconds": self.interval_seconds,
            "average_gpu_util_percent": sum(gpu) / len(gpu) if gpu else None,
            "peak_vram_mb": max(vram) if vram else None,
            "average_cpu_percent": sum(cpu) / len(cpu) if cpu else None,
            "peak_ram_used_gb": max(ram) if ram else None,
        }


__all__ = ["CSV_FIELDS", "LiveStatus", "PerformanceMonitor", "query_gpu"]
