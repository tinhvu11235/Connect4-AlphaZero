"""Verify that PyTorch performs real CUDA computation on the selected device."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connect4_ai.utils.device import resolve_device


def _nvidia_smi() -> dict[str, str] | None:
    fields = [
        "name",
        "driver_version",
        "compute_cap",
        "memory.total",
    ]
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    return dict(zip(fields, values, strict=True))


def verify(*, allow_cpu: bool = False) -> dict[str, object]:
    torch.manual_seed(20260922)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260922)

    device = resolve_device(allow_cpu=allow_cpu)
    using_cuda = device.type == "cuda"
    if using_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    left = torch.randn((2048, 2048), device=device)
    right = torch.randn((2048, 2048), device=device)
    if using_cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    product = left @ right
    if using_cuda:
        torch.cuda.synchronize(device)
    matmul_seconds = time.perf_counter() - started

    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 7 * 6, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 7),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn((256, 1, 7, 6), device=device)
    targets = torch.randint(0, 7, (256,), device=device)
    first_parameter = next(model.parameters())
    parameter_before = first_parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    gradients = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = torch.sqrt(sum((gradient**2).sum() for gradient in gradients))
    optimizer.step()
    if using_cuda:
        torch.cuda.synchronize(device)

    vm = psutil.virtual_memory()
    result: dict[str, object] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "passed",
        "host": {
            "operating_system": platform.platform(),
            "cpu_logical_count": os.cpu_count(),
            "cpu_physical_count": psutil.cpu_count(logical=False),
            "system_ram_bytes": vm.total,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "pytorch": {
            "version": torch.__version__,
            "compiled_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cudnn_available": torch.backends.cudnn.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "selected_device": str(device),
        },
        "nvidia_smi": _nvidia_smi(),
        "matrix_multiplication": {
            "device": str(product.device),
            "shape": list(product.shape),
            "all_finite": bool(torch.isfinite(product).all().item()),
            "elapsed_seconds": matmul_seconds,
        },
        "forward_backward": {
            "model_device": str(first_parameter.device),
            "input_device": str(inputs.device),
            "target_device": str(targets.device),
            "output_device": str(logits.device),
            "loss": float(loss.detach().item()),
            "gradient_norm": float(gradient_norm.item()),
            "gradients_finite": all(
                bool(torch.isfinite(gradient).all().item()) for gradient in gradients
            ),
            "parameter_updated": not torch.equal(
                parameter_before, first_parameter.detach()
            ),
        },
    }

    if using_cuda:
        properties = torch.cuda.get_device_properties(device)
        result["pytorch"].update(  # type: ignore[union-attr]
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "device_total_memory_bytes": properties.total_memory,
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            }
        )

    checks = [
        result["matrix_multiplication"]["all_finite"],  # type: ignore[index]
        result["forward_backward"]["gradients_finite"],  # type: ignore[index]
        result["forward_backward"]["parameter_updated"],  # type: ignore[index]
    ]
    if not all(checks):
        result["status"] = "failed"
        raise RuntimeError("PyTorch computation verification failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Explicitly allow a CPU-only diagnostic run.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = verify(allow_cpu=args.allow_cpu)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
