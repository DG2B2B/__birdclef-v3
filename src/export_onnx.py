"""
BirdCLEF 2026 – ONNX / OpenVINO Export & CPU Optimization
Converts PyTorch student models to ONNX for fast CPU inference.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
import time
import gc


# ──────────────────────────────────────────────
# ONNX Export
# ──────────────────────────────────────────────

def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model in eval mode.
        output_path: Where to save the .onnx file.
        input_shape: Dummy input shape.
        opset_version: ONNX opset version.
        dynamic_batch: If True, export with dynamic batch dimension.

    Returns:
        Path to the exported .onnx file.
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # Verify
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"[OK] ONNX exported: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    return output_path


# ──────────────────────────────────────────────
# ONNX verification
# ──────────────────────────────────────────────

def verify_onnx(
    onnx_path: str | Path,
    pytorch_model: torch.nn.Module,
    input_shape: tuple = (4, 3, 224, 224),
    tolerance: float = 1e-4,
) -> bool:
    """Verify ONNX model outputs match PyTorch within tolerance."""
    import onnxruntime as ort

    onnx_path = Path(onnx_path)

    # PyTorch forward
    pytorch_model.eval()
    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        pt_out = pytorch_model.forward_clip(dummy).numpy()

    # ONNX forward
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

    max_diff = np.max(np.abs(pt_out - onnx_out))
    print(f"Max diff PyTorch vs ONNX: {max_diff:.6f}")

    if max_diff < tolerance:
        print("[OK] ONNX verification passed")
        return True
    else:
        print(f"[WARN] Difference above tolerance ({tolerance})")
        return False


# ──────────────────────────────────────────────
# CPU Benchmark
# ──────────────────────────────────────────────

def benchmark_onnx(
    onnx_path: str | Path,
    n_samples: int = 1000,
    batch_size: int = 64,
    input_shape: tuple = (3, 224, 224),
    n_warmup: int = 10,
) -> dict:
    """Benchmark ONNX inference speed on CPU.

    Args:
        onnx_path: Path to .onnx file.
        n_samples: Total samples to process.
        batch_size: Batch size.
        input_shape: Single sample shape (C, H, W).
        n_warmup: Warmup iterations.

    Returns:
        Dict with timing stats.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Warmup
    for _ in range(n_warmup):
        dummy = np.random.randn(batch_size, *input_shape).astype(np.float32)
        sess.run(None, {"input": dummy})

    # Benchmark
    times = []
    for _ in range(0, n_samples, batch_size):
        dummy = np.random.randn(batch_size, *input_shape).astype(np.float32)
        t0 = time.perf_counter()
        sess.run(None, {"input": dummy})
        times.append(time.perf_counter() - t0)

    total_time = sum(times)
    samples_processed = (len(times)) * batch_size
    ms_per_sample = (total_time / samples_processed) * 1000
    ms_per_batch = np.mean(times) * 1000

    print(f"ONNX Benchmark ({onnx_path}):")
    print(f"  Total time:     {total_time:.2f}s for {samples_processed} samples")
    print(f"  Per sample:     {ms_per_sample:.2f} ms")
    print(f"  Per batch ({batch_size}): {ms_per_batch:.2f} ms")
    # Estimate total inference time for full test set
    # Assuming ~60 segments × N soundscapes
    # If 300 soundscapes × 60 segments = 18 000 segments
    total_est = 18000 * ms_per_sample / 1000
    print(f"  Estimated test set ({18000} segments): {total_est:.0f}s ({total_est/60:.1f} min)")

    return {
        "ms_per_sample": ms_per_sample,
        "ms_per_batch": ms_per_batch,
        "total_estimated_minutes": total_est / 60,
    }


# ──────────────────────────────────────────────
# Batch exporter
# ──────────────────────────────────────────────

def export_all_students(
    model_paths: dict[str, str | Path],  # {name: .pth path}
    output_dir: str | Path,
    n_classes: int = 234,
    use_sed_head: bool = True,
) -> dict[str, Path]:
    """Export all student models to ONNX.

    Args:
        model_paths: Dict mapping model name → checkpoint path.
        output_dir: Output directory.
        n_classes: Number of output classes.
        use_sed_head: Whether model uses SED head.

    Returns:
        Dict mapping model name → onnx path.
    """
    from src.models_sed import create_model

    output_dir = Path(output_dir) / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, ckpt_path in model_paths.items():
        print(f"\nExporting {name}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        backbone = cfg.get("backbone", name)

        model = create_model(
            backbone_name=backbone,
            n_classes=n_classes,
            pretrained=False,
            use_sed_head=use_sed_head,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        onnx_path = output_dir / f"{name}.onnx"
        export_to_onnx(model, onnx_path)
        verify_onnx(onnx_path, model)
        benchmark_onnx(onnx_path, n_samples=500)

        results[name] = onnx_path
        del model
        gc.collect()

    return results


if __name__ == "__main__":
    print("ONNX export module ready.")
    print("  - export_to_onnx()")
    print("  - verify_onnx()")
    print("  - benchmark_onnx()")
    print("  - export_all_students()")
