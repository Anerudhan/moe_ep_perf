"""Expose JIT sources when importing FlashInfer directly from its checkout."""

from pathlib import Path


root = Path(__file__).resolve().parent / "flashinfer"
data = root / "flashinfer" / "data"
data.mkdir(parents=True, exist_ok=True)
for source in ("csrc", "include", "3rdparty/cutlass", "3rdparty/spdlog", "3rdparty/cccl"):
    target = root / source
    if not target.is_dir():
        raise RuntimeError(f"Missing FlashInfer source directory: {target}")
    link = data / target.name
    if link.exists() or link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected FlashInfer data path: {link}")
    else:
        link.symlink_to(target, target_is_directory=True)
