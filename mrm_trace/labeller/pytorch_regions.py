"""
PyTorch weight-tensor address extraction for region pre-classification.

perf mem captures raw load/store events — there are no malloc/free pairs for the
labeller's address tracker, and PyTorch symbols don't match ggml_* rules.  But
for a loaded model we know exactly which virtual address ranges hold weight data:
torch.nn.Parameter.data_ptr() + nbytes.

This module extracts those ranges before perf collection starts.  The labeller
pre-registers them so every perf sample that falls inside a weight page is
immediately classified as ``weight`` instead of ``unknown``.

Only CPU-resident tensors are included — GPU (CUDA/MPS) tensors are not accessible
to perf mem's hardware PMU on the host CPU's memory bus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch.nn as nn

_PAGE_SIZE = 4096


@dataclass
class TensorRegion:
    """A contiguous virtual address range backed by a single model tensor."""
    name: str           # parameter name, e.g. "model.layers.0.self_attn.q_proj.weight"
    region_type: str    # always "weight" (buffers may be "activation" in future)
    addr_start: int     # data_ptr()
    addr_end: int       # data_ptr() + nbytes  (exclusive)
    size_bytes: int
    shape: list
    dtype: str

    def pages(self):
        """Yield every 4 KB page number covered by this region."""
        page_start = self.addr_start >> 12
        page_end = (self.addr_end - 1) >> 12
        yield from range(page_start, page_end + 1)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TensorRegion":
        return TensorRegion(**d)


def extract_pytorch_regions(model: "nn.Module") -> List[TensorRegion]:
    """
    Return one TensorRegion per unique CPU tensor in the model.

    Iterates named_parameters() and named_buffers(), deduplicates by data_ptr()
    (weight-tied tensors appear once), and skips non-CPU tensors.

    Args:
        model: any torch.nn.Module that has already been loaded to CPU (or
               device_map="auto" with some layers on CPU).
    Returns:
        List of TensorRegion sorted by addr_start.
    """
    seen_ptrs: set = set()
    regions: list = []

    def _add(name: str, tensor, region_type: str) -> None:
        try:
            # Skip non-CPU tensors
            if tensor.device.type != "cpu":
                return
            ptr = tensor.data_ptr()
            if ptr == 0 or ptr in seen_ptrs:
                return
            seen_ptrs.add(ptr)
            nb = tensor.nbytes
            regions.append(TensorRegion(
                name=name,
                region_type=region_type,
                addr_start=ptr,
                addr_end=ptr + nb,
                size_bytes=nb,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
            ))
        except Exception:
            pass  # contiguous check or device access can fail on some backends

    for name, param in model.named_parameters():
        _add(name, param.data, "weight")

    for name, buf in model.named_buffers():
        # Buffers include LayerNorm running_mean/var, positional embeddings, etc.
        # Classify as weight — they are read-only model state during inference.
        _add(name, buf, "weight")

    regions.sort(key=lambda r: r.addr_start)
    return regions


def parameter_device_summary(model: "nn.Module") -> dict:
    """
    Census of where the model's parameters and buffers actually live.

    ``extract_pytorch_regions`` deliberately skips non-CPU tensors, because a
    CUDA ``data_ptr()`` is a device address that a CPU PMU can never sample.
    That makes an empty ``pytorch_regions.json`` the correct result for a
    GPU-resident model — but indistinguishable, after the fact, from the
    extractor having silently failed.  Recording this census alongside turns
    "no weight regions" into a statement a reader can check:

        {"cuda:0": {"tensors": 290, "bytes": 988123136}}

    says the weights were on the GPU; ``{}`` or a cpu entry with zero regions
    extracted would say something broke.
    """
    summary: dict = {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        try:
            dev = str(tensor.device)
            nb = int(tensor.nbytes)
        except Exception:
            continue
        entry = summary.setdefault(dev, {"tensors": 0, "bytes": 0})
        entry["tensors"] += 1
        entry["bytes"] += nb
    return summary
