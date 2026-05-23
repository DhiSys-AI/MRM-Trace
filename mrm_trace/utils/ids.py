from datetime import datetime, timezone
from typing import Optional


def generate_run_id(
    model_id: str,
    context_length: int,
    batch_size: int,
    output_tokens: int,
    rep_index: int,
    warmup: bool = False,
    ts: Optional[str] = None,
) -> str:
    if ts is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = "warmup" if warmup else f"rep{rep_index}"
    return f"{model_id}_ctx{context_length}_batch{batch_size}_out{output_tokens}_{kind}_{ts}"
