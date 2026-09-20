"""Visual prefix tokens for SemIf sliced-head scoring (#48).

Patch embeddings are prepended to the text embeddings. The letter-slot
readout still happens at the last position. Pixels never enter the text prompt.
"""

from __future__ import annotations

from typing import Any, Optional

PREFIX_COUNT = 32


def pool_patches(hidden: Any, count: int = PREFIX_COUNT) -> Any:
    """Reduce a ViT sequence [B, T, D] to [B, count, D] without a generation loop."""
    import torch
    import torch.nn.functional as F

    if hidden.dim() != 3:
        raise ValueError("hidden must be [batch, tokens, dim]")
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")
    _batch, tokens, _dim = hidden.shape
    if tokens == count:
        return hidden
    pooled = F.adaptive_avg_pool1d(hidden.transpose(1, 2), count)
    return pooled.transpose(1, 2)


class VisualPrefixProjector:
    """Maps encoder patch dim → LLM hidden size. Untrained; small gain so text still leads."""

    def __init__(self, in_dim: int, out_dim: int, device: Any, dtype: Any):
        import torch
        import torch.nn as nn

        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        self.norm.to(device=device, dtype=dtype)
        self.proj.to(device=device, dtype=dtype)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def __call__(self, patches: Any) -> Any:
        import torch

        if patches.dim() == 2:
            patches = patches.unsqueeze(0)
        x = patches.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        return self.proj(self.norm(x))


def sliced_logits_with_prefix(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    visual_prefix: Any,
    slots: list[int],
) -> Any:
    """One forward: visual tokens + text tokens → sliced letter logits. No decode."""
    import torch
    import torch.nn.functional as F

    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    lm_head = getattr(model, "lm_head", None)
    if base is None or lm_head is None or not hasattr(lm_head, "weight"):
        raise RuntimeError("Model has no transformer + lm_head for visual prefix")
    embed = base.embed_tokens(input_ids)
    prefix = visual_prefix
    if prefix.dim() == 2:
        prefix = prefix.unsqueeze(0)
    prefix = prefix.to(device=embed.device, dtype=embed.dtype)
    inputs_embeds = torch.cat([prefix, embed], dim=1)
    vis_len = prefix.shape[1]
    vis_mask = torch.ones(
        (attention_mask.shape[0], vis_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    mask = torch.cat([vis_mask, attention_mask], dim=1)
    out = base(
        inputs_embeds=inputs_embeds,
        attention_mask=mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        hidden = out[0]
    rep = hidden[:, -1, :]
    slots_tensor = torch.as_tensor(slots, dtype=torch.long, device=rep.device)
    sliced_weight = lm_head.weight[slots_tensor]
    sliced_bias = lm_head.bias[slots_tensor] if getattr(lm_head, "bias", None) is not None else None
    return F.linear(rep, sliced_weight, sliced_bias)[0]


def uses_visual_prefix(vision: Optional[dict]) -> bool:
    """Untrained patch prefix is opt-in. Dual-rate default is compact state.vision only."""
    import os

    if os.environ.get("SEMIF_VISION_PREFIX", "0") not in {"1", "true", "yes"}:
        return False
    if not isinstance(vision, dict):
        return False
    backend = str(vision.get("backend") or "")
    if backend in {"", "stub", "synthetic"}:
        return False
    return True
