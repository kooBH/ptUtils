"""Exponential moving average of model weights.

Motivation (see logs/20260915_lv2_WER_variance_EMA.md): with batch_size 1 the
per-epoch eval WER of the lv2 separators swings by ~20 %-abs between consecutive
epochs (e.g. lv2_v108.2 : 39.0 -> 33.8 -> 19.9 -> 41.5 -> 29.6). The saved
``bestmodel_wer.pt`` is therefore the minimum of a noisy sequence rather than a
stably good operating point. An EMA of the weights averages out that
step-to-step wander and is evaluated / saved alongside the raw model.

Usage
-----
    ema = ModelEMA(model, decay=0.999)
    ...
    loss.backward(); optimizer.step()
    ema.update(model)                 # every optimizer step
    ...
    with ema.swap(model):             # model temporarily holds the EMA weights
        evaluate(model)
    torch.save(ema.state_dict(), "lastmodel_ema.pt")   # plain model state_dict

``decay`` is the per-step factor; the effective averaging horizon is roughly
``1 / (1 - decay)`` steps (0.999 -> ~1000 steps, a third of a 3000-item epoch;
0.9995 -> ~2000). Buffers (non-float or non-trainable) are copied, not averaged.
"""
import contextlib
import copy

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, warmup_steps: int = 0):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0,1), got {decay}")
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.n_updates = 0
        # Detached float copy of every entry of the state_dict.
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _decay(self) -> float:
        # Ramp the decay in from 0 so the very first updates are not dominated by
        # the (possibly random) init: d = min(decay, (1+n)/(10+n)).
        if self.warmup_steps <= 0:
            return self.decay
        return min(self.decay, (1.0 + self.n_updates) / (10.0 + self.n_updates))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self._decay()
        self.n_updates += 1
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.is_floating_point():
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def state_dict(self) -> dict:
        """EMA weights as a plain model ``state_dict`` (loadable by ``model.load_state_dict``)."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict) -> None:
        for k in self.shadow:
            self.shadow[k].copy_(sd[k])

    @contextlib.contextmanager
    def swap(self, model: torch.nn.Module):
        """Temporarily load the EMA weights into ``model`` (restored on exit)."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield model
        finally:
            model.load_state_dict(backup)
