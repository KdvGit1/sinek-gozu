"""Sayısal doğrulama: FP32 ile FP64 karşılaştırması (TASARIM.md Bölüm 5.4)."""
from __future__ import annotations

import copy

import torch

from .detection import detection_loss


def precision_check(model, tiles, heat, reg, pos) -> dict:
    """Aynı girdiyle modeli FP32 ve FP64'te çalıştırır; kayıp ve gradyan farklarını döndürür.
    Beklenen: göreli farklar ~1e-5 veya altı. Büyükse döngülerde sayısal bir sorun var demektir."""
    res = {}
    grads = {}
    for name, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
        m = copy.deepcopy(model).to(dtype)
        m.zero_grad()
        out = m(tiles.to(dtype))
        loss = detection_loss(out, heat.to(dtype), reg.to(dtype), pos)["total"]
        loss.backward()
        res[f"loss_{name}"] = loss.item()
        grads[name] = torch.cat([p.grad.flatten().double() for p in m.parameters() if p.grad is not None])
        res[f"state_{name}"] = out["state"].double()
    res["loss_rel_diff"] = abs(res["loss_fp32"] - res["loss_fp64"]) / max(abs(res["loss_fp64"]), 1e-12)
    g32, g64 = grads["fp32"], grads["fp64"]
    res["grad_rel_diff"] = ((g32 - g64).norm() / g64.norm().clamp(min=1e-12)).item()
    s32, s64 = res.pop("state_fp32"), res.pop("state_fp64")
    res["state_rel_diff"] = ((s32 - s64).norm() / s64.norm().clamp(min=1e-12)).item()
    return res
