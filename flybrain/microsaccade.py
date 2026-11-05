"""
H51 (günlük §61): mikrosakkad girişi. Biyoloji: sabitlenmiş görüntü kaybolur; sinek fotoreseptörleri fotomekanik
mikrosakkadlar yapar (Juusola vd. 2017). Karo T adım boyunca küçük bir yörüngede 1–2 px kaydırılır; kodlayıcı her adımda
yeniden çalışır ve fotoreseptör / L1–L3 sürüşü adım adım değişir. Topoloji ve parametreler aynı; ek parametre yok.
"""
from __future__ import annotations

import torch
from torch import nn

_DIRS = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def offsets(T: int, amp: int) -> list[tuple[int, int]]:
    """Adım başına (dy, dx): merkezden başlayıp 8 yönü dolaşan küçük yörünge (genlik amp piksel)."""
    return [(dy * amp, dx * amp) for dy, dx in (_DIRS[t % len(_DIRS)] for t in range(T))]


def shift(x: torch.Tensor, dy: int, dx: int, amp: int) -> torch.Tensor:
    """[B, C, H, W] içeriği (dy, dx) kadar kaydırır; kenar pikseli uzatılır (replicate)."""
    if dy == 0 and dx == 0:
        return x
    H, W = x.shape[-2:]
    p = nn.functional.pad(x, (amp, amp, amp, amp), mode="replicate")
    return p[..., amp - dy:amp - dy + H, amp - dx:amp - dx + W]


def jitter_drive(model, tiles: torch.Tensor, amp: int) -> torch.Tensor:
    """[S, B, C, k] adım adım sürüş (S = T)."""
    return torch.stack([model.encode(shift(tiles, dy, dx, amp))[0] for dy, dx in offsets(model.cfg.steps, amp)])


def forward_jitter(model, tiles: torch.Tensor, amp: int, state=None) -> dict:
    """Mikrosakkadlı ileri geçiş: model.forward ile aynı çıktı biçimi (köprüsüz saf model)."""
    if amp <= 0:
        return model.forward(tiles, state)
    v = model.core(jitter_drive(model, tiles, amp), state)
    return model._pack(model.head_forward(model.readout(v, None)), v)
