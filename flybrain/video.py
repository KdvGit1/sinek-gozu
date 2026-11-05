"""
§89 H61 videoya biyolojik uyum: zamanlı ileri geçiş, aşamalı ödül / ceza kaybı, H47 sonraki kare tahmini, hareket ölçümleri.

Zamanlı ileri geçiş: klipteki kareler sırayla verilir; nöron durumu (v) ve kısa süreli plastisite (STP) belleği kareden kareye
taşınır, hiç sıfırlanmaz (yalnız sızıntıyla söner; kullanıcı kararı §88). İlk karede ağ dinlenimden başlar ve `adim_ilk` adım
(durağan rejimdeki T = 10) çalışır; sonraki her kare `adim` adım (≈ 33 ms'lik video karesi). Okuma her karede.

Aşamalı şekillendirme (kullanıcı fikri, H61; hayvan eğitimindeki "shaping"): ödül = pozitif terim, ceza = negatif terim.
  aşama 1 "varlık": insanın kutusunun içindeki her kolon ödül alır (boyut / ofset önemsiz); yanlış yere yanma hafif cezalı
  aşama 2 "kutu": eski hedefler (merkez kolon + Gauss) + ofset / boyut kaybı açılır; ceza normal
  aşama 3 "temizlik": yanlış yere yanmanın cezası artar
H47: her karede okumadan (fotoreseptör / L1–L3 okunmaz) doğrusal kod çözücüyle sonraki karenin kolon yamaları tahmin edilir;
öğretmen olayın kendisi (dış model yok).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .detection import focal_loss
from .microsaccade import shift


def video_ileri(model, klip: torch.Tensor, adim: int, adim_ilk: int, state=None, stp=None, ozellik: bool = False):
    """klip [K, B, C, H, W] → kare başına çıktı sözlükleri (ve istenirse okuma özellikleri), son (v, stp).
    state None → dinlenimden başlar (yeni sahne); ilk karede adim_ilk adım."""
    outs, feats = [], []
    v, s = state, stp
    for k in range(klip.shape[0]):
        d = model.drive(klip[k])
        n = adim_ilk if (k == 0 and v is None) else adim
        v, s = model.core(d, state=v, steps=n, stp_state=s, return_stp=True)
        f = model.readout(v, None)
        outs.append(model._pack(model.head_forward(f), v))
        if ozellik:
            feats.append(f)
    return outs, feats, v, s


def odak_kaybi(logits, heat, pos, neg_w: float = 1.0, alpha: float = 2, beta: float = 4):
    """CenterNet odak kaybı; negatif (ceza) terimi neg_w ile ölçeklenir. neg_w = 1 → flybrain.detection.focal_loss ile aynı."""
    if neg_w == 1.0:
        return focal_loss(logits, heat, pos, alpha, beta)
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    pos_l = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_l = -((1 - heat) ** beta) * (p ** alpha) * torch.log(1 - p) * (~pos)
    return (pos_l.sum() + neg_w * neg_l.sum()) / pos.sum().clamp(min=1)


def asama_kaybi(out: dict, hedef: tuple, asama: int, neg_w: float, w_offset: float = 1.0, w_size: float = 0.5) -> dict:
    """hedef = (heat, reg, pos, icerde) [B, 2C, ...] (BilateralTargets, reg_mode="box": icerde = kutunun içindeki kolonlar).
    Aşama 1: pozitif = kutunun içindeki her kolon (ısı 1), ofset / boyut yok. Aşama 2–3: eski merkez hedefleri + regresyon."""
    heat, reg, pos, icerde = (t.to(out["heat"].device) for t in hedef)
    if asama == 1:
        pos1 = pos | icerde
        l_heat = odak_kaybi(out["heat"], torch.maximum(heat, pos1.float()), pos1, neg_w)
        z = l_heat.new_zeros(())
        return {"total": l_heat, "heat": l_heat.detach(), "offset": z, "size": z}
    l_heat = odak_kaybi(out["heat"], heat, pos, neg_w)
    n = pos.sum().clamp(min=1)
    l_off = (nn.functional.l1_loss(out["offset"], reg[..., :2], reduction="none").sum(-1) * pos).sum() / n
    l_size = (nn.functional.l1_loss(out["logsize"], reg[..., 2:], reduction="none").sum(-1) * pos).sum() / n
    return {"total": l_heat + w_offset * l_off + w_size * l_size, "heat": l_heat.detach(), "offset": l_off.detach(),
            "size": l_size.detach()}


class SonrakiKare(nn.Module):
    """H47: kolonun okuma özelliklerinden (medulla ve sonrası + VPN; girdi nöronları okunmaz) sonraki karenin o kolondaki
    z-normalize gri yamasını tahmin eden doğrusal kod çözücü (bütün kolonlarda paylaşılan; CNN yok)."""

    def __init__(self, n_ozellik: int, patch: int):
        super().__init__()
        self.lin = nn.Linear(n_ozellik, patch * patch)

    def forward(self, f):                                              # [B, C, F] → [B, C, p²]
        return self.lin(f)


def kolon_yamalari(model, tiles: torch.Tensor) -> torch.Tensor:
    """tiles [B, Cin, H, W] → [B, C, p²] z-normalize gri yamalar (iki göz: sol göz aynalanmış karo; encode ile aynı düzen)."""
    g = tiles.mean(1, keepdim=True)
    p = model.eye.sample(g)
    if model.n_eyes == 2:
        p = torch.cat([p, model.eye.sample(torch.flip(g, dims=[3]))], dim=1)
    p = p.reshape(p.shape[0], p.shape[1], -1)
    return (p - p.mean(-1, keepdim=True)) / (p.std(-1, keepdim=True) + 0.02)


@torch.no_grad()
def yon_seciciligi(model, tiles: torch.Tensor, tipler: dict, adim: int, adim_ilk: int, hiz: int = 4, K: int = 6) -> dict:
    """Yapay hareket: karo kare başına `hiz` px sağa / sola / aşağı / yukarı kayar (K kare). Tip başına yanıt (3. kareden
    sonuna ortalama), DSI = (en çok − en az) / (en çok + en az) ve tercih edilen yön. tipler: ad → nöron indeksleri."""
    yon = {"sağ": (0, 1), "sol": (0, -1), "aşağı": (1, 0), "yukarı": (-1, 0)}
    R = {}
    for ad, (dy, dx) in yon.items():
        klip = torch.stack([shift(tiles, dy * hiz * k, dx * hiz * k, hiz * K) for k in range(K)])
        acts = []
        v, s = None, None
        for k in range(K):
            d = model.drive(klip[k])
            n = adim_ilk if k == 0 else adim
            v, s = model.core(d, state=v, steps=n, stp_state=s, return_stp=True)
            if k >= 2:
                acts.append(torch.tanh(torch.relu(v)))
        R[ad] = torch.stack(acts).mean(0)                              # [N, B]
    sonuc = {}
    for t_ad, idx in tipler.items():
        r = {y: float(R[y][idx].mean()) for y in yon}
        mx, mn = max(r.values()), min(r.values())
        sonuc[t_ad] = {"dsi": (mx - mn) / max(mx + mn, 1e-9), "tercih": max(r, key=r.get)}
    return sonuc


def sonme_ozeti(model, tip_adlari: np.ndarray, ornek=("Mi1", "Tm3", "Mi4", "Mi9", "T4a", "T5a", "LC11", "HSE")) -> dict:
    """Zaman sabitleri: λ = sigmoid(leak_raw) tip başına; τ = 1/λ adım. Dağılım ve seçili tipler."""
    lam = torch.sigmoid(model.leak_raw.detach().float().cpu())
    tau = 1 / lam
    q = torch.quantile(tau, torch.tensor([0.05, 0.5, 0.95])).tolist()
    tid = model.type_id.cpu().numpy()
    sec = {}
    for t_ad in ornek:
        m = np.flatnonzero(tip_adlari == t_ad)
        if len(m):
            sec[t_ad] = float(tau[int(tid[m[0]])])
    return {"tau_p05": q[0], "tau_ortanca": q[1], "tau_p95": q[2], "tau_en_uzun": float(tau.max()), "tipler": sec}
