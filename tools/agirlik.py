#!/usr/bin/env python3
"""Sıkıştırılmış ağırlıklar: eğitim kontrol noktasından (best.pt, ~111 MB) yalnız öğrenilen parametreleri ve eğitim ayarlarını
alır (~0,2 MB). Konnektomdan türetilen sabit tamponlar (seyrek bağlantı matrisi, okuma matrisleri, tip indeksleri) yüklemede
data/connectome/processed/ altındaki işlenmiş konnektomdan yeniden kurulur; sonuç kontrol noktasıyla birebir aynıdır.

Kullanım:
  py -3 tools/agirlik.py disa-aktar results/flybrain/uzun_75/best.pt agirliklar/uzun_75_termal.pt
  py -3 tools/agirlik.py dogrula agirliklar/uzun_75_termal.pt results/flybrain/uzun_75/best.pt
Python'dan:
  from tools.agirlik import yukle
  model = yukle("agirliklar/uzun_75_termal.pt", "cuda")"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.train_flybrain import build  # noqa: E402


def disa_aktar(kontrol_noktasi: str | Path, cikis: str | Path, ad: str = "") -> dict:
    ck = torch.load(kontrol_noktasi, map_location="cpu", weights_only=False)
    m = build(argparse.Namespace(**ck["args"]), torch.device("cpu"))
    adlar = {k for k, _ in m.named_parameters()}
    paket = {"bicim": "sinek-gozu/1", "ad": ad or Path(kontrol_noktasi).parent.name, "args": ck["args"],
             "iterasyon": ck.get("iter"), "ema": bool(ck.get("ema_agirlik", False)),
             "parametreler": {k: v.clone() for k, v in ck["model"].items() if k in adlar}}
    Path(cikis).parent.mkdir(parents=True, exist_ok=True)
    torch.save(paket, cikis)
    return paket


def yukle(yol: str | Path, cihaz="cpu"):
    """Sıkıştırılmış ağırlıktan değerlendirme kipinde FlyEyeDetector."""
    paket = torch.load(yol, map_location="cpu", weights_only=False)
    if paket.get("bicim") != "sinek-gozu/1":
        raise ValueError(f"{yol}: sıkıştırılmış ağırlık değil (best.pt için train_flybrain.build + load_state_dict)")
    m = build(argparse.Namespace(**paket["args"]), torch.device(cihaz))
    eksik, fazla = m.load_state_dict(paket["parametreler"], strict=False)
    param = {k for k, _ in m.named_parameters()}
    if fazla or (set(eksik) & param):
        raise RuntimeError(f"uyumsuz ağırlık: eksik parametre {sorted(set(eksik) & param)[:5]}, fazla {fazla[:5]}")
    return m.to(cihaz).eval()


@torch.no_grad()
def dogrula(sikistirilmis: str | Path, kontrol_noktasi: str | Path) -> float:
    """İki yolla kurulan modelin aynı rastgele karolarda en büyük çıktı farkı (0 beklenir)."""
    ck = torch.load(kontrol_noktasi, map_location="cpu", weights_only=False)
    a = build(argparse.Namespace(**ck["args"]), torch.device("cpu"))
    a.load_state_dict(ck["model"])
    a.eval()
    b = yukle(sikistirilmis, "cpu")
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, a.cfg.in_channels, a.eye.height, a.eye.width, generator=g)
    oa, ob = a(x), b(x)
    return max(float((oa[k] - ob[k]).abs().max()) for k in ("heat", "offset", "logsize"))


def main():
    ap = argparse.ArgumentParser()
    alt = ap.add_subparsers(dest="komut", required=True)
    d = alt.add_parser("disa-aktar")
    d.add_argument("kontrol_noktasi")
    d.add_argument("cikis")
    d.add_argument("--ad", default="")
    v = alt.add_parser("dogrula")
    v.add_argument("sikistirilmis")
    v.add_argument("kontrol_noktasi")
    a = ap.parse_args()
    if a.komut == "disa-aktar":
        p = disa_aktar(a.kontrol_noktasi, a.cikis, a.ad)
        n = sum(t.numel() for t in p["parametreler"].values())
        print(f"{a.cikis}: {len(p['parametreler'])} tensör, {n} parametre, iterasyon {p['iterasyon']}")
    else:
        print(f"en büyük çıktı farkı: {dogrula(a.sikistirilmis, a.kontrol_noktasi):.3g}")


if __name__ == "__main__":
    main()
