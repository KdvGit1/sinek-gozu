"""§63: Faz 1/2 mekanizmaları — özdeşlik başlangıcı (yorulma, modülatör, gürültü eval'de) ve geri yayılım."""
import argparse
import shlex
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tools.train_flybrain as tf  # noqa: E402

# §47 uzun eğitim ayarı (results/flybrain/uzun12s_ayar.txt); mekanizma testleri bu tabanın üstüne kurulur
AYAR = ("--adapt column --photo-aug --hard-neg 0.2 --bridge 0 --head-context none --head-hidden 0 --merkez "
        "--okuma-fotoreseptorsuz --lamina-girisi --agirlik-norm l2 --aktivasyon sat --init-gain 2.0 --alpha-lr-mult 10")


def build(extra=""):
    a = tf.make_parser().parse_args(shlex.split(
        "--modality thermal --patch 8 --threshold 5 --readout voltage --pad replicate --steps 3 --iki-lob " + AYAR + " " + extra))
    torch.manual_seed(0)
    return tf.build(a, torch.device("cpu"))


def main():
    base = build()
    sd = base.state_dict()
    torch.manual_seed(1)
    x = torch.rand(2, 1, base.eye.height, base.eye.width)
    base.eval()
    with torch.no_grad():
        ref = base(x)["heat"]
    for flag, ident in [("--yorulma", True), ("--modulator", True), ("--gurultu 0.05", True), ("--yumusak", False),
                        ("--sant", False), ("--mikrosakkad 4", False), ("--superpozisyon", False)]:
        m = build(flag)
        miss, unexp = m.load_state_dict(sd, strict=False)
        m.eval()
        with torch.no_grad():
            h = m(x)["heat"]
        d = float((h - ref).abs().max())
        m.train()
        out = m(x)
        out["heat"].sum().backward()
        g = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
        print(f"{flag:16s} yeni param {len(miss):2d}  çıktı farkı {d:.2e}  {'(özdeş olmalı)' if ident else ''}  grad>0 {g > 0}",
              flush=True)
        if ident:
            assert d < 1e-3, flag


if __name__ == "__main__":
    main()
