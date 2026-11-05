"""§79 termal → RGB aktarımı: kodlayıcının ilk katmanı 1 → 3 kanala /3 ile genişletilince, R = G = B (gri) bir karo RGB
modelde termal modeldekiyle aynı çıktıyı vermeli. Kaynak: sıkıştırılmış termal ağırlık. CPU'da çalışır. Kullanım: py -3 tests/test_rgb79.py"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tools.train_flybrain as tf  # noqa: E402
from tools.agirlik import yukle  # noqa: E402

if __name__ == "__main__":
    ck_path = ROOT / "agirliklar" / "uzun_75_termal.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    dev = torch.device("cpu")
    torch.manual_seed(0)
    mt = yukle(ck_path, dev)
    a_r = argparse.Namespace(**{**ck["args"], "modality": "rgb640"})
    mr = tf.build(a_r, dev)
    notes = tf.load_init(mr, str(ck_path))
    print("\n".join(n for n in notes if "encoder" in n or "uyuşmuyor" in n) or "(not yok)")
    assert any("termal → RGB" in n for n in notes), "kanal genişletme uygulanmadı"
    mt.eval(); mr.eval()
    g = torch.rand(2, 1, mt.eye.height, mt.eye.width)
    with torch.no_grad():
        ot, orr = mt(g), mr(g.expand(-1, 3, -1, -1).contiguous())
    d = float((ot["heat"] - orr["heat"]).abs().max())
    print(f"ısı farkı (gri karo, termal ↔ RGB): {d:.2e}")
    assert d < 1e-4, d
    c = torch.rand(2, 3, mt.eye.height, mt.eye.width)
    with torch.no_grad():
        oc = mr(c)
    print("renkli karo: çıktı biçimi", tuple(oc["heat"].shape))
    print("✓ §79 RGB aktarım testi geçti")
