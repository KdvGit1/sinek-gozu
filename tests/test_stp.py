"""§65 H53: işaretli kısa süreli plastisite — özdeşlik başlangıcı, φ'ye gradyan (yorulmanın düz bölge hatası olmasın),
φ > 0 kolaylaştırır (çıkış artar), φ < 0 yorar (azalır)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_faz1 import build  # noqa: E402


def main():
    base = build("--iki-lob")
    sd = base.state_dict()
    torch.manual_seed(1)
    x = torch.rand(2, 1, base.eye.height, base.eye.width)
    base.eval()
    with torch.no_grad():
        ref = base(x)["heat"]
    m = build("--iki-lob --stp")
    miss, _ = m.load_state_dict(sd, strict=False)
    assert sorted(miss) == ["stp_phi", "stp_tau"], miss
    m.eval()
    with torch.no_grad():
        d = float((m(x)["heat"] - ref).abs().max())
    print(f"özdeşlik farkı {d:.2e}")
    assert d < 1e-5
    m.train()
    m(x)["heat"].sum().backward()
    g = float(m.stp_phi.grad.abs().sum())
    print(f"φ gradyanı {g:.3e} (yorulmanın U gradyanı ~4,5e-5 ile çarpılıyordu)")
    assert g > 0
    m.eval()
    acts = {}
    for val in (-3.0, 3.0):
        with torch.no_grad():
            m.stp_phi.fill_(val)
            drive, _ = m.encode(x)
            acts[val] = float(m._act(m.core(drive)).mean())
    print(f"ortalama etkinlik: φ<0 (yorulma) {acts[-3.0]:.4f}  φ>0 (kolaylaşma) {acts[3.0]:.4f}")
    assert acts[3.0] > acts[-3.0]
    print("✓ H53 testleri geçti")


if __name__ == "__main__":
    main()
