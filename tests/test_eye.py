"""Göz örnekleyicisi testleri.  Çalıştırma: python tests/test_eye.py  (pytest ile de çalışır)"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # Windows konsolunda ✓ işareti
sys.path.insert(0, str(ROOT))
from flybrain.eye import HexEye, Tiler, _max_rectangle  # noqa: E402

PROCESSED = ROOT / "data" / "connectome" / "processed" / "right_ol_thr5"


def eye(p):
    return HexEye.from_processed(PROCESSED, patch=p)


def test_patches_do_not_overlap_and_count_is_796():
    for p in (1, 2, 3, 4, 8):
        e = eye(p)
        assert e.n_columns == 796
        assert e.footprint().sum() == 796 * p * p, f"p={p}: yamalar çakışıyor"


def test_neighbors_are_adjacent_patches():
    e = eye(4)
    nb = e.neighbors()
    c = e.centers()
    d = [np.linalg.norm(c[i] - c[j]) for i in range(e.n_columns) for j in nb[i] if j >= 0]
    # Tuğla düzeninde komşu merkezler: yatay-diyagonal ~ p·√1.25, dikey p
    assert max(d) <= 4 * 1.12 + 1e-6 and min(d) >= 4 - 1e-6
    assert (nb >= 0).sum(1).max() == 6


def test_sample_returns_exact_pixels():
    e = eye(4)
    B, C = 2, 3
    tiles = torch.arange(B * C * e.height * e.width, dtype=torch.float32).reshape(B, C, e.height, e.width)
    s = e.sample(tiles)
    assert s.shape == (B, e.n_columns, C, 4, 4)
    for i in (0, 100, 795):
        t, l = e.top[i], e.left[i]
        assert torch.equal(s[1, i, 2], tiles[1, 2, t:t + 4, l:l + 4])


def test_render_inverts_sample():
    e = eye(4)
    tiles = torch.rand(1, 3, e.height, e.width)
    rec = e.render(e.sample(tiles)[0])
    fp = torch.as_tensor(e.footprint())
    assert torch.allclose(rec[:, fp], tiles[0][:, fp])


def test_max_rectangle_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(30):
        m = rng.random((7, 9)) > 0.3
        y, x, h, w = _max_rectangle(m)
        assert m[y:y + h, x:x + w].all()
        best = max(((y1 - y0) * (x1 - x0)) for y0 in range(7) for y1 in range(y0 + 1, 8)
                   for x0 in range(9) for x1 in range(x0 + 1, 10) if m[y0:y1, x0:x1].all()) if m.any() else 0
        assert h * w == best


def test_tiler_covers_every_pixel():
    for p in (2, 4, 8):
        t = Tiler(eye(p), overlap=0.25)
        for H, W in ((1080, 1920), (512, 640), (100, 90)):
            assert t.coverage(H, W).min() >= 1, f"p={p}, {H}x{W}: görülmeyen piksel var"


def test_tiler_extract_matches_image():
    e = eye(4)
    t = Tiler(e, overlap=0.25)
    img = torch.rand(3, 300, 500)
    tiles, org = t.extract(img)
    assert tiles.shape[1:] == (3, e.height, e.width)
    for k in (0, len(org) // 2, len(org) - 1):
        y, x = org[k]
        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(300, y + e.height), min(500, x + e.width)
        assert torch.equal(tiles[k][:, y0 - y:y1 - y, x0 - x:x1 - x], img[:, y0:y1, x0:x1])


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"✓ {name}")
            except AssertionError as ex:
                fails += 1
                print(f"✗ {name}: {ex}")
    sys.exit(1 if fails else 0)
