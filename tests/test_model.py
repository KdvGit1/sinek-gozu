"""Model testleri.  Çalıştırma: python tests/test_model.py  (pytest ile de çalışır)"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # Windows konsolunda ✓ işareti
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from flybrain.checks import precision_check  # noqa: E402
from flybrain.detection import TargetBuilder, decode, detection_loss, match  # noqa: E402
from flybrain.model import FlyEyeDetector, ModelConfig, _SparseMatmul, _csr  # noqa: E402

PROCESSED = str(ROOT / "data" / "connectome" / "processed" / "right_ol_thr5")
_MODEL = {}


def model(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _MODEL:
        torch.manual_seed(0)
        _MODEL[key] = FlyEyeDetector(ModelConfig(processed_dir=PROCESSED, **kw))
    return _MODEL[key]


def test_sparse_matmul_gradcheck():
    rng = np.random.default_rng(0)
    N, E = 12, 40
    post, pre = rng.integers(0, N, E), rng.integers(0, N, E)
    pair = np.unique(np.stack([post, pre], 1), axis=0)          # tekrarsız kenarlar
    post, pre = pair[:, 0], pair[:, 1]
    o = np.lexsort((pre, post)); post, pre = post[o], pre[o]
    ot = np.lexsort((post, pre))
    args = (_csr(post, pre, N), torch.as_tensor(pre), _csr(pre[ot], post[ot], N), torch.as_tensor(post[ot]),
            torch.as_tensor(ot), torch.as_tensor(post), torch.as_tensor(pre))
    vals = torch.randn(len(post), dtype=torch.float64, requires_grad=True)
    x = torch.randn(N, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v, xx: _SparseMatmul.apply(v, xx, *args), (vals, x))


def test_forward_shapes_thermal_and_rgb():
    for ch in (1, 3):
        m = model(in_channels=ch, patch=4)
        out = m(torch.rand(2, ch, m.eye.height, m.eye.width))
        assert out["heat"].shape == (2, 796)
        assert out["offset"].shape == (2, 796, 2) and out["logsize"].shape == (2, 796, 2)
        assert torch.isfinite(out["state"]).all()


def test_reservoir_freezes_connectome():
    m = model(in_channels=1, patch=4, train_connectome=False)
    assert all(not p.requires_grad for p in m.connectome_parameters())
    out = m(torch.rand(1, 1, m.eye.height, m.eye.width))
    out["heat"].sum().backward()
    assert m.alpha_raw.grad is None and m.encoder[0].weight.grad is not None


def test_state_is_stable_over_many_steps():
    m = model(in_channels=1, patch=4)
    drive, _ = m.encode(torch.rand(1, 1, m.eye.height, m.eye.width))
    with torch.no_grad():
        v = m.core(drive, steps=100)
    assert torch.isfinite(v).all() and v.abs().max() < 100


def test_targets_decode_roundtrip():
    m = model(in_channels=1, patch=4)
    tb = TargetBuilder(m.eye)
    gt = torch.tensor([[40., 50., 12., 7.], [70., 90., 14., 8.]])
    heat, reg, pos = tb([gt])
    assert pos.sum() == 2
    logit = torch.where(pos, torch.tensor(10.), torch.tensor(-10.))
    dets = decode({"heat": logit, "offset": reg[..., :2], "logsize": reg[..., 2:]}, m.eye, 0.5)[0]
    tp, fp, fn = match(dets, gt, iou_thr=0.9)
    assert (tp, fp, fn) == (2, 0, 0)


def test_loss_decreases_on_one_batch():
    torch.manual_seed(0)
    m = FlyEyeDetector(ModelConfig(processed_dir=PROCESSED, in_channels=1, patch=4))
    tb = TargetBuilder(m.eye)
    tiles = torch.rand(2, 1, m.eye.height, m.eye.width) * 0.3
    gt = [torch.tensor([[40., 50., 12., 7.]]), torch.tensor([[60., 70., 12., 7.]])]
    for b, g in enumerate(gt):
        y, x, h, w = g[0].int().tolist()
        tiles[b, :, y:y + h, x:x + w] = 1.0
    heat, reg, pos = tb(gt)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        L = detection_loss(m(tiles), heat, reg, pos)["total"]
        L.backward()
        opt.step()
        losses.append(L.item())
    assert losses[-1] < 0.5 * losses[0], losses[::5]


def test_fp32_matches_fp64():
    m = model(in_channels=1, patch=4)
    tb = TargetBuilder(m.eye)
    heat, reg, pos = tb([torch.tensor([[40., 50., 12., 7.]])])
    r = precision_check(m, torch.rand(1, 1, m.eye.height, m.eye.width), heat, reg, pos)
    assert r["loss_rel_diff"] < 1e-4 and r["grad_rel_diff"] < 1e-3 and r["state_rel_diff"] < 1e-4, r


def test_box_regression_any_column_recovers_box():
    """reg_mode="box": kutunun içindeki her kolon tepe olarak seçildiğinde, hedefleri öğrenilmiş bir model gerçek
    kutuyu birebir vermeli (30.09 teşhisi: tespitlerin ~yarısı merkez değil komşu kolondan çıkıyordu)."""
    from flybrain.eye import HexEye
    eye = HexEye.from_processed(PROCESSED, 4)
    gt = torch.tensor([[50.0, 60.0, 14.0, 9.0]])
    heat, reg, pos, regm = TargetBuilder(eye, reg_mode="box")([gt])
    cols = torch.nonzero(regm[0]).flatten()
    assert len(cols) > 1 and bool(regm[0][pos[0]].all()), "merkez kolon ve komşuları dahil olmalı"
    for c in cols.tolist():
        logits = torch.full((1, eye.n_columns), -10.0)
        logits[0, c] = 10.0
        out = {"heat": logits, "offset": reg[..., :2], "logsize": reg[..., 2:]}
        d = decode(out, eye, score_thr=0.5)[0]
        assert len(d) == 1 and torch.allclose(d[0, :4], gt[0], atol=1e-3), (c, d)


def test_fast_backward_matches_reference():
    """Hızlı geri yayılım (SDDMM + önceden hesaplanmış okuma transpozu) eski/genel yolla aynı gradyanı vermeli.
    GPU varsa GPU'da (SDDMM yolu), yoksa CPU'da (okuma transpozu) test edilir."""
    import flybrain.model as M
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = FlyEyeDetector(ModelConfig(processed_dir=PROCESSED, readout="voltage", bridge_dim=8)).to(dev)
    x = torch.rand(40, 1, m.eye.height, m.eye.width, device=dev)          # 40 > SDDMM_CHUNK: parçalama da denenir

    def grads():
        m.zero_grad()
        out = m(x)
        (out["heat"].sigmoid().pow(2).sum() + out["offset"].sum() + out["logsize"].pow(2).sum()).backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}

    fast = grads()
    old_edge, old_ro = M._edge_grad, m._readout_mm
    M._edge_grad = lambda g, x, crow, col, post, pre: (g[post] * x[pre]).sum(-1)
    m._readout_mm = lambda k, a: m._spmm(getattr(m, f"{k}_crow"), getattr(m, f"{k}_col"), getattr(m, f"{k}_val"), a)
    try:
        ref = grads()
    finally:
        M._edge_grad = old_edge
        del m._readout_mm
    assert fast.keys() == ref.keys()
    for n in ref:
        rel = float((fast[n] - ref[n]).abs().max() / ref[n].abs().max().clamp(min=1e-12))
        assert rel < 1e-4, f"{n}: göreli fark {rel:.2e}"


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
