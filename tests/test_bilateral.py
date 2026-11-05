"""H35 (§56): iki optik loblu model — kolon eşlemesi, ileri geçiş biçimleri, görüş alanı B maskesi, hedefler ve
iki gözlü kod çözme."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flybrain.connectome import Connectome  # noqa: E402
from flybrain.detection import BilateralTargets, TargetBuilder, decode, fuse_two_eyes  # noqa: E402
from flybrain.model import FlyEyeDetector, ModelConfig  # noqa: E402

BI = ROOT / "data" / "connectome" / "processed" / "bi_ol_merkez_thr5"
SAF = dict(patch=8, readout="voltage", head_hidden=0, head_context="none", readout_photo=False, input_lamina=True,
           weight_norm="l2", activation="sat", init_gain=2.0, adapt="column", steps=3)


def _conn():
    return Connectome.load(BI)


def test_kolonlar():
    c = _conn()
    assert c.n_eyes == 2 and c.n_columns == 2 * c.n_eye_columns
    col = c.neuron_column
    eye = c.nodes.eye.fillna("").to_numpy()
    assert (col[(eye == "R") & (col >= 0)] < c.n_eye_columns).all()
    assert (col[(eye == "L") & (col >= 0)] >= c.n_eye_columns).all()
    assert (col[eye == ""] == -1).all()
    # iki gözde de fotoreseptör ve L1–L3 girdisi var
    t = c.nodes.type.to_numpy()
    for e, lo, hi in (("R", 0, c.n_eye_columns), ("L", c.n_eye_columns, c.n_columns)):
        m = (eye == e) & np.isin(t, ["R7", "R8", "L1"]) & (col >= lo) & (col < hi)
        assert m.sum() > 500


def test_ileri_ve_decode():
    c = _conn()
    for view in ("A", "B"):
        m = FlyEyeDetector(ModelConfig(processed_dir=str(BI), view=view, **SAF), c)
        x = torch.rand(2, 1, m.eye.height, m.eye.width)
        out = m(x)
        assert out["heat"].shape == (2, 2 * m.eye.n_columns)
        assert out["eyes"][1] == view
        if view == "B":
            drive, _ = m.encode(x)
            vis = m.eye_visible.bool()
            assert (drive[:, ~vis] == 0).all() and (drive[:, vis] != 0).any()
        d = decode(out, m.eye, score_thr=0.0, max_det=20)
        assert len(d) == 2 and all(t.shape[1] == 5 for t in d)


def test_hedefler_ayna():
    c = _conn()
    m = FlyEyeDetector(ModelConfig(processed_dir=str(BI), **SAF), c)
    tb = TargetBuilder(m.eye)
    bt = BilateralTargets(tb, m.eye.width)
    box = torch.tensor([[100.0, 40.0, 20.0, 10.0]])                 # sol yarıda bir insan
    heat, reg, pos = bt([box])
    C = m.eye.n_columns
    assert heat.shape == (1, 2 * C) and pos[0, :C].sum() == 1 and pos[0, C:].sum() == 1
    cy, cx = m.eye.centers()[pos[0, :C].numpy().argmax()]
    cyl, cxl = m.eye.centers()[pos[0, C:].numpy().argmax()]
    assert abs(cx - 45) < 8 and abs(cxl - (m.eye.width - 45)) < 8   # sol göz aynalı konumda görüyor
    vis = torch.as_tensor(m.eye.centers()[:, 1] >= m.eye.width * 0.4)
    heat_b, _, pos_b = BilateralTargets(tb, m.eye.width, vis)([box])
    assert pos_b[0, :C].sum() == 0 and pos_b[0, C:].sum() == 1       # B: sağ göz sol yarıyı görmez


def test_uzlasma():
    a = torch.tensor([[10.0, 10, 20, 10, 0.8], [50, 50, 20, 10, 0.6]])
    b = torch.tensor([[11.0, 10, 20, 10, 0.6]])
    f = fuse_two_eyes(a, b)
    assert len(f) == 2
    assert torch.isclose(f[:, 4].max(), torch.tensor(0.7)) and torch.isclose(f[:, 4].min(), torch.tensor(0.3))
    g = fuse_two_eyes(a, b, zone=(0.0, 30.0))                        # ikinci kutu ortak şeridin dışında: cezasız
    assert torch.isclose(g[:, 4].min(), torch.tensor(0.6))


if __name__ == "__main__":
    for f in (test_kolonlar, test_ileri_ve_decode, test_hedefler_ayna, test_uzlasma):
        f()
        print("ok", f.__name__, flush=True)
