#!/usr/bin/env python3
"""Görüntülerde insan tespiti (sıkıştırılmış ağırlıkla; eğitim verisi gerekmez, işlenmiş konnektom gerekir).

Kullanım:
  py -3 tools/tespit.py agirliklar/uzun_75_termal.pt resim.jpg klasor/ --cikis tespitler --esik 0.3
  py -3 tools/tespit.py agirliklar/uzun_80_rgb.pt foto.jpg --cikis tespitler

Her görüntü için <cikis>/<ad>.jpg (kutular çizili) ve <cikis>/<ad>.txt (YOLO biçimi + skor:
"0 xc yc w h skor", görüntüye göre normalize) yazılır. Değerlendirme ayarı rapordakiyle aynıdır
(FLYBRAIN_DECODE=merkez7p2, FLYBRAIN_OVERLAP=0.5; ortam değişkeniyle değiştirilebilir).
RGB modeli uzun kenarı 640 piksele küçültülmüş görüntülerle eğitildi; büyük RGB görüntüler bu yüzden 640'a
küçültülerek işlenir ve kutular özgün boyuta geri ölçeklenir. Termal görüntüler olduğu gibi (gri tona çevrilerek) girer."""
from __future__ import annotations

import os

os.environ.setdefault("FLYBRAIN_DECODE", "merkez7p2")
os.environ.setdefault("FLYBRAIN_OVERLAP", "0.5")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flybrain.detection import decode, decode_cpu, merge_tiles, merge_top  # noqa: E402
from flybrain.model import HIZ75  # noqa: E402
from tools.agirlik import yukle  # noqa: E402
from tools.train_flybrain import IMAGE_EXTS, drop_outside, tiler_for  # noqa: E402

RGB_UZUN_KENAR = 640


def goruntu_oku(yol: Path, kanal: int) -> tuple[torch.Tensor, float]:
    """[C, H, W] 0–1 ve ölçek (işlenen / özgün). Eğitimdeki load_image ile aynı dönüşüm."""
    with Image.open(yol) as im:
        if kanal == 1:
            if im.mode.startswith("I;16") or im.mode in ("I", "F"):
                a = np.asarray(im, dtype=np.float32)
                lo, hi = np.percentile(a, (0.5, 99.5))
                return torch.from_numpy(np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1))[None], 1.0
            return torch.from_numpy(np.asarray(im.convert("L"), dtype=np.float32)[None] / 255.0), 1.0
        im = im.convert("RGB")
        k = min(1.0, RGB_UZUN_KENAR / max(im.size))
        if k < 1.0:
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
        a = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(np.ascontiguousarray(a)), k


@torch.no_grad()
def tespit_et(model, img: torch.Tensor, pad: str, cihaz, chunk: int = 256) -> torch.Tensor:
    """img [C, H, W] → [N, 5] (y, x, h, w, skor) piksel, skora göre azalan (en çok 300)."""
    _, H, W = img.shape
    karolar, kokenler = tiler_for(model, pad).extract(img)
    dets = []
    for i in range(0, len(karolar), chunk):
        o = model(karolar[i:i + chunk].to(cihaz))
        dets += (decode_cpu(o, model.eye, score_thr=0.01, max_det=100) if HIZ75 else
                 [d.cpu() for d in decode(o, model.eye, score_thr=0.01, max_det=100)])
    if HIZ75:
        return merge_top(dets, kokenler, H, W, 0.5, 300, True)
    m = drop_outside(merge_tiles(dets, kokenler, 0.5), H, W)
    return m[m[:, 4].argsort(descending=True)][:300] if len(m) else m


def goruntuler(girdiler: list[str]) -> list[Path]:
    out = []
    for g in map(Path, girdiler):
        if g.is_dir():
            out += sorted(p for p in g.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        elif g.is_file():
            out.append(g)
        else:
            sys.exit(f"bulunamadı: {g}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("agirlik", help="sıkıştırılmış ağırlık (agirliklar/*.pt)")
    ap.add_argument("girdi", nargs="+", help="görüntü dosyaları ya da klasörler")
    ap.add_argument("--cikis", default="tespitler")
    ap.add_argument("--esik", type=float, default=0.3, help="çizilecek / yazılacak en düşük skor")
    ap.add_argument("--cihaz", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cihaz = torch.device(a.cihaz)
    model = yukle(a.agirlik, cihaz)
    paket = torch.load(a.agirlik, map_location="cpu", weights_only=False)
    pad, kanal = paket["args"].get("pad", "zero"), model.cfg.in_channels
    cikis = Path(a.cikis)
    cikis.mkdir(parents=True, exist_ok=True)
    for yol in goruntuler(a.girdi):
        img, k = goruntu_oku(yol, kanal)
        d = tespit_et(model, img, pad, cihaz)
        d = d[d[:, 4] >= a.esik] if len(d) else d
        if len(d):
            d[:, :4] /= k                                            # özgün görüntü pikseline
        with Image.open(yol) as im:
            ciz = im.convert("RGB")
        W, H = ciz.size
        kalem = ImageDraw.Draw(ciz)
        satirlar = []
        for y, x, h, w, s in d.tolist():
            kalem.rectangle([x, y, x + w, y + h], outline=(255, 40, 40), width=max(1, round(max(W, H) / 400)))
            kalem.text((x, max(0, y - 11)), f"{s:.2f}", fill=(255, 40, 40))
            satirlar.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f} {s:.4f}")
        ciz.save(cikis / f"{yol.stem}.jpg", quality=92)
        (cikis / f"{yol.stem}.txt").write_text("\n".join(satirlar) + ("\n" if satirlar else ""), encoding="utf-8")
        print(f"{yol.name}: {len(satirlar)} tespit (skor ≥ {a.esik})")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
