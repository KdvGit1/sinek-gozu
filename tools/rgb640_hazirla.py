#!/usr/bin/env python3
"""
§79 RGB verisini uzun kenarı 640 olacak şekilde küçültür (kullanıcı kararı 14.10: "640'a küçült").
data/yolo/rgb → data/yolo/rgb640. YOLO etiketleri normalize olduğu için aynen kopyalanır.
WiSARD RGB 1920×1080 / 2720×1530 → 640×360 (insanlar medyan ~61 px → ~20 px; termal WiSARD ile aynı ölçek). Uzun kenarı
zaten ≤ 640 olanlar (RGBTDronePerson görünür 640×512) olduğu gibi kopyalanır. YOLO da eğitimde 640'a küçülterek görüyor.
Kullanım: py -3 tools/rgb640_hazirla.py [--isci 4]   (dosya olarak çalıştırılmalı: Windows'ta çok süreçli)
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "data" / "yolo" / "rgb", ROOT / "data" / "yolo" / "rgb640"
LONG = 640


def isle(rel: str):
    s, d = SRC / rel, DST / rel
    if d.is_file():
        return rel, "var"
    d.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(s)
    W, H = im.size
    if max(W, H) > LONG:
        k = LONG / max(W, H)
        im = im.convert("RGB").resize((max(1, round(W * k)), max(1, round(H * k))), Image.LANCZOS)
        im.save(d, quality=95) if d.suffix.lower() in (".jpg", ".jpeg") else im.save(d)
        r = "küçültüldü"
    else:
        shutil.copy2(s, d)
        r = "kopyalandı"
    ls = SRC / Path(rel.replace("images/", "labels/", 1)).with_suffix(".txt")
    if ls.is_file():
        ld = DST / Path(rel.replace("images/", "labels/", 1)).with_suffix(".txt")
        ld.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ls, ld)
    return rel, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isci", type=int, default=4)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(SRC / "manifest.csv", encoding="utf-8")))
    rels = [r["output"].replace("\\", "/") for r in rows]
    DST.mkdir(parents=True, exist_ok=True)
    t0, say = time.time(), {}
    with Pool(a.isci) as p:
        for i, (_, r) in enumerate(p.imap_unordered(isle, rels, chunksize=16)):
            say[r] = say.get(r, 0) + 1
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(rels)} ({time.time() - t0:.0f} sn) {say}", flush=True)
    shutil.copy2(SRC / "manifest.csv", DST / "manifest.csv")
    y = (SRC / "rgb.yaml").read_text(encoding="utf-8")
    (DST / "rgb640.yaml").write_text(y.replace(str(SRC), str(DST)).replace(SRC.as_posix(), DST.as_posix()), encoding="utf-8")
    print(f"bitti: {len(rels)} görüntü, {time.time() - t0:.0f} sn, {say}")


if __name__ == "__main__":
    main()
