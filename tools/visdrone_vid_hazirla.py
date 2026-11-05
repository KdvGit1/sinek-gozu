#!/usr/bin/env python3
"""
§89 VisDrone2019-VID-train → video eğitimi için hazır kareler (H61; kullanıcı kararı 15.10: lisans eğitime engel değil).
- Etiket satırı: kare, kimlik, x, y, w, h, skor, sınıf, kesilme, örtülme. İnsan = sınıf 1 (pedestrian) ve 2 (people), skor 1.
  Yok sayılan bölgeler (sınıf 0 ya da skor 0) görüntünün ortalama rengiyle maskelenir (convert_to_yolo ile aynı kural);
  araçlar / others atılır (arka plan).
- Ölçek klip başına: insan boyu (√(w·h)) ortancası RGB modelinin alıştığı boya (rgb640 WiSARD ortancası 17 px) gelsin
  diye s = min(1, 17 / ortanca); uzun kenar en az 480 px. Lanczos, JPEG kalite 90.
- Çıktı (E:, SSD değil): data/video/visdrone_vid/images/<klip>/<kare:07d>.jpg, labels/... (YOLO: 0 xc yc w h),
  ozet.json (klip başına özgün boyut, ölçek, kare, insan kutusu, ortanca boy). Kimlikler ozet'e ayrıca yazılmaz (gerekirse
  ham etiketten okunur).
Kullanım: py -3 tools/visdrone_vid_hazirla.py [--isci 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
HAM = ROOT / "data/raw/visdrone/VisDrone2019-VID-train"
CIKTI = ROOT / "data/video/visdrone_vid"
HEDEF_BOY = 17.0


def klip(seq: str) -> dict:
    a = np.loadtxt(HAM / "annotations" / f"{seq}.txt", delimiter=",", ndmin=2)
    kareler = sorted((HAM / "sequences" / seq).glob("*.jpg"))
    with Image.open(kareler[0]) as im:
        W, H = im.size
    insan = a[np.isin(a[:, 7], [1, 2]) & (a[:, 6] == 1)]
    orta = float(np.median(np.sqrt(insan[:, 4] * insan[:, 5]))) if len(insan) else 0.0
    s = min(1.0, HEDEF_BOY / max(orta, 1e-6)) if len(insan) else 0.45   # insansız klip: tipik ölçek (yalnız negatif)
    s = max(s, 480 / max(W, H))
    nW, nH = max(1, round(W * s)), max(1, round(H * s))
    (CIKTI / "images" / seq).mkdir(parents=True, exist_ok=True)
    (CIKTI / "labels" / seq).mkdir(parents=True, exist_ok=True)
    n_kutu = 0
    for p in kareler:
        f = int(p.stem)
        r = a[a[:, 0] == f]
        kisi = r[np.isin(r[:, 7], [1, 2]) & (r[:, 6] == 1)]
        yok = r[(r[:, 7] == 0) | (r[:, 6] == 0)]
        hedef = CIKTI / "images" / seq / f"{f:07d}.jpg"
        if not hedef.is_file():
            with Image.open(p) as im:
                im = im.convert("RGB")
                if len(yok):
                    ort = im.resize((1, 1)).getpixel((0, 0))
                    d = ImageDraw.Draw(im)
                    for x, y, w, h in yok[:, 2:6]:
                        d.rectangle([x, y, x + w, y + h], fill=ort)
                im.resize((nW, nH), Image.LANCZOS).save(hedef, quality=90)
        satir = []
        for x, y, w, h in kisi[:, 2:6]:
            x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            satir.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        n_kutu += len(satir)
        (CIKTI / "labels" / seq / f"{f:07d}.txt").write_text("\n".join(satir) + ("\n" if satir else ""), encoding="utf-8")
    return {"klip": seq, "ozgun": [W, H], "olcek": round(s, 4), "yeni": [nW, nH], "kare": len(kareler),
            "insan_kutusu": n_kutu, "ortanca_boy_ozgun": round(orta, 1), "ortanca_boy_yeni": round(orta * s, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isci", type=int, default=5)
    a = ap.parse_args()
    seqs = sorted(p.stem for p in (HAM / "annotations").glob("*.txt"))
    t0 = time.time()
    with Pool(a.isci) as pool:
        ozet = []
        for i, r in enumerate(pool.imap_unordered(klip, seqs)):
            ozet.append(r)
            print(f"[{i + 1}/{len(seqs)}] {r['klip']}: {r['kare']} kare, ölçek {r['olcek']}, {r['yeni'][0]}×{r['yeni'][1]}, "
                  f"insan {r['insan_kutusu']}, boy {r['ortanca_boy_yeni']} px", flush=True)
    ozet.sort(key=lambda r: r["klip"])
    (CIKTI / "ozet.json").write_text(json.dumps({"kaynak": str(HAM), "hedef_boy_px": HEDEF_BOY, "klipler": ozet},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ {len(ozet)} klip, {sum(r['kare'] for r in ozet)} kare, {sum(r['insan_kutusu'] for r in ozet)} insan kutusu, "
          f"{time.time() - t0:.0f} sn → {CIKTI}")


if __name__ == "__main__":
    sys.exit(main())
