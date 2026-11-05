#!/usr/bin/env python3
"""
§89 H61 video bölmesi (kullanıcı kararı 15.10): her veri seti VİDEO bazında ~%70 / %15 / %15 (kare sayısı değil; setler
olabildiğince dengeli). Ana modeller (termal uzun_75, RGB uzun_80_rgb) WiSARD ve BIRDSAI'nin eski eğitim videolarını gördü;
sızıntıyı azaltmak için ana modelin görmediği videolar önce test / val'e yerleşir (eski test → yeni test, eski val → yeni val;
artanı karşı bölmeyi doldurur), eksik kalan yer eski eğitim videolarından (tohumlu rastgele) doldurulur ve bunlar
`ana_model_gordu: true` diye işaretlenir; değerlendirme "ana modelin görmediği" alt kümeyi ayrıca raporlar. Artan
görülmemiş videolar eğitime gider.
Video olmayan kaynaklar alınmaz: RGBTDronePerson (numaralı ama kesintisiz video değil), HIT-UAV (görüntü), VisDrone-DET
(aynı uçuştan ~500 kare arayla anlık görüntüler: her kare bir sahne kesmesi). En az `--en-az` kare (daha kısa dizi atılır).
Kaynaklar: termal = WiSARD IR + BIRDSAI (data/yolo/thermal, manifest ile dizilere); RGB = WiSARD VIS (data/yolo/rgb640) +
VisDrone-VID (data/video/visdrone_vid, tools/visdrone_vid_hazirla.py).
Çıktı: data/video/video_bolme.json (+ özet results/rapor/89_video_bolme.md)
Kullanım: py -3 tools/video_bolme.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tools.canli_video as cv  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
CIKTI = ROOT / "data/video/video_bolme.json"
SETLER = {"thermal": ("wisard", "birdsai"), "rgb640": ("wisard", "visdrone_vid")}


def videolar(modality: str, en_az: int) -> dict:
    """set → [{anahtar, eski_bolme, ana_model_gordu, kareler[...]}]"""
    by = {}
    d = cv.diziler(modality, test_setleri=("*",))
    for (ds, sp, key), paths in d.items():
        if ds not in SETLER[modality] or len(paths) < en_az:
            continue
        by.setdefault(ds, []).append({"anahtar": key.split("/")[0], "eski_bolme": sp, "ana_model_gordu": sp == "train",
                                      "kareler": [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths]})
    if modality == "rgb640":
        kok = ROOT / "data/video/visdrone_vid/images"
        for sd in sorted(p for p in kok.iterdir() if p.is_dir()) if kok.is_dir() else []:
            kare = sorted(sd.glob("*.jpg"))
            if len(kare) >= en_az:
                by.setdefault("visdrone_vid", []).append({"anahtar": sd.name, "eski_bolme": "yok", "ana_model_gordu": False,
                                                          "kareler": [str(p.relative_to(ROOT)).replace("\\", "/") for p in kare]})
    return by


def bol(vids: list, rng: random.Random, oran=(0.70, 0.15, 0.15)) -> dict:
    n = len(vids)
    n_val = max(1, round(oran[1] * n))
    n_test = max(1, round(oran[2] * n))
    havuz = {k: [v for v in vids if v["eski_bolme"] == k] for k in ("test", "val", "yok", "train")}
    for h in havuz.values():
        rng.shuffle(h)
    out = {"train": [], "val": [], "test": []}
    hedef_n = {"val": n_val, "test": n_test}
    for hedef in ("val", "test"):                                       # 1) eski val → yeni val, eski test → yeni test
        while len(out[hedef]) < hedef_n[hedef] and havuz[hedef]:
            out[hedef].append(havuz[hedef].pop())
    artan = havuz["yok"] + havuz["val"] + havuz["test"]                 # 2) görülmemiş artanlar sırayla val / test
    havuz["val"], havuz["test"], havuz["yok"] = [], [], []
    sira = ["val", "test"]
    while artan and any(len(out[h]) < hedef_n[h] for h in sira):
        h = sira[0] if len(out[sira[0]]) < hedef_n[sira[0]] else sira[1]
        if len(out[h]) < hedef_n[h]:
            out[h].append(artan.pop())
        sira.reverse()
    havuz["yok"] = artan
    for hedef in ("val", "test"):                                       # 3) eksik: eski eğitimden (ana model gördü)
        while len(out[hedef]) < hedef_n[hedef] and havuz["train"]:
            out[hedef].append(havuz["train"].pop())
    out["train"] = havuz["train"] + havuz["yok"] + havuz["test"] + havuz["val"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-az", type=int, default=24)
    ap.add_argument("--tohum", type=int, default=89)
    a = ap.parse_args()
    rng = random.Random(a.tohum)
    J = {"olusturma": time.strftime("%d.%m.%Y %H:%M"), "kural": __doc__.strip().splitlines()[0], "tohum": a.tohum,
         "en_az_kare": a.en_az, "modaliteler": {}}
    L = ["# §89 Video bölmesi (%70 / %15 / %15, video bazında)", "",
         "Ana modelin görmediği videolar önce test / val'e yerleştirildi; eksik kalan eski eğitim videolarından dolduruldu",
         "(\"ana model gördü\"). Video olmayan kaynaklar (RGBT, HIT-UAV, VisDrone-DET anlık görüntüleri) alınmadı.", "",
         "| modalite | set | eğitim (video / kare) | val (video / kare, ana model gördü) | test (video / kare, ana model gördü) | atılan kısa |",
         "|---|---|---|---|---|---|"]
    for mod in ("thermal", "rgb640"):
        by = videolar(mod, a.en_az)
        J["modaliteler"][mod] = {}
        for ds in SETLER[mod]:
            vids = by.get(ds, [])
            sp = bol(vids, rng)
            J["modaliteler"][mod][ds] = sp
            f = lambda x: f"{len(x)} / {sum(len(v['kareler']) for v in x):,}".replace(",", ".")
            g = lambda x: sum(v["ana_model_gordu"] for v in x)
            L.append(f"| {mod} | {ds} | {f(sp['train'])} | {f(sp['val'])}, {g(sp['val'])} | {f(sp['test'])}, {g(sp['test'])} | — |")
            print(mod, ds, {k: len(v) for k, v in sp.items()}, "gördü val/test", g(sp["val"]), g(sp["test"]), flush=True)
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    CIKTI.write_text(json.dumps(J, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results/rapor/89_video_bolme.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"✓ {CIKTI}")


if __name__ == "__main__":
    main()
