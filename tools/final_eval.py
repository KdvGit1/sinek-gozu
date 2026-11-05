#!/usr/bin/env python3
"""
Son değerlendirme: sinek modeli(leri), çekirdeksiz kontrol ve YOLO baseline'ı aynı ölçüm koduyla karşılaştırır.

Her model için val ve test kümesinde:
  * mAP50 (akademik standart) ve AP@IoU 0,3 (küçük insanlarda kutu birkaç piksel kayınca da "buldu" sayılır)
  * Pratik ölçüt (arama-kurtarma): görüntü başına ortalama 0,1 / 0,5 / 1 yanlış alarm kabul edildiğinde duyarlılık
    (FPPI = false positives per image; eşleşme IoU ≥ 0,3)
  * hepsi toplam ve kaynak set bazında
Tespitler bir kez çıkarılıp results/son_degerlendirme/<model>_<küme>.pt olarak saklanır (metrikler yeniden
hesaplanmak istenirse modeller tekrar çalıştırılmaz).

Kullanım:
  py -3 tools/final_eval.py --fly agirliklar/uzun_75_termal.pt --yolo thermal --tag son_termal
  py -3 tools/final_eval.py --modality rgb640 --yolo rgb640 --fly agirliklar/uzun_80_rgb.pt --tag son_rgb
  python tools/final_eval.py ... --limit 40 --tag _deneme          # hızlı deneme
Çıktı: results/rapor/<tag>.{md,json,png}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# standart değerlendirme ayarı (rapordaki sayılar): komşu kolonlarla merkez + p² kod çözme, karo bindirmesi 0,5
os.environ.setdefault("FLYBRAIN_DECODE", "merkez7p2")
os.environ.setdefault("FLYBRAIN_OVERLAP", "0.5")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flybrain.detection import DECODE_MODE, average_precision, decode, decode_cpu, match_flags, merge_tiles, merge_top  # noqa: E402
from flybrain.model import HIZ75  # noqa: E402
from tools.train_flybrain import (EVAL_OVERLAP, _EvalImages, build, dataset_of, drop_outside, list_images,  # noqa: E402
                                  load_boxes, tiler_for)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CACHE = ROOT / "results" / "son_degerlendirme"
REPORT = ROOT / "results" / "rapor"
FPPI_POINTS = (0.1, 0.5, 1.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------- tespit toplama
@torch.no_grad()
def collect_fly(run_dir: Path, images, device, chunk=256):
    """run_dir: eğitim klasörü (içinde best.pt) ya da sıkıştırılmış ağırlık dosyası (tools/agirlik.py, ör. agirliklar/*.pt)."""
    if run_dir.is_file():
        from tools.agirlik import yukle
        model = yukle(run_dir, device)
        ck = torch.load(run_dir, map_location="cpu", weights_only=False)
        ck["iter"] = ck.get("iterasyon")
    else:
        ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        model = build(argparse.Namespace(**ck["args"]), device)
        model.load_state_dict(ck["model"])
    model.eval()
    pad = ck["args"].get("pad", "zero")
    loader = torch.utils.data.DataLoader(_EvalImages(images, model.cfg.in_channels, tiler_for(model, pad)),
                                         batch_size=None, num_workers=3)
    out, pending, n = {}, [], 0

    def flush():
        tiles = torch.cat([p[1] for p in pending]).to(device)
        dets = []
        for i in range(0, len(tiles), chunk):
            o = model(tiles[i:i + chunk])
            dets += (decode_cpu(o, model.eye, score_thr=0.01, max_det=100) if HIZ75 else     # §77: aynı sonuç, hızlı
                     [d.cpu() for d in decode(o, model.eye, score_thr=0.01, max_det=100)])
        s = 0
        for k, t, org, H, W, gts in pending:
            if HIZ75:                                                      # §77: önek NMS, aynı sonuç
                m = merge_top(dets[s:s + len(t)], org, H, W, 0.5, 300, True)
            else:
                m = drop_outside(merge_tiles(dets[s:s + len(t)], org, 0.5), H, W)
                m = m[m[:, 4].argsort(descending=True)][:300] if len(m) else m
            s += len(t)
            out[k] = (m, gts)
        pending.clear()

    for item in loader:
        pending.append(item)
        n += len(item[1])
        if n >= chunk:
            flush()
            n = 0
    if pending:
        flush()
    return [out[k] for k in range(len(images))], {"best_iter": ck["iter"], "train_hours": ck.get("train_sec", 0) / 3600}


@torch.no_grad()
def collect_yolo(modality: str, images):
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "results" / "baseline" / f"yolov8n_{modality}" / "weights" / "best.pt"))
    res = []
    for i in range(0, len(images), 32):
        chunk = images[i:i + 32]
        for path, r in zip(chunk, model.predict([str(p) for p in chunk], imgsz=640, conf=0.001, iou=0.7,
                                                max_det=300, verbose=False)):
            H, W = r.orig_shape
            b = r.boxes
            xyxy, sc = b.xyxy.cpu(), b.conf.cpu()
            d = torch.stack([xyxy[:, 1], xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], sc], 1) \
                if len(sc) else torch.zeros(0, 5)
            res.append((d, load_boxes(path, H, W)))
    return res, {}


# ---------------------------------------------------------------------------- metrikler
def recall_at_fppi(flags, n_gt, n_img, targets=FPPI_POINTS):
    if not flags or n_gt == 0:
        return {f"R@FPPI{t}": 0.0 for t in targets}
    f = sorted(flags, key=lambda x: -x[0])
    tp = np.cumsum([x[1] for x in f])
    fp = np.cumsum([not x[1] for x in f])
    fppi, rec = fp / n_img, tp / n_gt
    return {f"R@FPPI{t}": float(rec[fppi <= t].max()) if (fppi <= t).any() else 0.0 for t in targets}


def pr_curve(flags, n_gt):
    f = sorted(flags, key=lambda x: -x[0])
    tp = np.cumsum([x[1] for x in f])
    fp = np.cumsum([not x[1] for x in f])
    return tp / max(n_gt, 1), tp / np.maximum(tp + fp, 1)


def metrics(pairs, images):
    """pairs: [(dets, gts)] görüntü sırasıyla. Toplam ve set bazında metrikler + eğriler için bayraklar."""
    groups = {"TOPLAM": list(range(len(images)))}
    for i, p in enumerate(images):
        groups.setdefault(dataset_of(p), []).append(i)
    res, curves = {}, {}
    for g, idx in groups.items():
        f50, f30, n_gt = [], [], 0
        for i in idx:
            d, gt = pairs[i]
            f50 += match_flags(d, gt, 0.5)
            f30 += match_flags(d, gt, 0.3)
            n_gt += len(gt)
        res[g] = {"görüntü": len(idx), "insan": n_gt, "mAP50": average_precision(f50, n_gt),
                  "AP@IoU0.3": average_precision(f30, n_gt), **recall_at_fppi(f30, n_gt, len(idx))}
        curves[g] = {"pr50": pr_curve(f50, n_gt), "f30": f30, "n_gt": n_gt, "n_img": len(idx)}
    return res, curves


# ---------------------------------------------------------------------------- ana akış
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", nargs="*", default=[], help="results/flybrain altındaki eğitim adları ya da agirliklar/*.pt")
    ap.add_argument("--yolo", default="thermal", help="YOLO modalitesi; boş bırakılırsa YOLO atlanır")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--modality", default="thermal", help="§79: görüntü kümesi (thermal | rgb640); YOLO adı da --yolo ile")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="son_degerlendirme")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)
    device = torch.device("cuda")
    CACHE.mkdir(parents=True, exist_ok=True)
    models = [(n, "fly") for n in args.fly] + ([(f"YOLOv8n_{args.yolo}", "yolo")] if args.yolo else [])
    allres, allcurves, info = {}, {}, {}
    for split in args.splits:
        images = list_images(args.modality, split)
        if args.limit:
            images = images[:: max(1, len(images) // args.limit)][:args.limit]
        for name, kind in models:
            safe = name.replace("/", "__").replace("\\", "__")
            dec = f"_{DECODE_MODE}" if kind == "fly" and DECODE_MODE != "taban" else ""   # §50: kod çözme kipi anahtarda
            if kind == "fly" and EVAL_OVERLAP != 0.25:
                dec += f"_ov{EVAL_OVERLAP:g}"                                                 # §55: bindirme de
            cf = CACHE / f"{safe}_{split}{dec}{'_lim' + str(args.limit) if args.limit else ''}.pt"
            keys = [str(p.relative_to(ROOT)) for p in images]
            cached = torch.load(cf, weights_only=False) if cf.is_file() and not args.no_cache else None
            # önbellek yalnızca aynı görüntü listesiyle çıkarıldıysa geçerli (veri kümesi değişince bayatlar)
            if cached is not None and len(cached) == 3 and cached[2] == keys:
                pairs, meta, _ = cached
                log(f"{name} / {split}: önbellekten ({len(pairs)} görüntü)")
            else:
                if cached is not None:
                    log(f"{name} / {split}: önbellek farklı görüntü listesine ait, yeniden çıkarılıyor")
                log(f"{name} / {split}: tespitler çıkarılıyor ({len(images)} görüntü)")
                t = time.time()
                # .pt ile biten ad: sıkıştırılmış ağırlık dosyası (ör. agirliklar/uzun_75_termal.pt); yoksa eğitim klasörü
                yol = ROOT / name if name.endswith(".pt") else ROOT / "results" / "flybrain" / name
                pairs, meta = (collect_fly(yol, images, device) if kind == "fly" else collect_yolo(args.yolo, images))
                meta["sn_goruntu_basina"] = (time.time() - t) / len(images)
                torch.save((pairs, meta, keys), cf)
            r, c = metrics(pairs, images)
            allres.setdefault(name, {})[split] = r
            allcurves.setdefault(name, {})[split] = c
            info[name] = meta
            log(f"   {split}: " + "  ".join(f"{g} mAP50 {v['mAP50']:.3f}" for g, v in r.items()))

    # ---- tablo
    L = ["# Son değerlendirme: sinek modeli, çekirdeksiz kontrol ve YOLO", "",
         f"Oluşturulma: {time.strftime('%d.%m.%Y %H:%M')}" + (f" (DENEME, --limit {args.limit})" if args.limit else ""), "",
         "- **mAP50**: akademik standart (kutu gerçek kutuyla ≥ %50 örtüşmeli).",
         "- **AP@IoU0,3**: gevşek eşik; ~12 px'lik insanlarda kutu birkaç piksel kaysa da 'buldu' sayılır.",
         "- **R@FPPI x**: görüntü başına ortalama x yanlış alarm kabul edildiğinde bulunan insan oranı (IoU ≥ 0,3). "
         "Arama-kurtarma için pratik ölçüt.", ""]
    for split in args.splits:
        groups = list(next(iter(allres.values()))[split].keys())
        for g in groups:
            first = next(iter(allres.values()))[split][g]
            L += [f"## {split} — {g} ({first['görüntü']} görüntü, {first['insan']} insan)", "",
                  "| Model | mAP50 | AP@IoU0,3 | R@FPPI 0,1 | R@FPPI 0,5 | R@FPPI 1 |", "|---|---|---|---|---|---|"]
            for name in allres:
                v = allres[name][split][g]
                L.append(f"| {name} | **{v['mAP50']:.3f}** | {v['AP@IoU0.3']:.3f} | {v['R@FPPI0.1']:.1%} | "
                         f"{v['R@FPPI0.5']:.1%} | {v['R@FPPI1.0']:.1%} |")
            L.append("")
    L += ["## Model bilgisi", ""] + [f"- {n}: {json.dumps(m, ensure_ascii=False)}" for n, m in info.items()]
    (REPORT / f"{args.tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (REPORT / f"{args.tag}.json").write_text(json.dumps({"sonuclar": allres, "model_bilgisi": info},
                                                        ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- grafik: test kümesinde set bazında PR (IoU 0,5) ve FPPI–duyarlılık (IoU 0,3)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    split = "test" if "test" in args.splits else args.splits[0]
    groups = list(next(iter(allcurves.values()))[split].keys())
    fig, ax = plt.subplots(2, len(groups), figsize=(5 * len(groups), 8.5), squeeze=False)
    for j, g in enumerate(groups):
        for name in allcurves:
            c = allcurves[name][split][g]
            rec, prec = c["pr50"]
            ax[0, j].plot(rec, prec, label=f"{name} (mAP50 {allres[name][split][g]['mAP50']:.2f})")
            f = sorted(c["f30"], key=lambda x: -x[0])
            tp = np.cumsum([x[1] for x in f])
            fp = np.cumsum([not x[1] for x in f])
            ax[1, j].plot(fp / c["n_img"], tp / max(c["n_gt"], 1), label=name)
        ax[0, j].set(title=f"{split} — {g}: kesinlik–duyarlılık (IoU≥0,5)", xlabel="duyarlılık", ylabel="kesinlik",
                     xlim=(0, 1), ylim=(0, 1.02))
        ax[1, j].set(title=f"{split} — {g}: yanlış alarm–duyarlılık (IoU≥0,3)", xscale="log", xlim=(0.01, 10),
                     ylim=(0, 1.02), xlabel="görüntü başına yanlış alarm (FPPI, log)", ylabel="bulunan insan oranı")
        for a in ax[:, j]:
            a.grid(alpha=.3)
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORT / f"{args.tag}.png", dpi=100)
    log(f"✓ {REPORT / args.tag}.md / .json / .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
