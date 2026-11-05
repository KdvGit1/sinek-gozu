#!/usr/bin/env python3
"""§89e: YOLOv8n'i sinek video modelleriyle AYNI video val karelerinde ölçer (train_video.vval ile aynı pencereler: her val
videosundan 3 ayrık pencere × 20 kare, ilk 2 kare (ısınma) puanlanmaz; aynı eşleme / AP kodu). YOLO kare kare çalışır (zaman yok).
Eğitim GPU'yu kullanırken çalıştırılabilsin diye varsayılan CPU (az iş parçacığı).
Kullanım: py -3 tools/video_val_yolo.py --modality thermal [--cihaz cpu] [--kare 20 --pencere 3]"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tools.train_flybrain as tf  # noqa: E402
import tools.train_video as TV  # noqa: E402
from flybrain.detection import average_precision, match_flags  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["thermal", "rgb640"])
    ap.add_argument("--cihaz", default="cpu")
    ap.add_argument("--kare", type=int, default=20)
    ap.add_argument("--pencere", type=int, default=3)
    ap.add_argument("--isinma", type=int, default=2)
    ap.add_argument("--aralik", default="wisard:3,birdsai:1,visdrone_vid:3")
    ap.add_argument("--is-parcacigi", type=int, default=2)
    ap.add_argument("--bolum", default="val", choices=["val", "test"], help="§92: test bölümü yalnız son modeller için")
    a = ap.parse_args()
    torch.set_num_threads(a.is_parcacigi)
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "results/baseline" / f"yolov8n_{a.modality}" / "weights/best.pt"))
    aralik = {k: int(v) for k, v in (x.split(":") for x in a.aralik.split(","))}
    vids = TV.videolar(a.modality, a.bolum)
    fl, ng, n_kare, t0 = {}, {}, 0, time.time()
    for v in vids:
        for pen in TV.pencereler(v, aralik, a.kare, a.pencere):
            yollar = [p for k, p in enumerate(pen) if k >= a.isinma]
            for p, r in zip(yollar, model.predict([str(p) for p in yollar], imgsz=640, conf=0.001, iou=0.7, max_det=300,
                                                  device=a.cihaz, verbose=False)):
                H, W = r.orig_shape
                xyxy, sc = r.boxes.xyxy.cpu(), r.boxes.conf.cpu()
                d = torch.stack([xyxy[:, 1], xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], sc], 1) \
                    if len(sc) else torch.zeros(0, 5)
                gts = tf.load_boxes(p, H, W)
                f = match_flags(d, gts, 0.5)
                for g in ("toplam", v["set"]) + (() if v["gordu"] else ("ana_model_gormedi",)):
                    fl.setdefault(g, []).extend(f)
                    ng[g] = ng.get(g, 0) + len(gts)
                n_kare += 1
    r = {g: average_precision(fl[g], ng[g]) for g in fl}
    print(json.dumps({"modality": a.modality, "bolum": a.bolum, "kare": n_kare, "insan": ng, "mAP50": r, "sn": round(time.time() - t0)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
