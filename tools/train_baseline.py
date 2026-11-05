#!/usr/bin/env python3
"""
Karşılaştırma (baseline) modeli: YOLOv8n, convert_to_yolo.py çıktısıyla eğitilir (ADIMLAR.md Adım 4).

Sinek beyni modeliyle aynı veri ve aynı metrik (mAP@0.5) kullanılır. Eğitimden sonra test kümesinde
hem toplam hem de kaynak set bazında (hit_uav, birdsai, visdrone...) değerlendirme yapılır.

Notlar:
  * Görüntüler 640x512 olduğu için imgsz=640'ta küçültme olmaz. VisDrone test görüntüleri (1920x1080'e kadar)
    YOLO'nun standart davranışıyla 640'a küçültülür.
  * Varsayılan ağırlık COCO ön eğitimli yolov8n.pt (standart baseline pratiği). Sıfırdan eğitim için
    --weights yolov8n.yaml.
  * Ultralytics AGPL-3.0 lisanslı; burada yalnızca karşılaştırma için kullanılıyor, ürüne girmez.

Kullanım:
  python tools/train_baseline.py --modality thermal
  python tools/train_baseline.py --modality thermal rgb          # sırayla
  python tools/train_baseline.py --modality thermal --epochs 3   # hızlı deneme
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "yolo"
OUT = PROJECT_ROOT / "results" / "baseline"


def metrics_dict(m) -> dict:
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
            "precision": float(m.box.mp), "recall": float(m.box.mr)}


def per_dataset_test(model, data_yaml: Path, args) -> dict:
    """Test kümesini kaynak sete göre ayırıp her biri için ayrı değerlendirir."""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    test_dir = Path(cfg["path"]) / cfg["test"]
    groups: dict[str, list[str]] = {}
    for p in sorted(test_dir.iterdir()):
        # Dosya adı: <set>_<10 hex>_<orijinal ad> (convert_to_yolo.output_name); set adında "_" olabilir
        m = re.match(r"^(.+?)_[0-9a-f]{10}_", p.name)
        groups.setdefault(m.group(1) if m else "?", []).append(str(p))
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for ds, files in groups.items():
            lst = Path(tmp) / f"{ds}.txt"
            lst.write_text("\n".join(files), encoding="utf-8")
            sub = Path(tmp) / f"{ds}.yaml"
            sub.write_text(yaml.safe_dump({**cfg, "test": str(lst)}, allow_unicode=True), encoding="utf-8")
            m = model.val(data=str(sub), split="test", imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                          project=str(OUT), name=f"_tmp_{ds}", exist_ok=True, plots=False, verbose=False)
            out[ds] = {"images": len(files), **metrics_dict(m)}
            shutil.rmtree(OUT / f"_tmp_{ds}", ignore_errors=True)
            print(f"  test/{ds:16s} görüntü {len(files):5d}  mAP50 {out[ds]['mAP50']:.3f}")
    return out


def run(modality: str, args) -> None:
    from ultralytics import YOLO

    data_yaml = DATA / modality / f"{modality}.yaml"
    if not data_yaml.is_file():
        sys.exit(f"{data_yaml} yok. Önce: python tools/convert_to_yolo.py convert")
    name = f"yolov8n_{modality}{('_' + args.name) if args.name else ''}"   # §60: H39 varyantları ayrı klasörde
    print(f"\n=== {name}  ({data_yaml})")
    model = YOLO(args.weights)
    extra = {"optimizer": args.optimizer} if args.optimizer else {}
    if args.lr0:
        extra.update(lr0=args.lr0, momentum=args.momentum)
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, patience=args.patience, seed=0, deterministic=True,
                project=str(OUT), name=name, exist_ok=True, plots=True, **extra)

    best = OUT / name / "weights" / "best.pt"
    model = YOLO(str(best))
    res = {"modality": modality, "weights_init": args.weights, "epochs": args.epochs, "imgsz": args.imgsz}
    res["val"] = metrics_dict(model.val(data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch,
                                        workers=args.workers, project=str(OUT), name=f"{name}_val",
                                        exist_ok=True))
    if not args.no_test:                                   # §63: kontrol koşularında test kümesi kullanılmaz
        res["test"] = metrics_dict(model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch,
                                             workers=args.workers, project=str(OUT), name=f"{name}_test",
                                             exist_ok=True))
        res["test_by_dataset"] = per_dataset_test(model, data_yaml, args)
    (OUT / name / "sonuclar.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {name}: val mAP50 {res['val']['mAP50']:.3f}, test mAP50 {res.get('test', {}).get('mAP50', float('nan')):.3f}"
          f"  → {OUT / name / 'sonuclar.json'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="YOLOv8n karşılaştırma modelini eğit")
    ap.add_argument("--modality", nargs="+", default=["thermal"], choices=["thermal", "rgb", "rgb640"])
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--name", default="", help="çıktı adı eki (ör. sifirdan → yolov8n_thermal_sifirdan)")
    ap.add_argument("--optimizer", default="", help="§63: açık iyileştirici (ör. SGD); boş = ultralytics otomatik")
    ap.add_argument("--lr0", type=float, default=0.0)
    ap.add_argument("--momentum", type=float, default=0.937)
    ap.add_argument("--no-test", action="store_true", help="test kümesinde değerlendirme yapma (kontrol koşuları)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    for m in args.modality:
        run(m, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
