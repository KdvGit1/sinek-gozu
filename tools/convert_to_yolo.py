#!/usr/bin/env python3
"""
Drone insan tespiti veri setlerini tek bir YOLO formatına dönüştürür.

Her kaynak (dataset + modalite) configs/datasets.yaml içinde tanımlanır. Betik:
  * COCO JSON, Pascal VOC XML, YOLO txt ve BIRDSAI (MOT CSV) formatlarını okur,
  * sınıfları tek "person" sınıfına eşler (crowd gibi kutuları atar ya da maskeler),
  * RGB ve termal görüntüleri ayrı çıktı klasörlerine yazar,
  * video/sekans bazlı train/val/test bölmesi yapar (aynı videonun kareleri
    farklı bölmelere düşüp metrikleri şişirmesin diye),
  * her çıktı dosyasının kaynağını ve lisansını manifest.csv'ye yazar,
  * lisans politikasına uymayan setleri (ör. NonCommercial) varsayılan olarak atlar.

Kullanım:
  python tools/convert_to_yolo.py inspect <klasör>          # indirdiğin seti incele
  python tools/convert_to_yolo.py convert --dry-run          # sadece istatistik
  python tools/convert_to_yolo.py convert                    # dönüştür
  python tools/convert_to_yolo.py convert --only wisard hit_uav --mode hardlink
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
    from PIL import Image, ImageDraw
except ImportError as e:  # pragma: no cover
    sys.exit(f"Eksik paket: {e.name}. Kurulum: pip install -r requirements.txt")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sınıf eşleme hedefleri: çıktı sınıf adı, None (kutuyu at) veya IGNORE (bölgeyi maskele).
IGNORE = "ignore_region"


# --------------------------------------------------------------------------- #
# Veri yapıları
# --------------------------------------------------------------------------- #
@dataclass
class Box:
    """Piksel koordinatlarında sol-üst köşe + genişlik/yükseklik."""
    x: float
    y: float
    w: float
    h: float
    label: str  # kaynak sınıf adı/id'si (string)


@dataclass
class Sample:
    image: Path
    rel: str  # kaynak köküne göre göreli yol (posix)
    boxes: list[Box]
    size: tuple[int, int] | None = None  # (w, h), biliniyorsa
    split: str | None = None  # kaynak tarafından belirlenmişse


@dataclass
class Stats:
    images: Counter = field(default_factory=Counter)        # split -> adet
    boxes: Counter = field(default_factory=Counter)         # split -> adet
    empty: Counter = field(default_factory=Counter)         # split -> kutusuz görüntü
    dropped: Counter = field(default_factory=Counter)       # neden -> adet
    unmapped: Counter = field(default_factory=Counter)      # kaynak sınıf -> adet
    missing_images: int = 0


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def resolve(path: str | os.PathLike, base: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return p if p.is_absolute() else (base / p)


def iter_images(root: Path):
    if not root.is_dir():
        raise FileNotFoundError(2, "klasör yok", str(root))
    for dirpath, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                yield Path(dirpath) / f


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def stable_fraction(key: str) -> float:
    """Anahtarı [0,1) aralığına deterministik olarak eşler (seed'den bağımsız, tekrarlanabilir)."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16) / float(1 << 48)


def trailing_int(name: str) -> int | None:
    m = re.search(r"(\d+)(?!.*\d)", name)
    return int(m.group(1)) if m else None


def path_filter(src: dict):
    """include_regex / exclude_regex ile göreli yolu filtreleyen fonksiyon döndürür."""
    inc = re.compile(src["include_regex"], re.I) if src.get("include_regex") else None
    exc = re.compile(src["exclude_regex"], re.I) if src.get("exclude_regex") else None

    def ok(rel: str) -> bool:
        if inc and not inc.search(rel):
            return False
        if exc and exc.search(rel):
            return False
        return True

    return ok


def split_from_path(rel: str, split_regex: dict | None) -> str | None:
    if not split_regex:
        return None
    for split, pattern in split_regex.items():
        if re.search(pattern, rel, re.I):
            return split
    return None


# --------------------------------------------------------------------------- #
# Okuyucular — her biri Sample üretir
# --------------------------------------------------------------------------- #
def read_yolo(src: dict, base: Path):
    """YOLO txt: <cls> <xc> <yc> <w> <h> (normalize). Etiket dosyası aranma sırası:
    1) labels_root + göreli yol, 2) yol içinde /images/ -> /labels/, 3) görüntüyle aynı klasör."""
    root = resolve(src["image_root"], base)
    labels_root = resolve(src["labels_root"], base) if src.get("labels_root") else None
    ok = path_filter(src)
    missing_label = 0
    for img in sorted(iter_images(root)):
        rel = rel_posix(img, root)
        if not ok(rel):
            continue
        candidates = []
        if labels_root:
            candidates.append((labels_root / rel).with_suffix(".txt"))
        parts = img.parts
        if "images" in parts:
            i = len(parts) - 1 - parts[::-1].index("images")
            candidates.append(Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
        candidates.append(img.with_suffix(".txt"))
        label_file = next((c for c in candidates if c.is_file()), None)
        if label_file is None:
            missing_label += 1
            if not src.get("keep_unlabeled_as_empty", False):
                continue
            yield Sample(img, rel, [])
            continue

        size = image_size(img)
        W, H = size
        boxes = []
        for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
            vals = line.split()
            if len(vals) < 5:
                continue
            c, xc, yc, w, h = vals[0], *map(float, vals[1:5])
            boxes.append(Box((xc - w / 2) * W, (yc - h / 2) * H, w * W, h * H, c))
        yield Sample(img, rel, boxes, size)
    if missing_label:
        print(f"    ! {missing_label} görüntünün etiket dosyası bulunamadı"
              f"{' (boş kabul edildi)' if src.get('keep_unlabeled_as_empty') else ' (atlandı)'}")


def read_coco(src: dict, base: Path):
    ann_paths = src["annotation"]
    ann_paths = ann_paths if isinstance(ann_paths, list) else [ann_paths]
    root = resolve(src["image_root"], base)
    ok = path_filter(src)
    for ann_path in ann_paths:
        ann_file = resolve(ann_path, base)
        data = json.loads(ann_file.read_text(encoding="utf-8"))
        cats = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
        by_image = defaultdict(list)
        for a in data.get("annotations", []):
            by_image[a["image_id"]].append(a)
        # Bir json dosyası tek bir split'e aitse (ör. train.json) config'te "split_of_annotation" ile verilir.
        fixed_split = (src.get("split_of_annotation") or {}).get(Path(ann_path).name)
        for im in data.get("images", []):
            img = root / im["file_name"]
            rel = Path(im["file_name"]).as_posix()
            if not ok(rel):
                continue
            boxes = []
            for a in by_image.get(im["id"], []):
                if a.get("iscrowd", 0) and src.get("iscrowd_as_ignore", True):
                    label = "__iscrowd__"
                else:
                    label = cats.get(a["category_id"], str(a["category_id"]))
                x, y, w, h = a["bbox"]
                boxes.append(Box(x, y, w, h, label))
            size = (im["width"], im["height"]) if "width" in im and "height" in im else None
            yield Sample(img, rel, boxes, size, split=fixed_split)


def read_voc(src: dict, base: Path):
    ann_root = resolve(src["annotation_root"], base)
    img_root = resolve(src["image_root"], base)
    ok = path_filter(src)
    # Görüntüleri isim (stem) ile indeksle; VOC'taki <filename> alanı her zaman güvenilir değil.
    index = {}
    for img in iter_images(img_root):
        index.setdefault(img.stem, img)
    if not ann_root.is_dir():
        raise FileNotFoundError(2, "klasör yok", str(ann_root))
    for xml_file in sorted(ann_root.rglob("*.xml")):
        tree = ET.parse(xml_file).getroot()
        stem = Path(tree.findtext("filename") or xml_file.stem).stem
        img = index.get(stem) or index.get(xml_file.stem)
        if img is None:
            yield Sample(img_root / f"{xml_file.stem}.MISSING", xml_file.stem, [])
            continue
        rel = rel_posix(img, img_root)
        if not ok(rel):
            continue
        size = None
        s = tree.find("size")
        if s is not None:
            w, h = int(float(s.findtext("width", "0"))), int(float(s.findtext("height", "0")))
            size = (w, h) if w > 0 and h > 0 else None
        boxes = []
        for obj in tree.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:  # döndürülmüş kutular (robndbox) desteklenmiyor — normal_xml kullan
                continue
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            label = obj.findtext("name", "").strip()
            if obj.findtext("difficult", "0").strip() == "1" and src.get("difficult_as_ignore", False):
                label = "__difficult__"
            boxes.append(Box(x1, y1, x2 - x1, y2 - y1, label))
        yield Sample(img, rel, boxes, size, split=split_from_path(xml_file.relative_to(ann_root).as_posix(),
                                                                  src.get("split_regex")))


def read_birdsai(src: dict, base: Path):
    """BIRDSAI: annotations/<video>.csv, images/<video>/*.jpg
    CSV: frame, id, x, y, w, h, class(0=hayvan,1=insan), species, occlusion, noise"""
    root = resolve(src["root"], base)
    frame_offset = int(src.get("frame_offset", 0))
    ok = path_filter(src)
    if not (root / "annotations").is_dir():
        raise FileNotFoundError(2, "klasör yok", str(root / "annotations"))
    per_video = {str(k): int(v) for k, v in (src.get("frame_offset_per_video") or {}).items()}
    for csv_file in sorted((root / "annotations").glob("*.csv")):
        video = csv_file.stem
        offset = per_video.get(video, frame_offset)
        img_dir = root / "images" / video
        if not img_dir.is_dir():
            print(f"    ! görüntü klasörü yok: {img_dir}")
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        by_frame = {}
        for i, p in enumerate(imgs):
            n = trailing_int(p.stem)
            by_frame[n if n is not None else i] = p
        rows = defaultdict(list)
        with csv_file.open(newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) < 7 or not r[0].strip().lstrip("-").isdigit():
                    continue
                frame = int(r[0]) - offset
                x, y, w, h = map(float, r[2:6])
                label = str(int(float(r[6])))
                if src.get("drop_noisy") and len(r) > 9 and r[9].strip() == "1":
                    label = "__noisy__"
                rows[frame].append(Box(x, y, w, h, label))
        unmatched = [fr for fr in rows if fr not in by_frame]
        if unmatched:
            print(f"    ! {video}: {len(unmatched)} CSV karesi görüntüyle eşleşmedi "
                  f"(frame_offset ayarını kontrol et)")
        for frame, img in sorted(by_frame.items()):
            rel = rel_posix(img, root)
            if ok(rel):
                yield Sample(img, rel, rows.get(frame, []))


def read_visdrone(src: dict, base: Path):
    """VisDrone-DET: annotations/<ad>.txt, satır: left,top,w,h,score,category,truncation,occlusion
    category 0 = 'ignored regions', score 0 = değerlendirmede yok sayılan kutu."""
    root = resolve(src["root"], base)
    ok = path_filter(src)
    if not (root / "annotations").is_dir():
        raise FileNotFoundError(2, "klasör yok", str(root / "annotations"))
    for ann in sorted((root / "annotations").glob("*.txt")):
        img = next((root / "images" / f"{ann.stem}{e}" for e in (".jpg", ".png")
                    if (root / "images" / f"{ann.stem}{e}").is_file()), root / "images" / f"{ann.stem}.jpg")
        rel = rel_posix(img, root)
        if not ok(rel):
            continue
        boxes = []
        for line in ann.read_text(encoding="utf-8").splitlines():
            v = [p for p in line.strip().split(",") if p != ""]
            if len(v) < 6:
                continue
            x, y, w, h = map(float, v[:4])
            label = "__iscrowd__" if v[4] == "0" else v[5]
            boxes.append(Box(x, y, w, h, label))
        yield Sample(img, rel, boxes)


READERS = {"yolo": read_yolo, "coco": read_coco, "voc": read_voc, "birdsai": read_birdsai,
           "visdrone": read_visdrone}


# --------------------------------------------------------------------------- #
# Dönüştürme
# --------------------------------------------------------------------------- #
def group_key(sample: Sample, src: dict) -> str:
    """Bölme için grup anahtarı: varsayılan olarak görüntünün klasörü (video/sekans)."""
    rx = src.get("group_regex")
    if rx:
        m = re.search(rx, sample.rel, re.I)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return str(Path(sample.rel).parent.as_posix())


def assign_split(group: str, dataset: str, ratios: dict) -> str:
    f = stable_fraction(f"{dataset}/{group}")
    if f < ratios["test"]:
        return "test"
    if f < ratios["test"] + ratios["val"]:
        return "val"
    return "train"


def map_boxes(sample: Sample, src: dict, size, cfg, stats: Stats):
    """Kaynak kutularını çıktı sınıflarına çevirir. (yolo_lines, ignore_boxes) döndürür."""
    W, H = size
    class_map = {str(k).lower(): v for k, v in (src.get("classes") or {}).items()}
    class_map.setdefault("__iscrowd__", IGNORE)
    class_map.setdefault("__difficult__", IGNORE)
    class_map.setdefault("__noisy__", None)
    out_ids = {name: i for i, name in enumerate(cfg["output_classes"])}
    min_px = float(cfg.get("min_box_px", 2))

    lines, ignores = [], []
    for b in sample.boxes:
        key = b.label.lower()
        if key not in class_map:
            stats.unmapped[b.label] += 1
            continue
        target = class_map[key]
        if target is None:
            stats.dropped[f"sınıf:{b.label}"] += 1
            continue
        # Görüntü sınırlarına kırp
        x1, y1 = max(0.0, b.x), max(0.0, b.y)
        x2, y2 = min(float(W), b.x + b.w), min(float(H), b.y + b.h)
        if x2 - x1 < min_px or y2 - y1 < min_px:
            stats.dropped["çok küçük / görüntü dışı"] += 1
            continue
        if target == IGNORE:
            ignores.append((x1, y1, x2, y2))
            continue
        if target not in out_ids:
            raise ValueError(f"'{target}' output_classes içinde yok ({src['_name']})")
        xc, yc = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        lines.append(f"{out_ids[target]} {xc:.6f} {yc:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines, ignores


def place_image(src_img: Path, dst: Path, mode: str, ignores, policy: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if ignores and policy == "mask":
        # Etiketlenmemiş kalabalık bölgelerini ortalama renkle doldur ki model onları
        # "arka plan" diye öğrenmesin.
        with Image.open(src_img) as im:
            # 16-bit (radyometrik) termal görüntüleri 8-bit'e indirme: sıcaklık bilgisi kaybolur.
            if im.mode.startswith("I;16") or im.mode in ("I", "F"):
                im = im.convert("I")
            elif im.mode == "L":
                im = im.copy()
            else:
                im = im.convert("RGB")
        mean = im.resize((1, 1)).getpixel((0, 0))
        draw = ImageDraw.Draw(im)
        for x1, y1, x2, y2 in ignores:
            draw.rectangle([x1, y1, x2, y2], fill=mean)
        if dst.suffix.lower() in {".jpg", ".jpeg"}:
            im.save(dst, quality=95)
        else:
            im.save(dst)
        return
    if mode == "hardlink":
        try:
            os.link(src_img, dst)
            return
        except OSError:
            pass  # farklı disk vb. — kopyalamaya düş
    elif mode == "symlink":
        try:
            os.symlink(src_img.resolve(), dst)
            return
        except OSError:
            pass  # Windows'ta geliştirici modu kapalıysa symlink izni yok
    shutil.copy2(src_img, dst)


def output_name(dataset: str, source: str, rel: str, suffix: str) -> str:
    """Çakışmasız ve Windows yol limiti için kısa dosya adı."""
    h = hashlib.sha1(f"{source}/{rel}".encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(rel).stem)[-40:]
    return f"{dataset}_{h}_{stem}{suffix.lower()}"


MANIFEST_FIELDS = ("output", "dataset", "source", "original", "license", "boxes")


def clean_previous(mod_root: Path, dataset: str):
    """Aynı setin önceki çalıştırmadan kalan çıktılarını sil (split değişmiş olabilir)."""
    pattern = re.compile(rf"^{re.escape(dataset)}_[0-9a-f]{{10}}_")
    for sub in ("images", "labels"):
        d = mod_root / sub
        if not d.is_dir():
            continue
        for split_dir in d.iterdir():
            if split_dir.is_dir():
                for f in split_dir.iterdir():
                    if pattern.match(f.name):
                        f.unlink()


def license_allowed(ds: dict, cfg: dict, allow_any: bool) -> bool:
    allowed = [s.lower() for s in cfg.get("license_policy", {}).get("allow", [])]
    return allow_any or not allowed or str(ds.get("license", "")).lower() in allowed


def convert(cfg: dict, cfg_dir: Path, args) -> int:
    out_root = resolve(cfg.get("output_root", "data/yolo"), cfg_dir)
    ratios = {"val": 0.1, "test": 0.1, **(cfg.get("split_ratios") or {})}
    crowd_policy = cfg.get("ignore_region_policy", "drop")  # drop | mask
    keep_empty = cfg.get("keep_empty_images", True)
    min_groups = int(cfg.get("min_groups_for_group_split", 10))

    datasets = cfg["datasets"]
    if args.only:
        unknown = set(args.only) - set(datasets)
        if unknown:
            sys.exit(f"Config'te olmayan dataset: {', '.join(sorted(unknown))}")
        datasets = {k: v for k, v in datasets.items() if k in args.only}

    manifest_rows = defaultdict(list)       # modality -> rows
    report = defaultdict(dict)              # modality -> dataset/source -> stats
    cleaned_datasets = defaultdict(set)     # modality -> bu çalıştırmada yeniden üretilen setler
    written_modalities = set()

    for ds_name, ds in datasets.items():
        if not ds.get("enabled", True) and not args.only:
            print(f"- {ds_name}: devre dışı (enabled: false), atlandı")
            continue
        if not license_allowed(ds, cfg, args.allow_any_license):
            print(f"- {ds_name}: lisans '{ds.get('license')}' politikada izinli değil, ATLANDI "
                  f"(zorlamak için --allow-any-license)")
            continue
        print(f"- {ds_name}  [{ds.get('license', '?')}]")
        for src in ds["sources"]:
            src_name = f"{ds_name}/{src['name']}"
            if not src.get("enabled", True):
                print(f"  · {src_name}: devre dışı, atlandı")
                continue
            src["_name"] = src_name
            modality = src["modality"]
            reader = READERS[src["format"]]
            stats = Stats()
            print(f"  · {src_name} ({src['format']}, {modality})")

            try:
                samples = list(reader(src, cfg_dir))
                # Resmi bölme listeleri (ör. VOC ImageSets/Main/train.txt): uzantısız görüntü adı -> split
                list_split = {}
                for split_name, list_path in (src.get("split_lists") or {}).items():
                    for line in resolve(list_path, cfg_dir).read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            list_split[Path(line.split()[0]).stem] = split_name
            except FileNotFoundError as e:
                print(f"    ! dosya bulunamadı: {e.filename} — atlandı")
                continue

            # Kareleri seyrelt (ardışık video kareleri neredeyse aynıdır)
            stride = int(src.get("frame_stride", 1))
            if stride > 1:
                by_group = defaultdict(list)
                for s in samples:
                    by_group[group_key(s, src)].append(s)
                samples = [s for g in by_group.values() for s in sorted(g, key=lambda s: s.rel)[::stride]]

            # Bölme önceliği: kaynağa sabit split > annotation dosyasından gelen > split_lists > yol regex'i
            # > grup hash'i
            for s in samples:
                s.split = (src.get("split") or s.split or list_split.get(s.image.stem)
                           or split_from_path(s.rel, src.get("split_regex")))
            if list_split:
                unlisted = sum(1 for s in samples if s.image.stem not in list_split)
                if unlisted:
                    print(f"    ! {unlisted} görüntü split_lists içinde yok; grup hash'iyle bölünecek")
            groups = {group_key(s, src) for s in samples if s.split is None}
            per_image = 0 < len(groups) < min_groups
            if per_image:
                print(f"    ! sadece {len(groups)} grup (klasör) var; görüntü bazlı bölmeye düşüldü. "
                      f"Ardışık kareler train/val arasında sızabilir — group_regex tanımlamayı düşün.")

            if not args.dry_run and ds_name not in cleaned_datasets[modality]:
                clean_previous(out_root / modality, ds_name)
                cleaned_datasets[modality].add(ds_name)

            for s in samples:
                if not s.image.is_file():
                    stats.missing_images += 1
                    continue
                split = s.split or assign_split(s.rel if per_image else group_key(s, src), ds_name, ratios)
                size = s.size or image_size(s.image)
                lines, ignores = map_boxes(s, src, size, cfg, stats)
                if not lines and not keep_empty:
                    stats.dropped["boş görüntü"] += 1
                    continue
                stats.images[split] += 1
                stats.boxes[split] += len(lines)
                if not lines:
                    stats.empty[split] += 1
                if ignores:
                    stats.dropped[f"ignore bölge ({crowd_policy})"] += len(ignores)
                if args.dry_run:
                    continue

                name = output_name(ds_name, src["name"], s.rel, s.image.suffix)
                mod_root = out_root / modality
                img_dst = mod_root / "images" / split / name
                lbl_dst = (mod_root / "labels" / split / name).with_suffix(".txt")
                place_image(s.image, img_dst, args.mode, ignores, crowd_policy)
                lbl_dst.parent.mkdir(parents=True, exist_ok=True)
                lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                manifest_rows[modality].append({
                    "output": f"images/{split}/{name}", "dataset": ds_name, "source": src["name"],
                    "original": str(s.image), "license": ds.get("license", ""), "boxes": len(lines),
                })
                written_modalities.add(modality)

            print(f"    görüntü: {dict(stats.images)}  kutu: {dict(stats.boxes)}  boş: {dict(stats.empty)}")
            if stats.dropped:
                print(f"    atılan: {dict(stats.dropped)}")
            if stats.unmapped:
                print(f"    ! EŞLENMEMİŞ sınıflar (atıldı, config'te 'classes' altına ekle): {dict(stats.unmapped)}")
            if stats.missing_images:
                print(f"    ! {stats.missing_images} görüntü dosyası bulunamadı")
            report[modality][src_name] = {
                "license": ds.get("license", ""), "images": dict(stats.images), "boxes": dict(stats.boxes),
                "empty": dict(stats.empty), "dropped": dict(stats.dropped),
                "unmapped": dict(stats.unmapped), "missing_images": stats.missing_images,
            }

    if args.dry_run:
        print("\n(dry-run: hiçbir dosya yazılmadı)")
        return 0

    for modality in written_modalities:
        mod_root = out_root / modality
        # --only ile çalışınca diğer setlerin kayıtlarını koru, yeniden üretilenleri değiştir.
        redone = cleaned_datasets[modality]
        manifest_path, report_path = mod_root / "manifest.csv", mod_root / "report.json"
        rows = []
        if manifest_path.is_file():
            with manifest_path.open(newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r["dataset"] not in redone]
        rows += manifest_rows[modality]
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS))
            w.writeheader()
            w.writerows(rows)
        old_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        old_report = {k: v for k, v in old_report.items() if k.split("/")[0] not in redone}
        report_path.write_text(json.dumps({**old_report, **report[modality]}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        data_yaml = {
            "path": str(mod_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": dict(enumerate(cfg["output_classes"])),
        }
        for split in SPLITS:
            (mod_root / "images" / split).mkdir(parents=True, exist_ok=True)
        with (mod_root / f"{modality}.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)
        print(f"\n✓ {modality}: {mod_root / f'{modality}.yaml'}")
    return 0


# --------------------------------------------------------------------------- #
# inspect: indirilen bir klasörün yapısını ve etiketlerini özetler
# --------------------------------------------------------------------------- #
def inspect(root: Path, max_files: int = 2000) -> int:
    if not root.exists():
        sys.exit(f"Klasör yok: {root}")
    ext_count, dir_images = Counter(), Counter()
    jsons, xmls, txts, csvs = [], [], [], []
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            ext = p.suffix.lower()
            ext_count[ext] += 1
            if ext in IMAGE_EXTS:
                dir_images[Path(dirpath).relative_to(root).as_posix()] += 1
            elif ext == ".json":
                jsons.append(p)
            elif ext == ".xml":
                xmls.append(p)
            elif ext == ".txt":
                txts.append(p)
            elif ext == ".csv":
                csvs.append(p)

    print(f"Kök: {root}\nUzantılar: {dict(ext_count.most_common())}")
    print(f"\nGörüntü içeren klasörler ({len(dir_images)}), ilk 25:")
    for d, n in dir_images.most_common(25):
        print(f"  {n:>7}  {d}")

    sample_img = next(iter_images(root), None)
    if sample_img:
        print(f"\nÖrnek görüntü: {sample_img.relative_to(root)}  boyut={image_size(sample_img)}")

    for j in jsons[:10]:
        try:
            data = json.loads(j.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"\nJSON okunamadı {j.relative_to(root)}: {e}")
            continue
        if isinstance(data, dict) and "annotations" in data and "images" in data:
            cats = {c["id"]: c.get("name") for c in data.get("categories", [])}
            cnt = Counter(a["category_id"] for a in data["annotations"])
            print(f"\nCOCO: {j.relative_to(root)}  görüntü={len(data['images'])} kutu={len(data['annotations'])}")
            print(f"  örnek file_name: {data['images'][0]['file_name'] if data['images'] else '-'}")
            for cid, n in cnt.most_common():
                print(f"  sınıf {cid} '{cats.get(cid)}': {n}")
        else:
            keys = list(data)[:10] if isinstance(data, dict) else f"list[{len(data)}]"
            print(f"\nJSON (COCO değil): {j.relative_to(root)}  anahtarlar={keys}")

    if xmls:
        names = Counter()
        for x in xmls[:max_files]:
            try:
                names.update(o.findtext("name", "").strip() for o in ET.parse(x).getroot().findall("object"))
            except ET.ParseError:
                pass
        print(f"\nVOC XML: {len(xmls)} dosya (ilk {min(len(xmls), max_files)} tarandı) sınıflar: {dict(names)}")

    if txts:
        ids, yolo_like = Counter(), 0
        for t in txts[:max_files]:
            lines = t.read_text(encoding="utf-8", errors="replace").splitlines()
            rows = [l.split() for l in lines if l.strip()]
            if rows and all(len(r) >= 5 and _is_yolo_row(r) for r in rows):
                yolo_like += 1
                ids.update(r[0] for r in rows)
        print(f"\nTXT: {len(txts)} dosya, YOLO benzeri: {yolo_like} (taranan {min(len(txts), max_files)}) "
              f"sınıf id'leri: {dict(ids)}")
        if txts:
            print(f"  örnek: {txts[0].relative_to(root)}")

    for c in csvs[:3]:
        with c.open(encoding="utf-8", errors="replace") as f:
            head = [next(f, "").rstrip() for _ in range(3)]
        print(f"\nCSV: {c.relative_to(root)}\n  " + "\n  ".join(head))
    if csvs:
        print(f"  (toplam {len(csvs)} CSV)")
    return 0


def _is_yolo_row(r):
    try:
        int(r[0])
        return all(0.0 <= float(v) <= 1.0001 for v in r[1:5])
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Drone insan tespiti veri setlerini YOLO formatına dönüştürür.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="İndirilen bir veri seti klasörünün yapısını/etiketlerini özetle")
    p_ins.add_argument("root", type=Path)

    p_conv = sub.add_parser("convert", help="Config'teki setleri dönüştür")
    p_conv.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    p_conv.add_argument("--only", nargs="+", metavar="DATASET", help="Sadece bu setleri işle")
    p_conv.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy",
                        help="Görüntüleri kopyala (varsayılan) ya da disk tasarrufu için bağla")
    p_conv.add_argument("--dry-run", action="store_true", help="Dosya yazmadan istatistikleri göster")
    p_conv.add_argument("--allow-any-license", action="store_true",
                        help="license_policy dışındaki (ör. NC) setleri de işle")

    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        return inspect(args.root)

    cfg_path = args.config.resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    # Config içindeki göreli yollar proje köküne göre çözülür (config configs/ altındaysa).
    base = cfg_path.parent.parent if cfg_path.parent.name == "configs" else cfg_path.parent
    return convert(cfg, base, args)


if __name__ == "__main__":
    sys.exit(main())
