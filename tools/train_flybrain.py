#!/usr/bin/env python3
"""
Sinek beyni modelini (FlyEyeDetector) gerçek veriyle eğitir (TASARIM.md Bölüm 6, madde 5).

Veri: tools/convert_to_yolo.py çıktısı (data/yolo/<modalite>/images|labels/<split>).
Görüntü asla küçültülmez:
  * Eğitim: her görüntüden göz boyutunda karolar kesilir. Karoların bir kısmı rastgele bir insanı
    içine alacak şekilde seçilir (insanlar karonun küçük bir bölümünde olduğu için), kalanı rastgele konumdan.
  * Değerlendirme: tam görüntü Tiler ile karolara bölünür, tespitler birleştirilir (NMS),
    mAP@0.5 YOLO baseline ile aynı yöntemle (101 noktalı AP) hesaplanır.

Deney değişkenleri (TASARIM.md): --patch (p), --threshold (sinaps eşiği), --reservoir, --bridge.
Nöron başına kanal d = 1 (ana model); d > 1 henüz modelde yok.

Kullanım:
  python tools/train_flybrain.py --modality thermal                     # ana deney (p=4, eşik 5)
  python tools/train_flybrain.py --modality thermal --iters 200 --name deneme   # hızlı deneme
  python tools/train_flybrain.py --modality thermal --resume            # kaldığı yerden devam
  python tools/train_flybrain.py --eval-only results/flybrain/<ad>/best.pt

Çıktı: results/flybrain/<ad>/ (last.pt, best.pt, log.csv, config.json, sonuclar.json)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flybrain.detection import (BilateralTargets, TargetBuilder, average_precision, box_iou, decode, decode_cpu,  # noqa: E402
                                detection_loss, match, match_flags, merge_tiles, merge_top)
from flybrain.eye import Tiler  # noqa: E402
from flybrain.model import HIZ75, FlyEyeDetector, ModelConfig  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA = ROOT / "data" / "yolo"
OUT = ROOT / "results" / "flybrain"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ veri
def list_images(modality: str, split: str) -> list[Path]:
    d = DATA / modality / "images" / split
    if not d.is_dir():
        sys.exit(f"{d} yok. Önce: python tools/convert_to_yolo.py convert")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_image(path: Path, channels: int) -> torch.Tensor:
    """[C, H, W], 0–1. Termal görüntüler (RGB olarak kaydedilmiş olsa da) tek kanala çevrilir.
    16-bit görüntüler kendi aralığına göre ölçeklenir."""
    with Image.open(path) as im:
        if channels == 1:
            if im.mode.startswith("I;16") or im.mode in ("I", "F"):
                a = np.asarray(im, dtype=np.float32)
                lo, hi = np.percentile(a, (0.5, 99.5))
                a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
                return torch.from_numpy(a)[None]
            a = np.asarray(im.convert("L"), dtype=np.float32)[None]
        else:
            a = np.asarray(im.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(a)) / 255.0


def load_image_boxes(path: Path, channels: int):
    img = load_image(path, channels)
    return img, load_boxes(path, img.shape[1], img.shape[2])


def load_boxes(img: Path, H: int, W: int) -> torch.Tensor:
    """YOLO etiketi → [M, 4] (y, x, h, w) piksel."""
    lp = label_path(img)
    rows = []
    if lp.is_file():
        for line in lp.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) >= 5:
                xc, yc, w, h = map(float, v[1:5])
                rows.append(((yc - h / 2) * H, (xc - w / 2) * W, h * H, w * W))
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)


def crop(img: torch.Tensor, oy: int, ox: int, eh: int, ew: int, pad_mode: str = "zero") -> torch.Tensor:
    """[C, eh, ew] keser; görüntü dışı Tiler.extract ile aynı şekilde doldurulur:
    "zero" = 0 ile (ilk eğitim), "replicate" = kenar pikseli uzatılır."""
    C, H, W = img.shape
    if oy >= 0 and ox >= 0 and oy + eh <= H and ox + ew <= W:      # §46: tamamen içeride → dilim (aynı pikseller)
        return img[:, oy:oy + eh, ox:ox + ew].clone()
    if pad_mode == "replicate":
        ys = torch.arange(oy, oy + eh).clamp(0, H - 1)
        xs = torch.arange(ox, ox + ew).clamp(0, W - 1)
        return img[:, ys][:, :, xs]
    out = torch.zeros((C, eh, ew), dtype=img.dtype)
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(H, oy + eh), min(W, ox + ew)
    if y1 > y0 and x1 > x0:
        out[:, y0 - oy:y1 - oy, x0 - ox:x1 - ox] = img[:, y0:y1, x0:x1]
    return out


class TileDataset(torch.utils.data.Dataset):
    """Her öğe: bir görüntüden `per_image` karo ve karo koordinatındaki kutular.
    targets (TargetBuilder) verilirse ısı/ofset hedefleri de burada, yani veri yükleyici işçisinde üretilir;
    ana süreç GPU'yu beslemekle meşgul kalır."""

    def __init__(self, images: list[Path], channels: int, eye_hw, inscribed, per_image: int, pos_frac: float,
                 seed: int, pad_mode: str = "zero", targets: TargetBuilder | None = None,
                 scale_aug: tuple[float, float] | None = None, size_balance: bool = False, photo_aug: bool = False,
                 hard_pool: str | None = None, hard_frac: float = 0.0):
        """scale_aug (H1): görüntü karo kesilmeden önce rastgele s ∈ [lo, hi] (log-düzgün) ile ölçeklenir.
        size_balance (H1): pozitif karo için insan, boyutun kareköküyle orantılı olasılıkla seçilir (büyükler daha sık).
        photo_aug (H2): karo başına rastgele gama / kontrast / parlaklık.
        hard_pool / hard_frac (H5): zor negatif havuzu dosyası (ana süreç yazar); karoların hard_frac'ı oradan."""
        self.images, self.channels, self.pad_mode = images, channels, pad_mode
        self.scale_aug, self.size_balance, self.photo_aug = scale_aug, size_balance, photo_aug
        self.hard_pool, self.hard_frac = hard_pool, hard_frac
        self._pool, self._pool_mtime = None, None
        self.eh, self.ew = eye_hw
        self.ry, self.rx, self.rh, self.rw = inscribed
        self.per_image, self.pos_frac, self.seed = per_image, pos_frac, seed
        self.targets = targets
        self._rng = None

    def __len__(self):
        return len(self.images)

    def rng(self) -> random.Random:
        """Süreç başına tek üreteç: kalıcı işçilerde de her epoch farklı karolar kesilir."""
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = random.Random(hash((self.seed, info.seed if info else torch.initial_seed())))
        return self._rng

    def _load(self, i, rng):
        img = load_image(self.images[i], self.channels)
        _, H, W = img.shape
        boxes = load_boxes(self.images[i], H, W)
        sa = getattr(self, "scale_aug", None)
        if sa:
            s = math.exp(rng.uniform(math.log(sa[0]), math.log(sa[1])))
            nh, nw = max(8, round(H * s)), max(8, round(W * s))
            img = torch.nn.functional.interpolate(img[None], size=(nh, nw), mode="bilinear", align_corners=False,
                                                  antialias=s < 1)[0].clamp(0, 1)
            boxes = boxes * torch.tensor([nh / H, nw / W, nh / H, nw / W])
        sr = getattr(self, "size_range", None)
        if sr and len(boxes):                                       # H12: bu göz yalnız bu boyuttaki insanlardan sorumlu
            sz = torch.sqrt(boxes[:, 2] * boxes[:, 3])
            boxes = boxes[(sz >= sr[0]) & (sz < sr[1])]
        return img, boxes

    def _hard_entries(self):
        """Zor negatif havuzu: ana süreç dosyayı yeniledikçe yeniden okunur. [(görüntü indeksi, cy, cx), ...]"""
        f = getattr(self, "hard_pool", None)
        if not f or getattr(self, "hard_frac", 0) <= 0:
            return None
        try:
            mt = os.stat(f).st_mtime
        except OSError:
            return None
        if mt != getattr(self, "_pool_mtime", None):
            try:
                self._pool, self._pool_mtime = torch.load(f, weights_only=False), mt
            except Exception:                                       # yazılırken okunduysa sonraki çağrıda
                return getattr(self, "_pool", None)
        return self._pool

    def __getitem__(self, i):
        rng = self.rng()
        img, boxes = self._load(i, rng)
        _, H, W = img.shape
        tiles, tboxes = [], []
        pool = self._hard_entries()
        for _ in range(self.per_image):
            src_img, src_boxes = img, boxes
            if pool and rng.random() < self.hard_frac:              # H5: zor negatifin etrafından karo
                j, cy, cx = pool[rng.randrange(len(pool))]
                src_img, src_boxes = load_image_boxes(self.images[j], self.channels)   # havuz orijinal koordinatta
                oy = int(cy - self.ry - rng.uniform(0, self.rh))
                ox = int(cx - self.rx - rng.uniform(0, self.rw))
            elif len(boxes) and rng.random() < self.pos_frac:
                if getattr(self, "size_balance", False):
                    wts = torch.sqrt(boxes[:, 2] * boxes[:, 3]).tolist()
                    k = rng.choices(range(len(boxes)), weights=wts)[0]
                else:
                    k = rng.randrange(len(boxes))
                y, x, h, w = boxes[k].tolist()
                oy = int(y + h / 2 - self.ry - rng.uniform(0, self.rh))
                ox = int(x + w / 2 - self.rx - rng.uniform(0, self.rw))
            else:
                oy = rng.randint(-self.ry, max(-self.ry, H - self.ry - self.rh))
                ox = rng.randint(-self.rx, max(-self.rx, W - self.rx - self.rw))
            t = crop(src_img, oy, ox, self.eh, self.ew, self.pad_mode)
            if getattr(self, "photo_aug", False):                      # H2: gama / kontrast / parlaklık
                t = t.clamp(0, 1) ** math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
                m = t.mean()
                t = ((t - m) * rng.uniform(0.6, 1.4) + m + rng.uniform(-0.2, 0.2)).clamp(0, 1)
            b = src_boxes.clone()
            if len(b):
                b[:, 0] -= oy
                b[:, 1] -= ox
            if rng.random() < 0.5:                                       # yatay çevirme
                t = t.flip(-1)
                if len(b):
                    b[:, 1] = self.ew - b[:, 1] - b[:, 3]
            if rng.random() < 0.5:                                       # dikey çevirme (yukarıdan bakış)
                t = t.flip(-2)
                if len(b):
                    b[:, 0] = self.eh - b[:, 0] - b[:, 2]
            tiles.append(t)
            tboxes.append(b)
        if self.targets is not None:
            return torch.stack(tiles), tboxes, self.targets(tboxes)
        return torch.stack(tiles), tboxes


def collate(batch):
    tiles, boxes = torch.cat([b[0] for b in batch]), [x for b in batch for x in b[1]]
    if len(batch[0]) == 3:                                     # hedefler işçide üretildiyse
        return tiles, boxes, tuple(torch.cat([b[2][k] for b in batch]) for k in range(len(batch[0][2])))
    return tiles, boxes


# ============================================================================ değerlendirme
def dataset_of(path: Path) -> str:
    m = re.match(r"^(.+?)_[0-9a-f]{10}_", path.name)
    return m.group(1) if m else "?"


EVAL_OVERLAP = float(os.environ.get("FLYBRAIN_OVERLAP", "0.25"))   # §52: değerlendirmede karo bindirmesi (0,5 önerilen)


def tiler_for(model: FlyEyeDetector, pad_mode: str) -> Tiler:
    return Tiler(model.eye, overlap=EVAL_OVERLAP, pad_mode="replicate" if pad_mode == "replicate" else "constant")


def drop_outside(dets: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Merkezi görüntünün dışında (dolgu bölgesinde) kalan tespitleri atar; orada gerçek insan olamaz."""
    if not len(dets):
        return dets
    cy, cx = dets[:, 0] + dets[:, 2] / 2, dets[:, 1] + dets[:, 3] / 2
    return dets[(cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)]


def parse_scales(spec: str) -> list[tuple[float, float, float]]:
    """H30 (günlük §49) çok ölçekli bakış: "s:min:max,..." → [(ölçek, en küçük boyut, en büyük boyut)], boyutlar
    özgün görüntü pikselinde (√(h·w)); o ölçekteki tespitlerden yalnız bu aralıktakiler tutulur. Boş = tek ölçek."""
    out = []
    for part in [p for p in spec.split(",") if p.strip()]:
        f = [float(x) for x in part.split(":")]
        out.append((f[0], f[1] if len(f) > 1 else 0.0, f[2] if len(f) > 2 else 1e9))
    return out


def parse_scale_weights(spec: str) -> dict[float, float]:
    """Ölçek başına skor çarpanı: "0.5=0.6,0.25=0.4" (verilmeyen ölçek 1). Ana süreçte uygulanır (işçiler bilmez)."""
    out = {}
    for part in [p for p in spec.split(",") if p.strip()]:
        k, v = part.split("=")
        out[float(k)] = float(v)
    return out


EVAL_SCALES = parse_scales(os.environ.get("FLYBRAIN_MS", ""))   # ör. "1:0:40,0.5:24" — varsayılan tek ölçek
EVAL_SCALE_W = parse_scale_weights(os.environ.get("FLYBRAIN_MS_W", ""))


class _EvalImages(torch.utils.data.Dataset):
    """Değerlendirme: görüntü okuma + karolama veri yükleyici işçilerinde yapılır.
    scales verilirse görüntü her ölçekte ayrıca karolanır (aynı sinek gözü, görüntü küçültülerek; "uzaktan bakış")."""

    def __init__(self, images: list[Path], channels: int, tiler: Tiler, scales=None):
        self.images, self.channels, self.tiler, self.scales = images, channels, tiler, scales or []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, k):
        img = load_image(self.images[k], self.channels)
        _, H, W = img.shape
        if not self.scales:
            tiles, org = self.tiler.extract(img)
            return k, tiles, org, H, W, load_boxes(self.images[k], H, W)
        all_t, all_o = [], []
        for s, lo, hi in self.scales:
            im_s = img if s == 1.0 else torch.nn.functional.interpolate(
                img[None], scale_factor=s, mode="bilinear", antialias=True, align_corners=False)[0]
            t, o = self.tiler.extract(im_s)
            all_t.append(t)
            all_o += [(oy, ox, s, lo, hi) for oy, ox in o]
        return k, torch.cat(all_t), all_o, H, W, load_boxes(self.images[k], H, W)


@torch.no_grad()
def evaluate(model: FlyEyeDetector, images: list[Path], channels: int, device, chunk: int = 256,
             conf_thr: float = 0.3, pad_mode: str = "zero", outside: bool = True, workers: int = 3,
             scales=None, scale_weights=None) -> dict:
    """Tam görüntü karolama + mAP@0.5 (toplam ve kaynak set bazında) + conf_thr'de kesinlik/duyarlılık.
    outside=True: merkezi görüntü dışında kalan tespitler atılır.
    Hız için birkaç görüntünün karoları tek yığında işlenir (karolar birbirinden bağımsız; sonuç aynı)."""
    model.eval()
    flags, n_gt = {}, {}
    TP = FP = FN = 0
    t0 = time.time()
    scales = EVAL_SCALES if scales is None else scales
    scale_weights = EVAL_SCALE_W if scale_weights is None else scale_weights
    loader = torch.utils.data.DataLoader(_EvalImages(images, channels, tiler_for(model, pad_mode), scales),
                                         batch_size=None, num_workers=workers)
    pending, n_tiles, done = [], 0, 0

    def flush():
        nonlocal TP, FP, FN, done
        tiles = torch.cat([it[1] for it in pending]).to(device, non_blocking=True)
        dets = []
        for i in range(0, len(tiles), chunk):
            o = model(tiles[i:i + chunk])
            dets += (decode_cpu(o, model.eye, score_thr=0.01, max_det=100) if HIZ75 else    # §75: aynı sonuç, hızlı
                     [d.cpu() for d in decode(o, model.eye, score_thr=0.01, max_det=100)])
        s = 0
        for k, t, org, H, W, gts in pending:
            td = dets[s:s + len(t)]
            if org and len(org[0]) == 5:                                  # çok ölçekli: özgün koordinata taşı + süz
                moved = []
                for d, (oy, ox, sc, lo, hi) in zip(td, org):
                    if len(d):
                        d = d.clone()
                        d[:, :4] /= sc
                        sz = (d[:, 2] * d[:, 3]).clamp(min=0).sqrt()
                        d = d[(sz >= lo) & (sz < hi)]
                        d[:, 4] *= scale_weights.get(sc, 1.0)
                    moved.append(d)
                td, org = moved, [(oy / sc, ox / sc) for oy, ox, sc, _, _ in org]
            s += len(t)
            if HIZ75:                                                     # §75: önek NMS, aynı sonuç
                merged = merge_top(td, org, H, W, 0.5, 300, outside)
            else:
                merged = merge_tiles(td, org, 0.5)
                if outside:
                    merged = drop_outside(merged, H, W)
                merged = merged[merged[:, 4].argsort(descending=True)][:300] if len(merged) else merged
            ds = dataset_of(images[k])
            flags.setdefault(ds, []).extend(match_flags(merged, gts, 0.5))
            n_gt[ds] = n_gt.get(ds, 0) + len(gts)
            tp, fp, fn = match(merged[merged[:, 4] >= conf_thr] if len(merged) else merged, gts, 0.5)
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            done += 1
            if done % 500 == 0:
                log(f"    değerlendirme {done}/{len(images)} ({time.time() - t0:.0f} sn)")
        pending.clear()

    for item in loader:
        pending.append(item)
        n_tiles += len(item[1])
        if n_tiles >= chunk:
            flush()
            n_tiles = 0
    if pending:
        flush()
    model.train()
    all_flags = [f for v in flags.values() for f in v]
    res = {"images": len(images), "mAP50": average_precision(all_flags, sum(n_gt.values())),
           f"precision@{conf_thr}": TP / max(TP + FP, 1), f"recall@{conf_thr}": TP / max(TP + FN, 1),
           "by_dataset": {ds: {"gt": n_gt[ds], "mAP50": average_precision(flags[ds], n_gt[ds])}
                          for ds in sorted(flags)},
           "seconds": round(time.time() - t0, 1)}
    return res


@torch.no_grad()
def mine_hard_negatives(model, train_imgs, channels, device, args, path: Path, chunk: int = 256,
                        score_thr: float = 0.2, cap: int = 4000) -> int:
    """H5: rastgele train görüntülerinde tespit çalıştırır; hiçbir insana değmeyen (IoU < 0,1) yüksek skorlu
    tespitlerin merkezlerini havuza yazar: [(görüntü indeksi, cy, cx)]. Dosya atomik olarak değiştirilir;
    veri işçileri değişikliği görüp yeniden okur."""
    rng = random.Random(len(train_imgs) + int(time.time()))
    idx = rng.sample(range(len(train_imgs)), min(args.mine_images, len(train_imgs)))
    model.eval()
    tiler = tiler_for(model, args.pad)
    found = []
    pending, n = [], 0

    def flush():
        tiles = torch.cat([t for _, t, _, _ in pending]).to(device)
        dets = []
        for a in range(0, len(tiles), chunk):
            o = model(tiles[a:a + chunk])
            dets += (decode_cpu(o, model.eye, score_thr=score_thr, max_det=50) if HIZ75 else    # §75: aynı sonuç
                     [d.cpu() for d in decode(o, model.eye, score_thr=score_thr, max_det=50)])
        s = 0
        for k, t, org, gts in pending:
            m = merge_tiles(dets[s:s + len(t)], org, 0.5)
            s += len(t)
            if not len(m):
                continue
            if len(gts):
                m = m[box_iou(m[:, :4], gts).max(1).values < 0.1]
            for y, x, h, w, sc in m.tolist():
                found.append((sc, idx[k], y + h / 2, x + w / 2))
        pending.clear()

    for k, i in enumerate(idx):
        img = load_image(train_imgs[i], channels)
        tiles, org = tiler.extract(img)
        pending.append((k, tiles, org, load_boxes(train_imgs[i], img.shape[1], img.shape[2])))
        n += len(tiles)
        if n >= chunk:
            flush()
            n = 0
    if pending:
        flush()
    model.train()
    found.sort(key=lambda f: -f[0])
    pool = [(i, cy, cx) for _, i, cy, cx in found[:cap]]
    tmp = path.with_suffix(".tmp")
    torch.save(pool, tmp)
    os.replace(tmp, path)
    return len(pool)


# ============================================================================ eğitim
def build(args, device) -> FlyEyeDetector:
    net = "right_ol_merkez" if getattr(args, "merkez", False) else "right_ol"     # §41: optik lob + merkez beyin
    if getattr(args, "iki_lob", False):                                         # H35 (§56): sağ + sol optik lob + merkez
        net = "bi_ol_merkez"
    cfg = ModelConfig(processed_dir=str(ROOT / "data" / "connectome" / "processed" / f"{net}_thr{args.threshold}"),
                      in_channels=1 if args.modality == "thermal" else 3, patch=args.patch, steps=args.steps,
                      train_connectome=not args.reservoir, bridge_dim=args.bridge,
                      readout=getattr(args, "readout", "relu"), core=not getattr(args, "no_core", False),
                      topology=getattr(args, "topology", "fly"), topology_seed=getattr(args, "topology_seed", 0),
                      head_context=getattr(args, "head_context", "none"),
                      adapt=getattr(args, "adapt", "none"), quality=getattr(args, "quality", False),
                      channels=getattr(args, "channels", 1), bridge_drop=getattr(args, "bridge_drop", 0.0),
                      head_hidden=getattr(args, "head_hidden", 64),
                      readout_photo=not getattr(args, "okuma_fotoreseptorsuz", False),
                      readout_norm=getattr(args, "okuma_norm", "layer"), type_gain=getattr(args, "tip_kazanci", False),
                      encoder_act="none" if getattr(args, "kodlayici_dogrusal", False) else "relu",
                      init_gain=getattr(args, "init_gain", 0.8),
                      weight_norm=getattr(args, "agirlik_norm", "in_syn"), activation=getattr(args, "aktivasyon", "relu"),
                      init_bias=getattr(args, "init_bias", 0.1), input_lamina=getattr(args, "lamina_girisi", False),
                      view=getattr(args, "gorus", "A"), view_overlap=getattr(args, "gorus_ortak", 0.2),
                      homeostasis=getattr(args, "homeostaz", "kapali") != "kapali",
                      microsaccade=getattr(args, "mikrosakkad", 0), act_soft=getattr(args, "yumusak", False),
                      noise_sigma=getattr(args, "gurultu", 0.0), fatigue=getattr(args, "yorulma", False),
                      shunt=getattr(args, "sant", False), shunt_e_inh=getattr(args, "sant_e_inh", 0.0),
                      modulators=getattr(args, "modulator", False), superposition=getattr(args, "superpozisyon", False),
                      microsaccade_views=getattr(args, "mikrosakkad_gorus", False),
                      lamina_weights=getattr(args, "lamina_agirlik", False),
                      stp=getattr(args, "stp", False))
    return FlyEyeDetector(cfg).to(device)


def type_input_coverage(model, names) -> torch.Tensor:
    """Tip başına ortalama girdi kapsamı: modeldeki (≥ eşik sinapslı, modeldeki nöronlardan) girdi / FlyWire'daki bütün
    girdi (saturation_audit ile aynı tanım). Önbellek results/flybrain/_cache/."""
    import pandas as pd
    cdir = OUT / "_cache"
    cdir.mkdir(parents=True, exist_ok=True)
    f = cdir / f"kapsam_{Path(model.cfg.processed_dir).name}.npy"
    if f.is_file():
        cov = np.load(f)
    else:
        from tools.build_connectome import read_edges
        nodes = pd.read_csv(Path(model.cfg.processed_dir) / "nodes.csv", low_memory=False)
        tot = read_edges().groupby("post_root_id").syn_count.sum()
        cov = (nodes.in_syn.to_numpy() / tot.reindex(nodes.root_id).fillna(0).clip(lower=1).to_numpy()).clip(0, 1)
        np.save(f, cov)
    tid = model.type_id.cpu()
    nt = len(names)
    c = torch.zeros(nt).index_add_(0, tid, torch.as_tensor(cov, dtype=torch.float32))
    return c / torch.bincount(tid, minlength=nt).clamp(min=1).float()


def setup_homeostasis_v2(model, args, kf: Path, names, tiles, device) -> dict:
    """§71 H41c: hedef ölüler = gerçek girdide aktif oranı < %1 + girdi kapsamı ≥ %50 + uyaranı mevcut (hareket / renk /
    polarizasyon / saat tipi değil); gerçek doygun adaylar = ort. r > 0,9 ve karolar arası std < 0,05."""
    from flybrain.homeostasis import FUNCTION_PATTERNS, H41C, type_stats
    P0, m0, s0 = type_stats(model, tiles)
    cov = type_input_coverage(model, names).to(device)
    pats = list(FUNCTION_PATTERNS.values()) + [r"(LNv|^DN1|^DN3|^LPN)"]
    excl = torch.tensor([any(re.search(p, n) for p in pats) for n in names], device=device)
    dead = (P0 < 0.01) & (cov >= 0.5) & ~excl
    sat = (m0 > H41C["sat_r"]) & (s0 < H41C["sat_std"])
    nt = len(names)
    st = {"cfg": dict(H41C), "dead": dead, "sat": sat, "frozen": torch.zeros(nt, dtype=torch.bool, device=device),
          "push_n": torch.zeros(nt, dtype=torch.long, device=device), "P_start": torch.zeros(nt, device=device)}
    info = {"varyant": "v2 (H41c)", "olu_gercek_girdi": int((P0 < 0.01).sum()), "olu_yapisal_disari": int(((P0 < 0.01) & (cov < 0.5)).sum()),
            "olu_uyaransiz_disari": int(((P0 < 0.01) & (cov >= 0.5) & excl).sum()), "hedef_olu": int(dead.sum()),
            "gercek_doygun": int(sat.sum()), "kurallar": H41C,
            "hedef_tipler": [n for n, d in zip(names, dead.tolist()) if d], "doygun_tipler": [n for n, x in zip(names, sat.tolist()) if x]}
    kf.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"homeostaz v2 (H41c): uyku karosu {len(tiles)}, gerçek girdide ölü tip {info['olu_gercek_girdi']} → yapısal dışarı "
        f"{info['olu_yapisal_disari']}, uyaransız dışarı {info['olu_uyaransiz_disari']}, hedef {info['hedef_olu']}; gerçek doygun {info['gercek_doygun']}")
    return {"tiles": tiles, "v2": st}


def setup_homeostasis(model, args, out: Path, device) -> dict:
    """H41b: uyku karoları (etiketsiz train), başlangıçta ölü (< %1) ve doygun (> %90) tip kümeleri. Kümeler dosyaya
    yazılır; kesintiden devamda aynı kümeler okunur."""
    from flybrain.connectome import Connectome
    from flybrain.homeostasis import FUNCTION_PATTERNS, type_active_fraction, type_names
    t = Tiler(model.eye, overlap=0.25, pad_mode="replicate" if args.pad == "replicate" else "constant")
    tiles = torch.cat([t.extract(load_image(p, model.cfg.in_channels))[0]
                       for p in random.Random(7).sample(list_images(args.modality, "train"), args.uyku_goruntu)])
    kf = out / "homeostaz_kume.json"
    names = type_names(Connectome.load(model.cfg.processed_dir))
    if args.homeostaz == "v2":                                          # §71 H41c
        return setup_homeostasis_v2(model, args, kf, names, tiles, device)
    if kf.is_file():
        k = json.loads(kf.read_text(encoding="utf-8"))
        dead = torch.tensor([n in set(k["olu"]) for n in names], device=device)
        sat = torch.tensor([n in set(k["doygun"]) for n in names], device=device)
        log(f"homeostaz kümeleri dosyadan: ölü {int(dead.sum())}, doygun {int(sat.sum())}")
        return {"tiles": tiles, "dead": dead, "sat": sat}
    P0 = type_active_fraction(model, tiles)
    dead, sat = P0 < 0.01, P0 > 0.90
    info = {"varyant": args.homeostaz, "olu_baslangic": int(dead.sum()), "doygun_sayi": int(sat.sum())}
    if args.homeostaz == "uyaransiz_haric":
        excl = torch.tensor([any(re.search(p, n) for p in FUNCTION_PATTERNS.values()) for n in names], device=device)
        ck = torch.load(OUT / args.homeo_ref / "best.pt", map_location="cpu", weights_only=False)
        ref = build(argparse.Namespace(**ck["args"]), device)
        ref.load_state_dict(ck["model"])
        Pr = type_active_fraction(ref, tiles)
        rn = type_names(Connectome.load(ref.cfg.processed_dir))
        alive = {n for n, pr in zip(rn, Pr.tolist()) if pr >= 0.01}
        del ref
        torch.cuda.empty_cache()
        keep = torch.tensor([n in alive for n in names], device=device)          # 12 sa'da canlı, iki lobda ölü
        info.update({"disari_uyaransiz": int((dead & excl & ~keep).sum()),
                     "12sa_canli_iki_lobda_olu_uyaransiz": [n for n, d, e, kk in zip(names, dead.tolist(), excl.tolist(),
                                                                                      keep.tolist()) if d and e and kk]})
        dead = dead & ~(excl & ~keep)
    info.update({"olu": [n for n, d in zip(names, dead.tolist()) if d], "doygun": [n for n, x in zip(names, sat.tolist()) if x],
                 "olu_uygulanan": int(dead.sum())})
    kf.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"homeostaz ({args.homeostaz}): uyku karosu {len(tiles)}, ölü tip {info['olu_baslangic']} → uygulanan "
        f"{info['olu_uygulanan']}, doygun tip {info['doygun_sayi']}")
    return {"tiles": tiles, "dead": dead, "sat": sat}


def keep_awake():
    """Windows: eğitim sürerken bilgisayarın uyku moduna geçmesini engeller. Sistem ayarını DEĞİŞTİRMEZ;
    yalnızca bu süreç çalıştığı sürece geçerli bir istek (video oynatıcıların yaptığı gibi), süreç bitince kalkar."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def progress(it: int, train_sec: float, args) -> float:
    """Eğitimin ilerlemesi [0, 1]. --hours verilmişse geçen EĞİTİM süresine göre (değerlendirme süresi hariç;
    bilgisayar başka işle yavaşlasa da plan kaymaz), yoksa iterasyona göre."""
    if args.hours > 0:
        if getattr(args, "lr_dongu_saat", 0) > 0:                       # süresiz: döngü sayısı (sınırsız)
            return train_sec / (args.hours * 3600)
        return min(1.0, train_sec / (args.hours * 3600))
    return min(1.0, it / max(1, args.iters))


def lr_at(it: int, args, prog: float | None = None) -> float:
    """Isınma (iterasyonla) + kosinüs (ilerlemeyle), en düşük %5."""
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    if prog is None:                                                    # iterasyon tabanlı (eski davranış)
        t = (it - args.warmup) / max(1, args.iters - args.warmup)
    else:
        t = prog
    if getattr(args, "lr_dongu_saat", 0) > 0 and prog is not None:      # süresiz eğitim: her döngüde kosinüs yeniden
        t = prog % 1.0
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def load_init(model, path: str) -> list[str]:
    """--init-from: başka bir eğitimin ağırlıklarıyla başla (iyileştirici ve plan sıfırdan). Yeni eklenen parçalar
    (ör. kalite dalı: başlığın son katmanı 5 → 6 çıktı) için örtüşen kısım kopyalanır, gerisi kendi başlangıcında kalır."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    src = ck["model"] if "model" in ck else ck["parametreler"]   # best.pt ya da sıkıştırılmış ağırlık (tools/agirlik.py)
    own, notes = model.state_dict(), []
    for k, v in own.items():
        if k not in src:
            notes.append(f"{k}: kaynakta yok (yeni)")
            continue
        s = src[k]
        if s.shape == v.shape:
            own[k] = s
        elif k == "encoder.0.weight" and s.dim() == 2 and s.shape[0] == v.shape[0] and v.shape[1] == 3 * s.shape[1]:
            # §79 termal (1 kanal) → RGB (3 kanal): yama düzeni [ham kanallar..., z kanalları...] × p²; her blok üç
            # kanala /3 ile dağıtılır → R = G = B olan (gri) bir görüntü başlangıçta termaldeki gibi işlenir
            pp = model.cfg.patch * model.cfg.patch
            nb = s.shape[1] // pp
            own[k] = v.clone()
            for j in range(nb):
                for c in range(3):
                    own[k][:, (j * 3 + c) * pp:(j * 3 + c + 1) * pp] = s[:, j * pp:(j + 1) * pp] / 3
            notes.append(f"{k}: {tuple(s.shape)} → {tuple(v.shape)} (termal → RGB, kanal başına /3)")
        elif s.dim() == v.dim() and s.shape[1:] == v.shape[1:]:
            n = min(s.shape[0], v.shape[0])
            own[k] = v.clone()
            own[k][:n] = s[:n]
            notes.append(f"{k}: {tuple(s.shape)} → {tuple(v.shape)} (ilk {n} satır kopyalandı)")
        else:
            notes.append(f"{k}: biçim uyuşmuyor {tuple(s.shape)} / {tuple(v.shape)} (atlandı)")
    model.load_state_dict(own)
    return notes


def save(path: Path, model, opt, it, best, args, train_sec: float = 0.0, **extra):
    """extra (§67): ema (EMA durum sözlüğü), lr_olcek, plato_kotu, ema_agirlik (bu dosyadaki model EMA mı)."""
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it, "best": best,
                "train_sec": train_sec, "args": vars(args), "config": model.cfg.to_dict(), **extra}, path)


def make_parser() -> argparse.ArgumentParser:
    """Eğitim seçenekleri (araçlar da kullanır: speed_profile.py)."""
    ap = argparse.ArgumentParser(description="Sinek beyni modelini gerçek veriyle eğit")
    ap.add_argument("--modality", default="thermal", choices=["thermal", "rgb", "rgb640"],
                    help="rgb640 (§79): RGB, uzun kenar 640'a küçültülmüş kopya (data/yolo/rgb640)")
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--threshold", type=int, default=5, choices=[1, 2, 5])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--reservoir", action="store_true", help="konnektom sabit, sadece kodlayıcı + başlık")
    ap.add_argument("--bridge", type=int, default=0, help="ince ayrıntı köprüsü boyutu (0 = kapalı)")
    ap.add_argument("--pad", default="replicate", choices=["replicate", "zero"],
                    help="görüntü dışı dolgu; zero = ilk eğitimdeki (kenarda sahte tespit yapıyordu)")
    ap.add_argument("--readout", default="relu", choices=["relu", "voltage"],
                    help="okuma: relu(v) etkinliği ya da zar gerilimi v")
    ap.add_argument("--no-core", action="store_true",
                    help="çekirdeksiz kontrol: köprü → doğrudan başlık (--bridge > 0 gerekir)")
    ap.add_argument("--topology", default="fly", choices=["fly", "random"],
                    help="random: rastgele topoloji kontrolü (derece ve α grupları korunur)")
    ap.add_argument("--topology-seed", type=int, default=0)
    ap.add_argument("--scale-aug", default="", help="H1: eğitimde rastgele ölçek aralığı, ör. 0.5,2 (boş = kapalı)")
    ap.add_argument("--size-balance", action="store_true", help="H1: pozitif karoda büyük insanlar daha sık")
    ap.add_argument("--photo-aug", action="store_true", help="H2: karo başına gama/kontrast/parlaklık artırma")
    ap.add_argument("--adapt", default="none", choices=["none", "local", "column"],
                    help="H2: fotoreseptör adaptasyonu (local: ~4 kolonluk pencere, column: yalnız kolonun kendi yaması)")
    ap.add_argument("--giou", type=float, default=0.0, help="H3: GIoU kutu kaybı ağırlığı (0 = kapalı)")
    ap.add_argument("--w-size", type=float, default=0.5, help="H22: boyut (log h, log w) kaybının ağırlığı")
    ap.add_argument("--okuma-fotoreseptorsuz", action="store_true",
                    help="§44: fotoreseptörleri okuma (konnektomu atlayan gizli köprü)")
    ap.add_argument("--okuma-norm", default="layer", choices=["layer", "batch"],
                    help="G1: batch = okumada özellik başına sabit standartlaştırma")
    ap.add_argument("--tip-kazanci", action="store_true", help="G4: hücre tipi başına öğrenilen kazanç")
    ap.add_argument("--init-gain", type=float, default=0.8, help="G3: başlangıç kazancı (|girdi ağırlıkları| toplamı)")
    ap.add_argument("--alpha-lr-mult", type=float, default=1.0, help="G2: α (konnektom ağırlıkları) öğrenme oranı çarpanı")
    ap.add_argument("--agirlik-norm", default="in_syn", choices=["in_syn", "l2"],
                    help="G6: l2 = her nöronun Σw² = 1 (Xavier benzeri; sinyal derinlikte sönmez)")
    ap.add_argument("--aktivasyon", default="relu", choices=["relu", "sat"], help="G6: sat = tanh(relu(v)), doymalı")
    ap.add_argument("--lamina-girisi", action="store_true",
                    help="G8: kodlayıcı L1–L3'e de girdi verir (FlyWire'da eksik R→L sinapslarının vekili)")
    ap.add_argument("--init-bias", type=float, default=0.1,
                    help="G7: nöron dinlenme etkinliği (tonik salım); yüksek değer relu'nun kesmediği doğrusal bölge")
    ap.add_argument("--kodlayici-dogrusal", action="store_true", help="kodlayıcıda ReLU yok (mercek + fotoreseptör)")
    ap.add_argument("--merkez", action="store_true",
                    help="optik lob + merkez beyin modülü (2.781 nöron, ileri + geri besleme; günlük §41)")
    ap.add_argument("--head-hidden", type=int, default=64,
                    help="başlık gizli birim sayısı; 0 + --head-context none = doğrusal okuma (B5)")
    ap.add_argument("--bridge-drop", type=float, default=0.0, help="H16: eğitimde köprüyü karo bazında kapatma olasılığı")
    ap.add_argument("--channels", type=int, default=1, help="H14: nöron başına kanal d (1 = biyolojik)")
    ap.add_argument("--quality", action="store_true", help="H4: kalite dalı (skor = ısı × kalite)")
    ap.add_argument("--hard-neg", type=float, default=0.0, help="H5: zor negatif karo oranı (0 = kapalı)")
    ap.add_argument("--mine-every", type=int, default=1000, help="H5: kaç iterasyonda bir zor negatif toplanır")
    ap.add_argument("--mine-images", type=int, default=300, help="H5: her toplamada taranan train görüntüsü")
    ap.add_argument("--head-context", default="none", choices=["none", "hex7", "hex19", "hex7d"],
                    help="hex7: başlık kolonun 6 altıgen komşusunu da görür")
    ap.add_argument("--reg-mode", default="center", choices=["center", "box"],
                    help="box: boyut/ofset kutunun içindeki bütün kolonlarda eğitilir (tepe komşu kolondan çıksa da doğru)")
    ap.add_argument("--final-splits", default="val,test",
                    help="eğitim sonunda tam değerlendirilecek kümeler; pilotlarda 'val' (test son modele saklanır)")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--hours", type=float, default=0,
                    help="süre tabanlı eğitim: bu kadar saat EĞİTİM (değerlendirme hariç); --iters yok sayılır")
    ap.add_argument("--eval-minutes", type=float, default=0,
                    help="> 0 ise ara değerlendirme bu kadar dakikalık eğitimde bir (--eval-every yerine)")
    ap.add_argument("--batch", type=int, default=32, help="karo sayısı / iterasyon")
    ap.add_argument("--per-image", type=int, default=4, help="bir görüntüden kesilen karo")
    ap.add_argument("--pos-frac", type=float, default=0.7, help="insan içeren karo oranı (hedef)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-images", type=int, default=400, help="ara değerlendirmede val görüntü sayısı")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="bu kontrol noktasının ağırlıklarıyla başla (ince ayar pilotu / eğitimi sürdürme); plan sıfırdan")
    ap.add_argument("--iki-lob", action="store_true",
                    help="H35: iki optik loblu konnektom (sağ + sol + merkez); sol göz aynalanmış karoyu görür")
    ap.add_argument("--gorus", default="A", choices=["A", "B"],
                    help="H35 görüş alanı: A iki göz de bütün karo, B her göz kendi yarısı + ortak şerit")
    ap.add_argument("--gorus-ortak", type=float, default=0.2, help="H35 B: ortak şeridin karo genişliğine oranı")
    ap.add_argument("--homeostaz", default="kapali", choices=["kapali", "hepsi", "uyaransiz_haric", "v2"],
                    help="H41b: ölü bant homeostazı (eşik + kazanç, gradyansız). hepsi: bütün ölü tipler; "
                         "uyaransiz_haric: hareket/renk/polarizasyon tipleri dışarıda (12 sa modelinde canlı olanlar hariç)")
    ap.add_argument("--uyku-her", type=int, default=500, help="H41b: kaç iterasyonda bir uyku evresi")
    ap.add_argument("--mikrosakkad", type=int, default=0, help="H51: titreşim genliği (px), eğitim + değerlendirme")
    ap.add_argument("--lamina-agirlik", action="store_true",
                    help="§71 H57: yön koruyan lamina girişi (L1–L3 altı bakışı öğrenilen ağırlıklarla toplar)")
    ap.add_argument("--mikrosakkad-gorus", action="store_true",
                    help="§71 hata #8: süperpozisyonla birlikte mikrosakkad (her adımda kaydırılmış 7 bakış). Verilmezse "
                         "süperpozisyon açıkken mikrosakkad uygulanmaz (eski davranış)")
    ap.add_argument("--yumusak", action="store_true", help="H52: tip başına keskinliği öğrenilen softplus aktarımı")
    ap.add_argument("--gurultu", type=float, default=0.0, help="H43: eğitimde gerilim gürültüsü σ")
    ap.add_argument("--yorulma", action="store_true", help="H40: sinaptik yorgunluk + nöron adaptasyonu (özdeşlik başı)")
    ap.add_argument("--sant", action="store_true", help="H42: iletkenlik tabanlı (şant) sinaps")
    ap.add_argument("--sant-e-inh", type=float, default=0.0, help="H42: baskılayıcı denge potansiyeli")
    ap.add_argument("--modulator", action="store_true", help="H45: içsel modülatör (DA/SER/OCT), özdeşlik başı")
    ap.add_argument("--superpozisyon", action="store_true", help="H44: nöral süperpozisyon girişi (6 bakış)")
    ap.add_argument("--stp", action="store_true", help="H53: işaretli kısa süreli plastisite (kolaylaşma/yorulma, tip başına)")
    ap.add_argument("--uyku-goruntu", type=int, default=20, help="H41b: uyku karoları için etiketsiz train görüntüsü")
    ap.add_argument("--homeo-ref", default="uzun12s_sinek", help="H41b-2: 'canlı mı' listesi için tek gözlü referans")
    ap.add_argument("--init-opt", action="store_true", help="--init-from ile iyileştiricinin (Adam) durumunu da al")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="§67: ağırlıkların üstel ortalaması (ör. 0.999); ara değerlendirme ve best.pt EMA modeliyle")
    ap.add_argument("--ema-ikisi", action="store_true", help="§67: EMA açıkken ham modeli de değerlendir (teşhis)")
    ap.add_argument("--plato", type=int, default=0,
                    help="§67: dinamik öğrenme oranı: ısınmadan sonra sabit lr; val bu kadar değerlendirme boyunca "
                         "iyileşmezse lr yarıya (0 = eski kosinüs planı)")
    ap.add_argument("--plato-min", type=float, default=0.0625,
                    help="§67: lr ölçeği bunun altına inince eğitim biter (varsayılan 1/16)")
    ap.add_argument("--plato-taban", type=float, default=0.0,
                    help="§75: > 0 ise plato yarılaması bu ölçekte durur ve eğitim plato yüzünden HİÇ erken bitmez "
                         "(süre --toplam-saat ile dolar); 0 = eski davranış (--plato-min'de durur)")
    ap.add_argument("--set-us", type=float, default=0.0,
                    help="§67 (H17): set dengeli örnekleme; görüntü ağırlığı = (setin görüntü sayısı)^-us "
                         "(0 = düzgün, 0.5 = karekök, 1 = her set eşit)")
    ap.add_argument("--toplam-saat", type=float, default=0,
                    help="§65: döngülü eğitimde (--lr-dongu-saat) toplam eğitim süresi sınırı (saat); 0 = süresiz")
    ap.add_argument("--lr-dongu-saat", type=float, default=0,
                    help="süresiz eğitim: öğrenme oranı her bu kadar saatte kosinüsle yeniden başlar; durdurulana kadar sürer")
    ap.add_argument("--ckpt-minutes", type=float, default=30, help="bu kadar dakikada bir last.pt kaydet")
    ap.add_argument("--eval-only", default=None, help="kontrol noktası yolu: sadece val + test değerlendir")
    return ap


def main(argv=None) -> int:
    ap = make_parser()
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.eval_only:
        ck = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        for k in ("modality", "patch", "threshold", "steps", "reservoir", "bridge"):
            setattr(args, k, ck["args"][k])
        args.readout = ck["args"].get("readout", "relu")
        args.no_core = ck["args"].get("no_core", False)
        args.topology = ck["args"].get("topology", "fly")
        args.topology_seed = ck["args"].get("topology_seed", 0)
        args.head_context = ck["args"].get("head_context", "none")
        args.pad = ck["args"].get("pad", "zero")                   # --pad eklenmeden önceki modeller: zero
        model = build(args, device)
        model.load_state_dict(ck["model"])
        ch = model.cfg.in_channels
        res = {s: evaluate(model, list_images(args.modality, s), ch, device, pad_mode=args.pad)
               for s in ("val", "test")}
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    name = args.name or (f"{args.modality}_p{args.patch}_thr{args.threshold}"
                         f"{'_rezervuar' if args.reservoir else ''}{f'_kopru{args.bridge}' if args.bridge else ''}")
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)

    model = build(args, device)
    ch = model.cfg.in_channels
    params = [p for p in model.parameters() if p.requires_grad]
    alpha_mult = getattr(args, "alpha_lr_mult", 1.0)
    if alpha_mult != 1.0 and model.alpha_raw.requires_grad:              # G2: α ayrı öğrenme oranıyla
        others = [p for p in params if p is not model.alpha_raw]
        opt = torch.optim.Adam([{"params": others, "mult": 1.0}, {"params": [model.alpha_raw], "mult": alpha_mult}],
                               args.lr)
    else:
        opt = torch.optim.Adam(params, args.lr)
    start, best, train_sec0 = 0, -1.0, 0.0
    if args.init_from and not (args.resume and (out / "last.pt").is_file()):
        ck0 = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if Path(ck0["config"]["processed_dir"]).name != Path(model.cfg.processed_dir).name:
            # başka konnektom (ör. sağ lob → iki lob): parametreler hücre tipi / tip çifti / özellik ADLARIYLA taşınır
            from flybrain.connectome import Connectome
            from flybrain.transfer import transfer_by_names
            src = build(argparse.Namespace(**ck0["args"]), torch.device("cpu"))
            src.load_state_dict(ck0["model"])
            rep = transfer_by_names(model, src.to(device), Connectome.load(model.cfg.processed_dir),
                                    Connectome.load(src.cfg.processed_dir))
            log(f"  adla aktarım ({Path(src.cfg.processed_dir).name} → {Path(model.cfg.processed_dir).name}): {rep}")
            del src
            args.init_opt = False                                             # iyileştirici durumu taşınamaz
        else:
            for note in load_init(model, args.init_from):
                log(f"  başlangıç ağırlıkları: {note}")
        if args.init_opt:
            try:
                opt.load_state_dict(torch.load(args.init_from, map_location="cpu", weights_only=False)["opt"])
                log("  iyileştirici durumu da alındı")
            except (ValueError, KeyError) as e:
                log(f"  iyileştirici durumu alınamadı ({e}); sıfırdan")
        log(f"başlangıç: {args.init_from}")
    if args.resume and (out / "last.pt").is_file():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best, train_sec0 = ck["iter"], ck["best"], ck.get("train_sec", 0.0)
        log(f"devam: iterasyon {start}, eğitim süresi {train_sec0 / 3600:.2f} sa, en iyi val mAP50 {best:.4f}")
    ema, lr_scale, bad, stop_flag = None, 1.0, 0, [False]
    if args.ema > 0:                                                     # §67: EMA modeli (başlangıçta modelin kopyası)
        import copy
        ema = copy.deepcopy(model).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
    if args.resume and (out / "last.pt").is_file():
        ckr = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        if ema is not None and "ema" in ckr:
            ema.load_state_dict(ckr["ema"])
        lr_scale, bad = ckr.get("lr_olcek", 1.0), ckr.get("plato_kotu", 0)
        del ckr
    ema_pairs = list(zip(ema.parameters(), model.parameters())) if ema is not None else []
    st = lambda: dict(ema=ema.state_dict() if ema is not None else None, lr_olcek=lr_scale, plato_kotu=bad)
    if args.lr_dongu_saat > 0:                                           # süresiz eğitim: döngü uzunluğu = hours
        args.hours = args.lr_dongu_saat
    if args.hours > 0:
        args.iters = 10 ** 12                                            # durma koşulu süre
    (out / "config.json").write_text(json.dumps({"args": vars(args), "model": model.cfg.to_dict(),
                                                 "parametreler": model.n_parameters(), "cihaz": str(device)},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"{name}: {model.n_parameters()}  cihaz {device}  göz {model.eye.height}x{model.eye.width} px")

    train_imgs = list_images(args.modality, "train")
    val_imgs = list_images(args.modality, "val")
    val_sub = random.Random(0).sample(val_imgs, min(args.eval_images, len(val_imgs)))
    tiler = Tiler(model.eye)
    tb = TargetBuilder(model.eye, reg_mode=args.reg_mode, quality=args.quality)
    scale_aug = tuple(float(v) for v in args.scale_aug.split(",")) if args.scale_aug else None
    hard_pool = str(out / "zor_negatif.pt") if args.hard_neg > 0 else None
    ds = TileDataset(train_imgs, ch, (model.eye.height, model.eye.width),
                     (tiler.ry, tiler.rx, tiler.rh, tiler.rw), args.per_image, args.pos_frac, args.seed,
                     args.pad, targets=(tb if model.n_eyes == 1 else BilateralTargets(
                         tb, model.eye.width, None if model.eye_visible is None
                         else model.eye_visible[:model.n_eye_columns].cpu())),
                     scale_aug=scale_aug, size_balance=args.size_balance,
                     photo_aug=args.photo_aug, hard_pool=hard_pool, hard_frac=args.hard_neg)
    sampler = None
    if args.set_us > 0:                                                  # §67 (H17): set dengeli örnekleme
        from collections import Counter
        cnt = Counter(dataset_of(q) for q in train_imgs)
        wts = torch.tensor([cnt[dataset_of(q)] ** (-args.set_us) for q in train_imgs], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(wts, num_samples=len(train_imgs), replacement=True,
                                                         generator=torch.Generator().manual_seed(args.seed))
        tot = sum(c ** (1 - args.set_us) for c in cnt.values())
        log(f"set dengeli örnekleme (us {args.set_us}): "
            + ", ".join(f"{k} %{100 * cnt[k] ** (1 - args.set_us) / tot:.0f}" for k in sorted(cnt)))
    loader_kw = dict(batch_size=max(1, args.batch // args.per_image), shuffle=sampler is None, sampler=sampler,
                     collate_fn=collate, num_workers=args.workers, drop_last=True, pin_memory=device.type == "cuda")
    if args.workers > 0:
        # Kalıcı işçiler: Windows'ta her epoch işçileri yeniden başlatmak ~10 sn sürüyordu. Rastgelelik
        # işçi başına üreteçten geldiği için her epoch yine farklı karolar kesilir.
        loader_kw.update(persistent_workers=True, prefetch_factor=4)
    loader = torch.utils.data.DataLoader(ds, **loader_kw)
    log(f"train {len(train_imgs)} görüntü, val {len(val_imgs)} (ara değerlendirme {len(val_sub)})")

    log_path = out / "log.csv"
    new_log = not (args.resume and log_path.is_file())
    logf = log_path.open("w" if new_log else "a", newline="", encoding="utf-8")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["iter", "lr", "loss", "heat", "offset", "size", "val_mAP50", "sn", "val_ham"])

    homeo = None
    if args.homeostaz != "kapali":                                        # H41b (§62): ölü bant homeostazı
        homeo = setup_homeostasis(model, args, out, device)
    keep_awake()
    it, t0 = start, time.time()
    t_win, it_win, t_ckpt = t0, it, t0
    eval_spent = 0.0                                                     # değerlendirme/kayıt süresi (eğitim sayılmaz)
    train_sec = lambda: train_sec0 + (time.time() - t0) - eval_spent
    last_eval_sec = train_sec0
    use_prog = args.hours > 0
    run_sum, run_n = torch.zeros(4, device=device), 0
    model.train()
    finished = lambda: it >= args.iters or (use_prog and args.lr_dongu_saat <= 0
                                            and progress(it, train_sec(), args) >= 1.0)         or (getattr(args, "toplam_saat", 0) > 0 and train_sec() >= args.toplam_saat * 3600) or stop_flag[0]
    while not finished():
        for tiles, boxes, tg in loader:
            if finished():
                break
            if args.plato > 0:                                         # §67: ısınma + sabit × plato ölçeği
                lr_now = (args.lr * (it + 1) / args.warmup if it < args.warmup else args.lr) * lr_scale
            else:
                lr_now = lr_at(it, args, progress(it, train_sec(), args) if use_prog else None)
            for g in opt.param_groups:
                g["lr"] = lr_now * g.get("mult", 1.0)
            nb = dict(device=device, non_blocking=True)
            outp = model(tiles.to(**nb))
            heat, reg, pos = (t.to(**nb) for t in tg[:3])
            L = detection_loss(outp, heat, reg, pos, regm=tg[3].to(**nb) if len(tg) > 3 else None,
                               w_giou=args.giou, w_size=args.w_size, qreg=tg[4].to(**nb) if len(tg) > 5 else None,
                               qm=tg[5].to(**nb) if len(tg) > 5 else None)
            opt.zero_grad(set_to_none=True)
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            if ema_pairs:                                              # §67: EMA ← d·EMA + (1 − d)·model
                with torch.no_grad():
                    torch._foreach_mul_([e for e, _ in ema_pairs], args.ema)
                    torch._foreach_add_([e for e, _ in ema_pairs], [m_.detach() for _, m_ in ema_pairs],
                                        alpha=1 - args.ema)
            it += 1
            # Kayıplar GPU'da biriktirilir; her iterasyonda .item()/isfinite ile GPU'yu beklemek CPU ile GPU'nun
            # üst üste çalışmasını engelliyordu. NaN kontrolü 100 iterasyonda bir yapılır.
            run_sum += torch.stack([L["total"].detach(), L["heat"], L["offset"], L["size"]])
            run_n += 1

            if it % 100 == 0:
                m = (run_sum / run_n).tolist()
                run_sum.zero_()
                run_n = 0
                now = time.time()
                speed = (it - it_win) / (now - t_win)                    # son pencerenin hızı (başlangıç hariç)
                t_win, it_win = now, it
                if not all(math.isfinite(x) for x in m):
                    if not (out / "last.pt").is_file():
                        sys.exit(f"! iterasyon {it}: kayıp NaN/sonsuz ve geri dönülecek kontrol noktası yok")
                    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
                    model.load_state_dict(ck["model"])
                    opt.load_state_dict(ck["opt"])
                    log(f"! iterasyon {it}: kayıp NaN/sonsuz → son kontrol noktasına (it {ck['iter']}) dönüldü")
                    continue
                if use_prog and args.lr_dongu_saat > 0:
                    left = f"döngü {int(progress(it, train_sec(), args)) + 1}, süresiz ({train_sec() / 3600:.1f} sa)"
                    total = "∞"
                elif use_prog:
                    left = f"kalan ~{(args.hours * 3600 - train_sec()) / 3600:.1f} sa (süre tabanlı)"
                    total = "∞"
                else:
                    left = f"kalan ~{(args.iters - it) / speed / 3600:.1f} sa"
                    total = str(args.iters)
                log(f"it {it}/{total}  kayıp {m[0]:.3f} (ısı {m[1]:.3f}, ofset {m[2]:.3f}, boyut {m[3]:.3f})"
                    f"  lr {lr_now:.2e}  {speed:.1f} it/s ({speed * args.batch:.0f} karo/sn)  {left}")
                logw.writerow([it, f"{lr_now:.3e}", *[f"{x:.4f}" for x in m], "", round(now - t0)])
                logf.flush()
                if now - t_ckpt > args.ckpt_minutes * 60:                # uzun eğitimde zamanlı kontrol noktası
                    t_s = time.time()
                    save(out / "last.pt", model, opt, it, best, args, train_sec(), **st())
                    eval_spent += time.time() - t_s
                    t_ckpt = now
            if homeo is not None and it % args.uyku_her == 0:              # H41b: uyku evresi (gradyansız)
                t_h = time.time()
                if "v2" in homeo:                                       # §71 H41c
                    from flybrain.homeostasis import homeo_step_v2, type_stats
                    P, mr, sr = type_stats(model, homeo["tiles"])
                    rep = homeo_step_v2(model, homeo["v2"], P, mr, sr)
                else:
                    from flybrain.homeostasis import homeo_step, type_active_fraction
                    P = type_active_fraction(model, homeo["tiles"])
                    rep = homeo_step(model, P, homeo["dead"], homeo["sat"])
                log(f"  uyku evresi: {rep}")
                with (out / "homeostaz.csv").open("a", encoding="utf-8") as hf:
                    hf.write(f"{it}," + ",".join(f"{v}" for v in rep.values()) + chr(10))
                eval_spent += time.time() - t_h
            if args.hard_neg > 0 and it % args.mine_every == 0:          # H5: zor negatif havuzunu yenile
                t_m = time.time()
                n_pool = mine_hard_negatives(model, train_imgs, ch, device, args, out / "zor_negatif.pt")
                log(f"  zor negatif havuzu: {n_pool} konum ({time.time() - t_m:.0f} sn)")
                eval_spent += time.time() - t_m
            if args.eval_minutes > 0:
                do_eval = train_sec() - last_eval_sec >= args.eval_minutes * 60 or finished()
            else:
                do_eval = it % args.eval_every == 0 or it == args.iters
            if do_eval:
                t_e = time.time()
                last_eval_sec = train_sec()
                if ema is not None:                                    # §67: tamponlar (homeostaz vb.) modelden
                    with torch.no_grad():
                        for be, bm in zip(ema.buffers(), model.buffers()):
                            be.copy_(bm)
                    r = evaluate(ema, val_sub, ch, device, pad_mode=args.pad)
                    rh = evaluate(model, val_sub, ch, device, pad_mode=args.pad) if args.ema_ikisi else None
                    ema.eval()
                else:
                    r, rh = evaluate(model, val_sub, ch, device, pad_mode=args.pad), None
                log(f"  val({len(val_sub)}) mAP50 {r['mAP50']:.4f}{' (EMA)' if ema is not None else ''}  "
                    + "  ".join(f"{k}:{v['mAP50']:.3f}" for k, v in r["by_dataset"].items())
                    + (f"  | ham {rh['mAP50']:.4f}  " + "  ".join(f"{k}:{v['mAP50']:.3f}"
                                                               for k, v in rh["by_dataset"].items()) if rh else ""))
                logw.writerow([it, "", "", "", "", "", f"{r['mAP50']:.4f}", round(time.time() - t0),
                               f"{rh['mAP50']:.4f}" if rh else ""])
                logf.flush()
                improved = r["mAP50"] > best
                if improved:
                    best = r["mAP50"]
                    save(out / "best.pt", ema if ema is not None else model, opt, it, best, args, last_eval_sec,
                         ema_agirlik=ema is not None)
                if args.plato > 0 and it >= args.warmup:                  # §67: dinamik öğrenme oranı
                    bad = 0 if improved else bad + 1
                    if bad >= args.plato:
                        taban = getattr(args, "plato_taban", 0.0)                # §75: tabanda kalır, erken durma yok
                        lr_scale, bad = (max(lr_scale * 0.5, taban) if taban > 0 else lr_scale * 0.5), 0
                        log(f"  plato: {args.plato} değerlendirmedir iyileşme yok → lr ölçeği {lr_scale:.4f} "
                            f"(lr {args.lr * lr_scale:.2e})" + (" [taban]" if taban > 0 and lr_scale <= taban else ""))
                        if taban <= 0 and lr_scale < args.plato_min:
                            stop_flag[0] = True
                            log(f"  plato: lr ölçeği {args.plato_min} altına indi → eğitim bitiyor")
                save(out / "last.pt", model, opt, it, best, args, last_eval_sec, **st())
                eval_spent += time.time() - t_e
    # Süre dolduğunda döngü veri beklerken de çıkabilir: son durum her zaman kaydedilir; hiç değerlendirme
    # yapılmadıysa (çok kısa eğitim) bir tane yapılır ki best.pt olsun.
    if not (out / "best.pt").is_file():
        mb = ema if ema is not None else model
        r = evaluate(mb, val_sub, ch, device, pad_mode=args.pad)
        best = r["mAP50"]
        save(out / "best.pt", mb, opt, it, best, args, train_sec(), ema_agirlik=ema is not None)
    save(out / "last.pt", model, opt, it, best, args, train_sec(), **st())
    log(f"eğitim bitti: iterasyon {it}, eğitim süresi {train_sec() / 3600:.2f} sa")
    logf.close()

    splits = [s for s in args.final_splits.split(",") if s]
    log(f"son değerlendirme (en iyi model, tam {' + '.join(splits)})")
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    res = {"name": name, "best_iter": ck["iter"], "parametreler": model.n_parameters(),
           **{s: evaluate(model, val_imgs if s == "val" else list_images(args.modality, s), ch, device,
                          pad_mode=args.pad) for s in splits}}
    (out / "sonuclar.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"✓ {name}: " + ", ".join(f"{s} mAP50 {res[s]['mAP50']:.4f}" for s in splits) + f"  → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
