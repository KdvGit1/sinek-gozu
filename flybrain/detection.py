"""
Kolon tabanlı tespit (CenterNet benzeri): hedef üretimi, kayıp ve kutu çözümleme.

Kutu biçimi: (y, x, h, w) — sol-üst köşe + yükseklik/genişlik, piksel.
Her insan, merkezinin düştüğü kolona "pozitif" olarak atanır. Isı hedefi, kolon merkezinin kutu
merkezine uzaklığına göre Gauss olarak yayılır. O kolonda ayrıca ofset (kolon merkezine göre, p biriminde)
ve log boyut (p biriminde) öğrenilir.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from .eye import HexEye

try:  # C++ NMS (çok daha hızlı); yoksa aşağıdaki Python sürümü kullanılır
    from torchvision.ops import nms as _tv_nms
except ImportError:  # pragma: no cover
    _tv_nms = None


def column_map(eye: HexEye) -> np.ndarray:
    """[H, W] her pikselin kolon indeksi (göz dışı −1)."""
    m = np.full((eye.height, eye.width), -1, np.int64)
    for i, (t, l) in enumerate(zip(eye.top, eye.left)):
        m[t:t + eye.patch, l:l + eye.patch] = i
    return m


class TargetBuilder:
    def __init__(self, eye: HexEye, sigma_min: float | None = None, sigma_scale: float = 1 / 3,
                 reg_mode: str = "center", quality: bool = False, q_min: float = 0.3):
        """reg_mode: "center" — boyut/ofset yalnızca insan merkezinin düştüğü kolonda eğitilir (ilk tasarım);
                     "box"    — kutunun içine düşen bütün kolonlarda eğitilir (her kolon kendi merkezine göre ofset).
        30.09.2026 teşhisi: tespitlerin ~%50'si merkez kolondan değil komşudan çıkıyordu; "center" modunda o komşunun
        boyut/ofset tahmini hiç eğitilmemiş oluyordu. "box" modunda __call__ 4. çıktı olarak regresyon maskesi döner."""
        if reg_mode not in ("center", "box"):
            raise ValueError("reg_mode: center | box")
        self.eye = eye
        self.cmap = column_map(eye)
        self.centers = torch.as_tensor(eye.centers(), dtype=torch.float32)    # [C, 2] (y, x)
        self.sigma_min = sigma_min if sigma_min is not None else eye.patch / 2
        self.sigma_scale = sigma_scale
        self.reg_mode = reg_mode
        self.quality, self.q_min = quality, q_min

    def __call__(self, boxes_per_tile: list[torch.Tensor]):
        """boxes_per_tile: her karo için [M, 4] (y, x, h, w), karo koordinatında.
        Döner: heat [B, C], reg [B, C, 4] (dy, dx, log h, log w), pos [B, C] bool
               (+ reg_mode="box" ise regm [B, C] bool: boyut/ofset eğitilecek kolonlar)
               (+ quality=True ise her zaman 6'lı: heat, reg, pos, regm, qreg [B, C, 4], qm [B, C] bool —
                  kalite dalı için: ısısı ≥ q_min olan her kolonun en yakın kutusu, o kolona göre (dy, dx, log h, log w).
                  Tepe bir komşu kolonda çıkarsa o kolonun kutusu ne kadar iyi, bunu öğrenmek için.)"""
        B, C, p = len(boxes_per_tile), self.eye.n_columns, self.eye.patch
        heat = torch.zeros(B, C)
        reg = torch.zeros(B, C, 4)
        pos = torch.zeros(B, C, dtype=torch.bool)
        regm = torch.zeros(B, C, dtype=torch.bool)
        qreg = torch.zeros(B, C, 4) if self.quality else None
        qbest = torch.zeros(B, C) if self.quality else None
        cy_c, cx_c = self.centers[:, 0], self.centers[:, 1]
        for b, boxes in enumerate(boxes_per_tile):
            if self.reg_mode == "box" and len(boxes):
                # Büyükten küçüğe: çakışan kolonda küçük kutu kazanır (FCOS kuralı)
                for y, x, h, w in boxes[(boxes[:, 2] * boxes[:, 3]).argsort(descending=True)].tolist():
                    cy, cx = y + h / 2, x + w / 2
                    inside = (cy_c >= y) & (cy_c < y + h) & (cx_c >= x) & (cx_c < x + w)
                    iy, ix = int(cy), int(cx)
                    if 0 <= iy < self.eye.height and 0 <= ix < self.eye.width and self.cmap[iy, ix] >= 0:
                        inside[int(self.cmap[iy, ix])] = True             # merkez kolon her zaman dahil
                    if not inside.any():
                        continue
                    reg[b, inside] = torch.stack([(cy - cy_c[inside]) / p, (cx - cx_c[inside]) / p,
                                                  torch.full_like(cy_c[inside], np.log(max(h, 1) / p)),
                                                  torch.full_like(cy_c[inside], np.log(max(w, 1) / p))], 1)
                    regm[b, inside] = True
            for y, x, h, w in boxes.tolist():
                cy, cx = y + h / 2, x + w / 2
                iy, ix = int(cy), int(cx)
                if not (0 <= iy < self.eye.height and 0 <= ix < self.eye.width):
                    continue
                col = int(self.cmap[iy, ix])
                if col < 0:
                    continue                                   # merkez gözün dışında: başka karoda görülür
                d2 = ((self.centers - torch.tensor([cy, cx])) ** 2).sum(1)
                sigma = max(self.sigma_min, self.sigma_scale * float(np.sqrt(h * w)))
                g = torch.exp(-d2 / (2 * sigma ** 2))
                heat[b] = torch.maximum(heat[b], g)
                if self.quality:
                    g = g.clone()
                    g[col] = 1.0
                    m = (g >= self.q_min) & (g > qbest[b])
                    qbest[b, m] = g[m]
                    qreg[b, m] = torch.stack([(cy - cy_c[m]) / p, (cx - cx_c[m]) / p,
                                              torch.full_like(cy_c[m], np.log(max(h, 1) / p)),
                                              torch.full_like(cy_c[m], np.log(max(w, 1) / p))], 1)
                heat[b, col] = 1.0
                pos[b, col] = True
                if self.reg_mode == "center":
                    reg[b, col] = torch.tensor([(cy - self.centers[col, 0]) / p, (cx - self.centers[col, 1]) / p,
                                                np.log(max(h, 1) / p), np.log(max(w, 1) / p)])
        if self.quality:
            return heat, reg, pos, (regm if self.reg_mode == "box" else pos.clone()), qreg, qbest > 0
        if self.reg_mode == "box":
            return heat, reg, pos, regm
        return heat, reg, pos


def focal_loss(logits: torch.Tensor, heat: torch.Tensor, pos: torch.Tensor, alpha: float = 2, beta: float = 4):
    """CenterNet odak kaybı (pozitif sayısına bölünmüş)."""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    pos_l = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_l = -((1 - heat) ** beta) * (p ** alpha) * torch.log(1 - p) * (~pos)
    return (pos_l.sum() + neg_l.sum()) / pos.sum().clamp(min=1)


def local_iou(off: torch.Tensor, logsize: torch.Tensor, reg: torch.Tensor, giou: bool = False) -> torch.Tensor:
    """Kolon-yerel koordinatta (p biriminde; IoU ölçekten bağımsız) tahmin kutusu (ofset, log boyut) ile hedef
    kutu (reg: dy, dx, log h, log w) arasında eleman bazında IoU ya da GIoU."""
    ph, pw = torch.exp(logsize[..., 0]), torch.exp(logsize[..., 1])
    th, tw = torch.exp(reg[..., 2]), torch.exp(reg[..., 3])
    py1, px1, py2, px2 = off[..., 0] - ph / 2, off[..., 1] - pw / 2, off[..., 0] + ph / 2, off[..., 1] + pw / 2
    ty1, tx1, ty2, tx2 = reg[..., 0] - th / 2, reg[..., 1] - tw / 2, reg[..., 0] + th / 2, reg[..., 1] + tw / 2
    inter = (torch.minimum(py2, ty2) - torch.maximum(py1, ty1)).clamp(min=0) * \
            (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp(min=0)
    union = ph * pw + th * tw - inter
    iou = inter / union.clamp(min=1e-9)
    if not giou:
        return iou
    hull = (torch.maximum(py2, ty2) - torch.minimum(py1, ty1)) * (torch.maximum(px2, tx2) - torch.minimum(px1, tx1))
    return iou - (hull - union) / hull.clamp(min=1e-9)


def detection_loss(out: dict, heat: torch.Tensor, reg: torch.Tensor, pos: torch.Tensor,
                   w_offset: float = 1.0, w_size: float = 0.5, regm: torch.Tensor | None = None,
                   w_giou: float = 0.0, qreg: torch.Tensor | None = None, qm: torch.Tensor | None = None,
                   w_quality: float = 1.0) -> dict:
    """regm: boyut/ofset kaybının hesaplanacağı kolonlar (TargetBuilder reg_mode="box"); verilmezse pos.
    w_giou > 0 (H3): aynı kolonlarda ek olarak 1 − GIoU (örtüşme doğrudan optimize edilir).
    qreg/qm (H4): kalite dalı; hedef = o kolonun tahmin kutusunun (gradyansız) gerçek kutuyla IoU'su."""
    dev = out["heat"].device
    heat, reg, pos = heat.to(dev), reg.to(dev), pos.to(dev)
    rm = pos if regm is None else regm.to(dev)
    l_heat = focal_loss(out["heat"], heat, pos)
    n = rm.sum().clamp(min=1)
    l_off = (nn.functional.l1_loss(out["offset"], reg[..., :2], reduction="none").sum(-1) * rm).sum() / n
    l_size = (nn.functional.l1_loss(out["logsize"], reg[..., 2:], reduction="none").sum(-1) * rm).sum() / n
    total = l_heat + w_offset * l_off + w_size * l_size
    if w_giou > 0:
        g = local_iou(out["offset"], out["logsize"].clamp(-6, 6), reg, giou=True)
        total = total + w_giou * ((1 - g) * rm).sum() / n
    if qreg is not None and "quality" in out:
        qreg, qm = qreg.to(dev), qm.to(dev)
        with torch.no_grad():
            tq = local_iou(out["offset"], out["logsize"].clamp(-6, 6), qreg)
        lq = nn.functional.binary_cross_entropy_with_logits(out["quality"], tq, reduction="none")
        total = total + w_quality * (lq * qm).sum() / qm.sum().clamp(min=1)
    return {"total": total, "heat": l_heat.detach(), "offset": l_off.detach(), "size": l_size.detach()}


_GEOM_CACHE: dict = {}
# H24 (günlük §49): kutu merkezini tepe kolon + 6 komşusunun (merkez + ofset) tahminlerinin ısıyla ağırlıklı ortalamasından
# oku. Yalnız son işlem; model aynı. Ortam değişkeni FLYBRAIN_DECODE=merkez7 ile açılır (bütün araçlar aynı decode'u
# kullandığı için tek yerden). Varsayılan: kapalı (eski sonuçlar aynen tekrar üretilebilsin).
DECODE_MODE = os.environ.get("FLYBRAIN_DECODE", "taban")


def _eye_geometry(eye: HexEye, device, dtype):
    """Komşu tablosu ve kolon merkezleri (her decode çağrısında Python'da yeniden kurulmasın diye önbellekli)."""
    key = (id(eye), eye.patch, str(device), dtype)
    if key not in _GEOM_CACHE:
        _GEOM_CACHE[key] = (torch.as_tensor(eye.neighbors(), device=device),
                            torch.as_tensor(eye.centers(), dtype=dtype, device=device))
    return _GEOM_CACHE[key]


def fuse_two_eyes(a: torch.Tensor, b: torch.Tensor, iou_thr: float = 0.5, single: float = 0.5,
                  zone: tuple[float, float] | None = None) -> torch.Tensor:
    """§53/§56 iki göz uzlaşması: IoU ≥ iou_thr eşleşen çiftlerin kutu ve skor ortalaması; yalnız bir gözün gördüğü
    tespitin skoru ×single. zone = (x0, x1): yalnız merkezi bu aralıkta (iki gözün de görebildiği şerit) olan tek göz
    tespitleri cezalanır (görüş alanı B); None → her yer ortak (görüş alanı A)."""
    def pen(d):
        if not len(d):
            return d
        d = d.clone()
        if zone is None:
            d[:, 4] *= single
        else:
            cx = d[:, 1] + d[:, 3] / 2
            d[:, 4] *= torch.where((cx >= zone[0]) & (cx <= zone[1]), torch.full_like(cx, single), torch.ones_like(cx))
        return d
    if len(a) == 0 or len(b) == 0:
        return pen(a if len(b) == 0 else b)
    iou = box_iou(a[:, :4], b[:, :4]).cpu().numpy()                     # §67: tek aktarım, eşleme CPU'da
    pairs, lone_a, used_b = _greedy(iou, a[:, 4].argsort(descending=True).tolist(), iou_thr)
    if pairs:
        ii = torch.as_tensor([q[0] for q in pairs], device=a.device)
        jj = torch.as_tensor([q[1] for q in pairs], device=b.device)
        parts = [(a[ii] + b[jj]) / 2]
    else:
        parts = []
    parts += [pen(a[lone_a]), pen(b[torch.as_tensor(~used_b, device=b.device)])]
    return torch.cat(parts)


class BilateralTargets:
    """H35 (§56): iki gözlü model için hedefler. Sağ göz karonun kendisini, sol göz aynasını görür: sol gözün hedefleri
    aynalanmış kutulardan; sonuçlar kolon ekseninde birleştirilir ([B, 2C, ...]). visible [C] (görüş alanı B): gözün
    görmediği kolonlarda hedef yok (ısı 0, pozitif/regresyon yok)."""

    def __init__(self, tb: "TargetBuilder", width: int, visible: torch.Tensor | None = None):
        self.tb, self.width = tb, width
        self.visible = None if visible is None else visible.bool().cpu()

    def __call__(self, boxes_per_tile: list[torch.Tensor]):
        flipped = []
        for bx in boxes_per_tile:
            f = bx.clone()
            if len(f):
                f[:, 1] = self.width - f[:, 1] - f[:, 3]
            flipped.append(f)
        r, l = self.tb(boxes_per_tile), self.tb(flipped)
        out = []
        for a, b in zip(r, l):
            t = torch.cat([a, b], dim=1)
            if self.visible is not None:
                m = torch.cat([self.visible, self.visible])
                shape = (1, -1) + (1,) * (t.dim() - 2)
                t = t * m.view(shape).to(t.dtype) if t.dtype != torch.bool else t & m.view(shape)
            out.append(t)
        return tuple(out)


@torch.no_grad()
def decode(out: dict, eye: HexEye, score_thr: float = 0.3, max_det: int = 100) -> list[torch.Tensor]:
    """Karo başına tespitler: [K, 5] (y, x, h, w, skor), karo koordinatında, skora göre azalan.
    Tepe noktası = ısısı 6 komşusunun hepsinden büyük/eşit olan kolon.
    İki gözlü modelde (out["eyes"]) iki göz ayrı çözülür, sol gözünkiler aynadan geri çevrilir ve uzlaşmayla birleşir."""
    if out.get("eyes") is not None:
        _, view, ov = out["eyes"]
        C = eye.n_columns
        half = lambda sl: {k: out[k][:, sl] for k in ("heat", "offset", "logsize", "quality") if k in out}
        dr = decode(half(slice(0, C)), eye, score_thr, max_det)
        dl = decode(half(slice(C, 2 * C)), eye, score_thr, max_det)
        W = eye.width
        x0, x1 = W * (0.5 - ov / 2), W * (0.5 + ov / 2)
        res = []
        for a, b in zip(dr, dl):
            if len(b):
                b = b.clone()
                b[:, 1] = W - b[:, 1] - b[:, 3]
            if view == "B":                                              # her göz yalnız gördüğü yarıdan
                ca, cb = a[:, 1] + a[:, 3] / 2, b[:, 1] + b[:, 3] / 2
                a, b = a[ca >= x0], b[cb <= x1]
            f = fuse_two_eyes(a, b, zone=(x0, x1) if view == "B" else None)
            res.append(f[f[:, 4].argsort(descending=True)] if len(f) else f)
        return res
    boxes, valid = _decode_boxes(out, eye, score_thr, max_det)
    return [boxes[b][valid[b]] for b in range(boxes.shape[0])]


def _decode_boxes(out: dict, eye: HexEye, score_thr: float, max_det: int):
    """decode'un tek göz yolunun vektörel kısmı → boxes [B, k, 5], valid [B, k] (aynı cihazda)."""
    prob = torch.sigmoid(out["heat"])                                   # [B, C]
    nb, centers = _eye_geometry(eye, prob.device, prob.dtype)          # [C, 6], [C, 2]
    nbp = torch.where(nb >= 0, prob[:, nb.clamp(min=0)], torch.zeros(()).to(prob))   # [B, C, 6]
    peak = (prob >= nbp.max(-1).values) & (prob >= score_thr)
    k = min(max_det, prob.shape[1])
    score, idx = torch.where(peak, prob, torch.full_like(prob, -1.0)).topk(k, dim=1)  # [B, k]
    p = eye.patch
    off = out["offset"].gather(1, idx[..., None].expand(-1, -1, 2))
    size = torch.exp(out["logsize"].gather(1, idx[..., None].expand(-1, -1, 2))) * p
    cy = centers[idx, 0] + off[..., 0] * p
    cx = centers[idx, 1] + off[..., 1] * p
    if DECODE_MODE.startswith("merkez7"):                              # "merkez7" ya da "merkez7p2" (ağırlık ısı²; §52)
        nidx = nb[idx].clamp(min=0)                                     # [B, k, 6]
        pn = torch.gather(prob, 1, nidx.reshape(prob.shape[0], -1)).reshape(nidx.shape) * (nb[idx] >= 0)
        on = torch.gather(out["offset"], 1, nidx.reshape(prob.shape[0], -1, 1).expand(-1, -1, 2)).reshape(*nidx.shape, 2)
        s0 = score.clamp(min=0)
        if DECODE_MODE == "merkez7p2":
            s0, pn = s0 ** 2, pn ** 2
        wsum = (s0 + pn.sum(-1)).clamp(min=1e-9)
        cy = (cy * s0 + ((centers[nidx, 0] + on[..., 0] * p) * pn).sum(-1)) / wsum
        cx = (cx * s0 + ((centers[nidx, 1] + on[..., 1] * p) * pn).sum(-1)) / wsum
    h, w = size[..., 0], size[..., 1]
    valid = score >= 0
    if "quality" in out:                                                # H4: skor = ısı × kalite
        score = score * torch.sigmoid(out["quality"].gather(1, idx))
    boxes = torch.stack([cy - h / 2, cx - w / 2, h, w, score], dim=-1)  # [B, k, 5]
    return boxes, valid


@torch.no_grad()
def decode_cpu(out: dict, eye: HexEye, score_thr: float = 0.3, max_det: int = 100) -> list[torch.Tensor]:
    """§75: decode ile birebir aynı sonuç (CPU tensörleri). Vektörel kısım aynı cihazda aynı işlemlerle; kutular tek
    aktarımla CPU'ya; iki göz uzlaşması CPU'da (eskisi karo başına GPU ↔ CPU eşitlemesi yapıyordu; val800'de ~78 sn)."""
    if out.get("eyes") is None:
        bx, va = _decode_boxes(out, eye, score_thr, max_det)
        bx, va = bx.cpu(), va.cpu()
        return [bx[b][va[b]] for b in range(bx.shape[0])]
    _, view, ov = out["eyes"]
    C = eye.n_columns
    half = lambda sl: {k: out[k][:, sl] for k in ("heat", "offset", "logsize", "quality") if k in out}
    br, vr = _decode_boxes(half(slice(0, C)), eye, score_thr, max_det)
    bl, vl = _decode_boxes(half(slice(C, 2 * C)), eye, score_thr, max_det)
    br, vr, bl, vl = br.cpu(), vr.cpu(), bl.cpu(), vl.cpu()
    W = eye.width
    x0, x1 = W * (0.5 - ov / 2), W * (0.5 + ov / 2)
    res = []
    for t in range(br.shape[0]):
        a, b = br[t][vr[t]], bl[t][vl[t]]
        if len(b):
            b = b.clone()
            b[:, 1] = W - b[:, 1] - b[:, 3]
        if view == "B":
            ca, cb = a[:, 1] + a[:, 3] / 2, b[:, 1] + b[:, 3] / 2
            a, b = a[ca >= x0], b[cb <= x1]
        f = fuse_two_eyes(a, b, zone=(x0, x1) if view == "B" else None)
        res.append(f[f[:, 4].argsort(descending=True)] if len(f) else f)
    return res


def _greedy(iou: np.ndarray, order, thr: float):
    """§67 açgözlü eşleme (eski torch döngüsüyle aynı kural): sıradaki her i için kullanılmamışlar içinde IoU'su en
    büyük j (eşitlikte ilki; kullanılmış = −1); IoU ≥ thr ve j kullanılmamışsa eşleşir. → (eşleşen [(i, j)],
    eşleşmeyen i'ler, kullanılan maskesi)."""
    used = np.zeros(iou.shape[1], dtype=bool)
    pairs, lone = [], []
    for i in order:
        row = np.where(used, np.float32(-1), iou[i])
        j = int(row.argmax())
        if iou[i, j] >= thr and not used[j]:
            used[j] = True
            pairs.append((i, j))
        else:
            lone.append(i)
    return pairs, lone, used


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(y, x, h, w) kutuları arası IoU matrisi."""
    ay1, ax1, ay2, ax2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    by1, bx1, by2, bx2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ih = (torch.minimum(ay2[:, None], by2[None]) - torch.maximum(ay1[:, None], by1[None])).clamp(min=0)
    iw = (torch.minimum(ax2[:, None], bx2[None]) - torch.maximum(ax1[:, None], bx1[None])).clamp(min=0)
    inter = ih * iw
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter).clamp(min=1e-9)


def nms(dets: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
    """dets [K, 5] (y, x, h, w, skor) → bastırılmamış tespitler."""
    if len(dets) == 0:
        return dets
    order = dets[:, 4].argsort(descending=True)
    dets = dets[order]
    if _tv_nms is not None:
        xyxy = torch.stack([dets[:, 1], dets[:, 0], dets[:, 1] + dets[:, 3], dets[:, 0] + dets[:, 2]], 1)
        return dets[_tv_nms(xyxy, dets[:, 4], iou_thr)]
    keep = torch.ones(len(dets), dtype=torch.bool)
    iou = box_iou(dets[:, :4], dets[:, :4])
    for i in range(len(dets)):
        if keep[i]:
            keep[i + 1:] &= iou[i, i + 1:] <= iou_thr
    return dets[keep]


def merge_tiles(dets_per_tile: list[torch.Tensor], origins: list[tuple[int, int]], iou_thr: float = 0.5):
    """Karo tespitlerini görüntü koordinatına taşır ve karolar arası çiftleri NMS ile birleştirir."""
    moved = []
    for d, (oy, ox) in zip(dets_per_tile, origins):
        if len(d):
            d = d.clone()
            d[:, 0] += oy
            d[:, 1] += ox
            moved.append(d)
    return nms(torch.cat(moved), iou_thr) if moved else torch.zeros(0, 5)


def merge_top(dets_per_tile: list[torch.Tensor], origins: list[tuple[int, int]], H: int, W: int,
              iou_thr: float = 0.5, top: int = 300, outside: bool = True) -> torch.Tensor:
    """§75: merge_tiles → (outside ise) görüntü dışını at → skora göre ilk `top` ile aynı sonuç, daha hızlı. Açgözlü
    NMS'de bir kutunun kalması yalnız kendinden yüksek skorlu kutulara bağlı: NMS skora göre sıralı bir önekte koşar;
    önekte (görüntü içinde) en az `top` kalan varsa ilk `top`'u tam kümeyle aynıdır, yoksa önek iki katına çıkar.
    (Görüntü başına ~5.000 kutu; val800'de 50 sn → ~6 sn.)"""
    moved = []
    for d, (oy, ox) in zip(dets_per_tile, origins):
        if len(d):
            d = d.clone()
            d[:, 0] += oy
            d[:, 1] += ox
            moved.append(d)
    if not moved:
        return torch.zeros(0, 5)
    allb = torch.cat(moved)
    allb = allb[allb[:, 4].argsort(descending=True)]
    n = len(allb)
    M = min(n, max(4 * top, 1024))
    while True:
        kept = nms(allb[:M], iou_thr)
        if outside and len(kept):
            cy, cx = kept[:, 0] + kept[:, 2] / 2, kept[:, 1] + kept[:, 3] / 2
            kept = kept[(cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)]
        if len(kept) >= top or M == n:
            return kept[kept[:, 4].argsort(descending=True)][:top] if len(kept) else kept
        M = min(n, 2 * M)


def match_flags(dets: torch.Tensor, gts: torch.Tensor, iou_thr: float = 0.5) -> list[tuple[float, bool]]:
    """AP hesabı için: her tespitin (skor, doğru_pozitif_mi) çifti. Yüksek skordan başlayarak açgözlü eşleme."""
    if len(dets) == 0:
        return []
    order = dets[:, 4].argsort(descending=True).tolist()
    sc = dets[:, 4].tolist()
    if len(gts) == 0:
        return [(sc[i], False) for i in order]
    pairs, _, _ = _greedy(box_iou(dets[:, :4], gts).cpu().numpy(), order, iou_thr)
    hit = {i for i, _ in pairs}
    return [(sc[i], i in hit) for i in order]


def average_precision(flags: list[tuple[float, bool]], n_gt: int) -> float:
    """101 noktalı enterpolasyonlu AP (Ultralytics / COCO ile aynı yöntem)."""
    if n_gt == 0 or not flags:
        return 0.0
    flags = sorted(flags, key=lambda f: -f[0])
    tp = np.cumsum([f[1] for f in flags])
    fp = np.cumsum([not f[1] for f in flags])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def match(dets: torch.Tensor, gts: torch.Tensor, iou_thr: float = 0.3) -> tuple[int, int, int]:
    """Açgözlü eşleme: (doğru pozitif, yanlış pozitif, kaçırılan)."""
    if len(dets) == 0:
        return 0, 0, len(gts)
    if len(gts) == 0:
        return 0, len(dets), 0
    pairs, _, _ = _greedy(box_iou(dets[:, :4], gts).cpu().numpy(), dets[:, 4].argsort(descending=True).tolist(),
                          iou_thr)
    tp = len(pairs)
    return tp, len(dets) - tp, len(gts) - tp
