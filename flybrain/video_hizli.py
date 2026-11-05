"""
§89b video yükleyici hızlandırması (RGB pilotunda GPU veri bekliyordu: VisDrone klibi ~350 ms). Sonuç değiştirmeyen iki parça:
  kesit_yukle  — görüntüyü çözüp YALNIZ kesilecek karoyu ondalık sayıya çevirir (eskiden bütün kare çevriliyordu; kare
                 başına ~7 ms). tf.crop(tf.load_image(p), oy, ox, eh, ew, "replicate") ile birebir aynı (8 bit görüntüler).
  HizliHedef   — BilateralTargets(TargetBuilder(reg_mode="box")) ile aynı hedefler, kutular üzerinde döngü yerine matris
                 işlemleriyle (VisDrone'da klipte ~100 kutu: ~50 ms → birkaç ms). Eşit alanlı iki kutu aynı kolonu
                 kapladığında hangisinin kazanacağı (sıralama bağı) farklı olabilir; başka fark yok (test: tests/test_video89b.py).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def boyut(path) -> tuple[int, int]:
    """(H, W) — görüntüyü çözmeden."""
    with Image.open(path) as im:
        return im.size[1], im.size[0]


def kesit_yukle(path, channels: int, oy: int, ox: int, eh: int, ew: int) -> torch.Tensor:
    """[C, eh, ew] 0–1; görüntü dışı kenar pikseli uzatılarak (replicate) doldurulur. 16 bit görüntüde None (eski yol).
    path: dosya yolu ya da JPEG baytları (§89d RAM deposu; aynı çözücü, aynı pikseller)."""
    if isinstance(path, (bytes, bytearray)):
        import io
        path = io.BytesIO(path)
    with Image.open(path) as im:
        if channels == 1:
            if im.mode.startswith("I;16") or im.mode in ("I", "F"):
                return None
            a = np.asarray(im.convert("L"))[..., None]
        else:
            a = np.asarray(im.convert("RGB"))
    H, W = a.shape[:2]
    ys = np.clip(np.arange(oy, oy + eh), 0, H - 1)
    xs = np.clip(np.arange(ox, ox + ew), 0, W - 1)
    k = a[ys][:, xs]                                                    # [eh, ew, C] uint8
    return torch.from_numpy(np.ascontiguousarray(k.transpose(2, 0, 1))).float() / 255.0


class HizliHedef:
    """BilateralTargets(TargetBuilder(eye, reg_mode="box"), width, visible) ile aynı çıktı: (ısı, reg, pozitif, içeride)."""

    def __init__(self, tb, width: int, visible: torch.Tensor | None = None):
        self.tb, self.width = tb, width
        self.visible = None if visible is None else visible.bool().cpu()
        self.cy_c, self.cx_c = tb.centers[:, 0], tb.centers[:, 1]
        self.cmap = torch.as_tensor(tb.cmap, dtype=torch.long)
        self.p = tb.eye.patch
        self.H, self.W = tb.eye.height, tb.eye.width

    def _goz(self, boxes_per_tile):
        B, C, p = len(boxes_per_tile), len(self.cy_c), self.p
        heat = torch.zeros(B, C)
        reg = torch.zeros(B, C, 4)
        pos = torch.zeros(B, C, dtype=torch.bool)
        regm = torch.zeros(B, C, dtype=torch.bool)
        for b, boxes in enumerate(boxes_per_tile):
            if not len(boxes):
                continue
            # eski kod kutuları .tolist() ile Python float'a (64 bit) çevirip merkezi orada hesaplıyor; piksel sınırına düşen
            # merkezlerin aynı kolona yuvarlanması için aynı hassasiyet: hesap 64 bit, tensörlerle karşılaştırma 32 bit
            y, x, h, w = boxes.double().unbind(1)
            cy64, cx64 = y + h / 2, x + w / 2
            cy, cx = cy64.float(), cx64.float()
            iy, ix = torch.trunc(cy64).long(), torch.trunc(cx64).long()    # Python int() gibi sıfıra doğru (−0,5 → 0)
            ok = (iy >= 0) & (iy < self.H) & (ix >= 0) & (ix < self.W)
            col = torch.full_like(iy, -1)
            col[ok] = self.cmap[iy[ok], ix[ok]]
            # regresyon: kutunun içindeki kolonlar (+ merkez kolon); çakışmada küçük alanlı kutu kazanır
            ins = (self.cy_c[None] >= y.float()[:, None]) & (self.cy_c[None] < (y + h).float()[:, None]) & \
                  (self.cx_c[None] >= x.float()[:, None]) & (self.cx_c[None] < (x + w).float()[:, None])  # [M, C]
            m = col >= 0
            ins[torch.nonzero(m).squeeze(1), col[m]] = True
            if ins.any():
                # eski döngü: kutular alana göre azalan sırada (aynı argsort) gezilir, sonraki üstüne yazar → kolonun sahibi,
                # o sırada en son gelen kaplayan kutu (eşit alan bağları da eskisiyle aynı çözülür)
                sira = (boxes[:, 2] * boxes[:, 3]).argsort(descending=True)
                rank = torch.empty_like(sira)
                rank[sira] = torch.arange(len(sira))
                sahip = torch.where(ins, rank[:, None], torch.tensor(-1)).argmax(0)            # [C]
                kap = ins.any(0)
                s = sahip[kap]
                reg[b, kap] = torch.stack([(cy[s] - self.cy_c[kap]) / p, (cx[s] - self.cx_c[kap]) / p,
                                           torch.log(torch.clamp(h[s], min=1) / p).float(),
                                           torch.log(torch.clamp(w[s], min=1) / p).float()], 1)
                regm[b, kap] = True
            # ısı: merkez kolonu gözdeki kutular
            g_ok = col >= 0
            if g_ok.any():
                sig = torch.clamp(self.tb.sigma_scale * torch.sqrt(h[g_ok] * w[g_ok]), min=self.tb.sigma_min)
                d2 = (self.cy_c[None] - cy[g_ok][:, None]) ** 2 + (self.cx_c[None] - cx[g_ok][:, None]) ** 2
                heat[b] = torch.exp(-d2 / (2 * sig[:, None] ** 2).float()).max(0).values      # eskisi gibi 32 bit bölme
                heat[b, col[g_ok]] = 1.0
                pos[b, col[g_ok]] = True
        return heat, reg, pos, regm

    def __call__(self, boxes_per_tile):
        flipped = []
        for bx in boxes_per_tile:
            f = bx.clone()
            if len(f):
                f[:, 1] = self.width - f[:, 1] - f[:, 3]
            flipped.append(f)
        r, l = self._goz(boxes_per_tile), self._goz(flipped)
        out = []
        for a, b in zip(r, l):
            t = torch.cat([a, b], dim=1)
            if self.visible is not None:
                mv = torch.cat([self.visible, self.visible])
                shape = (1, -1) + (1,) * (t.dim() - 2)
                t = t * mv.view(shape).to(t.dtype) if t.dtype != torch.bool else t & mv.view(shape)
            out.append(t)
        return tuple(out)
