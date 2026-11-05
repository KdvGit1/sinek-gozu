"""
Altıgen bileşik göz: görüntüyü sağ optik lobun 796 kolonuna örnekler.

Geometri (TASARIM.md Bölüm 3.3):
  * Kolonların (p, q) koordinatları FlyWire'dan gelir. Komşuluk sinapslardan doğrulanmıştır:
    komşular (±1,0), (0,±1), ±(1,1).
  * Piksel düzleminde "tuğla" (offset) düzeni kullanılır: aynı (q−p) değerli kolonlar dikey bir
    sütun oluşturur, komşu sütunlar yarım yama kaydırılmıştır. Böylece p×p'lik kare yamalar
    çakışmadan ve boşluksuz dizilir; her kolonun yine 6 komşusu olur.
  * Görüntü asla küçültülmez. Büyük kare, gözün ayak izine sığan karolara bölünür (Tiler).

Temel kullanım:
    eye = HexEye.from_processed("data/connectome/processed/right_ol_thr5", patch=4)
    tiler = Tiler(eye, overlap=0.25)
    tiles, origins = tiler.extract(img)          # img: [C, H, W] tensör
    patches = eye.sample(tiles)                  # [N, 796, C, p, p]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))


def _max_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """İkili maskedeki en büyük tamamen dolu dikdörtgen: (y0, x0, h, w). Histogram yöntemi, O(H·W)."""
    H, W = mask.shape
    heights = np.zeros(W, dtype=int)
    best = (0, 0, 0, 0)
    best_area = 0
    for y in range(H):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[int] = []
        for x in range(W + 1):
            h = heights[x] if x < W else 0
            start = x
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                width = x - (stack[-1] + 1 if stack else 0)
                area = heights[top] * width
                if area > best_area:
                    best_area = area
                    x0 = stack[-1] + 1 if stack else 0
                    best = (y - heights[top] + 1, x0, int(heights[top]), width)
                start = top
            stack.append(x)
    return best


@dataclass
class HexEye:
    column_id: np.ndarray      # [n] FlyWire kolon id'leri (modeldeki kolon sırası)
    p: np.ndarray              # [n]
    q: np.ndarray              # [n]
    patch: int                 # yama kenarı (piksel)
    top: np.ndarray            # [n] yamanın sol-üst köşesi (göz koordinatında, piksel)
    left: np.ndarray           # [n]
    height: int                # gözün sınırlayıcı kutusu (piksel)
    width: int

    # ------------------------------------------------------------------ kurulum
    @classmethod
    def from_columns(cls, columns: pd.DataFrame, patch: int) -> "HexEye":
        if patch < 1:
            raise ValueError("patch >= 1 olmalı")
        c = columns.sort_values("column_id").reset_index(drop=True)
        bc = (c.q - c.p).to_numpy()            # tuğla sütunu
        br2 = (c.p + c.q).to_numpy()           # dikey konum × 2
        left = (bc - bc.min()) * patch
        top = ((br2 - br2.min()) * patch) // 2
        return cls(column_id=c.column_id.to_numpy(), p=c.p.to_numpy(), q=c.q.to_numpy(), patch=patch,
                   top=top.astype(np.int64), left=left.astype(np.int64),
                   height=int(top.max() + patch), width=int(left.max() + patch))

    @classmethod
    def from_processed(cls, processed_dir: str | Path, patch: int) -> "HexEye":
        return cls.from_columns(pd.read_csv(Path(processed_dir) / "columns.csv"), patch)

    @property
    def n_columns(self) -> int:
        return len(self.column_id)

    # ------------------------------------------------------------------ geometri
    def footprint(self) -> np.ndarray:
        """[height, width] bool: gözün gördüğü pikseller."""
        m = np.zeros((self.height, self.width), bool)
        for t, l in zip(self.top, self.left):
            m[t:t + self.patch, l:l + self.patch] = True
        return m

    def inscribed_rect(self) -> tuple[int, int, int, int]:
        """Ayak izinin içine tamamen sığan en büyük dikdörtgen (y0, x0, h, w). Karolama adımı buna göre."""
        return _max_rectangle(self.footprint())

    def centers(self) -> np.ndarray:
        """[n, 2] kolon merkezleri (y, x), göz koordinatında piksel."""
        half = self.patch / 2
        return np.stack([self.top + half, self.left + half], axis=1)

    def neighbors(self) -> np.ndarray:
        """[n, 6] komşu kolon indeksleri (yoksa -1). Sıra: NEIGHBOR_OFFSETS."""
        pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.p, self.q))}
        out = np.full((self.n_columns, 6), -1, dtype=np.int64)
        for i, (a, b) in enumerate(zip(self.p, self.q)):
            for k, (dp, dq) in enumerate(NEIGHBOR_OFFSETS):
                out[i, k] = pos.get((int(a) + dp, int(b) + dq), -1)
        return out

    def neighbors2(self) -> np.ndarray:
        """[n, 18] iki halka komşu (yoksa -1): önce 6 birinci halka (NEIGHBOR_OFFSETS sırası), sonra 12 ikinci halka
        (iki komşu ofsetinin toplamı olup 0 ya da birinci halka olmayan ofsetler; sıra her kolonda aynı)."""
        ring1 = set(NEIGHBOR_OFFSETS)
        ring2 = sorted({(a1 + a2, b1 + b2) for a1, b1 in NEIGHBOR_OFFSETS for a2, b2 in NEIGHBOR_OFFSETS}
                       - ring1 - {(0, 0)})
        assert len(ring2) == 12
        pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.p, self.q))}
        out = np.full((self.n_columns, 18), -1, dtype=np.int64)
        for i, (a, b) in enumerate(zip(self.p, self.q)):
            for k, (dp, dq) in enumerate(list(NEIGHBOR_OFFSETS) + ring2):
                out[i, k] = pos.get((int(a) + dp, int(b) + dq), -1)
        return out

    # ------------------------------------------------------------------ örnekleme
    def _pixel_index(self, device=None) -> torch.Tensor:
        """[n, p, p] her yamadaki piksellerin düzleştirilmiş (y·W + x) indeksi. Cihaz başına bir kez hesaplanıp
        saklanır (§46: her adımda NumPy'da kurulup GPU'ya kopyalanıyordu; değerler aynı)."""
        cache = self.__dict__.setdefault("_idx_cache", {})
        key = str(device)
        if key not in cache:
            r = np.arange(self.patch)
            ys = self.top[:, None, None] + r[None, :, None]
            xs = self.left[:, None, None] + r[None, None, :]
            cache[key] = torch.as_tensor(ys * self.width + xs, dtype=torch.long, device=device)
        return cache[key]

    def __getstate__(self):
        """Veri işçilerine gönderilirken önbellek dışarıda kalır (Windows'ta GPU tensörü süreçler arası taşınamaz)."""
        d = dict(self.__dict__)
        d.pop("_idx_cache", None)
        return d

    def sample(self, tiles: torch.Tensor) -> torch.Tensor:
        """tiles: [B, C, height, width] → [B, n, C, p, p] (her kolonun gördüğü yama)."""
        B, C, H, W = tiles.shape
        if (H, W) != (self.height, self.width):
            raise ValueError(f"karo boyutu {(H, W)} ≠ göz boyutu {(self.height, self.width)}")
        idx = self._pixel_index(tiles.device)                      # [n, p, p]
        flat = tiles.reshape(B, C, H * W)
        out = flat[:, :, idx.reshape(-1)]                           # [B, C, n·p·p]
        return out.reshape(B, C, self.n_columns, self.patch, self.patch).permute(0, 2, 1, 3, 4)

    def render(self, values: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        """Kolon başına değerleri görüntüye boyar (görselleştirme).
        values: [n, C] → [C, height, width] (her yama tek renk)
                [n, C, p, p] → [C, height, width] (yamalar yerine konur)"""
        if values.dim() == 2:
            values = values[:, :, None, None].expand(-1, -1, self.patch, self.patch)
        n, C = values.shape[:2]
        out = torch.full((C, self.height * self.width), fill, dtype=values.dtype, device=values.device)
        idx = self._pixel_index(values.device).reshape(-1)
        out[:, idx] = values.permute(1, 0, 2, 3).reshape(C, -1)
        return out.reshape(C, self.height, self.width)


def replicate_pad(img: torch.Tensor, top: int, bottom: int, left: int, right: int) -> torch.Tensor:
    """[C, H, W] görüntüyü kenar pikselini uzatarak doldurur (dolgu görüntü boyutundan büyük olsa da çalışır)."""
    C, H, W = img.shape
    ys = torch.arange(-top, H + bottom, device=img.device).clamp(0, H - 1)
    xs = torch.arange(-left, W + right, device=img.device).clamp(0, W - 1)
    return img[:, ys][:, :, xs]


class Tiler:
    """Büyük bir kareyi gözün ayak izine göre karolara böler (küçültme yok).

    Her karoda gözün iç dikdörtgeni (inscribed rect) görüntünün bir bölümünü eksiksiz kaplar;
    karolar bu dikdörtgenlere göre `overlap` oranında örtüşerek dizilir. Böylece karo sınırında
    bölünen insanlar da en az bir karoda bütün olarak görülür. Görüntü dışına taşan kısımlar
    `pad_value` ile doldurulur.
    """

    def __init__(self, eye: HexEye, overlap: float = 0.25, pad_value: float = 0.0, pad_mode: str = "constant"):
        """pad_mode: "constant" (pad_value ile doldur) | "replicate" (kenar pikselini uzat).
        Termalde 0 ile doldurmak görüntü kenarında yapay, çok sert bir "soğuk" sınır yaratır ve modelin
        kenarda sahte tespit yapmasına yol açar (29.09.2026 bulgusu); replicate bu sınırı ortadan kaldırır."""
        if not 0 <= overlap < 1:
            raise ValueError("overlap [0, 1) aralığında olmalı")
        if pad_mode not in ("constant", "replicate"):
            raise ValueError("pad_mode: constant | replicate")
        self.eye = eye
        self.overlap = overlap
        self.pad_value = pad_value
        self.pad_mode = pad_mode
        self.ry, self.rx, self.rh, self.rw = eye.inscribed_rect()

    @staticmethod
    def _starts(size: int, win: int, stride: int) -> list[int]:
        if size <= win:
            return [0]
        s = list(range(0, size - win + 1, stride))
        if s[-1] != size - win:
            s.append(size - win)
        return s

    def origins(self, H: int, W: int) -> list[tuple[int, int]]:
        """Gözün sınırlayıcı kutusunun görüntüdeki sol-üst köşeleri (negatif olabilir)."""
        sy = max(1, int(self.rh * (1 - self.overlap)))
        sx = max(1, int(self.rw * (1 - self.overlap)))
        return [(y - self.ry, x - self.rx)
                for y in self._starts(H, self.rh, sy) for x in self._starts(W, self.rw, sx)]

    def extract(self, img: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """img: [C, H, W] → karolar [N, C, eye.height, eye.width] ve karo kökenleri."""
        C, H, W = img.shape
        org = self.origins(H, W)
        eh, ew = self.eye.height, self.eye.width
        pad_t = max(0, -min(o[0] for o in org))
        pad_l = max(0, -min(o[1] for o in org))
        pad_b = max(0, max(o[0] for o in org) + eh - H)
        pad_r = max(0, max(o[1] for o in org) + ew - W)
        if self.pad_mode == "replicate":
            padded = replicate_pad(img, pad_t, pad_b, pad_l, pad_r)
        else:
            padded = torch.nn.functional.pad(img, (pad_l, pad_r, pad_t, pad_b), value=self.pad_value)
        tiles = torch.stack([padded[:, y + pad_t:y + pad_t + eh, x + pad_l:x + pad_l + ew] for y, x in org])
        return tiles, org

    def coverage(self, H: int, W: int) -> np.ndarray:
        """[H, W] her pikselin kaç karonun göz ayak izinde görüldüğü (test/görselleştirme için)."""
        cov = np.zeros((H, W), np.int32)
        fp = self.eye.footprint()
        for y, x in self.origins(H, W):
            y0, x0 = max(0, y), max(0, x)
            y1, x1 = min(H, y + fp.shape[0]), min(W, x + fp.shape[1])
            cov[y0:y1, x0:x1] += fp[y0 - y:y1 - y, x0 - x:x1 - x]
        return cov
