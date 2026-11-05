"""
§89d video eğitim karelerini RAM'de tutmak (kullanıcı onayı 18.10). Ölçüm: Windows dosya önbelleği ısıtılan kareleri tutmuyordu
(beklemede yalnız 1,85 GB; 8 GB boş RAM varken), rastgele kare okuma RGB 11,5 ms, termal 29 ms → eğitim diski bekliyordu
(RGB 1,7–1,9 yığın/sn, yükleyici kodu ne olursa olsun).
Çözüm: eğitim karelerinin JPEG baytları ve etiket metinleri bir kez tek bir paket dosyasına yazılır (E:, SSD değil:
data/video/paket/<modalite>_train.bin + _indeks.json); eğitim başında paket sırayla okunup paylaşılan belleğe (SharedMemory)
konur; yükleyici işçileri aynı belleğe bağlanır, kopyalamadan okur. Görüntü yine PIL ile aynı baytlardan çözülür, etiket
aynı kodla ayrıştırılır → kareler ve hedefler disktekinin birebir aynısı (tests/test_video89b.py).
"""
from __future__ import annotations

import io
import json
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PAKET = ROOT / "data/video/paket"


def _etiket_yolu(p: Path) -> Path:
    parts = list(p.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def paket_olustur(modality: str, kareler: list[str], log=print) -> tuple[Path, Path]:
    """kareler: proje köküne göre göreli yollar. Paket ve indeks yazılır (aynı kare listesiyle varsa atlanır)."""
    PAKET.mkdir(parents=True, exist_ok=True)
    bin_p, idx_p = PAKET / f"{modality}_train.bin", PAKET / f"{modality}_train_indeks.json"
    if bin_p.is_file() and idx_p.is_file():
        J = json.loads(idx_p.read_text(encoding="utf-8"))
        if J.get("kareler") == kareler and bin_p.stat().st_size == J["boyut"]:
            return bin_p, idx_p
    t0 = time.time()
    kayit, off = [], 0
    with open(bin_p, "wb") as f:
        for i, rel in enumerate(kareler):
            p = ROOT / rel
            img = p.read_bytes()
            lp = _etiket_yolu(p)
            lbl = lp.read_bytes() if lp.is_file() else b""
            with Image.open(io.BytesIO(img)) as im:
                W, H = im.size
            f.write(img)
            f.write(lbl)
            kayit.append([off, len(img), off + len(img), len(lbl), H, W])
            off += len(img) + len(lbl)
            if (i + 1) % 5000 == 0:
                log(f"  paket {modality}: {i + 1}/{len(kareler)} kare, {off / 1e9:.2f} GB ({time.time() - t0:.0f} sn)")
    idx_p.write_text(json.dumps({"kareler": kareler, "kayit": kayit, "boyut": off}), encoding="utf-8")
    log(f"  paket {modality} yazıldı: {len(kareler)} kare, {off / 1e9:.2f} GB, {time.time() - t0:.0f} sn → {bin_p}")
    return bin_p, idx_p


class KareDeposu:
    """Ana süreçte yukle() paketi paylaşılan belleğe koyar; işçilere pickle ile yalnız bellek adı ve indeks gider."""

    def __init__(self, bin_p: Path, idx_p: Path):
        J = json.loads(idx_p.read_text(encoding="utf-8"))
        self.bin_p, self.n = str(bin_p), J["boyut"]
        self.idx = {str(ROOT / k): tuple(v) for k, v in zip(J["kareler"], J["kayit"])}
        self.ad, self._shm, self._sahip = None, None, False

    def kare_bayt(self, path) -> int:
        r = self.idx[str(path)]
        return r[1] + r[3]

    def yukle(self, log=print, butce: int | None = None, gruplar: list[list[str]] | None = None):
        """butce (bayt) paketten küçükse KISMİ depo (§89d, RGB paketi 5,2 GB, 16 GB RAM'e eğitimle birlikte sığmıyor):
        gruplar (video başına kare yolları, öncelik sırasıyla) bütçeye sığdıkça bütün olarak alınır, sığmayan atlanır;
        depoda olmayan kareler eskisi gibi diskten okunur (KlipVeri bir klibi yalnız bütün kareleri depodaysa RAM'den okur)."""
        t0 = time.time()
        n_paket, n_kare = self.n, len(self.idx)
        if butce is not None and butce < self.n and gruplar:
            secili, top = [], 0
            for g in gruplar:
                b = sum(self.kare_bayt(p) for p in g)
                if top + b <= butce:
                    secili += g
                    top += b
            eski = sorted(((self.idx[str(p)], str(p)) for p in secili), key=lambda x: x[0][0])
            self.idx, kosu, yeni_off = {}, [], 0                        # kosu: (paketteki başlangıç, uzunluk) — bitişikler birleşir
            for r, k in eski:
                L = r[1] + r[3]
                self.idx[k] = (yeni_off, r[1], yeni_off + r[1], r[3], r[4], r[5])
                if kosu and kosu[-1][0] + kosu[-1][1] == r[0]:
                    kosu[-1][1] += L
                else:
                    kosu.append([r[0], L])
                yeni_off += L
            self.n = max(yeni_off, 1)
        else:
            kosu = [[0, self.n]]
        self._shm = shared_memory.SharedMemory(create=True, size=self.n)
        self._sahip = True
        buf, pos, parca = self._shm.buf, 0, 64 * 2 ** 20
        with open(self.bin_p, "rb", buffering=0) as f:
            for bas, L in kosu:
                f.seek(bas)
                son = pos + L
                while pos < son:
                    k = f.readinto(buf[pos:min(son, pos + parca)])
                    if not k:
                        raise IOError("paket beklenenden kısa")
                    pos += k
        self.ad = self._shm.name
        ek = "" if self.n >= n_paket else f" (KISMİ: paketin {n_paket / 1e9:.2f} GB'ının, {n_kare} karenin; bütçe {butce / 1e9:.2f} GB)"
        log(f"  RAM deposu: {self.n / 1e9:.2f} GB, {len(self.idx)} kare{ek}, {time.time() - t0:.0f} sn")
        return self

    def __getstate__(self):
        return {"bin_p": self.bin_p, "n": self.n, "idx": self.idx, "ad": self.ad, "_shm": None, "_sahip": False}

    def _buf(self):
        if self._shm is None:
            self._shm = shared_memory.SharedMemory(name=self.ad)
        return self._shm.buf

    def var(self, path) -> bool:
        return str(path) in self.idx

    def boyut(self, path) -> tuple[int, int]:
        r = self.idx[str(path)]
        return r[4], r[5]

    def resim_bayt(self, path) -> bytes:
        r = self.idx[str(path)]
        return bytes(self._buf()[r[0]:r[0] + r[1]])

    def kutular(self, path) -> torch.Tensor:
        """tools.train_flybrain.load_boxes ile aynı ayrıştırma (YOLO → (y, x, h, w) piksel)."""
        r = self.idx[str(path)]
        H, W = r[4], r[5]
        rows = []
        if r[3]:
            metin = bytes(self._buf()[r[2]:r[2] + r[3]]).decode("utf-8")
            for line in metin.splitlines():
                v = line.split()
                if len(v) >= 5:
                    xc, yc, w, h = map(float, v[1:5])
                    rows.append(((yc - h / 2) * H, (xc - w / 2) * W, h * H, w * W))
        return torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)

    def kapat(self):
        if self._shm is not None:
            self._shm.close()
            if self._sahip:
                self._shm.unlink()
            self._shm = None
