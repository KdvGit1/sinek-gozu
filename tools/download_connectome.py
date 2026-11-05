#!/usr/bin/env python3
"""
Sinek beyni (Drosophila) konnektom verisini indirir.

1) FlyWire v783 (tam beyin, dişi, ~139 bin nöron) — Codex'in herkese açık Google Cloud deposundan.
   Görme sistemi (optik lob) hücre tipleri ve altıgen kolon koordinatları dahil.
   Lisans: CC BY 4.0 (Dorkenwald et al. 2024; Schlegel et al. 2024; Matsliah et al. 2024).
2) flyvis optik lob modeli konnektomu (Lappalainen et al. 2024) — 65 hücre tipi, altıgen kolon kuralları.
   Lisans: MIT.

Her dosya indirildikten sonra MD5 ile doğrulanır; zaten inmiş ve doğru olan dosya tekrar indirilmez.

Kullanım:
  python tools/download_connectome.py              # varsayılan küme (~225 MB)
  python tools/download_connectome.py --list       # indirmeden listele
  python tools/download_connectome.py --full-synapses   # + 2,7 GB sinaps tablosu (gerekmedikçe indirme)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data" / "connectome"

GCS_BUCKET = "flywire-data"
GCS_PREFIX = "codex/data/fafb/783/"

# (dosya, açıklama)
FLYWIRE_FILES = [
    ("neurons.csv.gz", "nöron id'leri, baskın nörotransmiter tahmini"),
    ("classification.csv.gz", "süper sınıf / sınıf / alt sınıf, beyin tarafı (sol/sağ)"),
    ("consolidated_cell_types.csv.gz", "birleştirilmiş hücre tipi etiketleri"),
    ("visual_neuron_types.csv.gz", "optik lob hücre tipleri (Matsliah et al. 2024)"),
    ("column_assignment.csv.gz", "optik lob nöronlarının altıgen kolon (p,q) atamaları"),
    ("coordinates.csv.gz", "nöron konumları (3B)"),
    ("cell_stats.csv.gz", "morfolojik ölçümler (uzunluk, alan, hacim)"),
    ("connections_no_threshold.csv.gz", "tüm nöron→nöron bağlantıları (eşiksiz), neuropil ve nt tipi ile"),
]
FULL_SYNAPSES = ("fafb_v783_princeton_synapse_table.csv.gz", "tek tek sinaps tablosu (2,7 GB)")

FLYVIS_FILES = [
    ("https://raw.githubusercontent.com/TuragaLab/flyvis/main/flyvis/connectome/fib25-fib19_v2.2.json",
     "fib25-fib19_v2.2.json", "optik lob kolon modeli: 65 hücre tipi, bağlantı kuralları (FIB-25/FIB-19)"),
    ("https://raw.githubusercontent.com/TuragaLab/flyvis/main/license", "LICENSE_flyvis_MIT.txt",
     "flyvis MIT lisans metni"),
]


def gcs_meta(name: str) -> dict:
    url = (f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o/"
           + urllib.request.quote(GCS_PREFIX + name, safe=""))
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def download(url: str, dst: Path, size: int | None = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
        total = size or int(r.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct // 10 != last // 10:
                    print(f"      {pct:3d}%  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", flush=True)
                    last = pct
    tmp.replace(dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FlyWire ve flyvis konnektom verisini indir")
    ap.add_argument("--list", action="store_true", help="sadece listele")
    ap.add_argument("--full-synapses", action="store_true", help="2,7 GB sinaps tablosunu da indir")
    args = ap.parse_args(argv)

    files = FLYWIRE_FILES + ([FULL_SYNAPSES] if args.full_synapses else [])
    manifest = []

    print("FlyWire v783 (Codex, CC BY 4.0):")
    for name, desc in files:
        meta = gcs_meta(name)
        size = int(meta["size"])
        dst = OUT / "flywire_783" / name
        print(f"  {size / 1e6:8.1f} MB  {name}  — {desc}")
        manifest.append({"file": f"flywire_783/{name}", "source": meta["mediaLink"], "bytes": size,
                         "md5": meta.get("md5Hash"), "license": "CC-BY-4.0", "description": desc})
        if args.list:
            continue
        if dst.is_file() and md5_b64(dst) == meta.get("md5Hash"):
            print("      zaten var, MD5 doğru — atlandı")
            continue
        download(meta["mediaLink"], dst, size)
        if meta.get("md5Hash") and md5_b64(dst) != meta["md5Hash"]:
            dst.unlink()
            sys.exit(f"  ! MD5 uyuşmadı, dosya silindi: {name}. Tekrar çalıştırın.")
        print("      MD5 doğrulandı")

    print("\nflyvis (TuragaLab, MIT):")
    for url, name, desc in FLYVIS_FILES:
        dst = OUT / "flyvis" / name
        print(f"  {name}  — {desc}")
        if not args.list:
            download(url, dst)
            digest = hashlib.sha256(dst.read_bytes()).hexdigest()
            manifest.append({"file": f"flyvis/{name}", "source": url, "bytes": dst.stat().st_size,
                             "sha256": digest, "license": "MIT", "description": desc})

    if not args.list:
        (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✓ İndirildi: {OUT}\n  Kayıt: {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
