#!/usr/bin/env python3
"""
§89 H61 videoya biyolojik uyum: ana sinek modelini (termal uzun_75 / RGB uzun_80_rgb) ağırlıkları dondurmadan video
kliplerinde eğitir. Kullanıcı kararları (günlük §86–88): nöron durumu ve STP kareden kareye taşınır, klip içinde hiç
sıfırlanmaz (yalnız sızıntıyla söner); ilgisiz klipler dinlenimden başlar (yeni sahne); aşamalı şekillendirme (1 varlık +
hafif ceza, 2 kutu, 3 yanlış kutu temizliği); önce gradyan; H47 (sonraki kare tahmini, öğretmen olayın kendisi; dış model
yok); zaman sabitleri (λ) ayrı öğrenme oranıyla serbest.
Veri: data/video/video_bolme.json (tools/video_bolme.py, video bazında %70/15/15). Test kümesi kullanılmaz.
Değerlendirme (EMA modeli): video val (her val videosundan ortadaki ardışık pencere; her karo konumunun durumu kareden kareye
taşınır; ilk `isinma` kare puanlanmaz) → mAP50 toplam / set / "ana modelin görmediği" videolar; aynı karelerde durağan taban
(eğitim öncesi model, her kare tek başına, T = 10); durağan val800 (unutma bilgisi); T4/T5 yön seçiciliği; zaman sabitleri.
Durum: results/flybrain/<ad>/durum.md ve durum.json (doğruluk, hız, süre, kalan, aşama, son kayıt) — kullanıcı ve Claude okur.
Kaldığı yerden devam: --resume (last.pt). Kullanım (örnek):
  py -3 tools/train_video.py --modality thermal --name video_pilot/V1_termal --iters 12000
"""
from __future__ import annotations

import os

os.environ.setdefault("FLYBRAIN_DECODE", "merkez7p2")
os.environ.setdefault("FLYBRAIN_OVERLAP", "0.5")

import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tools.train_flybrain as tf  # noqa: E402
from flybrain.connectome import Connectome  # noqa: E402
from flybrain.detection import (BilateralTargets, TargetBuilder, average_precision, decode_cpu,  # noqa: E402
                                match_flags, merge_top)
from flybrain.eye import Tiler  # noqa: E402
from flybrain.video_hizli import HizliHedef, boyut, kesit_yukle  # noqa: E402
from flybrain.video import (SonrakiKare, asama_kaybi, kolon_yamalari, sonme_ozeti, video_ileri,  # noqa: E402
                            yon_seciciligi)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "results" / "flybrain"
BOLME = ROOT / "data/video/video_bolme.json"
ANA = {"thermal": "results/flybrain/uzun_75/best.pt", "rgb640": "results/flybrain/uzun_80_rgb/best.pt"}
HAREKET_TIPLERI = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "LC11", "LPLC2", "HSE", "VS1")
tr = lambda x, d=3: "—" if x is None else f"{x:.{d}f}".replace(".", ",")


def make_parser():
    ap = argparse.ArgumentParser(description="§89 H61 video uyumu")
    ap.add_argument("--modality", required=True, choices=["thermal", "rgb640"])
    ap.add_argument("--init", default="", help="başlangıç modeli (varsayılan: modalitenin ana modeli)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--toplam-saat", type=float, default=0.0, help="> 0: yalnız eğitim süresi (değerlendirme hariç)")
    ap.add_argument("--klip", type=int, default=6, help="klipteki kare sayısı (K)")
    ap.add_argument("--adim", type=int, default=3, help="kare başına döngü adımı")
    ap.add_argument("--adim-ilk", type=int, default=10, help="dinlenimden ilk karede adım (durağan rejim)")
    ap.add_argument("--isinma", type=int, default=2, help="puanlanmayan ilk kare sayısı")
    ap.add_argument("--aralik", default="wisard:3,birdsai:1,visdrone_vid:3", help="set:kare aralığı (BIRDSAI zaten 1/5)")
    ap.add_argument("--batch", type=int, default=8, help="yığındaki klip sayısı")
    ap.add_argument("--pos-frac", type=float, default=0.7)
    ap.add_argument("--set-us", type=float, default=0.5)
    ap.add_argument("--no-photo-aug", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--ram-depo-yok", action="store_true",
                    help="§89d: eğitim karelerini RAM'e (paylaşılan bellek) yükleme; varsayılan yükler (disk darboğazı)")
    ap.add_argument("--gpu-kesir", type=float, default=0.85,
                    help="§89d: PyTorch GPU belleği tavanı (12 GB'ın kesri). Windows sürücüsü taşanı sistem RAM'ine atıyor ve eğitim "
                    "~10 kat yavaşlıyordu (eğitim önbelleği ~5–6 GB + val800 önbelleği 5,2 GB); tavanda PyTorch önce önbelleğini boşaltır")
    ap.add_argument("--ram-pay", type=float, default=4.25,
                    help="§89d: RAM deposundan sonra boş kalacak pay (GB); eğitim süreci depodan sonra ~2,5–3 GB büyüyor, değerlendirmede daha da (3 GB payla boş RAM 0,3 GB'a indi, sonraki ölçüm ~60 kat yavaşladı) → hedef: en düşük boş RAM ≥ ~1,5 GB")
    ap.add_argument("--vval-pencere", type=int, default=3, help="her val videosundan ayrık pencere sayısı (1/4, 1/2, 3/4)")
    ap.add_argument("--eski-yukleyici", action="store_true",
                    help="§89b öncesi yükleyici (bütün kare float'a çevrilir, döngülü hedef); varsayılan hızlı yükleyici (aynı sonuç)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--alpha-lr-mult", type=float, default=3.0)
    ap.add_argument("--leak-lr-mult", type=float, default=10.0, help="zaman sabitleri (λ) için öğrenme oranı çarpanı")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--asama-sinir", default="2500,6000", help="aşama 2 ve 3'ün başladığı iterasyonlar")
    ap.add_argument("--neg-w", default="0.25,1.0,2.0", help="aşama 1/2/3'te yanlış yere yanmanın cezası (negatif ağırlık)")
    ap.add_argument("--w-offset", type=float, default=1.0)
    ap.add_argument("--w-size", type=float, default=0.5)
    ap.add_argument("--h47", type=float, default=0.0, help="H47 sonraki kare tahmini kaybının ağırlığı (0 = kapalı)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--vval-kare", type=int, default=20, help="her pencerede işlenen kare (§89d: 30 → 3 pencere × 20)")
    ap.add_argument("--val800", type=int, default=800, help="durağan val800 görüntü sayısı (0 = ölçme)")
    ap.add_argument("--ckpt-minutes", type=float, default=10.0)
    ap.add_argument("--plato", type=int, default=0, help="> 0: bu kadar değerlendirme iyileşme yoksa lr yarıya")
    ap.add_argument("--plato-taban", type=float, default=0.0625)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    return ap


# ---------------------------------------------------------------------------- veri
def videolar(modality: str, bolum: str) -> list:
    J = json.loads(BOLME.read_text(encoding="utf-8"))
    out = []
    for ds, sp in J["modaliteler"][modality].items():
        for v in sp[bolum]:
            out.append({"set": ds, "anahtar": v["anahtar"], "gordu": v["ana_model_gordu"],
                        "kareler": [str(ROOT / p) for p in v["kareler"]]})
    return out


class KlipVeri(torch.utils.data.Dataset):
    """Her öğe: bir videodan K kare (aralıklı), aynı karo konumunda; tutarlı çevirme ve fotometrik artırma; kare başına
    hedefler (BilateralTargets, reg_mode="box": (ısı, reg, pozitif, içeride)). Set seçimi set_us ile dengelenir."""

    def __init__(self, vids, channels, eye_hw, inscribed, K, aralik, pos_frac, set_us, photo_aug, targets, pad, seed,
                 uzunluk=10 ** 7):
        self.K, self.aralik, self.pos_frac, self.photo_aug = K, aralik, pos_frac, photo_aug
        self.ch, self.pad, self.seed, self.uzunluk = channels, pad, seed, uzunluk
        self.eh, self.ew = eye_hw
        self.ry, self.rx, self.rh, self.rw = inscribed
        self.targets = targets
        self.by = {}
        for v in vids:
            if len(v["kareler"]) > (K - 1) * aralik.get(v["set"], 1):
                self.by.setdefault(v["set"], []).append(v)
        self.setler = sorted(self.by)
        top = {s: sum(len(v["kareler"]) for v in self.by[s]) for s in self.setler}
        self.set_w = [top[s] ** (1 - set_us) for s in self.setler]
        self.vid_w = {s: [len(v["kareler"]) for v in self.by[s]] for s in self.setler}
        self._rng = None

    def __len__(self):
        return self.uzunluk

    def rng(self):
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = random.Random(hash((self.seed, info.seed if info else torch.initial_seed())))
        return self._rng

    def __getitem__(self, i):
        rng = self.rng()
        s = rng.choices(self.setler, weights=self.set_w)[0]
        v = rng.choices(self.by[s], weights=self.vid_w[s])[0]
        a = self.aralik.get(s, 1)
        span = (self.K - 1) * a
        i0 = rng.randrange(0, len(v["kareler"]) - span)
        yollar = [Path(v["kareler"][i0 + k * a]) for k in range(self.K)]
        depo = getattr(self, "depo", None)
        if depo is not None and all(depo.var(p) for p in yollar):         # §89d: RAM'den (aynı baytlar, aynı ayrıştırma)
            imgs = None
            H, W = depo.boyut(yollar[0])
            boxes = [depo.kutular(p) for p in yollar]
        elif getattr(self, "hizli", False):                             # §89b: önce kes, sonra çevir (aynı pikseller)
            depo = None
            imgs = None
            H, W = boyut(yollar[0])
            boxes = [tf.load_boxes(p, H, W) for p in yollar]
        else:
            imgs = [tf.load_image(p, self.ch) for p in yollar]
            _, H, W = imgs[0].shape
            boxes = [tf.load_boxes(p, im.shape[1], im.shape[2]) for p, im in zip(yollar, imgs)]
        son = boxes[-1]
        if len(son) and rng.random() < self.pos_frac:
            y, x, h, w = son[rng.randrange(len(son))].tolist()
            oy = int(y + h / 2 - self.ry - rng.uniform(0, self.rh))
            ox = int(x + w / 2 - self.rx - rng.uniform(0, self.rw))
        else:
            oy = rng.randint(-self.ry, max(-self.ry, H - self.ry - self.rh))
            ox = rng.randint(-self.rx, max(-self.rx, W - self.rx - self.rw))
        hf, vf = rng.random() < 0.5, rng.random() < 0.5
        if self.photo_aug:
            gam = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
            kon, par = rng.uniform(0.6, 1.4), rng.uniform(-0.2, 0.2)
        tiles, tb = [], []
        m0 = None
        for j, b in enumerate(boxes):
            kaynak = depo.resim_bayt(yollar[j]) if depo is not None else yollar[j]
            t = kesit_yukle(kaynak, self.ch, oy, ox, self.eh, self.ew) if imgs is None else None
            if t is None:
                im = imgs[j] if imgs is not None else tf.load_image(yollar[j], self.ch)
                t = tf.crop(im, oy, ox, self.eh, self.ew, self.pad)
            b = b.clone()
            if len(b):
                b[:, 0] -= oy
                b[:, 1] -= ox
            if self.photo_aug:
                t = t.clamp(0, 1) ** gam
                m0 = float(t.mean()) if m0 is None else m0                       # bütün karelerde aynı dönüşüm
                t = ((t - m0) * kon + m0 + par).clamp(0, 1)
            if hf:
                t = t.flip(-1)
                if len(b):
                    b[:, 1] = self.ew - b[:, 1] - b[:, 3]
            if vf:
                t = t.flip(-2)
                if len(b):
                    b[:, 0] = self.eh - b[:, 0] - b[:, 2]
            tiles.append(t)
            tb.append(b)
        return torch.stack(tiles), self.targets(tb)


# ---------------------------------------------------------------------------- değerlendirme
def pencereler(v, aralik, L, n):
    """§89d: videonun 1/(n+1), 2/(n+1), … noktalarında ortalanmış n ayrık pencere (her biri ayrı sahne: durum sıfırdan).
    n = 1 → eski tek orta pencere."""
    if n <= 1:
        return [pencere(v, aralik, L)]
    a = aralik.get(v["set"], 1)
    N = len(v["kareler"])
    L = min(L, max(1, ((N - 1) // a + 1) // n))
    out = []
    for j in range(n):
        merkez = int(N * (j + 1) / (n + 1))
        i0 = min(max(0, merkez - (L - 1) * a // 2), N - (L - 1) * a - 1)
        out.append([Path(v["kareler"][i0 + k * a]) for k in range(L)])
    return out


def pencere(v, aralik, L):
    a = aralik.get(v["set"], 1)
    n = len(v["kareler"])
    L = min(L, (n - 1) // a + 1)
    i0 = max(0, (n - (L - 1) * a - 1) // 2)
    return [Path(v["kareler"][i0 + k * a]) for k in range(L)]


@torch.no_grad()
def vval(m, vids, args, aralik, dev, durağan=False):
    """Video val: her videonun ortadaki penceresi, tam kare karolama; durum her karo konumunda kareden kareye taşınır.
    durağan=True: her kare tek başına, dinlenimden T = adim_ilk adım (karşılaştırma tabanı)."""
    m.eval()
    tiler = tf.tiler_for(m, "replicate")
    fl, ng = {}, {}
    for v, pen in ((v, pen) for v in vids for pen in pencereler(v, aralik, args.vval_kare, args.vval_pencere)):
        state = stp = None
        for k, p in enumerate(pen):
            img = tf.load_image(p, m.cfg.in_channels)
            _, H, W = img.shape
            tiles, org = tiler.extract(img)
            tiles = tiles.to(dev)
            if durağan:
                out = m(tiles)
            else:
                d = m.drive(tiles)
                n = args.adim_ilk if state is None else args.adim
                state, stp = m.core(d, state=state, steps=n, stp_state=stp, return_stp=True)
                if k < args.isinma:
                    continue
                out = m._pack(m.head_forward(m.readout(state, None)), state)
            if k < args.isinma:
                continue
            dets = decode_cpu(out, m.eye, score_thr=0.01, max_det=100)
            merged = merge_top(dets, org, H, W, 0.5, 300, True)
            gts = tf.load_boxes(p, H, W)
            f = match_flags(merged, gts, 0.5)
            for g in ("toplam", v["set"]) + (() if v["gordu"] else ("ana_model_gormedi",)):
                fl.setdefault(g, []).extend(f)
                ng[g] = ng.get(g, 0) + len(gts)
    return {g: average_precision(fl[g], ng[g]) for g in fl}


def dsi_karolari(vids, m, aralik, n=8):
    out = []
    for v in vids:
        for p in pencere(v, aralik, 3)[1:2]:
            img = tf.load_image(p, m.cfg.in_channels)
            b = tf.load_boxes(p, img.shape[1], img.shape[2])
            if not len(b):
                continue
            y, x, h, w = b[int(torch.argmax(b[:, 2] * b[:, 3]))].tolist()
            out.append(tf.crop(img, int(y + h / 2 - m.eye.height / 2), int(x + w / 2 - m.eye.width / 2), m.eye.height,
                               m.eye.width, "replicate"))
        if len(out) >= n:
            break
    return torch.stack(out)


# ---------------------------------------------------------------------------- yardımcılar
def bos_ram() -> int:
    """Windows: kullanılabilir fiziksel bellek (bayt)."""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.ullAvailPhys)


def eski_bitti(out: Path) -> bool:
    try:
        return json.loads((out / "durum.json").read_text(encoding="utf-8")).get("durum") == "BİTTİ"
    except (OSError, ValueError):
        return False


def yaz_durum(out: Path, D: dict):
    (out / "durum.json").write_text(json.dumps(D, ensure_ascii=False, indent=1), encoding="utf-8")
    e = D.get("son_degerlendirme") or {}
    L = [f"# Video eğitimi durumu: {D['ad']}", "",
         f"Güncellendi: {D['guncellendi']} · **{D['durum']}** · modalite {D['modalite']} · aşama {D['asama']}/3", "",
         f"- İterasyon: {D['iter']:,} / {D['iters']:,}".replace(",", ".") +
         (f" · eğitim süresi {tr(D['egitim_sa'], 2)} / {tr(D['toplam_saat'], 2)} sa" if D["toplam_saat"] else
          f" · eğitim süresi {tr(D['egitim_sa'], 2)} sa"),
         f"- Hız: {tr(D['hiz'], 1)} it/s · kalan (eğitim) ~{tr(D['kalan_sa'], 2)} sa · duvar saati {tr(D['duvar_sa'], 2)} sa · "
         f"lr {D['lr']:.2e} · son kayıt {D['son_kayit']}",
         f"- Bellek: GPU en çok {tr(D.get('gpu_gb'), 1)} GB (önbellek {tr(D.get('gpu_onbellek_gb'), 1)}, tepe "
         f"{tr(D.get('gpu_onbellek_tepe_gb'), 1)} GB; tavan {tr(12 * D.get('gpu_kesir', 0.85), 1)}) · boş RAM {tr(D.get('bos_ram_gb'), 1)} GB · kareler " +
         (f"RAM deposundan (kliplerin ~%{100 * D.get('ram_klip_payi', 1):.0f}'i)" if D.get('ram_deposu') else 'diskten'),
         f"- Son kayıp: toplam {tr(D['kayip'].get('total'))} · ısı {tr(D['kayip'].get('heat'))} · ofset "
         f"{tr(D['kayip'].get('offset'))} · boyut {tr(D['kayip'].get('size'))} · H47 {tr(D['kayip'].get('h47'))}", ""]
    if e:
        L += [f"## Son değerlendirme ({e['iter']:,}".replace(",", ".") + f" it, {e['zaman']})", "",
              "| ölçü | şimdi | eğitim öncesi (durağan taban) |", "|---|---|---|"]
        for g, val in e["vval"].items():
            L.append(f"| video val mAP50 · {g} | {tr(val, 4)} | {tr(D.get('taban', {}).get(g), 4)} |")
        L += [f"| durağan val800 | {tr(e.get('val800'), 4)} | {tr(D.get('taban_val800'), 4)} |", "",
              f"- Hareket: T4a–d DSI ortalama {tr(e['dsi']['t4_ort'], 2)} (tercih yönleri: {e['dsi']['t4_yonler']}), "
              f"T5a–d {tr(e['dsi']['t5_ort'], 2)} ({e['dsi']['t5_yonler']}); eğitim öncesi T4 {tr(D.get('taban_dsi_t4'), 2)}",
              f"- Sönme (τ = 1/λ adım): ortanca {tr(e['sonme']['tau_ortanca'], 2)}, %5–95 {tr(e['sonme']['tau_p05'], 2)}–"
              f"{tr(e['sonme']['tau_p95'], 2)}, en uzun {tr(e['sonme']['tau_en_uzun'], 1)} · " +
              ", ".join(f"{k} {tr(v, 2)}" for k, v in e["sonme"]["tipler"].items()), ""]
    L.append(f"En iyi video val: {tr(D.get('best'), 4)} ({D.get('best_iter')} it) · klasör `{D['klasor']}`")
    (out / "durum.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    args = make_parser().parse_args(argv)
    dev = torch.device("cuda")
    if args.gpu_kesir > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_kesir)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "egitim.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    init = Path(args.init or (ROOT / ANA[args.modality]))
    ck0 = torch.load(init, map_location="cpu", weights_only=False)
    yapi = dict(ck0["args"])                                            # modelin kurulum ayarları (araçlar bunu okur)
    model = tf.build(argparse.Namespace(**yapi), dev)
    model.load_state_dict(ck0["model"])
    aralik = {k: int(v) for k, v in (x.split(":") for x in args.aralik.split(","))}
    s2, s3 = (int(x) for x in args.asama_sinir.split(","))
    neg_w = [float(x) for x in args.neg_w.split(",")]
    tip = Connectome.load(model.cfg.processed_dir).nodes.type.fillna("?").astype(str).to_numpy()
    tipler = {t: torch.as_tensor(np.flatnonzero(tip == t)) for t in HAREKET_TIPLERI if (tip == t).any()}
    with torch.no_grad():
        F = model.readout(model.core(model.drive(torch.zeros(1, model.cfg.in_channels, model.eye.height,
                                                             model.eye.width, device=dev)), steps=1), None).shape[-1]
    h47 = SonrakiKare(F, model.eye.patch).to(dev) if args.h47 > 0 else None
    alpha, leak = model.alpha_raw, model.leak_raw
    digerleri = [p for p in model.parameters() if p.requires_grad and p is not alpha and p is not leak]
    if h47 is not None:
        if ck0.get("h47"):                                              # §89d: başlangıç modelinin H47 çözücüsü
            h47.load_state_dict(ck0["h47"])
        digerleri += list(h47.parameters())
    opt = torch.optim.Adam([{"params": digerleri, "mult": 1.0}, {"params": [alpha], "mult": args.alpha_lr_mult},
                            {"params": [leak], "mult": args.leak_lr_mult}], args.lr)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    ema_pairs = list(zip(ema.parameters(), model.parameters()))
    it, best, best_iter, train_sec0, lr_scale, kotu = 0, -1.0, None, 0.0, 1.0, 0
    D = {"ad": args.name, "modalite": args.modality, "iters": args.iters, "toplam_saat": args.toplam_saat,
         "klasor": str(out.relative_to(ROOT)).replace("\\", "/"), "kayip": {}, "baslangic_modeli": str(init),
         "iter": 0, "asama": 1, "lr": 0.0, "hiz": 0.0, "egitim_sa": 0.0, "duvar_sa": 0.0, "kalan_sa": 0.0, "best": None,
         "best_iter": None, "durum": "BAŞLIYOR", "son_kayit": "—", "guncellendi": time.strftime("%d.%m %H:%M:%S")}
    if args.resume and (out / "last.pt").is_file():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        if h47 is not None and ck.get("h47"):
            h47.load_state_dict(ck["h47"])
        it, best, best_iter, train_sec0 = ck["iter"], ck["best"], ck.get("best_iter"), ck["train_sec"]
        lr_scale, kotu = ck.get("lr_scale", 1.0), ck.get("plato_kotu", 0)
        D.update(ck.get("durum_taban", {}))
        if (out / "durum.json").is_file():                              # §89d: değerlendirme geçmişi kesintide silinmesin
            eski = json.loads((out / "durum.json").read_text(encoding="utf-8"))
            for k in ("gecmis", "son_degerlendirme", "son_durağan_vval"):
                if k in eski:
                    D[k] = eski[k]
            D["gecmis"] = [g for g in D.get("gecmis", []) if g["iter"] <= it]
            if D.get("gecmis"):
                eniyi = max(D["gecmis"], key=lambda g: g["vval"] if g["vval"] is not None else -1)
                D["best"], D["best_iter"] = eniyi["vval"], eniyi["iter"]
        log(f"devam: iterasyon {it}, eğitim süresi {train_sec0 / 3600:.2f} sa, en iyi {best:.4f}")
        if (it >= args.iters or (args.toplam_saat and train_sec0 / 3600 >= args.toplam_saat)) and eski_bitti(out):
            log("bu eğitim zaten bitmiş (durum BİTTİ) → atlanıyor")
            return 0
    else:
        log(f"başlangıç: {init} · ayarlar: {vars(args)}")
    model.train()

    tr_vids, va_vids = videolar(args.modality, "train"), videolar(args.modality, "val")
    log(f"videolar: eğitim {len(tr_vids)} ({sum(len(v['kareler']) for v in tr_vids):,} kare), val {len(va_vids)}; "
        + ", ".join(f"{s} {sum(v['set'] == s for v in tr_vids)}/{sum(v['set'] == s for v in va_vids)}"
                    for s in sorted({v['set'] for v in tr_vids})))
    tiler = Tiler(model.eye)
    tb = (BilateralTargets if args.eski_yukleyici else HizliHedef)(TargetBuilder(model.eye, reg_mode="box"), model.eye.width,
                          None if model.eye_visible is None else model.eye_visible[:model.n_eye_columns].cpu())
    ds = KlipVeri(tr_vids, model.cfg.in_channels, (model.eye.height, model.eye.width), (tiler.ry, tiler.rx, tiler.rh, tiler.rw),
                  args.klip, aralik, args.pos_frac, args.set_us, not args.no_photo_aug, tb, "replicate", args.seed + it)
    ds.hizli = not args.eski_yukleyici
    depo = None
    if not args.ram_depo_yok:                                            # §89d: eğitim kareleri RAM'de
        from flybrain.kare_deposu import KareDeposu, paket_olustur
        rel = [str(Path(p).relative_to(ROOT)).replace("\\", "/") for v in tr_vids for p in v["kareler"]]
        bin_p, idx_p = paket_olustur(args.modality, rel, log)
        # bütçe: şimdiki boş RAM − işçiler (~0,35 GB) − pay; ölçüm (§89d): RGB paketi (5,2 GB) + 5 işçi + eğitim süreci 16 GB'a
        # sığmıyor (4,1 GB depoyla ilk değerlendirmede boş RAM 0,02 GB'a indi) → sığmazsa KISMİ depo: örneklenme olasılığı /
        # bayt oranı en yüksek videolar bütün olarak RAM'e, gerisi diskten
        butce = int(bos_ram() - args.workers * 0.35 * 2 ** 30 - args.ram_pay * 2 ** 30)
        depo = KareDeposu(bin_p, idx_p)
        sw = sum(ds.set_w)
        olas = []                                                        # (klip olasılığı / bayt, kareler)
        for s, w in zip(ds.setler, ds.set_w):
            vt = sum(ds.vid_w[s])
            for v, n in zip(ds.by[s], ds.vid_w[s]):
                bayt = sum(depo.kare_bayt(p) for p in v["kareler"] if depo.var(p))
                olas.append((w / sw * n / vt / max(bayt, 1), w / sw * n / vt, v["kareler"]))
        olas.sort(key=lambda x: -x[0])
        if butce < 512 * 2 ** 20:
            log(f"  UYARI: boş RAM yetmiyor ({bos_ram() / 2 ** 30:.1f} GB) → kareler diskten")
            depo = None
        else:
            depo.yukle(log, butce, [g for _, _, g in olas])
            pay_ram = sum(o for _, o, g in olas if all(depo.var(p) for p in g))
            log(f"  kliplerin ~%{100 * pay_ram:.0f}'i RAM'den, gerisi diskten")
            D["ram_klip_payi"], D["ram_deposu"], D["ram_deposu_gb"] = pay_ram, True, depo.n / 2 ** 30
            ds.depo = depo
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers, persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers > 0 else None, pin_memory=True, drop_last=True)
    val800 = random.Random(0).sample(tf.list_images(args.modality, "val"), args.val800) if args.val800 else []
    dsi_t = dsi_karolari(va_vids, model, aralik).to(dev)

    def dsi_ozet(mm):
        r = yon_seciciligi(mm, dsi_t, tipler, args.adim, args.adim_ilk)
        t4 = [r[t] for t in ("T4a", "T4b", "T4c", "T4d") if t in r]
        t5 = [r[t] for t in ("T5a", "T5b", "T5c", "T5d") if t in r]
        return {"tipler": r, "t4_ort": float(np.mean([x["dsi"] for x in t4])), "t5_ort": float(np.mean([x["dsi"] for x in t5])),
                "t4_yonler": "/".join(x["tercih"] for x in t4), "t5_yonler": "/".join(x["tercih"] for x in t5)}

    if "taban" not in D:                                                 # eğitim öncesi model: aynı ölçümler (bir kez)
        t0 = time.time()
        D["taban"] = vval(ema, va_vids, args, aralik, dev, durağan=True)
        D["taban_zamanli"] = vval(ema, va_vids, args, aralik, dev)
        D["taban_val800"] = tf.evaluate(ema, val800, ema.cfg.in_channels, dev, pad_mode="replicate")["mAP50"] if val800 else None
        D["taban_dsi_t4"] = dsi_ozet(ema)["t4_ort"]
        log(f"taban (eğitim öncesi, {time.time() - t0:.0f} sn): video val durağan {D['taban']} · aynı model zamanlı "
            f"{D['taban_zamanli']} · val800 {D['taban_val800']} · T4 DSI {D['taban_dsi_t4']:.3f}")
    durum_taban = {k: D[k] for k in ("taban", "taban_zamanli", "taban_val800", "taban_dsi_t4")}
    torch.cuda.empty_cache()                                             # §89d: değerlendirme önbelleği eğitime kalmasın

    logp = out / "log.csv"
    yeni = not logp.is_file()
    logc = open(logp, "a", newline="", encoding="utf-8")
    logw = csv.writer(logc)
    if yeni:
        logw.writerow(["iter", "asama", "lr", "loss", "heat", "offset", "size", "h47", "vval", "val800", "t4_dsi", "tau_ortanca", "sn"])
    t0 = time.time()
    eval_spent = 0.0
    train_sec = lambda: train_sec0 + (time.time() - t0) - eval_spent
    t_win, it_win, t_ckpt = t0, it, t0
    hiz_ref, son_uyari = 0.0, -10 ** 9
    D["gpu_kesir"] = args.gpu_kesir
    run = torch.zeros(5, device=dev)
    run_n = 0

    def kaydet(yol, mm, **ek):
        torch.save({"model": mm.state_dict(), "args": yapi, "config": mm.cfg.to_dict(), "iter": it, "best": best,
                    "video": vars(args), **ek}, yol)

    def son_kayit():
        kaydet(out / "last.pt", model, ema=ema.state_dict(), opt=opt.state_dict(),
               h47=h47.state_dict() if h47 is not None else None, best_iter=best_iter, train_sec=train_sec(),
               lr_scale=lr_scale, plato_kotu=kotu, durum_taban=durum_taban)
        D["son_kayit"] = time.strftime("%d.%m %H:%M")

    def degerlendir():
        nonlocal best, best_iter, lr_scale, kotu
        te = time.time()
        torch.cuda.empty_cache()                                         # §89d: eğitim önbelleği + değerlendirme önbelleği > 12 GB
        r = vval(ema, va_vids, args, aralik, dev)
        v8 = tf.evaluate(ema, val800, ema.cfg.in_channels, dev, pad_mode="replicate")["mAP50"] if val800 else None
        dz = dsi_ozet(ema)
        sn = sonme_ozeti(ema, tip)
        e = {"iter": it, "zaman": time.strftime("%d.%m %H:%M"), "vval": r, "val800": v8, "dsi": dz, "sonme": sn}
        D["son_degerlendirme"] = e
        if r.get("toplam", -1) > (D.get("best") if D.get("best") is not None else -1):
            D["best"], D["best_iter"] = r["toplam"], it
        D.setdefault("gecmis", []).append({"iter": it, "vval": r.get("toplam"), "val800": v8, "t4_dsi": dz["t4_ort"],
                                           "tau_ortanca": sn["tau_ortanca"]})
        log(f"  video val {', '.join(f'{k} {v:.4f}' for k, v in r.items())} · val800 {v8} · T4 DSI {dz['t4_ort']:.3f} "
            f"({dz['t4_yonler']}) · T5 DSI {dz['t5_ort']:.3f} · τ ortanca {sn['tau_ortanca']:.2f} (%5–95 {sn['tau_p05']:.2f}–"
            f"{sn['tau_p95']:.2f}, Mi1 {sn['tipler'].get('Mi1', 0):.2f}, Mi4 {sn['tipler'].get('Mi4', 0):.2f}) "
            f"({time.time() - te:.0f} sn)")
        logw.writerow([it, asama(it), "", "", "", "", "", "", f"{r.get('toplam', 0):.4f}", "" if v8 is None else f"{v8:.4f}",
                       f"{dz['t4_ort']:.4f}", f"{sn['tau_ortanca']:.3f}", round(train_sec())])
        logc.flush()
        if r.get("toplam", -1) > best:
            best, best_iter, kotu = r["toplam"], it, 0
            kaydet(out / "best.pt", ema, ema_agirlik=True, h47=h47.state_dict() if h47 is not None else None)
            log(f"  en iyi: {best:.4f} → best.pt")
        elif args.plato > 0:
            kotu += 1
            if kotu >= args.plato:
                lr_scale, kotu = max(lr_scale * 0.5, args.plato_taban), 0
                log(f"  plato: {args.plato} değerlendirmedir iyileşme yok → lr ölçeği {lr_scale:.4f}")
        model.train()
        torch.cuda.empty_cache()
        return time.time() - te

    asama = lambda i: 1 if i < s2 else (2 if i < s3 else 3)
    bitti = lambda: it >= args.iters or (args.toplam_saat > 0 and train_sec() >= args.toplam_saat * 3600)
    params = [p for g in opt.param_groups for p in g["params"]]
    log(f"eğitim başlıyor: aşama sınırları {s2}/{s3}, ceza ağırlıkları {neg_w}, H47 {args.h47}, klip {args.klip} × "
        f"{args.adim} adım (ilk {args.adim_ilk}), ısınma {args.isinma}, aralık {aralik}")
    durum = "EĞİTİLİYOR"
    while not bitti():
        for clip, hedef in loader:
            if bitti():
                break
            a = asama(it)
            lr_now = args.lr * min(1.0, (it + 1) / args.warmup) * lr_scale
            for g in opt.param_groups:
                g["lr"] = lr_now * g["mult"]
            clip = clip.to(dev, non_blocking=True).transpose(0, 1)          # [K, B, C, H, W]
            hedef = [h.to(dev, non_blocking=True).transpose(0, 1) for h in hedef]
            outs, feats, _, _ = video_ileri(model, clip, args.adim, args.adim_ilk, ozellik=h47 is not None)
            parca = [asama_kaybi(outs[k], tuple(h[k] for h in hedef), a, neg_w[a - 1], args.w_offset, args.w_size)
                     for k in range(args.isinma, args.klip)]
            L = sum(p["total"] for p in parca) / len(parca)
            l47 = L.new_zeros(())
            if h47 is not None:
                ks = list(range(max(0, args.isinma - 1), args.klip - 1))
                for k in ks:
                    with torch.no_grad():
                        hedef_y = kolon_yamalari(model, clip[k + 1])
                    l47 = l47 + nn.functional.mse_loss(h47(feats[k]), hedef_y)
                l47 = l47 / len(ks)
            toplam = L + args.h47 * l47
            opt.zero_grad(set_to_none=True)
            toplam.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            with torch.no_grad():
                torch._foreach_mul_([e for e, _ in ema_pairs], args.ema)
                torch._foreach_add_([e for e, _ in ema_pairs], [m_.detach() for _, m_ in ema_pairs], alpha=1 - args.ema)
            it += 1
            run += torch.stack([toplam.detach(), sum(p["heat"] for p in parca) / len(parca),
                                sum(p["offset"] for p in parca) / len(parca), sum(p["size"] for p in parca) / len(parca),
                                l47.detach()])
            run_n += 1
            if it % 50 == 0:
                mv = (run / run_n).tolist()
                run.zero_()
                run_n = 0
                if not all(math.isfinite(x) for x in mv):
                    log("HATA: kayıp sonsuz / NaN — durduruldu")
                    durum = "HATA (NaN)"
                    son_kayit()
                    D.update({"durum": durum, "guncellendi": time.strftime("%d.%m %H:%M:%S")})
                    yaz_durum(out, D)
                    if depo is not None:
                        depo.kapat()
                    return 1
                now = time.time()
                hiz = (it - it_win) / max(now - t_win, 1e-6)
                t_win, it_win = now, it
                hiz_ref = max(hiz_ref * 0.999, hiz)                      # §89d: yavaşlama uyarısı (GPU taşması / disk)
                if hiz < 0.5 * hiz_ref and it - son_uyari >= 1000:
                    log(f"  UYARI: hız düştü ({hiz:.1f} it/s, önceki en iyi {hiz_ref:.1f}); GPU önbellek "
                        f"{torch.cuda.memory_reserved() / 2 ** 30:.1f} GB, boş RAM {bos_ram() / 2 ** 30:.1f} GB")
                    son_uyari = it
                kalan = ((args.iters - it) / max(hiz, 1e-6) / 3600) if not args.toplam_saat else \
                    max(0.0, args.toplam_saat - train_sec() / 3600)
                D.update({"iter": it, "asama": a, "lr": lr_now, "hiz": hiz, "egitim_sa": train_sec() / 3600,
                          "duvar_sa": (time.time() - t0) / 3600, "kalan_sa": kalan, "best": best, "best_iter": best_iter,
                          "kayip": dict(zip(("total", "heat", "offset", "size", "h47"), mv)), "durum": durum,
                          "guncellendi": time.strftime("%d.%m %H:%M:%S"), "son_kayit": D.get("son_kayit", "—"),
                          "gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "bos_ram_gb": bos_ram() / 2 ** 30,
                          "gpu_onbellek_gb": torch.cuda.memory_reserved() / 2 ** 30,
                          "gpu_onbellek_tepe_gb": torch.cuda.max_memory_reserved() / 2 ** 30,
                          "ram_deposu": depo is not None})
                if it % 100 == 0:
                    log(f"it {it}/{args.iters} aşama {a} kayıp {mv[0]:.3f} (ısı {mv[1]:.3f}, ofset {mv[2]:.3f}, boyut "
                        f"{mv[3]:.3f}, H47 {mv[4]:.3f}) lr {lr_now:.2e} {hiz:.1f} it/s")
                    logw.writerow([it, a, f"{lr_now:.3e}", *[f"{x:.4f}" for x in mv], "", "", "", "", round(train_sec())])
                    logc.flush()
                yaz_durum(out, D)
            if it % args.eval_every == 0:
                eval_spent += degerlendir()
                ts = time.time()
                son_kayit()
                eval_spent += time.time() - ts
                yaz_durum(out, D)
                t_win, it_win = time.time(), it                          # hız ölçümüne değerlendirme süresi girmesin
            if time.time() - t_ckpt > args.ckpt_minutes * 60:
                ts = time.time()
                son_kayit()
                eval_spent += time.time() - ts
                t_ckpt = time.time()
    it_ = getattr(loader, "_iterator", None)                            # §89d: son ölçümlerden önce işçileri ve depoyu bırak
    if it_ is not None:
        it_._shutdown_workers()
    if depo is not None:
        depo.kapat()
        depo = None
    if D.get("son_degerlendirme", {}).get("iter") != it:
        eval_spent += degerlendir()
    D["son_durağan_vval"] = vval(ema, va_vids, args, aralik, dev, durağan=True)   # ağırlık etkisi ↔ zamanlı işleme etkisi
    log(f"  son model durağan (her kare tek başına) video val: {D['son_durağan_vval']}")
    son_kayit()
    D.update({"durum": "BİTTİ", "guncellendi": time.strftime("%d.%m %H:%M:%S"), "iter": it, "egitim_sa": train_sec() / 3600})
    yaz_durum(out, D)
    log(f"eğitim bitti: iterasyon {it}, eğitim süresi {train_sec() / 3600:.2f} sa, en iyi video val {best:.4f} ({best_iter} it)")
    if depo is not None:
        depo.kapat()
    return 0


if __name__ == "__main__":
    sys.exit(main())
