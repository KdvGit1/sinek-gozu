#!/usr/bin/env python3
"""§95 son model ölçümleri: son termal model (uzun_75/best.pt ya da agirliklar/uzun_75_termal.pt) nasıl çalışıyor? Yalnız val kümesi (test kullanılmaz).
Alt komutlar:
  envanter   CPU: konnektom bileşimi, göz geometrisi, parametre sayıları, öğrenilmiş parametre dağılımları, başlık ağırlıkları
  dinamik    GPU: val karolarında adım adım (1–30) bölge etkinliği, varış, dengelenme; okuma özelliklerinin insan seçiciliği
             (AUC); adım adım ısı ayrımı; örnek karo haritaları (.npz)
  adim       GPU: çıkarımda adım sayısı T taraması (val800 mAP50)
  ablasyon   GPU: çıkarım ablasyonları (val800 mAP50)
  hiz        GPU: sinek ve YOLOv8n hız, hesap, parametre, etkinlik sayısı
  boyut      CPU: önbellekteki val tespitlerinden boyuta göre duyarlılık ve kutu kalitesi; veri kümesi görüntü / insan boyutları
Çıktılar results/rapor/95_veri/<ad>.json. Kod çözme merkez7p2, bindirme 0,5 (§52; final_eval ile aynı).
Kullanım: py -3 tools/olcum_95_son_model.py <alt komut> [...]"""
from __future__ import annotations

import os

os.environ.setdefault("FLYBRAIN_DECODE", "merkez7p2")
os.environ.setdefault("FLYBRAIN_OVERLAP", "0.5")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import flybrain.detection as det  # noqa: E402
import tools.train_flybrain as tf  # noqa: E402
from flybrain.connectome import Connectome  # noqa: E402
from flybrain.detection import BilateralTargets, TargetBuilder, box_iou, decode_cpu, merge_top  # noqa: E402
from tools.train_flybrain import build, dataset_of, evaluate, list_images, load_boxes, load_image  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RUN = ROOT / "results" / "flybrain"
OUT = ROOT / "results" / "rapor" / "95_veri"
CACHE = ROOT / "results" / "son_degerlendirme"
ZONES = ["retina", "lamina", "medulla", "lobula plakası", "lobula", "merkez beyin"]
# gorsellestirici.py ile aynı bölge eşlemesi (FlyWire ailesi → nöropil)
FAMILY_ZONE = {
    "Photo Receptors": 0,
    "Lamina Monopolar": 1, "Lamina Wide Field": 1, "Lamina Intrinsic": 1, "Lamina Tangential": 1, "Centrifugal": 1,
    "T1 Neuron": 1,
    "Transmedullary": 2, "Medulla Intrinsic": 2, "Distal Medulla": 2, "Distal Medulla Dorsal Rim Area": 2,
    "Proximal Medulla": 2, "Serpentine Medulla": 2, "Transmedullary Y": 2, "T2 Neuron": 2, "T3 Neuron": 2, "MeTu": 2,
    "MT": 2, "Medulla Lobula Tangential": 2, "Lobula Medulla Amacrine": 2, "Lobula Medulla Tangential": 2,
    "Proximal Distal Medulla Tangential": 2, "aMe": 2, "MeMe": 2, "cM": 2, "cML": 2, "MC": 2, "MeLp": 2,
    "Medulla Lobula Lobula Plate Amacrine": 2, "Weirdos": 2,
    "T4 Neuron": 3, "T5 Neuron": 3, "Lobula Plate Intrinsic": 3, "Translobula Plate": 3, "LPC": 3, "cLP": 3, "LPT": 3,
    "VS": 3, "HS": 3, "H": 3, "Lobula Lobula Plate Tangential": 3, "cLLP": 3, "cLLPM": 3, "cMLLP": 3, "LPLC": 3,
    "Y Neuron": 3,
    "LC": 4, "Lobula Intrinsic": 4, "LT": 4, "cL": 4, "cLM": 4, "LLPC": 4,
}


def zone_of(fam: str, cat: str) -> int:
    if cat == "central":
        return 5
    if fam in FAMILY_ZONE:
        return FAMILY_ZONE[fam]
    s = str(fam)
    if "Lobula Plate" in s:
        return 3
    if "Lobula" in s:
        return 4
    if "Lamina" in s:
        return 1
    if "Medulla" in s:
        return 2
    return 5 if cat == "boundary" else 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kaydet(ad: str, veri) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{ad}.json").write_text(json.dumps(veri, ensure_ascii=False, indent=1,
                                                    default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    log(f"yazıldı: 95_veri/{ad}.json")


def q(x, ps=(0, 5, 25, 50, 75, 95, 100)) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} | {"ort": float(x.mean()), "n": int(x.size)}


def yukle(dev, ad: str = "uzun_75"):
    """Eğitim klasörü results/flybrain/<ad>/best.pt yoksa sıkıştırılmış ağırlık agirliklar/uzun_75_termal.pt."""
    yol = RUN / ad / "best.pt"
    if not yol.is_file() and ad == "uzun_75":
        from tools.agirlik import yukle as yukle_sik
        sik = ROOT / "agirliklar" / "uzun_75_termal.pt"
        p = torch.load(sik, map_location="cpu", weights_only=False)
        m = yukle_sik(sik, dev)
        sd = m.state_dict()
        return m, {"args": p["args"], "model": {k: v.cpu() for k, v in sd.items()}, "iter": p["iterasyon"],
                   "ema_agirlik": p["ema"]}
    ck = torch.load(yol, map_location="cpu", weights_only=False)
    m = build(argparse.Namespace(**ck["args"]), dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def bolgeler(m, c: Connectome) -> np.ndarray:
    nd = c.nodes
    return np.array([zone_of(fa, ca) for fa, ca in zip(nd.family.fillna("?").astype(str),
                                                       nd.category.fillna("?").astype(str))], np.int64)


def val800() -> list[Path]:
    """Eğitimdeki ara değerlendirmeyle aynı alt küme (train_flybrain: random.Random(0).sample(val, 800))."""
    return random.Random(0).sample(list_images("thermal", "val"), 800)


# ============================================================================ envanter (CPU)
def envanter(a):
    m, ck = yukle(torch.device("cpu"))
    c = Connectome.load(m.cfg.processed_dir if Path(m.cfg.processed_dir).is_dir()
                        else ROOT / "data/connectome/processed/bi_ol_merkez_thr5")
    nd = c.nodes
    z = bolgeler(m, c)
    N, E = c.n_neurons, c.n_edges
    sign_n = nd.sign.to_numpy()
    sign_e = c.sign()
    is_out = nd.is_output.to_numpy().astype(bool)
    indeg = np.bincount(c.post, minlength=N)
    outdeg = np.bincount(c.pre, minlength=N)
    girdi = set(m.pr_idx.tolist())
    okunan = set(m.rc_col.tolist()) | set(m.ro_col.tolist())
    bolge = {}
    for k, ad in enumerate(ZONES):
        s = z == k
        bolge[ad] = {"noron": int(s.sum()), "tip": int(nd.type[s].nunique()), "uyarici": int((sign_n[s] > 0).sum()),
                     "baskilayici": int((sign_n[s] < 0).sum()), "vpn": int((is_out & s).sum()),
                     "girdi_alan": int(sum(1 for i in np.flatnonzero(s) if i in girdi)),
                     "okunan": int(sum(1 for i in np.flatnonzero(s) if i in okunan)),
                     "kenar_girdi": int(np.isin(c.post, np.flatnonzero(s)).sum())}
    nt = nd.type_nt.astype(str).value_counts().to_dict()
    pk = m.pr_kind.numpy()
    kinds = ["R1-6", "R7", "R8", "L1", "L2", "L3"]
    eye = m.eye
    ry, rx, rh, rw = eye.inscribed_rect()
    env = {
        "konnektom": {"noron": N, "kenar": E, "sinaps": float(c.syn.sum()), "tip": c.n_types, "alfa_grubu": c.n_pairs,
                      "goz": {k: int(v) for k, v in nd.eye.fillna("-").value_counts().items()},
                      "kategori": {k: int(v) for k, v in nd.category.value_counts().items()},
                      "noro_iletici": {k: int(v) for k, v in nt.items()},
                      "uyarici_kenar_orani": float((sign_e > 0).mean()),
                      "sinaps_kenar_basina": q(c.syn), "sinaps_255_ustu": int((c.syn > 255).sum()),
                      "girdi_derecesi": q(indeg), "cikti_derecesi": q(outdeg),
                      "girdisiz_noron": int((indeg == 0).sum()), "ciktisiz_noron": int((outdeg == 0).sum()),
                      "meta": c.meta},
        "bolgeler": bolge,
        "goz": {"kolon_goz_basina": eye.n_columns, "goz": m.n_eyes, "kolon_toplam": m.n_columns, "yama_px": eye.patch,
                "karo_yukseklik": eye.height, "karo_genislik": eye.width, "ic_dikdortgen": [ry, rx, rh, rw],
                "ayak_izi_piksel": int(eye.footprint().sum()), "karo_piksel": eye.height * eye.width},
        "girdi": {"girdi_noronu": int(m.pr_idx.numel()), "tur": {k: int((pk == i).sum()) for i, k in enumerate(kinds)},
                  "kodlayici_girdi": int(m.encoder[0].in_features), "kodlayici_gizli": int(m.encoder[0].out_features),
                  "kodlayici_cikti": int(m.to_photo.out_features),
                  "bakis_dagilimi": {int(k): int(v) for k, v in Counter(m.pr_view.tolist()).items()}},
        "okuma": {"kolonsal_tip": len(m.columnar_types), "cikis_tipi": len(m.output_types), "ozellik": m.n_features,
                  "okunan_noron": len(okunan), "girdi_okunan_kesisim": len(girdi & okunan),
                  "kolonsal_nnz": int(m.rc_col.numel()), "cikis_nnz": int(m.ro_col.numel()),
                  "vpn": int(is_out.sum()), "merkez": int(len(c.central_neurons))},
    }
    # parametreler
    sd = ck["model"]
    pars = {k: {"n": int(v.numel()), "sekil": list(v.shape)} for k, v in m.named_parameters()}
    egitilen = {k: p.requires_grad for k, p in m.named_parameters()}
    env["parametre"] = {"tensor": pars, "egitilen_toplam": int(sum(p.numel() for p in m.parameters() if p.requires_grad)),
                        "ozet": m.n_parameters(), "egitilen_bayrak": egitilen,
                        "en_iyi_iter": ck.get("iter"), "ema": bool(ck.get("ema_agirlik", False))}
    # öğrenilmiş parametreler
    sp = torch.nn.functional.softplus
    alpha = sp(m.alpha_raw.detach()).numpy()
    pair = m.pair.numpy()
    lam = torch.sigmoid(m.leak_raw.detach()).numpy()
    tip_say = np.bincount(m.type_id.numpy(), minlength=c.n_types)
    tip_ad = nd.groupby("type_id").type.first().reindex(range(c.n_types)).fillna("?").to_numpy()
    tau = 1 / lam
    canli = tip_say > 0
    sira = np.argsort(tau)
    w = m.weights().detach().numpy()
    gir_guc = np.bincount(c.post, weights=np.abs(w), minlength=N)
    gir_exc = np.bincount(c.post, weights=np.clip(w, 0, None), minlength=N)
    gir_inh = np.bincount(c.post, weights=np.clip(-w, 0, None), minlength=N)
    ogr = {"alfa_kazanc_grup": q(alpha), "alfa_kazanc_kenar": q(alpha[pair]), "alfa_baslangic": float(m.cfg.init_gain),
           "alfa_%10_degisen_grup": float((np.abs(alpha / m.cfg.init_gain - 1) > 0.1).mean()),
           "alfa_%10_degisen_kenar": float((np.abs(alpha[pair] / m.cfg.init_gain - 1) > 0.1).mean()),
           "noron_girdi_guc_abs": q(gir_guc), "noron_girdi_uyarici": q(gir_exc), "noron_girdi_baskilayici": q(gir_inh),
           "bias": q(m.bias.detach().numpy()), "bias_baslangic": float(m.cfg.init_bias),
           "lambda": q(lam[canli]), "tau": q(tau[canli]),
           "tau_noron_agirlikli": q(np.repeat(tau, tip_say)),
           "tau_en_yavas": [[str(tip_ad[i]), float(tau[i]), int(tip_say[i])] for i in sira[::-1] if canli[i]][:15],
           "tau_en_hizli": [[str(tip_ad[i]), float(tau[i]), int(tip_say[i])] for i in sira if canli[i]][:15]}
    tipler_ilgi = ["R1-6", "R7", "R8", "L1", "L2", "L3", "L4", "L5", "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm4",
                   "Tm9", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "LC11", "LC4", "LPLC2", "LC18", "HSE",
                   "VSm", "C2", "C3", "Dm3", "T2", "T3"]
    adx = {str(t): i for i, t in enumerate(tip_ad)}
    ogr["tau_secili"] = {t: float(tau[adx[t]]) for t in tipler_ilgi if t in adx}
    if hasattr(m, "stp_phi"):
        phi = torch.tanh(m.stp_phi.detach()).numpy()
        taus = (1 + sp(m.stp_tau.detach())).numpy()
        sp_ = np.argsort(phi)
        ogr["stp"] = {"phi": q(phi[canli]), "phi_noron_agirlikli": q(np.repeat(phi, tip_say)),
                      "kolaylasan_tip_(phi>0,05)": int(((phi > 0.05) & canli).sum()),
                      "yorulan_tip_(phi<-0,05)": int(((phi < -0.05) & canli).sum()),
                      "kolaylasan_noron": int(tip_say[phi > 0.05].sum()), "yorulan_noron": int(tip_say[phi < -0.05].sum()),
                      "tau_s": q(taus[canli]),
                      "en_kolaylasan": [[str(tip_ad[i]), float(phi[i]), int(tip_say[i])] for i in sp_[::-1] if canli[i]][:12],
                      "en_yorulan": [[str(tip_ad[i]), float(phi[i]), int(tip_say[i])] for i in sp_ if canli[i]][:12],
                      "secili": {t: float(phi[adx[t]]) for t in tipler_ilgi if t in adx}}
    if hasattr(m, "mod_beta"):
        mb = m.mod_beta.detach().numpy()
        mt = np.unique(m.type_id.numpy()[m.mod_idx.numpy()])
        so = np.argsort(-np.abs(mb))
        ogr["modulator"] = {"modulator_noron": int(m.mod_idx.numel()), "modulator_tip": int(m.n_mod_types),
                            "cift": int(mb.size), "beta": q(mb), "abs_beta_0,05_ustu": int((np.abs(mb) > 0.05).sum()),
                            "en_guclu": [[str(tip_ad[mt[m.modp_j[i]]]), str(tip_ad[m.modp_k[i]]), float(mb[i])]
                                         for i in so[:12]],
                            "tipler": [str(tip_ad[t]) for t in mt]}
    ogr["kolon_uyumu_c"] = float(sp(m.adapt_c.detach()))
    # kodlayıcı: ham ve z yarısı (girdi sırası: önce p² ham, sonra p² z)
    W1 = m.encoder[0].weight.detach().numpy()
    p2 = m.cfg.patch ** 2
    ogr["kodlayici"] = {"ham_yari_abs": float(np.abs(W1[:, :p2]).sum()), "z_yari_abs": float(np.abs(W1[:, p2:]).sum()),
                        "to_photo_agirlik": m.to_photo.weight.detach().numpy().round(4).tolist(),
                        "to_photo_bias": m.to_photo.bias.detach().numpy().round(4).tolist()}
    # başlık: LayerNorm (afin) + doğrusal → ısı için etkin ağırlık; komşu kolu
    ln, lin = m.head[0], m.head[1]
    weff = (ln.weight.detach() * lin.weight.detach()).numpy()                 # [5, F]
    beff = (lin.weight.detach() @ ln.bias.detach() + lin.bias.detach()).numpy()
    nct = len(m.columnar_types)
    adlar = list(m.columnar_types) + list(m.output_types)
    tip_bolge = {}
    for t, zz in zip(nd.type.astype(str), z):
        tip_bolge.setdefault(t, []).append(zz)
    fb = [ZONES[Counter(tip_bolge.get(n, [5])).most_common(1)[0][0]] for n in adlar]
    sira = np.argsort(-np.abs(weff[0]))
    nbw = m.head_nb.weight.detach().numpy().reshape(5, 6, nct)
    ogr["baslik"] = {"ozellik_adi": adlar, "ozellik_bolge": fb, "kolonsal_say": nct,
                     "isi_agirlik_etkin": weff[0].round(5).tolist(), "isi_bias_etkin": float(beff[0]),
                     "cikti_bias": lin.bias.detach().numpy().round(4).tolist(),
                     "grup_norm": {ad: [float(np.linalg.norm(weff[j, :nct])), float(np.linalg.norm(weff[j, nct:])),
                                        float(np.linalg.norm(nbw[j]))] for j, ad in
                                   enumerate(["isi", "ofset_y", "ofset_x", "log_h", "log_w"])},
                     "komsu_yon_norm": [[float(np.linalg.norm(nbw[j, d])) for d in range(6)] for j in range(5)],
                     "isi_en_buyuk": [[adlar[i], fb[i], float(weff[0, i])] for i in sira[:20]]}
    env["ogrenilen"] = ogr
    kaydet("envanter", env)


# ============================================================================ dinamik (GPU)
def karolar_sec(m, n_pos: int, n_neg: int, tohum: int = 0):
    """val800'den eşit aralıklı görüntüler: her görüntüden en çok insan merkezini içeren karo (pozitif) ya da
    rastgele bir karo (negatif). Hedefler eğitimdekiyle aynı (BilateralTargets)."""
    imgs = val800()
    tiler = tf.tiler_for(m, "replicate")
    tb = BilateralTargets(TargetBuilder(m.eye), m.eye.width)
    rng = random.Random(tohum)
    pos, neg = [], []
    for p in imgs:
        if len(pos) >= n_pos and len(neg) >= n_neg:
            break
        img = load_image(p, 1)
        H, W = img.shape[1:]
        gts = load_boxes(p, H, W)
        tiles, org = tiler.extract(img)
        best, bk = 0, None
        for k, (oy, ox) in enumerate(org):
            b = gts.clone()
            if len(b):
                b[:, 0] -= oy
                b[:, 1] -= ox
            n = int(tb.tb([b])[2].sum()) if len(b) else 0
            if n > best:
                best, bk = n, (k, b)
        if bk is not None and len(pos) < n_pos:
            pos.append((tiles[bk[0]], bk[1], dataset_of(p), p.name, org[bk[0]]))
        elif bk is None and len(neg) < n_neg:
            k = rng.randrange(len(org))
            neg.append((tiles[k], torch.zeros(0, 4), dataset_of(p), p.name, org[k]))
    secim = pos + neg
    t = torch.stack([s[0] for s in secim])
    hedef = tb([s[1] for s in secim])
    return t, hedef, secim


def auc(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """[F, n1], [F, n0] → [F] Mann–Whitney AUC = P(poz > neg) + ½·P(poz = neg). Eşitlikler yarım sayılır (alıcı alanı
    kolona ulaşmayan çıkış tipleri o kolonda sabit 0; ilk sürümde argsort eşitlikleri pozitife aleyhte sıralıyordu)."""
    ns = neg.sort(1).values.contiguous()
    pos = pos.contiguous()
    lo = torch.searchsorted(ns, pos, right=False).double()
    hi = torch.searchsorted(ns, pos, right=True).double()
    return ((lo + hi) / 2).mean(1).div(neg.shape[1]).float()


@torch.no_grad()
def dinamik(a):
    dev = torch.device("cuda")
    m, ck = yukle(dev)
    c = Connectome.load(ROOT / "data/connectome/processed/bi_ol_merkez_thr5")
    z = torch.as_tensor(bolgeler(m, c), device=dev)
    nd = c.nodes
    is_out = torch.as_tensor(nd.is_output.to_numpy().astype(bool), device=dev)
    okunan = torch.zeros(m.n_neurons, dtype=torch.bool, device=dev)
    okunan[m.rc_col] = True
    okunan[m.ro_col] = True
    girdi = torch.zeros_like(okunan)
    girdi[m.pr_idx] = True
    gruplar = {ad: z == k for k, ad in enumerate(ZONES)}
    gruplar |= {"VPN (çıkış)": is_out, "okunan": okunan, "girdi alan": girdi,
                "hepsi": torch.ones_like(okunan)}
    if a.yama:                                                           # ablasyonlu dinamik (yalnız grup + ısı)
        with yama(m, a.yama):
            return _dinamik(a, m, c, z, gruplar)
    return _dinamik(a, m, c, z, gruplar)


@torch.no_grad()
def _dinamik(a, m, c, z, gruplar):
    dev = z.device
    tiles, hedef, secim = karolar_sec(m, a.pozitif, a.negatif)
    heat_t, _, pos_m = hedef[0].to(dev), hedef[1], hedef[2].to(dev)
    neg_m = heat_t < 0.01                                                # insandan uzak kolonlar
    log(f"{len(secim)} karo ({int(pos_m.sum())} pozitif kolon, {int(neg_m.sum())} uzak negatif kolon); "
        f"setler {Counter(s[2] for s in secim)}")
    tiles = tiles.to(dev)
    drive = m.drive(tiles)                                               # [8, B, C, 6]
    T = a.adim
    v, s = None, None
    B = tiles.shape[0]
    N = m.n_neurons
    ilk_var = torch.full((N,), -1, dtype=torch.long, device=dev)        # karolar arası std > 1e-3 olduğu ilk adım
    hic_etkin = torch.zeros(N, dtype=torch.bool, device=dev)
    a_once = torch.zeros(N, B, device=dev)
    sonuc = {ad: [] for ad in gruplar}
    isi_auc, ozellik_auc, isi_ort = [], [], []
    g = torch.Generator(device="cpu").manual_seed(0)
    negi = torch.nonzero(neg_m.flatten()).squeeze(1)
    negi = negi[torch.randperm(len(negi), generator=g)[:20000].to(dev)]
    posi = torch.nonzero(pos_m.flatten()).squeeze(1)
    # örnek karo: HIT-UAV'dan en çok insanlı pozitif (yoksa ilk pozitif)
    adaylar = [i for i, s_ in enumerate(secim) if s_[2] == "hit_uav" and len(s_[1])]
    ornek = max(adaylar, key=lambda i: len(secim[i][1])) if adaylar else 0
    Ce = m.n_eye_columns
    col = torch.as_tensor(c.neuron_column, device=dev)
    zc = {}                                                              # bölge → (nöron, kolon) sağ göz
    for k, ad in enumerate(ZONES[:5]):
        sel = (z == k) & (col >= 0) & (col < Ce)
        zc[ad] = (torch.nonzero(sel).squeeze(1), col[sel])
    ornek_harita = {ad: [] for ad in zc}
    ornek_isi = []
    for t in range(1, T + 1):
        v, s = m.core(drive, state=v, steps=1, stp_state=s, return_stp=True)
        act = m._act(v)                                                 # [N, B] tanh(relu(v))
        sd = act.std(1)
        yeni = (ilk_var < 0) & (sd > 1e-3)
        ilk_var[yeni] = t
        hic_etkin |= (act > 0.01).any(1)
        d = (act - a_once).abs().mean(1)
        a_once = act
        for ad, msk in gruplar.items():
            x = act[msk]
            sonuc[ad].append({"adim": t, "etkin_ort": float(x.mean()), "etkin_oran_>0,01": float((x > 0.01).float().mean()),
                              "doygun_oran_>0,9": float((x > 0.9).float().mean()), "gerilim_ort": float(v[msk].mean()),
                              "gerilim_negatif_oran": float((v[msk] < 0).float().mean()), "degisim_ort": float(d[msk].mean()),
                              "karo_std_ort": float(sd[msk].mean())})
        f = m.readout(v)                                                # [B, C, F]
        out = m.head_forward(f)
        h = out[..., 0].flatten()
        isi_auc.append(float(auc(h[posi][None], h[negi][None])[0]))
        pr = torch.sigmoid(h)
        isi_ort.append([float(pr[posi].mean()), float(pr[negi].mean())])
        ff = f.reshape(-1, f.shape[-1]).T                               # [F, B·C]
        ozellik_auc.append(auc(ff[:, posi], ff[:, negi]).cpu().numpy().round(4).tolist())
        if t <= 12:
            for ad, (ni, ci) in zc.items():
                hm = torch.zeros(Ce, device=dev).index_add_(0, ci, act[ni, ornek])
                cnt = torch.zeros(Ce, device=dev).index_add_(0, ci, torch.ones_like(ci, dtype=torch.float))
                ornek_harita[ad].append((hm / cnt.clamp(min=1)).cpu().numpy())
            ornek_isi.append(torch.sigmoid(out[ornek, :Ce, 0]).cpu().numpy())
            ro = m._readout_mm("ro", act).reshape(m.n_columns, -1, act.shape[1])     # VPN etkinliği, alıcı alan ağırlıklı
            ornek_harita.setdefault("vpn", []).append(ro[:Ce, :, ornek].mean(1).cpu().numpy())
        if t == 10:                                                     # eğitimdeki okuma adımı: tespitler
            o10 = m._pack(out, v)
            dets10 = decode_cpu({k_: (v_[ornek:ornek + 1] if torch.is_tensor(v_) and v_.dim() > 1 and k_ != "state"
                                      else v_) for k_, v_ in o10.items() if k_ != "state"}, m.eye, 0.01, 100)[0]
        log(f"adım {t}: ısı AUC {isi_auc[-1]:.3f}, hepsi etkin {sonuc['hepsi'][-1]['etkin_ort']:.3f}")
    varis = {}
    for ad, msk in gruplar.items():
        iv = ilk_var[msk]
        ok = iv > 0
        varis[ad] = {"noron": int(msk.sum()), "farklilasan_oran": float(ok.float().mean()),
                     "ilk10_oran": float(((iv > 0) & (iv <= 10)).float().mean()),
                     "varis_ortanca": float(iv[ok].float().median()) if ok.any() else None,
                     "varis_ort": float(iv[ok].float().mean()) if ok.any() else None,
                     "hic_etkinlesmeyen_oran": float((~hic_etkin[msk]).float().mean()),
                     "varis_dagilimi": torch.bincount(iv[ok], minlength=T + 1)[1:].tolist() if ok.any() else []}
    # sürüş istatistikleri (kodlayıcı çıktısı, tür başına; gerçekten girdi alan bakışlar)
    kinds = ["R1-6", "R7", "R8", "L1", "L2", "L3"]
    surus = {}
    for kk, ad in enumerate(kinds):
        vw = list(range(1, 7)) if kk == 0 else ([0] if kk in (1, 2) else [7])
        x = drive[vw][..., kk]
        surus[ad] = {"ort": float(x.mean()), "std": float(x.std()), "p5": float(x.quantile(0.05)) if x.numel() < 2 ** 24
                     else None, "p95": float(x.quantile(0.95)) if x.numel() < 2 ** 24 else None,
                     "pozitif_kolon_ort": float(drive[vw][..., kk].mean(0).reshape(-1)[posi].mean()),
                     "uzak_kolon_ort": float(drive[vw][..., kk].mean(0).reshape(-1)[negi].mean())}
    kor = torch.corrcoef(drive[0].reshape(-1, 6).T).cpu().numpy().round(3).tolist()   # aynı yamada 6 kodlayıcı çıktısı
    son = np.array(ozellik_auc[9])
    adlar = list(m.columnar_types) + list(m.output_types)
    sira = np.argsort(-np.abs(son - 0.5))
    veri = {"karo": len(secim), "pozitif_kolon": int(len(posi)), "negatif_kolon_ornek": int(len(negi)),
            "setler": dict(Counter(s_[2] for s_ in secim)), "adim_ust": T, "egitim_adimi": int(m.cfg.steps),
            "grup": sonuc, "varis": varis, "isi_auc": isi_auc, "isi_olasilik_poz_neg": isi_ort,
            "ozellik_adi": adlar, "ozellik_auc": ozellik_auc,
            "en_secici_10": [[adlar[i], float(son[i])] for i in sira[:30]],
            "surus": surus, "surus_kor_merkez_bakis": kor,
            "ornek": {"indeks": ornek, "set": secim[ornek][2], "goruntu": secim[ornek][3], "koken": secim[ornek][4],
                      "insan": len(secim[ornek][1])}}
    if a.yama:
        slug = a.yama.lower().translate(str.maketrans("çğıöşüλα", "cgiosula")).replace(" ", "_")
        slug = "".join(ch for ch in slug if ch.isalnum() or ch == "_")
        kaydet(f"dinamik_{slug}", {k: veri[k] for k in ("karo", "pozitif_kolon", "grup", "varis", "isi_auc",
                                                       "isi_olasilik_poz_neg")} | {"yama": a.yama})
        return
    kaydet("dinamik", veri)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "ornek_karo.npz", karo=tiles[ornek].cpu().numpy(), kutular=secim[ornek][1].numpy(),
                        tespit=dets10.numpy(), isi=np.stack(ornek_isi),
                        surus=drive[:, ornek, :Ce].cpu().numpy(), **{f"h_{k}": np.stack(v_) for k, v_ in ornek_harita.items()},
                        top=m.eye.top, left=m.eye.left, patch=m.eye.patch)
    log("yazıldı: 95_veri/ornek_karo.npz")


# ============================================================================ adım taraması ve ablasyonlar (GPU)
def degerlendir(m, imgs, dev):
    r = evaluate(m, imgs, 1, dev, pad_mode="replicate")
    return {"mAP50": r["mAP50"], "kesinlik@0,3": r["precision@0.3"], "duyarlilik@0,3": r["recall@0.3"],
            "set": {k: v["mAP50"] for k, v in r["by_dataset"].items()}, "sn": r["seconds"]}


def adim(a):
    dev = torch.device("cuda")
    m, _ = yukle(dev)
    imgs = val800()
    f = OUT / "adim_taramasi.json"
    veri = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {"goruntu": len(imgs), "sonuc": {}}
    for T in [int(x) for x in a.tler.split(",")]:
        if str(T) in veri["sonuc"]:
            continue
        m.cfg.steps = T
        t0 = time.time()
        veri["sonuc"][str(T)] = degerlendir(m, imgs, dev)
        log(f"T = {T}: mAP50 {veri['sonuc'][str(T)]['mAP50']:.4f} ({time.time() - t0:.0f} sn)")
        kaydet("adim_taramasi", veri)
    m.cfg.steps = 10


@contextmanager
def yama(m, ad: str):
    """Çıkarım ablasyonu: modeli geçici değiştirir, sonra eski hâline getirir."""
    sakla = {k: v.detach().clone() for k, v in m.state_dict().items()}
    eski_drive = m.drive
    eski_dec, eski_ov = det.DECODE_MODE, tf.EVAL_OVERLAP
    sp_inv = lambda y: math.log(math.expm1(y))
    with torch.no_grad():
        if ad == "komşu kolu kapalı":
            m.head_nb.weight.zero_()
        elif ad == "kod çözme taban":
            det.DECODE_MODE = "taban"
        elif ad == "bindirme 0,25":
            tf.EVAL_OVERLAP = 0.25
        elif ad == "STP kapalı":
            m.stp_phi.zero_()
        elif ad == "modülatör kapalı":
            m.mod_beta.zero_()
        elif ad == "λ başlangıca (τ = 2)":
            m.leak_raw.zero_()
        elif ad == "α başlangıca (kazanç 2)":
            m.alpha_raw.fill_(sp_inv(m.cfg.init_gain))
        elif ad == "bias başlangıca (0,1)":
            m.bias.fill_(m.cfg.init_bias)
        elif ad in ("süperpozisyon yok", "R1-6 girdisi yok", "R7 + R8 girdisi yok", "lamina girdisi yok",
                    "yalnız lamina girdisi"):
            def yeni(tiles, f=eski_drive, ad=ad):
                d = f(tiles).clone()
                if ad == "süperpozisyon yok":                          # bütün R1-6 merkezden; L = 6 × merkez
                    d[1:7] = d[0]
                    d[7] = 6 * d[0]
                elif ad == "R1-6 girdisi yok":
                    d[..., 0] = 0
                elif ad == "R7 + R8 girdisi yok":
                    d[..., 1:3] = 0
                elif ad == "lamina girdisi yok":
                    d[..., 3:6] = 0
                else:
                    d[..., 0:3] = 0
                return d
            m.drive = yeni
        elif ad != "referans":
            raise ValueError(ad)
    try:
        yield
    finally:
        m.load_state_dict(sakla)
        if "drive" in m.__dict__:
            del m.__dict__["drive"]
        det.DECODE_MODE, tf.EVAL_OVERLAP = eski_dec, eski_ov


ABLASYONLAR = ["referans", "komşu kolu kapalı", "kod çözme taban", "bindirme 0,25", "süperpozisyon yok",
               "R1-6 girdisi yok", "R7 + R8 girdisi yok", "lamina girdisi yok", "yalnız lamina girdisi",
               "STP kapalı", "modülatör kapalı", "λ başlangıca (τ = 2)", "α başlangıca (kazanç 2)", "bias başlangıca (0,1)"]


def ablasyon(a):
    dev = torch.device("cuda")
    m, _ = yukle(dev)
    imgs = val800()
    f = OUT / "ablasyon.json"
    veri = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {"goruntu": len(imgs), "sonuc": {}}
    for ad in ABLASYONLAR:
        if ad in veri["sonuc"]:
            continue
        t0 = time.time()
        with yama(m, ad):
            veri["sonuc"][ad] = degerlendir(m, imgs, dev)
        log(f"{ad}: mAP50 {veri['sonuc'][ad]['mAP50']:.4f} ({time.time() - t0:.0f} sn)")
        kaydet("ablasyon", veri)


# ============================================================================ hız ve hesap (GPU)
@torch.no_grad()
def hiz(a):
    dev = torch.device("cuda")
    m, _ = yukle(dev)
    imgs = val800()[:a.goruntu]
    tiler = tf.tiler_for(m, "replicate")
    # karo sayıları ve uçtan uca süre (tek süreç; karolama dahil)
    sure, karo_say = [], {}
    yuk = [(p, load_image(p, 1)) for p in imgs]
    torch.cuda.synchronize()
    for p, img in yuk[:3]:                                              # ısınma
        t_, o_ = tiler.extract(img)
        m(t_.to(dev))
    torch.cuda.synchronize()
    t0 = time.time()
    for p, img in yuk:
        t1 = time.time()
        tl, org = tiler.extract(img)
        dets = []
        for i in range(0, len(tl), 256):
            dets += decode_cpu(m(tl[i:i + 256].to(dev)), m.eye, 0.01, 100)
        merge_top(dets, org, img.shape[1], img.shape[2], 0.5, 300, True)
        torch.cuda.synchronize()
        sure.append(time.time() - t1)
        karo_say.setdefault(dataset_of(p), []).append(len(tl))
    sinek_goruntu = (time.time() - t0) / len(yuk)
    # aşama dökümü: 256 karoluk yığın
    tl = torch.cat([tiler.extract(img)[0] for _, img in yuk[:8]])[:256].to(dev)
    asama = {}

    def zaman(fn, n=5):
        fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t1 = time.time()
            r = fn()
            torch.cuda.synchronize()
            ts.append(time.time() - t1)
        return float(np.median(ts)), r
    asama["surus (göz + kodlayıcı, 7 bakış × 2 göz)"], d = zaman(lambda: m.drive(tl))
    asama["çekirdek (T = 10)"], v = zaman(lambda: m.core(d))
    asama["okuma + başlık"], out = zaman(lambda: m._pack(m.head_forward(m.readout(v)), v))
    asama["kod çözme (CPU'ya aktarım dahil)"], _ = zaman(lambda: decode_cpu(out, m.eye, 0.01, 100))
    asama = {k: v_ / len(tl) for k, v_ in asama.items()}
    # analitik hesap (çarpma-toplama, karo başına)
    C, N, E, T = m.n_columns, m.n_neurons, int(m.col.numel()), m.cfg.steps
    k_in = m.encoder[0].in_features
    hesap = {"kodlayici": 7 * C * (k_in * m.encoder[0].out_features + m.encoder[0].out_features * m.to_photo.out_features),
             "cekirdek_sinaps": E * T, "cekirdek_eleman": 12 * N * T,
             "okuma": int(m.rc_col.numel() + m.ro_col.numel()),
             "baslik": C * (m.n_features * 5 + 6 * len(m.columnar_types) * 5)}
    hesap["toplam"] = sum(hesap.values())
    # YOLOv8n
    from ultralytics import YOLO
    y = YOLO(str(ROOT / "results/baseline/yolov8n_thermal/weights/best.pt"))
    info = y.info(detailed=False, verbose=False)
    ym = y.model.to(dev).eval()
    sekiller = []
    hooks = [lay.register_forward_hook(lambda mod, i, o, s=sekiller: s.append(
        (type(mod).__name__, list(o.shape) if torch.is_tensor(o) else [list(x.shape) for x in o if torch.is_tensor(x)])))
        for lay in ym.model]
    ym(torch.zeros(1, 3, 640, 640, device=dev))
    for h in hooks:
        h.remove()
    etkin = sum(int(np.prod(s_)) if s_ and isinstance(s_[0], int) else 0 for _, s_ in sekiller[:-1])
    y.predict([str(p) for p in imgs[:4]], imgsz=640, conf=0.001, iou=0.7, max_det=300, verbose=False)
    t0 = time.time()
    for i in range(0, len(imgs), 32):
        y.predict([str(p) for p in imgs[i:i + 32]], imgsz=640, conf=0.001, iou=0.7, max_det=300, verbose=False)
    yolo_goruntu = (time.time() - t0) / len(imgs)
    veri = {"goruntu": len(imgs), "sinek_sn_goruntu": sinek_goruntu, "sinek_sn_goruntu_ortanca": float(np.median(sure)),
            "karo_goruntu_basina": {k: q(v_) for k, v_ in karo_say.items()},
            "karo_goruntu_basina_hepsi": q([x for v_ in karo_say.values() for x in v_]),
            "asama_sn_karo_basina": asama, "hesap_karo_basina_carpma_toplama": hesap,
            "noron_adim_karo_basina": N * T,
            "yolo": {"katman": info[0] if info else None, "parametre": info[1] if info else None,
                     "gflops_640": info[3] if info and len(info) > 3 else None, "adim_stride": ym.stride.tolist(),
                     "katman_ciktilari": sekiller, "etkinlik_sayisi_640": etkin,
                     "sn_goruntu (yükleme + letterbox + NMS dahil, 32'lik yığın)": yolo_goruntu}}
    kaydet("hiz", veri)


# ============================================================================ boyut (CPU, önbellekten)
def esik_fppi(pairs, hedef_fppi=1.0, iou=0.3):
    """final_eval'in R@FPPI'siyle aynı eşleme (match_flags, IoU 0,3): görüntü başına ortalama yanlış alarm ≤ hedef olan
    en düşük skor eşiği."""
    fl = sorted((f for d, g in pairs for f in det.match_flags(d, g, iou)), key=lambda x: -x[0])
    fp = np.cumsum([not o for _, o in fl])
    i = int(np.searchsorted(fp, hedef_fppi * len(pairs), side="right"))
    return fl[max(0, i - 1)][0] if fl else 1.0


def boyut(a):
    from PIL import Image
    imgs = list_images("thermal", "val")
    BIN = [0, 8, 12, 16, 24, 32, 64, 1e9]
    ad_bin = ["<8", "8–12", "12–16", "16–24", "24–32", "32–64", "≥64"]
    sonuc = {"bin_px": ad_bin}
    for ad, dosya in (("sinek", "uzun_75_val_merkez7p2_ov0.5.pt"), ("yolo", "YOLOv8n_thermal_val.pt")):
        pairs, meta, keys = torch.load(CACHE / dosya, weights_only=False)
        assert keys == [str(p.relative_to(ROOT)) for p in imgs], "önbellek görüntü listesi farklı"
        esik = esik_fppi(pairs)
        r = {"esik_fppi1": esik}
        for iou_esik in (0.3, 0.5):
            bul, say, iou_top = np.zeros(7), np.zeros(7), np.zeros(7)
            set_bul = {}
            for (d, g), p in zip(pairs, imgs):
                if not len(g):
                    continue
                sz = torch.sqrt(g[:, 2] * g[:, 3]).numpy()
                b = np.digitize(sz, BIN) - 1
                dd = d[d[:, 4] >= esik] if len(d) else d
                if len(dd):
                    iou = box_iou(g, dd[:, :4])                                        # [G, D]
                    # açgözlü: yüksek skordan; her tespit bir insan
                    used = torch.zeros(len(dd), dtype=torch.bool)
                    hit = np.zeros(len(g), bool)
                    best = np.zeros(len(g))
                    for j in dd[:, 4].argsort(descending=True).tolist():
                        col = iou[:, j].clone()
                        col[torch.as_tensor(hit)] = -1
                        i = int(col.argmax())
                        if col[i] >= iou_esik:
                            hit[i] = True
                            best[i] = float(iou[i, j])
                else:
                    hit, best = np.zeros(len(g), bool), np.zeros(len(g))
                for k in range(7):
                    s = b == k
                    say[k] += s.sum()
                    bul[k] += hit[s].sum()
                    iou_top[k] += best[s & hit].sum()
                ds = dataset_of(p)
                sb = set_bul.setdefault(ds, [0, 0])
                sb[0] += int(hit.sum())
                sb[1] += len(g)
            r[f"iou{iou_esik}"] = {"insan": say.astype(int).tolist(), "bulunan": bul.astype(int).tolist(),
                                   "duyarlilik": (bul / np.maximum(say, 1)).round(4).tolist(),
                                   "eslesen_iou_ort": (iou_top / np.maximum(bul, 1)).round(4).tolist(),
                                   "set": {k: [v[0], v[1], round(v[0] / max(v[1], 1), 4)] for k, v in set_bul.items()}}
        sonuc[ad] = r
        log(f"{ad}: eşik {esik:.3f}, duyarlılık@0,3 {r['iou0.3']['duyarlilik']}")
    # veri kümesi: görüntü boyutu, insan boyutu, YOLO ölçeği, karo sayısı
    from flybrain.eye import HexEye, Tiler
    eye = HexEye.from_processed(ROOT / "data/connectome/processed/bi_ol_merkez_thr5", 8)
    tiler = Tiler(eye, overlap=0.5)
    ds_ = {}
    for p in imgs:
        with Image.open(p) as im:
            W, H = im.size
        g = load_boxes(p, H, W)
        e = ds_.setdefault(dataset_of(p), {"boyut": Counter(), "insan_px": [], "karo": [], "olcek": []})
        e["boyut"][f"{W}×{H}"] += 1
        e["karo"].append(len(tiler.origins(H, W)))
        s = 640 / max(H, W)
        e["olcek"].append(s)
        if len(g):
            e["insan_px"] += torch.sqrt(g[:, 2] * g[:, 3]).tolist()
    veri = {}
    for k, e in ds_.items():
        ip = np.array(e["insan_px"])
        ol = float(np.median(e["olcek"]))
        veri[k] = {"goruntu": len(e["karo"]), "boyutlar": dict(e["boyut"].most_common(4)),
                   "insan": int(ip.size), "insan_px": q(ip) if ip.size else None,
                   "yolo_olcek_ortanca": ol, "yolo_insan_px_ortanca": float(np.median(ip) * ol) if ip.size else None,
                   "sinek_kolon_genisligi_ortanca": float(np.median(ip) / 8) if ip.size else None,
                   "karo_goruntu_basina": q(e["karo"])}
    sonuc["veri_kumesi_val"] = veri
    tr = Counter(dataset_of(p) for p in list_images("thermal", "train"))
    sonuc["egitim_kumesi"] = dict(tr)
    tot = sum(c_ ** 0.5 for c_ in tr.values())
    sonuc["egitim_ornekleme_payi_set_us_0,5"] = {k: v ** 0.5 / tot for k, v in tr.items()}
    kaydet("boyut", sonuc)


def parlaklik(a):
    """Termal val800: insan kutusu içindeki ortalama piksel ile kutunun 1 kutu boyu çevresindeki halkanın ortalaması
    (0–1 ölçeğinde, modelin gördüğü load_image çıktısı). ON / OFF seçiciliğini yorumlamak için."""
    sonuc = {}
    for p in val800():
        img = load_image(p, 1)[0]
        H, W = img.shape
        g = load_boxes(p, H, W)
        e = sonuc.setdefault(dataset_of(p), {"ic": [], "halka": [], "goruntu": []})
        e["goruntu"].append(float(img.mean()))
        for y, x, h, w in g.tolist():
            y0, x0, y1, x1 = int(max(0, y)), int(max(0, x)), int(min(H, y + h)), int(min(W, x + w))
            if y1 <= y0 or x1 <= x0:
                continue
            Y0, X0, Y1, X1 = int(max(0, y - h)), int(max(0, x - w)), int(min(H, y + 2 * h)), int(min(W, x + 2 * w))
            dis = img[Y0:Y1, X0:X1]
            ic = img[y0:y1, x0:x1]
            e["ic"].append(float(ic.mean()))
            e["halka"].append(float((dis.sum() - ic.sum()) / max(dis.numel() - ic.numel(), 1)))
    veri = {}
    for k, e in sonuc.items():
        ic, ha = np.array(e["ic"]), np.array(e["halka"])
        veri[k] = {"insan": int(ic.size), "ic_ort": float(ic.mean()) if ic.size else None,
                   "halka_ort": float(ha.mean()) if ha.size else None,
                   "icten_parlak_oran": float((ic > ha).mean()) if ic.size else None,
                   "goruntu_ort": float(np.mean(e["goruntu"]))}
    kaydet("parlaklik", veri)


def _kutle_yaricap(g: np.ndarray, cy: float, cx: float) -> dict:
    """|gradyan| haritasında birimin merkezinden yarıçapa göre birikimli kütle: %50 ve %90 yarıçapı (px), sıfırdan farklı
    gradyanlı piksellerin oranı ve sınırlayıcı kutusu (etkin alıcı alan; Luo vd. 2016)."""
    g = np.abs(g)
    yy, xx = np.indices(g.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel()
    o = np.argsort(r)
    c = np.cumsum(g.ravel()[o])
    c /= max(c[-1], 1e-30)
    nz = np.argwhere(g > 0)
    return {"r50": float(r[o][np.searchsorted(c, 0.5)]), "r90": float(r[o][np.searchsorted(c, 0.9)]),
            "sifirdan_farkli_oran": float((g > 0).mean()),
            "kutu": [int(nz[:, 0].min()), int(nz[:, 1].min()), int(nz[:, 0].max()), int(nz[:, 1].max())] if len(nz) else None,
            "boyut": list(g.shape)}


def alan(a):
    """Bir çıktı biriminin girdi piksellerinden ne kadarını kullandığı (gradyan). Sinek: örnek karonun (HIT-UAV) sağ göz
    merkez kolonunun ısısı, T = 10. YOLO: aynı görüntü letterbox 640'ta, P3 / P4 / P5 (Detect girdileri) merkez biriminin
    kanal toplamı."""
    dev = torch.device("cuda")
    m, _ = yukle(dev)
    z = np.load(OUT / "ornek_karo.npz")
    t = torch.as_tensor(z["karo"], device=dev)[None].requires_grad_(True)
    out = m(t)
    cen = m.eye.centers()
    cy, cx = m.eye.height / 2, m.eye.width / 2
    k = int(np.argmin((cen[:, 0] - cy) ** 2 + (cen[:, 1] - cx) ** 2))
    out["heat"][0, k].backward()
    g = t.grad[0, 0].cpu().numpy()
    veri = {"sinek": {"kolon": k, "merkez_px": [float(cen[k, 0]), float(cen[k, 1])], "karo": list(g.shape)}
            | _kutle_yaricap(g, *cen[k])}
    # aynalı konum (sol göz aynı karonun aynasını görür): gradyanın sol yarı / sağ yarı payı
    veri["sinek"]["sag_yari_payi"] = float(np.abs(g[:, g.shape[1] // 2:]).sum() / max(np.abs(g).sum(), 1e-30))
    from ultralytics import YOLO
    import cv2
    y = YOLO(str(ROOT / "results/baseline/yolov8n_thermal/weights/best.pt"))
    ym = y.model.float().eval().to(dev)
    goruntu = list((ROOT / "data/yolo/thermal/images/val").glob(z_ad := str(json.loads((OUT / "dinamik.json").read_text(
        encoding="utf-8"))["ornek"]["goruntu"])))[0]
    im = cv2.imread(str(goruntu))[:, :, ::-1].astype(np.float32) / 255
    H, W = im.shape[:2]
    s_ = 640 / max(H, W)
    nh, nw = round(H * s_), round(W * s_)
    can = np.full((640, 640, 3), 114 / 255, np.float32)
    oy, ox = (640 - nh) // 2, (640 - nw) // 2
    can[oy:oy + nh, ox:ox + nw] = cv2.resize(im, (nw, nh))
    x = torch.as_tensor(can.transpose(2, 0, 1).copy(), device=dev)[None].requires_grad_(True)
    det = ym.model[-1]
    cikis = {}
    hooks = [ym.model[i].register_forward_hook(lambda mod, i_, o, i=i: cikis.__setitem__(i, o)) for i in det.f]
    ym(x)
    for h in hooks:
        h.remove()
    veri["yolo"] = {"goruntu": z_ad, "letterbox": [nh, nw, oy, ox], "detect_girdileri": list(det.f)}
    for i, adim_ in zip(det.f, [8, 16, 32]):
        o = cikis[i]
        x.grad = None
        o[0, :, o.shape[2] // 2, o.shape[3] // 2].sum().backward(retain_graph=True)
        g = x.grad[0].abs().sum(0).cpu().numpy()
        veri["yolo"][f"P{int(math.log2(adim_))}_adim{adim_}"] = {"sekil": list(o.shape)} | _kutle_yaricap(
            g, (o.shape[2] // 2 + 0.5) * adim_, (o.shape[3] // 2 + 0.5) * adim_)
    k0 = ym.model[0].conv
    veri["yolo"]["ilk_katman"] = {"kanal": k0.out_channels, "cekirdek": list(k0.kernel_size), "adim": list(k0.stride)}
    kaydet("alan", veri)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("komut", choices=["envanter", "dinamik", "adim", "ablasyon", "hiz", "boyut", "parlaklik", "alan"])
    ap.add_argument("--pozitif", type=int, default=64)
    ap.add_argument("--negatif", type=int, default=32)
    ap.add_argument("--adim", type=int, default=30, help="dinamik: kaç adım izlenir (eğitim T = 10)")
    ap.add_argument("--tler", default="1,2,3,4,5,6,7,8,10,12,15,20")
    ap.add_argument("--goruntu", type=int, default=100, help="hiz: ölçülen val görüntüsü")
    ap.add_argument("--yama", default="", help="dinamik: ABLASYONLAR'dan biriyle (ör. 'STP kapalı')")
    a = ap.parse_args()
    torch.manual_seed(0)
    {"envanter": envanter, "dinamik": dinamik, "adim": adim, "ablasyon": ablasyon, "hiz": hiz, "boyut": boyut, "parlaklik": parlaklik, "alan": alan}[a.komut](a)


if __name__ == "__main__":
    main()
