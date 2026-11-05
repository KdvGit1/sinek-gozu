"""
H41 (günlük §60): etkinliğe dayalı homeostaz ve "uyku evresi".

Biyoloji: homeostatik sinaptik ölçekleme (Turrigiano 1998): nöron, hedef ateşleme hızını korumak için bütün girdi
sinapslarını aynı oranda büyütür/küçültür. Bizde: hücre tipi k'nın kazancı g_k = exp(gain_raw_k), tipin bütün döngüsel
girdi akımını çarpar (model.cfg.type_gain). Uyku evresinde sabit bir karo kümesinde tip başına ortalama etkinlik m_k
ölçülür ve g_k ← g_k · (ρ_k / m_k)^η, adım başına değişim [1/(1+c), 1+c] ile sınırlı. Kazançlar GRADYAN ALMAZ
(kullanıcı kararı, 07.10.2026): yalnız uyku evresi değiştirir. Topoloji ve α değişmez.
"""
from __future__ import annotations

import numpy as np
import torch


def _drive(model, x):
    """§71 (hata #7): etkinlik ileri geçişin gerçek girdisiyle ölçülür (süperpozisyonda 6 bakış); eskiden tek bakıştı."""
    return model.drive(x) if hasattr(model, "drive") else model.encode(x)[0]


@torch.no_grad()
def type_activity(model, tiles: torch.Tensor, chunk: int = 24) -> torch.Tensor:
    """Tip başına ortalama etkinlik [n_types] (son adım; doymalı modelde tanh(relu(v)), yoksa relu(v)); bütün
    karolar ve tipin bütün nöronları üzerinden."""
    dev = model.alpha_raw.device
    sat = model.cfg.activation == "sat"
    n_t = int(model.type_id.max()) + 1
    s = torch.zeros(n_t, device=dev)
    cnt = torch.bincount(model.type_id, minlength=n_t).float()
    n = 0
    was = model.training
    model.eval()
    for a in range(0, len(tiles), chunk):
        x = tiles[a:a + chunk].to(dev)
        drive = _drive(model, x)                                            # §71: ileri geçişle aynı girdi
        v = model.core(drive)
        r = torch.tanh(torch.relu(v)) if sat else torch.relu(v)                    # [N, B]
        s.index_add_(0, model.type_id, r.sum(1))
        n += x.shape[0]
    model.train(was)
    return s / (cnt * max(n, 1)).clamp(min=1)


@torch.no_grad()
def sleep_step(model, target: torch.Tensor, mask: torch.Tensor, measured: torch.Tensor, eta: float = 0.5,
               clamp: float = 0.2, floor: float = 1e-3) -> dict:
    """Bir uyku adımı: mask'teki tiplerin kazancını hedefe doğru çeker. Döner: özet."""
    ratio = (target.clamp(min=floor) / measured.clamp(min=floor)) ** eta
    ratio = ratio.clamp(1 / (1 + clamp), 1 + clamp)
    upd = torch.where(mask, ratio.log(), torch.zeros_like(ratio))
    model.gain_raw.add_(upd)
    err = (measured[mask] - target[mask]).abs()
    return {"ort_mutlak_hata": float(err.mean()), "kazanc_ort": float(model.gain_raw[mask].exp().mean()),
            "kazanc_min": float(model.gain_raw[mask].exp().min()), "kazanc_max": float(model.gain_raw[mask].exp().max())}


def matched_targets(c_dst, c_src, act_src: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Hedef ρ: kaynak modelde aynı adlı tipin etkinliği. Kaynağında olmayan tipler maskelenir (dokunulmaz)."""
    src_names = c_src.nodes.drop_duplicates("type_id").set_index("type_id").type.sort_index().tolist()
    dst_names = c_dst.nodes.drop_duplicates("type_id").set_index("type_id").type.sort_index().tolist()
    idx = {t: i for i, t in enumerate(src_names)}
    tgt = torch.zeros(len(dst_names), device=device)
    mask = torch.zeros(len(dst_names), dtype=torch.bool, device=device)
    for j, t in enumerate(dst_names):
        if t in idx:
            tgt[j] = act_src[idx[t]]
            mask[j] = True
    return tgt, mask


# ------------------------------------------------------------------ H41b (§62): ölü bant, eşik + kazanç
FUNCTION_PATTERNS = {"hareket": r"^(T4|T5|LPLC|LPi|LPT|LLPC|HS|VS|H1$|H2$|Y\d|TmY|LC11$|LC4$|LC12$|LC15$|LC18$|LC9$|Tlp|Am1)",
                     "renk": r"^(R7|R8|Dm8|Tm5|Tm20|Tm29|MeTu)", "polarizasyon": r"DRA"}


def type_names(model_or_conn) -> list[str]:
    c = model_or_conn
    return c.nodes.drop_duplicates("type_id").set_index("type_id").type.sort_index().tolist()


@torch.no_grad()
def type_active_fraction(model, tiles: torch.Tensor, thr: float = 0.05, chunk: int = 24) -> torch.Tensor:
    """Tip başına aktif oran [n_types]: tipin (nöron, karo) çiftlerinin yüzde kaçında son adım r > thr."""
    dev = model.alpha_raw.device
    sat = model.cfg.activation == "sat"
    n_t = int(model.type_id.max()) + 1
    s = torch.zeros(n_t, device=dev)
    cnt = torch.bincount(model.type_id, minlength=n_t).float()
    n = 0
    was = model.training
    model.eval()
    for a in range(0, len(tiles), chunk):
        x = tiles[a:a + chunk].to(dev)
        drive = _drive(model, x)                                            # §71: ileri geçişle aynı girdi
        v = model.core(drive)
        r = torch.tanh(torch.relu(v)) if sat else torch.relu(v)
        s.index_add_(0, model.type_id, (r > thr).float().sum(1))
        n += x.shape[0]
    model.train(was)
    return s / (cnt * max(n, 1)).clamp(min=1)


@torch.no_grad()
def homeo_step(model, P: torch.Tensor, dead: torch.Tensor, sat: torch.Tensor, lo: float = 0.03, hi: float = 0.05,
               b_step: float = 0.05, g_step: float = 0.0953, sat_hi: float = 0.90, sat_lo: float = 0.85) -> dict:
    """Ölü bant kuralı. dead (başta aktif oranı < %1 olan uygun tipler): P < lo → eşik +b_step, P > hi → eşik −b_step
    (ofset ≥ 0). sat (başta > %90): P > sat_hi → log-kazanç −g_step (×0,91), P < sat_lo ve ofset < 0 → +g_step.
    %1–90 arasındaki tiplere dokunulmaz."""
    up = dead & (P < lo)
    down = dead & (P > hi)
    model.homeo_bias[up] += b_step
    model.homeo_bias[down] -= b_step
    model.homeo_bias.clamp_(min=0)
    gd = sat & (P > sat_hi)
    gu = sat & (P < sat_lo) & (model.homeo_gain < 0)
    model.homeo_gain[gd] -= g_step
    model.homeo_gain[gu] += g_step
    model.homeo_gain.clamp_(max=0)
    return {"ölü_hâlâ_<%1": int((dead & (P < 0.01)).sum()), "ölü_bantta": int((dead & (P >= lo) & (P <= hi)).sum()),
            "doygun_>%90": int((sat & (P > sat_hi)).sum()), "eşik_ort": float(model.homeo_bias[dead].mean()) if dead.any() else 0.0,
            "eşik_maks": float(model.homeo_bias.max()), "kazanç_min": float(model.homeo_gain.exp().min())}


# ------------------------------------------------------------------ H41c (§71): sınırlı ve yakınsayan homeostaz
H41C = {"lo": 0.03, "hi": 0.05, "b_step": 0.05, "b_max": 0.5, "decay": 0.02, "patience": 4, "min_gain": 0.005,
        "release": 0.2, "sat_r": 0.9, "sat_std": 0.05}


@torch.no_grad()
def type_stats(model, tiles: torch.Tensor, thr: float = 0.05, chunk: int = 24):
    """Tip başına aktif oran (r > thr), ortalama r ve karolar arası std (nöron başına std'nin tip ortalaması)."""
    dev = model.alpha_raw.device
    sat = model.cfg.activation == "sat"
    N = model.n_neurons
    act = torch.zeros(N, device=dev)
    s1 = torch.zeros(N, device=dev)
    s2 = torch.zeros(N, device=dev)
    n = 0
    was = model.training
    model.eval()
    for a in range(0, len(tiles), chunk):
        x = tiles[a:a + chunk].to(dev)
        v = model.core(_drive(model, x))
        r = torch.tanh(torch.relu(v)) if sat else torch.relu(v)
        act += (r > thr).float().sum(1)
        s1 += r.sum(1)
        s2 += (r ** 2).sum(1)
        n += x.shape[0]
    model.train(was)
    mean = s1 / max(n, 1)
    std = (s2 / max(n, 1) - mean ** 2).clamp(min=0).sqrt()
    nt = int(model.type_id.max()) + 1
    cnt = torch.bincount(model.type_id, minlength=nt).clamp(min=1).float()
    tav = lambda x: torch.zeros(nt, device=dev).index_add_(0, model.type_id, x) / cnt
    return tav(act / max(n, 1)), tav(mean), tav(std)


@torch.no_grad()
def homeo_step_v2(model, st: dict, P: torch.Tensor, mean_r: torch.Tensor, std_r: torch.Tensor) -> dict:
    """H41c uyku adımı. Ölü hedefler (st["dead"]: gerçek girdide ölü + girdisi modelde + uyaranı mevcut): P < lo → eşik
    +b_step, P > hi → −b_step. Bir tip patience uyku boyunca itildiği hâlde P en az min_gain artmadıysa VAZGEÇİLİR (frozen;
    görev o nöronun sessiz kalmasını istiyor) ve ofseti her uykuda ×(1 − release) ile sıfıra döner. Gerçek doygunluk
    (ort. r > sat_r ve karolar arası std < sat_std; st["sat"] tipleri) → eşik −b_step. Dokunulmayan tiplerin ofseti
    ×(1 − decay) söner (sızıntılı integral; anti-windup). Ofset [−b_max, b_max]; kazanç (homeo_gain) DEĞİŞMEZ."""
    c = st["cfg"]
    b = model.homeo_bias
    D, frozen = st["dead"], st["frozen"]
    up = D & ~frozen & (P < c["lo"])
    down = torch.zeros_like(up)          # aşağı itme yok: öğrenmeyle canlanan (P ≥ lo) tip itilmez, ofseti söner
    yeni = up & (st["push_n"] == 0)
    st["P_start"][yeni] = P[yeni]
    st["push_n"][up] += 1
    st["push_n"][~up] = 0
    b[up] += c["b_step"]
    give = up & (st["push_n"] >= c["patience"]) & ((P - st["P_start"]) < c["min_gain"])
    frozen |= give
    sat_now = st["sat"] & ~frozen & (mean_r > c["sat_r"]) & (std_r < c["sat_std"])
    b[sat_now] -= c["b_step"]
    rest = ~(up | down | sat_now) & ~frozen
    b[rest] *= (1 - c["decay"])
    b[frozen] *= (1 - c["release"])
    b.clamp_(-c["b_max"], c["b_max"])
    return {"hedef_olu": int(D.sum()), "itilen": int(up.sum()), "bantta": int((D & (P >= c["lo"]) & (P <= c["hi"])).sum()),
            "canlanan(P>=lo)": int((D & (P >= c["lo"])).sum()), "vazgecilen": int(frozen.sum()),
            "gercek_doygun": int(sat_now.sum()), "ofset_maks": float(b.abs().max()), "ofset_ort(hedef)": float(b[D].mean())
            if D.any() else 0.0}
