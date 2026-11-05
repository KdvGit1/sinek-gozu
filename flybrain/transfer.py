"""
H35 (günlük §56): bir konnektomda eğitilmiş modelin parametrelerini başka bir konnektomlu modele ADLARLA taşır
(ör. 12 saatlik sağ lob modeli → iki loblu model). Konnektom parametreleri hücre tipi / tip çifti başına olduğu için
ortak tiplerde birebir aktarılır; yeni tipler ve yeni tip çiftleri kendi başlangıç değerlerinde kalır.

  bias, leak_raw (, gain_raw)  : tip adı
  alpha_raw                    : (pre tipi, post tipi); nadir gruplar (pre tipi, "*nadir*")
  encoder.*, to_photo.*, adapt_c: biçim aynıysa birebir
  head (LayerNorm + Linear)    : okuma özelliği adı (kolonsal tip / çıkış tipi); yeni özellikler sıfır ağırlıkla
Yalnız d = 1, köprüsüz, doğrusal başlık (head_hidden = 0, head_context = none) desteklenir (saf ayar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .connectome import Connectome


def _type_names(c: Connectome) -> list[str]:
    return c.nodes.drop_duplicates("type_id").set_index("type_id").type.sort_index().tolist()


def _pair_keys(c: Connectome) -> dict:
    p = c.pairs
    return {(a, b): int(i) for i, a, b in zip(p.pair_id, p.pre_type, p.post_type)}


def _features(model) -> list[str]:
    return [f"c:{t}" for t in model.columnar_types] + [f"o:{t}" for t in model.output_types]


@torch.no_grad()
def transfer_by_names(dst, src, c_dst: Connectome, c_src: Connectome) -> dict:
    """src modelin parametrelerini dst'ye ad eşleşmesiyle kopyalar. Döner: özet sayılar."""
    rep = {}
    t_src = {t: i for i, t in enumerate(_type_names(c_src))}
    t_dst = _type_names(c_dst)
    for name in ("bias", "leak_raw", "gain_raw"):
        if hasattr(dst, name) and hasattr(src, name):
            P, S = getattr(dst, name), getattr(src, name)
            hit = [(j, t_src[t]) for j, t in enumerate(t_dst) if t in t_src]
            if hit:
                jd, js = map(list, zip(*hit))
                P[jd] = S[js]
            rep[name] = f"{len(hit)}/{len(t_dst)} tip"
    k_src, k_dst = _pair_keys(c_src), _pair_keys(c_dst)
    hit = [(j, k_src[k]) for k, j in k_dst.items() if k in k_src]
    if hit:
        jd, js = map(list, zip(*hit))
        dst.alpha_raw[jd] = src.alpha_raw[js]
    rep["alpha_raw"] = f"{len(hit)}/{len(k_dst)} α grubu"
    sd_src = src.state_dict()
    for k, v in dst.state_dict().items():
        if (k.startswith("encoder.") or k.startswith("to_photo.") or k == "adapt_c") and k in sd_src \
                and sd_src[k].shape == v.shape:
            v.copy_(sd_src[k])
    if dst.head_in is None and len(dst.head) == 2 and src.head_in is None and len(src.head) == 2:
        f_src = {f: i for i, f in enumerate(_features(src))}
        f_dst = _features(dst)
        hit = [(j, f_src[f]) for j, f in enumerate(f_dst) if f in f_src]
        jd, js = map(list, zip(*hit))
        nrm_d, lin_d = dst.head[0], dst.head[1]
        nrm_s, lin_s = src.head[0], src.head[1]
        if hasattr(nrm_d, "weight") and nrm_d.weight is not None:
            nrm_d.weight[jd] = nrm_s.weight[js]
            nrm_d.bias[jd] = nrm_s.bias[js]
        lin_d.weight.zero_()
        n_out = min(lin_d.weight.shape[0], lin_s.weight.shape[0])
        lin_d.weight[:n_out, jd] = lin_s.weight[:n_out, js]
        lin_d.bias[:n_out] = lin_s.bias[:n_out]
        rep["head"] = f"{len(hit)}/{len(f_dst)} okuma özelliği"
    return rep
