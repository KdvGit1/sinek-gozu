#!/usr/bin/env python3
"""
H35 (günlük §56): iki optik loblu konnektom — sağ optik lob + sol optik lob + iki lobun çıkışlarından girdi alan merkez
beyin nöronları (FlyWire v783, gerçek nöronlar ve gerçek sinapslar; karşı taraf bağlantıları dahil).

Kurallar sağ modelle (build_connectome.py --central) aynı, her iki yana simetrik uygulanır:
  * optik lob: visual_neuron_types side = right / left / "left,right" (iki yana uzanan görsel nöronlar)
  * merkez: optik lob dışı, o yanın VPN'lerinden ≥ min_syn sinaps alan ve girdisinin ≥ min_frac'ı o yanın optik
    lobundan gelen, aynı yandaki nöronlar (sağ için sağ modeldeki kümenin aynısı)
  * hücre tipi adları iki yanda ortak → tip çifti başına tek α iki lobda paylaşılır (parametre sayısı neredeyse aynı)
Göz eşlemesi: sol lobun kolonları (p, q) koordinatıyla sağ gözün aynı (p, q)'lu kolonuna eşlenir (column_id = sağdaki
eşi, eye = "L"). Model sol gözü aynalanmış görüntüyle besler (FlyWire'da iki gözün (p, q) ızgarası ayna simetrik
varsayılır; 796 sağ kolonun 772'si solda da var). Eşi olmayan sol kolonların nöronları kolonsuz kalır.

Çıktı: data/connectome/processed/bi_ol_merkez_thr{N}/ (sağ modelle aynı dosyalar; nodes.csv'de ek "eye" sütunu)
Kullanım: python tools/build_bilateral.py --thresholds 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.build_connectome import FW, OUT, PHOTORECEPTORS, assign_photoreceptor_columns, build, read_edges, type_signs  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=int, nargs="+", default=[5])
    ap.add_argument("--rare-pair-syn", type=int, default=50)
    ap.add_argument("--central-min-syn", type=int, default=20)
    ap.add_argument("--central-min-frac", type=float, default=0.1)
    ap.add_argument("--modulator-sign", type=int, choices=[-1, 1], default=1)
    args = ap.parse_args(argv)

    log("tablolar okunuyor")
    vis = pd.read_csv(FW / "visual_neuron_types.csv.gz")
    col = pd.read_csv(FW / "column_assignment.csv.gz")
    neurons = pd.read_csv(FW / "neurons.csv.gz")
    cls = pd.read_csv(FW / "classification.csv.gz").set_index("root_id")
    typ = pd.read_csv(FW / "consolidated_cell_types.csv.gz")[["root_id", "primary_type"]].set_index("root_id")
    edges = read_edges()
    log(f"  {len(edges):,} nöron çifti")

    # --- optik loblar
    ol = vis[vis.side.isin(["right", "left", "left,right"])][["root_id", "type", "family", "subsystem", "category",
                                                              "side"]].copy()
    ol["eye"] = ol.side.map({"right": "R", "left": "L"}).fillna("")
    ol_ids = set(ol.root_id.astype(np.int64))

    # --- merkez: her yan için o yanın VPN'lerinden girdi alan aynı yan nöronları
    in_tot = edges.groupby("post_root_id").syn_count.sum()
    central = []
    for side, e in (("right", "R"), ("left", "L")):
        vpn = set(ol[(ol.eye == e) & ol.category.eq("boundary")].root_id.astype(np.int64))
        V = edges[edges.pre_root_id.isin(vpn) & ~edges.post_root_id.isin(ol_ids)]
        s = V.groupby("post_root_id").syn_count.sum()
        frac = s / in_tot.reindex(s.index).clip(lower=1)                 # sağ modeldeki oran_ol ile aynı tanım
        ok = s.index[(s >= args.central_min_syn) & (frac >= args.central_min_frac)]
        ok = [r for r in ok if cls.side.get(r) == side]
        log(f"  merkez ({side}): {len(ok)} nöron")
        central.append(pd.DataFrame({"root_id": ok, "side_c": side}))
    central = pd.concat(central).drop_duplicates("root_id")
    ptype = typ.primary_type.reindex(central.root_id).to_numpy()
    ctype = np.where(pd.notna(ptype), ptype, "merkez_" + central.root_id.astype(str))
    cnodes = pd.DataFrame({"root_id": central.root_id.to_numpy(), "type": ctype, "family": "central",
                           "subsystem": cls.super_class.reindex(central.root_id).to_numpy(), "category": "central",
                           "side": central.side_c.to_numpy(), "eye": ""})
    nodes = pd.concat([ol, cnodes], ignore_index=True)
    nodes = nodes.sort_values(["type", "root_id"]).reset_index(drop=True)
    nodes["type_id"] = pd.Categorical(nodes.type).codes.astype(np.int32)

    # --- kolonlar: sağ göz ızgarası; sol kolonlar (p, q) ile sağdaki eşine
    colR = col[col.hemisphere == "right"]
    colL = col[col.hemisphere == "left"]
    pq2r = colR.groupby(["p", "q"]).column_id.first()
    l2r = colL.groupby("column_id")[["p", "q"]].first().apply(lambda r: pq2r.get((r.p, r.q), np.nan), axis=1)
    cmap_r = colR.set_index("root_id").column_id
    cmap_l = colL.set_index("root_id").column_id.map(l2r)
    nodes["column_id"] = np.where(nodes.eye == "R", cmap_r.reindex(nodes.root_id).to_numpy(),
                                  np.where(nodes.eye == "L", cmap_l.reindex(nodes.root_id).to_numpy(), np.nan))
    nodes["column_source"] = np.where(pd.notna(nodes.column_id), "flywire", "")
    # R1-6: en çok sinaps verdiği L1–L3'ün kolonu (her yan kendi kolonlarıyla), sol için sağ eşine çevrilir
    r16 = assign_photoreceptor_columns(nodes, edges, col)               # özgün column_id (sağ ya da sol)
    lcol = set(colL.column_id)
    r16 = r16.map(lambda c: l2r.get(c, np.nan) if c in lcol else c)
    m = (nodes.type == "R1-6") & nodes.column_id.isna() & nodes.eye.isin(["R", "L"])
    nodes.loc[m, "column_id"] = r16.reindex(nodes.loc[m, "root_id"]).to_numpy()
    nodes.loc[m & nodes.column_id.notna(), "column_source"] = "R1-6->L1-3"
    nodes["column_id"] = pd.to_numeric(nodes.column_id).astype("Int64")

    ts = type_signs(nodes, neurons, args.modulator_sign)
    nodes = nodes.join(ts, on="type")
    missing_sign = nodes.sign.isna()
    nodes.loc[missing_sign, ["type_nt", "sign"]] = ["UNKNOWN", 1]
    nodes["sign"] = nodes.sign.astype(np.int8)
    nodes["photoreceptor"] = nodes.type.where(nodes.type.isin(PHOTORECEPTORS), "")
    nodes["is_output"] = nodes.category.eq("boundary")
    nodes.insert(0, "idx", np.arange(len(nodes), dtype=np.int32))
    nodes = nodes.drop(columns=["side"])

    cols = colR.groupby("column_id")[["p", "q"]].first().reset_index()
    pr = nodes[(nodes.photoreceptor != "") & (nodes.eye == "R")].groupby(["column_id", "photoreceptor"]).size() \
        .unstack(fill_value=0)
    for t in PHOTORECEPTORS:
        cols[f"n_{t}"] = pr[t].reindex(cols.column_id).fillna(0).astype(int).to_numpy() if t in pr else 0
    cols["hex_x"] = (cols.q - cols.p) * np.sqrt(3) / 2
    cols["hex_y"] = (cols.p + cols.q) / 2
    cols["brick_col"] = cols.q - cols.p
    cols["brick_row2"] = cols.p + cols.q

    summary = {
        "neurons": int(len(nodes)),
        "neurons_by_eye": {k: int(v) for k, v in nodes.eye.replace("", "-").value_counts().items()},
        "central_neurons": int(nodes.category.eq("central").sum()),
        "central_by_side": {k: int(v) for k, v in central.side_c.value_counts().items()},
        "cell_types": int(nodes.type_id.max() + 1),
        "eye_columns": int(len(cols)),
        "left_columns_mapped": int(l2r.notna().sum()), "left_columns_total": int(len(l2r)),
        "left_neurons_with_column": int((nodes.eye.eq("L") & nodes.column_id.notna()).sum()),
        "right_neurons_with_column": int((nodes.eye.eq("R") & nodes.column_id.notna()).sum()),
        "output_neurons(boundary)": int(nodes.is_output.sum()),
        "photoreceptors": {f"{t}_{e}": int(((nodes.type == t) & (nodes.eye == e)).sum())
                           for t in PHOTORECEPTORS for e in "RL"},
        "neurons_unknown_sign": int(missing_sign.sum()),
        "central_filter": {"min_syn_from_vpn": args.central_min_syn, "min_frac_from_ol": args.central_min_frac},
    }
    log(f"düğümler: {json.dumps(summary, ensure_ascii=False)}")
    for thr in args.thresholds:
        out_dir = OUT / f"bi_ol_merkez_thr{thr}"
        log(f"eşik >= {thr}: {out_dir}")
        meta = build(thr, nodes, edges, args.rare_pair_syn, out_dir)
        cols.to_csv(out_dir / "columns.csv", index=False)
        # iki lob arası bağlantılar (karşı taraf)
        e = np.load(out_dir / "edges.npz")
        eye = nodes.eye.to_numpy()
        side = np.where(eye == "R", "R", np.where(eye == "L", "L", "?"))
        cross = int(((side[e["pre"]] == "R") & (side[e["post"]] == "L")).sum() +
                    ((side[e["pre"]] == "L") & (side[e["post"]] == "R")).sum())
        meta = {**summary, **meta, "edges_between_lobes_direct": cross, "source": "FlyWire v783 (CC BY 4.0)",
                "built": time.strftime("%Y-%m-%d")}
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        log("  " + ", ".join(f"{k}={meta[k]}" for k in ("edges", "synapses", "alpha_groups", "edges_between_lobes_direct",
                                                          "neurons_without_input", "spectral_radius_abs_syn")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
