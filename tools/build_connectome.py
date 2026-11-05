#!/usr/bin/env python3
"""
FlyWire v783'ten sağ optik lob alt ağını çıkarır ve modele hazır dosyalar üretir.

Tasarım kararları (TASARIM.md Bölüm 5):
  * Alt ağ: sağ optik lobdaki tüm görsel nöronlar (intrinsic + boundary/VPN), kırpma yok
  * Sinaps eşiği: parametre (--thresholds 1 2 5); her eşik için ayrı klasör
  * Ağırlık paylaşımı: tip çifti başına tek α. Toplam sinapsı --rare-pair-syn'den az olan
    tip çiftleri, presinaptik tip başına ortak bir "nadir" α paylaşır
  * İşaret (uyarıcı + / baskılayıcı −): hücre tipinin çoğunluk nörotransmiterinden (Dale ilkesi).
    Fotoreseptörler histaminerjik olduğu için − (FlyWire sınıflandırıcısında histamin sınıfı yok)

Fotoreseptör kolonları: R7/R8 kolonları FlyWire'da hazır. R1-6'nın kolonu yok; her R1-6, en çok
sinaps verdiği L1/L2/L3 nöronunun kolonuna atanır (nöral süperpozisyon: bir R1-6 terminali tek
bir lamina kartuşuna gider).

Çıktı: data/connectome/processed/right_ol_thr{N}/
  nodes.csv        nöron tablosu (indeks, tip, kolon, işaret, fotoreseptör/çıkış bayrakları)
  edges.npz        pre, post, syn, pair (int32 diziler)
  type_pairs.csv   α gruplarının listesi
  columns.csv      kolon tablosu: (p, q), düzlem koordinatları, kolondaki fotoreseptör sayıları
  meta.json        özet istatistikler

Kullanım:
  python tools/build_connectome.py                       # eşik 1, 2, 5
  python tools/build_connectome.py --thresholds 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
FW = ROOT / "data" / "connectome" / "flywire_783"
OUT = ROOT / "data" / "connectome" / "processed"

PHOTORECEPTORS = ("R1-6", "R7", "R8")
LAMINA_TARGETS = ("L1", "L2", "L3")
NT_SIGN = {"ACH": 1, "GABA": -1, "GLUT": -1, "DA": 1, "SER": 1, "OCT": 1}
MODULATORS = {"DA", "SER", "OCT"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_edges() -> pd.DataFrame:
    kw = dict(usecols=["pre_root_id", "post_root_id", "syn_count"],
              dtype={"pre_root_id": "int64", "post_root_id": "int64", "syn_count": "int32"})
    try:
        df = pd.read_csv(FW / "connections_no_threshold.csv.gz", engine="pyarrow", **kw)
    except (ImportError, ValueError):
        df = pd.read_csv(FW / "connections_no_threshold.csv.gz", **kw)
    return df.groupby(["pre_root_id", "post_root_id"], as_index=False, sort=False)["syn_count"].sum()


def type_signs(nodes: pd.DataFrame, neurons: pd.DataFrame, modulator_sign: int) -> pd.DataFrame:
    """Tip başına çoğunluk nörotransmiteri (tahmin skoruyla ağırlıklı) ve işaret."""
    nt = neurons.set_index("root_id")[["nt_type", "nt_type_score"]].reindex(nodes.root_id)
    nt.index = nodes.index
    df = pd.DataFrame({"type": nodes.type, "nt": nt.nt_type, "score": nt.nt_type_score.fillna(0)})
    votes = df.dropna(subset=["nt"]).groupby(["type", "nt"]).score.sum().reset_index()
    top = votes.sort_values("score", ascending=False).drop_duplicates("type").set_index("type")
    total = votes.groupby("type").score.sum()
    out = pd.DataFrame({"type_nt": top.nt, "type_nt_agreement": (top.score / total).round(3)})
    signs = {**NT_SIGN, **{m: modulator_sign for m in MODULATORS}}
    out["sign"] = out.type_nt.map(signs)
    for pr in PHOTORECEPTORS:                               # histamin: baskılayıcı
        out.loc[pr, ["type_nt", "sign"]] = ["HIST(override)", -1]
    return out


def assign_photoreceptor_columns(nodes, edges, col) -> pd.Series:
    """R1-6 nöronlarını en çok sinaps verdikleri L1/L2/L3'ün kolonuna ata."""
    r16 = nodes[nodes.type == "R1-6"].root_id
    colmap = col.set_index("root_id")
    e = edges[edges.pre_root_id.isin(r16)].copy()
    e["post_type"] = colmap.type.reindex(e.post_root_id).to_numpy()
    e["post_col"] = colmap.column_id.reindex(e.post_root_id).to_numpy()
    e = e[e.post_type.isin(LAMINA_TARGETS)]
    by_col = e.groupby(["pre_root_id", "post_col"]).syn_count.sum().reset_index()
    best = by_col.sort_values("syn_count", ascending=False).drop_duplicates("pre_root_id")
    return best.set_index("pre_root_id").post_col


def build(thr: int, nodes: pd.DataFrame, edges: pd.DataFrame, rare_pair_syn: int, out_dir: Path) -> dict:
    idx = pd.Index(nodes.root_id)
    e = edges[edges.syn_count >= thr]
    pi = idx.get_indexer(e.pre_root_id.to_numpy())
    qi = idx.get_indexer(e.post_root_id.to_numpy())
    keep = (pi >= 0) & (qi >= 0)
    pi, qi = pi[keep].astype(np.int32), qi[keep].astype(np.int32)
    syn = e.syn_count.to_numpy()[keep].astype(np.int32)

    # α grupları: tip çifti; nadir çiftler presinaptik tip başına tek grupta
    tid = nodes.type_id.to_numpy()
    pair = pd.DataFrame({"pre_type": tid[pi], "post_type": tid[qi], "syn": syn})
    tot = pair.groupby(["pre_type", "post_type"]).syn.agg(["sum", "size"]).reset_index()
    tot["rare"] = tot["sum"] < rare_pair_syn
    common = tot[~tot.rare].reset_index(drop=True)
    common["pair_id"] = np.arange(len(common))
    rare_pre = np.sort(tot[tot.rare].pre_type.unique())
    rare_id = pd.Series(len(common) + np.arange(len(rare_pre)), index=rare_pre)
    key = pd.MultiIndex.from_arrays([pair.pre_type, pair.post_type])
    ckey = pd.MultiIndex.from_arrays([common.pre_type, common.post_type])
    pid = ckey.get_indexer(key)
    is_rare = pid < 0
    pid[is_rare] = rare_id.reindex(pair.pre_type[is_rare]).to_numpy()
    pid = pid.astype(np.int32)

    type_names = nodes.drop_duplicates("type_id").set_index("type_id").type.sort_index()
    rare_rows = tot[tot.rare].groupby("pre_type").agg(syn_total=("sum", "sum"), n_edges=("size", "sum"),
                                                        n_type_pairs=("sum", "size")).reset_index()
    pairs_df = pd.concat([
        pd.DataFrame({"pair_id": common.pair_id, "pre_type": type_names.reindex(common.pre_type).to_numpy(),
                      "post_type": type_names.reindex(common.post_type).to_numpy(),
                      "syn_total": common["sum"], "n_edges": common["size"], "n_type_pairs": 1, "rare_group": False}),
        pd.DataFrame({"pair_id": rare_id.reindex(rare_rows.pre_type).to_numpy(),
                      "pre_type": type_names.reindex(rare_rows.pre_type).to_numpy(), "post_type": "*nadir*",
                      "syn_total": rare_rows.syn_total, "n_edges": rare_rows.n_edges,
                      "n_type_pairs": rare_rows.n_type_pairs, "rare_group": True}),
    ], ignore_index=True)
    pre_sign = nodes.sign.to_numpy()
    pairs_df["sign"] = pd.Series(pre_sign[pi]).groupby(pid).first().reindex(pairs_df.pair_id).to_numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "edges.npz", pre=pi, post=qi, syn=syn, pair=pid)
    pairs_df.to_csv(out_dir / "type_pairs.csv", index=False)
    in_syn = np.bincount(qi, weights=syn, minlength=len(nodes))
    nodes.assign(in_syn=in_syn.astype(np.int64)).to_csv(out_dir / "nodes.csv", index=False)

    meta = {
        "threshold_syn": thr,
        "neurons": int(len(nodes)),
        "edges": int(len(pi)),
        "synapses": int(syn.sum()),
        "autapses": int((pi == qi).sum()),
        "alpha_groups": int(len(pairs_df)),
        "alpha_groups_type_pair": int(len(common)),
        "alpha_groups_rare": int(len(rare_pre)),
        "rare_pair_syn": rare_pair_syn,
        "edges_in_rare_groups_pct": round(100 * float(is_rare.mean()), 2),
        "synapses_in_rare_groups_pct": round(100 * float(syn[is_rare].sum() / syn.sum()), 3),
        "excitatory_edges_pct": round(100 * float((pre_sign[pi] > 0).mean()), 2),
        "neurons_without_input": int((in_syn == 0).sum()),
    }
    try:  # başlangıç ölçeklemesi için |W| (sinaps sayısı) spektral yarıçapı
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigs
        A = csr_matrix((syn.astype(np.float64), (qi, pi)), shape=(len(nodes), len(nodes)))
        meta["spectral_radius_abs_syn"] = round(float(np.abs(eigs(A, k=1, which="LM",
                                                                    return_eigenvectors=False)[0])), 2)
    except Exception as ex:  # noqa: BLE001
        meta["spectral_radius_abs_syn"] = f"hesaplanamadı: {type(ex).__name__}"
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sağ optik lob alt ağını modele hazır dosyalara dönüştür")
    ap.add_argument("--thresholds", type=int, nargs="+", default=[1, 2, 5])
    ap.add_argument("--rare-pair-syn", type=int, default=50,
                    help="toplam sinapsı bundan az olan tip çiftleri presinaptik tip başına tek α paylaşır")
    ap.add_argument("--central", action="store_true",
                    help="merkez beyin modülünü ekle (günlük §41): tools/central_targets.py aday listesinden sağ yarıküre, "
                         "VPN'den ≥ --central-min-syn sinaps ve girdisinin ≥ --central-min-frac'ı optik lobdan gelen "
                         "nöronlar. Topoloji değişmez: gerçek nöronlar, gerçek sinapslar (ileri, geri besleme, kendi içi)")
    ap.add_argument("--central-min-syn", type=int, default=20)
    ap.add_argument("--central-min-frac", type=float, default=0.1)
    ap.add_argument("--modulator-sign", type=int, choices=[-1, 1], default=1,
                    help="dopamin/serotonin/oktopamin nöronlarının işareti (varsayılan +1)")
    args = ap.parse_args(argv)

    log("tablolar okunuyor")
    vis = pd.read_csv(FW / "visual_neuron_types.csv.gz")
    col = pd.read_csv(FW / "column_assignment.csv.gz")
    col = col[col.hemisphere == "right"]
    neurons = pd.read_csv(FW / "neurons.csv.gz")
    edges = read_edges()
    log(f"  {len(edges):,} nöron çifti")

    # --- düğümler: sağ optik lob
    nodes = vis[vis.side == "right"][["root_id", "type", "family", "subsystem", "category"]]
    n_central = 0
    if args.central:
        cand = pd.read_csv(OUT / "merkez_adaylar.csv")
        cand = cand[(cand.side == "right") & (cand.syn_vpn >= args.central_min_syn) &
                    (cand.oran_ol >= args.central_min_frac) & ~cand.post_root_id.isin(nodes.root_id)]
        # Tipi olmayan merkez nöronlar kendi tipini alır (başka nöronlarla α paylaşmaz)
        ctype = cand.primary_type.where(cand.primary_type.notna(), "merkez_" + cand.post_root_id.astype(str))
        central = pd.DataFrame({"root_id": cand.post_root_id.to_numpy(), "type": ctype.to_numpy(),
                                "family": "central", "subsystem": cand.ana_bolge.to_numpy(), "category": "central"})
        nodes = pd.concat([nodes, central], ignore_index=True)
        n_central = len(central)
        log(f"  merkez modül: {n_central} nöron ({central.type.nunique()} tip)")
    nodes = nodes.sort_values(["type", "root_id"]).reset_index(drop=True)
    nodes["type_id"] = pd.Categorical(nodes.type).codes.astype(np.int32)

    colmap = col.set_index("root_id").column_id
    nodes["column_id"] = colmap.reindex(nodes.root_id).to_numpy()
    nodes["column_source"] = np.where(nodes.column_id.notna(), "flywire", "")
    r16_col = assign_photoreceptor_columns(nodes, edges, col)
    m = (nodes.type == "R1-6") & nodes.column_id.isna()
    nodes.loc[m, "column_id"] = r16_col.reindex(nodes.loc[m, "root_id"]).to_numpy()
    nodes.loc[m & nodes.column_id.notna(), "column_source"] = "R1-6->L1-3"
    nodes["column_id"] = nodes.column_id.astype("Int64")

    ts = type_signs(nodes, neurons, args.modulator_sign)
    nodes = nodes.join(ts, on="type")
    missing_sign = nodes.sign.isna()
    nodes.loc[missing_sign, ["type_nt", "sign"]] = ["UNKNOWN", 1]
    nodes["sign"] = nodes.sign.astype(np.int8)
    nodes["photoreceptor"] = nodes.type.where(nodes.type.isin(PHOTORECEPTORS), "")
    nodes["is_output"] = nodes.category.eq("boundary")
    nodes.insert(0, "idx", np.arange(len(nodes), dtype=np.int32))

    # --- kolon tablosu (altıgen ızgara)
    cols = col.groupby("column_id")[["p", "q"]].first().reset_index()
    pr = nodes[nodes.photoreceptor != ""].groupby(["column_id", "photoreceptor"]).size().unstack(fill_value=0)
    for t in PHOTORECEPTORS:
        cols[f"n_{t}"] = pr[t].reindex(cols.column_id).fillna(0).astype(int).to_numpy() if t in pr else 0
    # Düzlem koordinatları. Komşuluk SİNAPSLARDAN doğrulandı: kolonlar arası sinapsların ~%88'i
    # (Δp,Δq) ∈ {(±1,0), (0,±1), ±(1,1)} ofsetlerinde; (1,-1) ikinci halka (~%2).
    # Buna uyan düzenli altıgen gömme (komşu mesafesi 1):
    #   x = (q - p)·√3/2 ,  y = (p + q)/2
    cols["hex_x"] = (cols.q - cols.p) * np.sqrt(3) / 2
    cols["hex_y"] = (cols.p + cols.q) / 2
    # Tuğla (offset) düzeni: aynı (q-p) değerine sahip kolonlar dikey bir sütun oluşturur,
    # komşu sütunlar yarım adım kaydırılmıştır. Piksel yamaları bu düzende çakışmadan dizilir.
    cols["brick_col"] = cols.q - cols.p         # yatay sütun indeksi
    cols["brick_row2"] = cols.p + cols.q        # dikey konum × 2 (yarım adımlar tam sayı kalsın)

    summary = {
        "neurons": int(len(nodes)),
        "cell_types": int(nodes.type_id.max() + 1),
        "columns": int(len(cols)),
        "output_neurons(boundary)": int(nodes.is_output.sum()),
        "central_neurons": n_central,
        "central_filter": ({"min_syn_from_vpn": args.central_min_syn, "min_frac_from_ol": args.central_min_frac}
                           if args.central else None),
        "photoreceptors": {t: int((nodes.type == t).sum()) for t in PHOTORECEPTORS},
        "R1-6_assigned_to_column": int((nodes.column_source == "R1-6->L1-3").sum()),
        "columns_with_R1-6": int((cols["n_R1-6"] > 0).sum()),
        "columns_with_R7": int((cols.n_R7 > 0).sum()),
        "columns_with_R8": int((cols.n_R8 > 0).sum()),
        "neurons_unknown_sign": int(missing_sign.sum()),
        "types_low_nt_agreement(<0.7)": int((ts.type_nt_agreement < 0.7).sum()),
        "modulator_sign": args.modulator_sign,
    }
    log(f"düğümler: {json.dumps(summary, ensure_ascii=False)}")

    for thr in args.thresholds:
        out_dir = OUT / (f"right_ol_merkez_thr{thr}" if args.central else f"right_ol_thr{thr}")
        log(f"eşik >= {thr}: {out_dir}")
        meta = build(thr, nodes, edges, args.rare_pair_syn, out_dir)
        cols.to_csv(out_dir / "columns.csv", index=False)
        meta = {**summary, **meta, "source": "FlyWire v783 (CC BY 4.0)", "built": time.strftime("%Y-%m-%d")}
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        log("  " + ", ".join(f"{k}={meta[k]}" for k in ("edges", "synapses", "alpha_groups",
                                                          "edges_in_rare_groups_pct", "excitatory_edges_pct",
                                                          "neurons_without_input", "spectral_radius_abs_syn")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
