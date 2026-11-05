"""
İşlenmiş konnektomu (tools/build_connectome.py çıktısı) yükler ve modelin ihtiyaç duyduğu
indeks yapılarını hazırlar.

Kolon sırası her yerde FlyWire column_id'ye göre artan sıradır (HexEye ile aynı).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

PHOTORECEPTOR_KINDS = ("R1-6", "R7", "R8")


@dataclass
class Connectome:
    nodes: pd.DataFrame
    columns: pd.DataFrame
    pre: np.ndarray            # [E] int64, (post, pre) sırasına göre sıralı
    post: np.ndarray           # [E]
    syn: np.ndarray            # [E] sinaps sayısı
    pair: np.ndarray           # [E] α grubu
    pairs: pd.DataFrame
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ yükleme
    @classmethod
    def load(cls, processed_dir: str | Path) -> "Connectome":
        d = Path(processed_dir)
        nodes = pd.read_csv(d / "nodes.csv")
        columns = pd.read_csv(d / "columns.csv").sort_values("column_id").reset_index(drop=True)
        e = np.load(d / "edges.npz")
        post, pre = e["post"].astype(np.int64), e["pre"].astype(np.int64)
        order = np.lexsort((pre, post))                 # CSR için (post, pre) sırası
        return cls(nodes=nodes, columns=columns, pre=pre[order], post=post[order],
                   syn=e["syn"][order].astype(np.float32), pair=e["pair"][order].astype(np.int64),
                   pairs=pd.read_csv(d / "type_pairs.csv"),
                   meta=json.loads((d / "meta.json").read_text(encoding="utf-8")))

    def randomized(self, seed: int = 0) -> "Connectome":
        """Rastgele topoloji kontrolü (TASARIM.md §3.7): bağlantıların presinaptik uçları tüm bağlantılar arasında
        karıştırılır. Korunanlar: nöron sayısı, her nöronun girdi ve çıktı bağlantı sayısı, her bağlantının sinaps
        sayısı ve α grubu (parametre sayısı birebir aynı), nöronların tipleri/kolonları/işaretleri.
        Değişen: kimin kime bağlı olduğu. İşaret yeni presinaptik nörondan gelir (Dale ilkesi korunur)."""
        rng = np.random.default_rng(seed)
        pre = self.pre[rng.permutation(self.n_edges)]
        order = np.lexsort((pre, self.post))
        return Connectome(nodes=self.nodes, columns=self.columns, pre=pre[order], post=self.post[order],
                          syn=self.syn[order], pair=self.pair[order], pairs=self.pairs,
                          meta={**self.meta, "randomized_seed": seed})

    # ------------------------------------------------------------------ boyutlar
    @property
    def n_neurons(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    @property
    def n_pairs(self) -> int:
        return int(self.pairs.pair_id.max() + 1)

    @property
    def n_types(self) -> int:
        return int(self.nodes.type_id.max() + 1)

    @cached_property
    def n_eyes(self) -> int:
        """§56 (H35): iki optik loblu konnektomda 2 (nodes.csv'de eye = "L" olan nöron varsa), yoksa 1."""
        return 2 if "eye" in self.nodes and self.nodes.eye.eq("L").any() else 1

    @property
    def n_eye_columns(self) -> int:
        """Bir gözün kolon sayısı (göz ızgarası, columns.csv)."""
        return len(self.columns)

    @property
    def n_columns(self) -> int:
        """Modeldeki toplam kolon sayısı: iki gözde sağ göz 0..C−1, sol göz C..2C−1 (sol göz aynalanmış görüntüyü görür)."""
        return len(self.columns) * self.n_eyes

    @cached_property
    def column_index(self) -> pd.Series:
        """FlyWire column_id → model kolon indeksi (0..n_columns-1)."""
        return pd.Series(np.arange(self.n_eye_columns), index=self.columns.column_id.to_numpy())

    @cached_property
    def neuron_column(self) -> np.ndarray:
        """[N] nöronun kolon indeksi, kolonu yoksa -1."""
        cid = self.nodes.column_id
        out = np.full(self.n_neurons, -1, np.int64)
        m = cid.notna().to_numpy()
        out[m] = self.column_index.reindex(cid[m].astype(int)).to_numpy()
        if self.n_eyes == 2:                                            # sol göz kolonları C..2C−1
            left = self.nodes.eye.eq("L").to_numpy() & (out >= 0)
            out[left] += self.n_eye_columns
        return out

    # ------------------------------------------------------------------ ağırlıklar
    def sign(self) -> np.ndarray:
        """[E] presinaptik nöronun işareti (+1 / −1)."""
        return self.nodes.sign.to_numpy()[self.pre].astype(np.float32)

    def normalized_syn(self, mode: str = "in_syn") -> np.ndarray:
        """[E] ağırlık tabanı. in_syn: her nöronun toplam girdi sinapsına bölünür, böylece
        tüm α = g iken her nöronun |girdi ağırlıkları toplamı| = g olur (kararlılık için)."""
        if mode == "none":
            return self.syn.copy()
        if mode == "in_syn":
            tot = np.bincount(self.post, weights=self.syn, minlength=self.n_neurons)
            return (self.syn / np.maximum(tot[self.post], 1)).astype(np.float32)
        if mode == "l2":
            # G6 (§45): kareler toplamının kökü → her nöronun Σ w² = 1 (Xavier benzeri). Bağımsız girdilerde
            # dalgalanma katmandan katmana korunur; "in_syn" ise ağırlıklı ortalama alıp √n kat söndürür.
            sq = np.bincount(self.post, weights=self.syn.astype(np.float64) ** 2, minlength=self.n_neurons)
            return (self.syn / np.maximum(np.sqrt(sq[self.post]), 1e-6)).astype(np.float32)
        raise ValueError(mode)

    # ------------------------------------------------------------------ girdi
    def photoreceptor_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kolonu bilinen fotoreseptörler: (nöron indeksi, kolon indeksi, tür 0=R1-6 1=R7 2=R8)."""
        kind = self.nodes.photoreceptor.map({k: i for i, k in enumerate(PHOTORECEPTOR_KINDS)})
        col = self.neuron_column
        m = kind.notna().to_numpy() & (col >= 0)
        idx = np.flatnonzero(m)
        return idx, col[idx], kind.to_numpy()[idx].astype(np.int64)

    # ------------------------------------------------------------------ okuma (readout)
    @cached_property
    def columnar_types(self) -> list[str]:
        return sorted(self.nodes.type[self.neuron_column >= 0].unique())

    def columnar_readout(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kolon × kolonsal tip ortalaması için seyrek matris (satır, sütun, değer).
        Satır = kolon·n_ctype + tip, sütun = nöron. Aynı kolondaki aynı tip nöronların ortalaması."""
        col = self.neuron_column
        tmap = {t: i for i, t in enumerate(self.columnar_types)}
        idx = np.flatnonzero(col >= 0)
        t = self.nodes.type.to_numpy()[idx]
        rows = col[idx] * len(tmap) + np.array([tmap[x] for x in t])
        cnt = np.bincount(rows, minlength=self.n_columns * len(tmap))
        return rows, idx, (1.0 / cnt[rows]).astype(np.float32)

    @cached_property
    def output_neurons(self) -> np.ndarray:
        """Çıkış (görsel projeksiyon, 'boundary') nöronlarının indeksleri."""
        return np.flatnonzero(self.nodes.is_output.to_numpy())

    @cached_property
    def central_neurons(self) -> np.ndarray:
        """Merkez beyin modülünün nöronları (günlük §41; yalnız `right_ol_merkez_*` konnektomlarında, yoksa boş)."""
        if "category" not in self.nodes:
            return np.zeros(0, np.int64)
        return np.flatnonzero(self.nodes.category.eq("central").to_numpy())

    @cached_property
    def output_types(self) -> list[str]:
        return sorted(set(self.nodes.type.to_numpy()[self.output_neurons]))

    def receptive_fields(self, hops: int = 4) -> np.ndarray:
        """[N, n_columns] her nöronun kolonlar üzerindeki alıcı alanı (satır toplamı 1 ya da 0).

        Kolonlu nöronlar kendi kolonlarında tek-sıcak (one-hot) başlar. Kolonsuz nöronların alanı,
        presinaptik partnerlerinin alanlarının sinaps ağırlıklı ortalamasıdır; `hops` kez yayılır.
        Yani alan tamamen konnektomdan hesaplanır (işaretler yok sayılır: 'nereden bilgi alıyor')."""
        from scipy.sparse import csr_matrix
        N, C = self.n_neurons, self.n_columns
        col = self.neuron_column
        has_col = col >= 0
        R = np.zeros((N, C), np.float32)
        R[np.flatnonzero(has_col), col[has_col]] = 1.0
        A = csr_matrix((self.syn, (self.post, self.pre)), shape=(N, N))
        for _ in range(hops):
            prop = np.asarray(A @ R)
            s = prop.sum(1, keepdims=True)
            prop = np.divide(prop, s, out=np.zeros_like(prop), where=s > 0)
            R = np.where(has_col[:, None], R, prop)
        return R
