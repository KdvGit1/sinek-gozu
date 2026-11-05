"""
Sinek optik lobu topolojili tespit modeli (TASARIM.md Bölüm 3).

  görüntü karosu ─▶ HexEye (796 kolon, p×p yama)
                 ─▶ OmmatidiumEncoder: yama → kolonun R1-6 / R7 / R8 sürüşü (+ isteğe bağlı köprü)
                 ─▶ ConnectomeCore: FlyWire sağ optik lob, döngüler dahil, T zaman adımı
                       v ← (1−λ)·v + λ·(W·relu(v) + b + I_foto)
                       W[post, pre] = işaret(pre) · softplus(α[tip çifti]) · sinaps / girdi_toplamı(post)
                 ─▶ Okuma: kolon × kolonsal tip ortalaması + çıkış (VPN) tiplerinin alıcı alan ağırlıklı etkinliği
                 ─▶ Başlık (her kolonda paylaşılan MLP): ısı (merkez burada mı?), ofset (dy, dx), log boyut (h, w)

Kararlar: d = 1, sürekli (oran tabanlı) nöron, FP32 eğitim; `train_connectome=False` → rezervuar deneyi.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .connectome import Connectome
from .eye import HexEye


@dataclass
class ModelConfig:
    processed_dir: str = "data/connectome/processed/right_ol_thr5"
    in_channels: int = 1              # termal 1, RGB 3
    patch: int = 4                    # yama kenarı p (piksel)
    steps: int = 10                   # zaman adımı T (derinlik 4–5 sinaps → ≥ 5)
    weight_norm: str = "in_syn"       # in_syn | l2 (G6: Σw²=1, Xavier benzeri) | none
    activation: str = "relu"          # relu | sat (G6: doymalı oran, tanh(relu(v)) ∈ [0, 1); gerçek nöronlar gibi)
    init_gain: float = 0.8            # tüm α = g → her nöronun |girdi ağırlıkları| toplamı g (<1: kararlı)
    init_leak: float = 0.5            # λ = dt/τ başlangıcı
    init_bias: float = 0.1
    train_connectome: bool = True     # False: α, bias, λ sabit → rezervuar (sadece kodlayıcı + başlık eğitilir)
    encoder_hidden: int = 32
    bridge_dim: int = 0               # ince ayrıntı köprüsü (0 = kapalı)
    head_hidden: int = 64
    rf_hops: int = 4
    rf_min: float = 1e-3              # alıcı alanda bundan küçük katkılar atılır (hız için)
    readout: str = "relu"             # relu: etkinlik relu(v) okunur | voltage: zar gerilimi v okunur
                                      # (sessiz nöronlar da derecelendirilmiş bilgi taşır; 29.09 teşhisi: relu'da
                                      #  okuma özelliklerinin %59'u ölüydü)
    core: bool = True                 # False: çekirdeksiz kontrol (kodlayıcı köprüsü → doğrudan başlık);
                                      # bridge_dim > 0 gerekir. Sinek çekirdeğinin katkısını ölçer.
    topology: str = "fly"             # fly: FlyWire bağlantıları | random: rastgele topoloji kontrolü
                                      # (derece, sinaps sayısı ve α grupları korunur; bkz. Connectome.randomized)
    topology_seed: int = 0
    head_context: str = "none"        # none: başlık her kolonda tek başına | hex7: kolon + 6 altıgen komşusu
                                      # hex7d (§71 H56b): doğrusal başlık + 6 komşunun kolon tipleri (sıfırdan, özdeş)
                                      # (altıgen ızgarada tek katmanlı evrişim gibi; kutu boyutu için)
                                      # | hex19: kolon + 2 halka (18 komşu; H7, saflık kontrolü)
    adapt: str = "none"               # none | local: fotoreseptör adaptasyonu (H2). Kodlayıcıya ham pikselin yanında
                                      # yerel ortalaması çıkarılıp yerel kontrasta bölünmüş hâli de verilir
                                      # (sineğin fotoreseptörlerindeki ışık uyumu / kazanç kontrolü karşılığı)
    readout_photo: bool = True        # §44: False → fotoreseptörler (R1-6, R7, R8) okunmaz. Çoğu ağdan girdi almaz;
                                      # gerilimleri kodlayıcı çıktısının kopyası → konnektomu atlayan gizli köprü
    readout_norm: str = "layer"       # layer: başlıkta LayerNorm | batch: özellik başına sabit standartlaştırma (G1;
                                      # BatchNorm, değerlendirmede sabit istatistiklerle doğrusal)
    type_gain: bool = False           # G4: hücre tipi başına öğrenilen kazanç (girdi akımını ölçekler; uyarılabilirlik)
    encoder_act: str = "relu"         # relu | none (none: kodlayıcı tamamen doğrusal = mercek + fotoreseptör)
    input_lamina: bool = False        # G8 (§45): kodlayıcı fotoreseptörlerin yanında her kolonun L1, L2, L3'üne de girdi
                                      # verir. FlyWire'da lamina eksik: R1-6 başına ort. 2,1 bağlantı; L1 girdisinin
                                      # yalnız %40'ı, L2/L3'ün %23'ü fotoreseptörden. Eksik R→L sinapslarının vekili.
                                      # Girdi alan nöronlar okunmaz (readout_photo=False ile birlikte).
    bridge_drop: float = 0.0          # H16/B3: eğitimde köprü karo bazında bu olasılıkla kapatılır (kalanlar 1/(1−p)
                                      # ile ölçeklenir); başlığı sinek çekirdeğini okumaya zorlar. Değerlendirmede etkisiz.
    channels: int = 1                 # H14: nöron başına kanal d. d > 1: her kanal kendi α'sıyla yayılır (köşegen),
                                      # ardından hücre tipi başına d×d karıştırma; topoloji aynı (TASARIM §5.6)
    quality: bool = False             # H4: başlığa 6. çıktı "bu kolonun kutusu ne kadar iyi" (IoU tahmini);
                                      # tespit skoru = ısı × kalite (FCOS merkezlilik / YOLOX IoU dalı karşılığı)
    view: str = "A"                   # H35 (§56), yalnız iki loblu konnektomda: A — iki göz de bütün karoyu görür
                                      # (sol göz aynalanmış karoyu); B — her göz kendi yarısını + ortada ortak şerit
    view_overlap: float = 0.2         # B'de ortak şeridin karo genişliğine oranı
    microsaccade: int = 0             # H51 (§63): T adım boyunca karo bu genlikte (px) titreşir (eğitim + değerlendirme)
    act_soft: bool = False            # H52: relu yerine tip başına keskinliği (β) öğrenilen softplus_β (dereceli nöron)
    noise_sigma: float = 0.0          # H43: eğitimde her adımda gerilime N(0, σ²) gürültü (stokastik rezonans)
    fatigue: bool = False             # H40: kısa süreli sinaptik yorgunluk (U, τ) + nöron adaptasyonu (β, τa), tip başına
    shunt: bool = False               # H42: iletkenlik tabanlı sinaps: G+·(E+ − v) + G−·(E− − v), E+ = 1, E− = shunt_e_inh
    shunt_e_inh: float = 0.0
    modulators: bool = False          # H45: DA/SER/OCT nöronlarının etkinliği innerve ettikleri tiplerin kazancını çarpar
    superposition: bool = False       # H44: her R1-6 kolonun yarım kolon kaydırılmış 6 bakışından birini alır; L1–L3 toplamı
    lamina_weights: bool = False      # §71 H57: L1–L3 altı bakışı tür başına öğrenilen ağırlıklarla toplar (yön korunur)
    microsaccade_views: bool = False  # §71 hata #8: süperpozisyon + mikrosakkad birlikte (adım başına kaydırılmış 7 bakış);
                                      # False iken eski davranış: süperpozisyon açıksa mikrosakkad uygulanmaz
    stp: bool = False                 # H53 (§65): işaretli kısa süreli plastisite: çıkış × (1 + φ_k·s), s = nöronun yakın
                                      # geçmiş etkinliği, φ_k = tanh(φ_raw) ∈ (−1, 1): + kolaylaşma, − yorulma; başta 0
    homeostasis: bool = False         # H41b (§62): tip başına gradyansız homeostatik eşik (homeo_bias, bias'a eklenir)
                                      # ve log-kazanç (homeo_gain, döngüsel girdiyi çarpar); yalnız uyku evresi değiştirir

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- seyrek çarpım
# Hız seçenekleri (§46; sonucu değiştirmezler, eşdeğerlik testiyle doğrulanır; ortam değişkeniyle açılır)
FAST_ADAPT = os.environ.get("FLYBRAIN_ADAPT_SEP", "0") == "1"   # ayrılabilir kutu ortalaması (k×k → 1×k + k×1)
CSR_INT32 = os.environ.get("FLYBRAIN_CSR32", "0") == "1"        # seyrek matris indeksleri 32 bit
# §75: tipe / α grubuna göre dağıtma index_select ile (geri yayılımı index_add; x[idx]'in geri yayılımı her çağrıda
# 100 bin indeksi sıralıyordu: modülatörde adım başına ~2,4 ms) ve döngüde GPU senkronizasyonu yok. İleri geçiş birebir
# aynı; geri yayılım yalnız toplama sırası kadar farklı (GPU belirlenimsizliği düzeyi). FLYBRAIN_HIZ75=0 → eski yol.
HIZ75 = os.environ.get("FLYBRAIN_HIZ75", "1") == "1"
# §80: süperpozisyonda 7 kaydırılmış bakış × 2 göz tek bir piksel toplamasıyla (kaydırma = kenarı uzatılmış indeks,
# sol göz = aynalanmış indeks; aynı pikseller, aynı kodlayıcı) ve forward'ın baştaki kullanılmayan encode'u atlanır
# (köprüsüz modelde sürüşü drive() yeniden hesaplıyordu). RGB'de kodlayıcı adımın en pahalı kısmıydı. FLYBRAIN_HIZ80=0 → eski yol.
HIZ80 = os.environ.get("FLYBRAIN_HIZ80", "1") == "1"
# §80b: birleşik kodlayıcı bir çağrıda en çok bu kadar yama elemanı açar (7 bakış × C × Cin × p² × yığın). Değerlendirmede
# 256 karoluk yığın RGB'de ~8 GB açıp GPU belleğini taşırıyordu (val800 83 → 600 sn); büyük yığın parçalara bölünür
# (örnek başına hesap → sonuç aynı). 64 M ≈ eski yolun tek bakış tepe belleği; eğitim yığını (32) bölünmez.
FUSED_ELEMS = 64_000_000

SDDMM_CHUNK = 32
_ZEROS: dict = {}   # cuSPARSE SDDMM yığın büyüdükçe orantısız yavaşlıyor (4× karo → 8× süre); parçalı çalıştırılır


def _edge_grad(g: torch.Tensor, x: torch.Tensor, crow, col, post, pre) -> torch.Tensor:
    """g_vals[e] = Σ_b g[post_e, b]·x[pre_e, b].
    GPU: seyrek örneklenmiş matris çarpımı (SDDMM); E×B'lik ara tensör oluşturmaz (29.09 profil: geri geçişin
    yarısı bu ara tensörlerin bellek trafiğiydi). CPU: doğrudan formül (gradcheck ile doğrulanan)."""
    if g.device.type != "cuda":
        return (g[post] * x[pre]).sum(-1)
    n = crow.numel() - 1
    key = (col.numel(), g.dtype, str(g.device))
    if key not in _ZEROS:                                   # §46: desen değerleri (beta = 0 → kullanılmıyor) bir kez
        _ZEROS[key] = torch.zeros(col.numel(), dtype=g.dtype, device=g.device)
    pat = torch.sparse_csr_tensor(crow, col, _ZEROS[key], (n, n), check_invariants=False)
    out = None
    for s in range(0, g.shape[1], SDDMM_CHUNK):
        part = torch.sparse.sampled_addmm(pat, g[:, s:s + SDDMM_CHUNK].contiguous(),
                                          x[:, s:s + SDDMM_CHUNK].T.contiguous(), beta=0.0).values()
        out = part if out is None else out + part
    return out


class _SparseMatmul(torch.autograd.Function):
    """y = W @ x, W: CSR (satır = post). Geri yayılım: gx = Wᵀ @ g, g_vals[e] = Σ_b g[post_e, b]·x[pre_e, b].
    PyTorch'un genel seyrek geri yayılımından yüzlerce kat hızlı ve belleği adım başına O(N·B)."""

    @staticmethod
    def forward(ctx, vals, x, crow, col, crow_t, col_t, perm_t, post, pre, vals_t=None):
        """vals_t: vals[perm_t] (transpoz değerleri); verilirse geri geçişte her adımda yeniden toplanmaz (§46)."""
        n = crow.numel() - 1
        W = torch.sparse_csr_tensor(crow, col, vals, (n, n), check_invariants=False)
        ctx.save_for_backward(vals, x, crow, col, crow_t, col_t, perm_t, post, pre,
                              vals_t if vals_t is not None else vals.new_zeros(0))
        return W @ x

    @staticmethod
    def backward(ctx, g):
        vals, x, crow, col, crow_t, col_t, perm_t, post, pre, vals_t = ctx.saved_tensors
        n = crow_t.numel() - 1
        g_vals = g_x = None
        if ctx.needs_input_grad[0]:
            g_vals = _edge_grad(g, x, crow, col, post, pre)
        if ctx.needs_input_grad[1]:
            vt = vals_t if vals_t.numel() else vals[perm_t]
            Wt = torch.sparse_csr_tensor(crow_t, col_t, vt, (n, n), check_invariants=False)
            g_x = Wt @ g
        return g_vals, g_x, None, None, None, None, None, None, None, None


class _FixedSpMM(torch.autograd.Function):
    """y = R @ x, R sabit (öğrenilmeyen) seyrek okuma matrisi. Transpoz önceden hesaplanır; PyTorch'un genel
    yolu her geri geçişte transpozu baştan kuruyordu (to_sparse_csr, 29.09 profili)."""

    @staticmethod
    def forward(ctx, x, crow, col, val, crow_t, col_t, val_t, n_rows):
        ctx.save_for_backward(crow_t, col_t, val_t)
        ctx.n_in = x.shape[0]
        R = torch.sparse_csr_tensor(crow, col, val, (n_rows, x.shape[0]), check_invariants=False)
        return R @ x

    @staticmethod
    def backward(ctx, g):
        crow_t, col_t, val_t = ctx.saved_tensors
        Rt = torch.sparse_csr_tensor(crow_t, col_t, val_t, (ctx.n_in, g.shape[0]), check_invariants=False)
        return Rt @ g.contiguous(), None, None, None, None, None, None, None


def _transpose_csr(crow: torch.Tensor, col: torch.Tensor, val: torch.Tensor, n_cols: int):
    """CSR (satır r, sütun c) → transpozun CSR'ı (crow_t, col_t, val_t)."""
    rows = np.repeat(np.arange(crow.numel() - 1), np.diff(crow.numpy()))
    c = col.numpy()
    o = np.lexsort((rows, c))
    return _csr(c[o], rows[o], n_cols), torch.as_tensor(rows[o]), val[torch.as_tensor(o)]


def _csr(rows: np.ndarray, cols: np.ndarray, n_rows: int):
    """(rows, cols) satır-sıralı olmalı → crow."""
    return torch.as_tensor(np.r_[0, np.cumsum(np.bincount(rows, minlength=n_rows))], dtype=torch.long)


def _inv_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))


# --------------------------------------------------------------------------- model
class FeatureNorm(nn.Module):
    """G1: özellik başına standartlaştırma ([B, C, F] → aynı şekil). Eğitimde yığın istatistikleri, değerlendirmede
    sabit (kayan ortalama) istatistikler → değerlendirmede tamamen doğrusal. Küçük ama tutarlı nöron sinyallerini
    okunabilir ölçeğe getirir; başlığı "düşündürmez"."""

    def __init__(self, f: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(f, affine=False, momentum=0.01)

    def forward(self, x):
        return self.bn(x.reshape(-1, x.shape[-1])).reshape(x.shape)


class FlyEyeDetector(nn.Module):
    def __init__(self, cfg: ModelConfig, connectome: Connectome | None = None):
        super().__init__()
        self.cfg = cfg
        c = connectome or Connectome.load(cfg.processed_dir)
        self.eye = HexEye.from_columns(c.columns, cfg.patch)
        N, C = c.n_neurons, c.n_columns
        self.n_neurons, self.n_columns = N, C
        self.n_eyes, self.n_eye_columns = c.n_eyes, c.n_eye_columns   # H35: iki gözde C = 2 × göz kolonu
        if cfg.view not in ("A", "B"):
            raise ValueError("view: A | B")
        if self.n_eyes == 2 and cfg.view == "B":                    # her göz karonun kendi yanını görür
            x = torch.as_tensor(self.eye.centers()[:, 1], dtype=torch.float32)
            vis = x >= self.eye.width * (0.5 - cfg.view_overlap / 2)    # (sol göz aynalı karoda: aynı koşul)
            self.register_buffer("eye_visible", torch.cat([vis, vis]).float(), persistent=False)
        else:
            self.eye_visible = None

        # --- çekirdek bağlantıları (CSR, satır = post). Rastgele topoloji kontrolünde yalnızca çekirdeğin
        #     kablolaması değişir; göz, fotoreseptör girişi ve okuma gerçek konnektomdan kalır.
        if cfg.topology not in ("fly", "random"):
            raise ValueError("topology: fly | random")
        cc = c.randomized(cfg.topology_seed) if cfg.topology == "random" else c
        post, pre = cc.post, cc.pre
        self.register_buffer("crow", _csr(post, pre, N))
        self.register_buffer("col", torch.as_tensor(pre))
        order_t = np.lexsort((post, pre))
        self.register_buffer("crow_t", _csr(pre[order_t], post[order_t], N))
        self.register_buffer("col_t", torch.as_tensor(post[order_t]))
        self.register_buffer("perm_t", torch.as_tensor(order_t))
        self.register_buffer("post", torch.as_tensor(post))
        self.register_buffer("pre", torch.as_tensor(pre))
        self.register_buffer("wbase", torch.as_tensor(cc.sign() * cc.normalized_syn(cfg.weight_norm)))
        self.register_buffer("pair", torch.as_tensor(cc.pair))
        self.register_buffer("type_id", torch.as_tensor(c.nodes.type_id.to_numpy(), dtype=torch.long))
        self._n_types = int(self.type_id.max()) + 1                      # §75: döngüde .max() senkronizasyonu yok
        self.register_buffer("central_idx", torch.as_tensor(c.central_neurons), persistent=False)
        self._csr_int32 = CSR_INT32

        if not cfg.core and cfg.bridge_dim <= 0:
            raise ValueError("çekirdeksiz model (core=False) için bridge_dim > 0 olmalı")
        if cfg.readout not in ("relu", "voltage"):
            raise ValueError("readout: relu | voltage")
        train = cfg.train_connectome and cfg.core
        d = cfg.channels
        a_shape, b_shape = ((c.n_pairs,), (c.n_types,)) if d == 1 else ((c.n_pairs, d), (c.n_types, d))
        self.alpha_raw = nn.Parameter(torch.full(a_shape, _inv_softplus(cfg.init_gain)), requires_grad=train)
        self.bias = nn.Parameter(torch.full(b_shape, cfg.init_bias), requires_grad=train)
        if d > 1:                                                     # tip başına kanal karıştırma, birimden başlar
            self.mix = nn.Parameter(torch.eye(d).repeat(c.n_types, 1, 1), requires_grad=train)
        if cfg.type_gain:
            self.gain_raw = nn.Parameter(torch.zeros(c.n_types), requires_grad=train)   # exp(0) = 1
        if cfg.homeostasis:                                           # H41b: gradyansız ofsetler (kontrol noktasına yazılır)
            self.register_buffer("homeo_bias", torch.zeros(c.n_types))
            self.register_buffer("homeo_gain", torch.zeros(c.n_types))
        if cfg.act_soft:                                              # H52: β = 1 + softplus(β_raw), başta 50 (≈ relu)
            self.beta_raw = nn.Parameter(torch.full((c.n_types,), _inv_softplus(49.0)), requires_grad=train)
        if cfg.fatigue:                                               # H40: özdeşlik başlangıcı (U ≈ 0, β ≈ 0)
            self.fat_u = nn.Parameter(torch.full((c.n_types,), -10.0), requires_grad=train)
            self.fat_tau = nn.Parameter(torch.zeros(c.n_types), requires_grad=train)
            self.ad_beta = nn.Parameter(torch.full((c.n_types,), -10.0), requires_grad=train)
            self.ad_tau = nn.Parameter(torch.zeros(c.n_types), requires_grad=train)
        if cfg.stp:                                                   # H53: özdeşlik başlangıcı φ = 0 (tanh'ın en dik yeri)
            self.stp_phi = nn.Parameter(torch.zeros(c.n_types), requires_grad=train)
            self.stp_tau = nn.Parameter(torch.zeros(c.n_types), requires_grad=train)
        if cfg.modulators:                                            # H45: modülatör tip j → innerve ettiği tip k
            tnt = c.nodes.type_nt.astype(str).to_numpy()
            tid = c.nodes.type_id.to_numpy()
            mod_n = np.flatnonzero(np.isin(tnt, ["DA", "SER", "OCT"]))
            mtypes = np.unique(tid[mod_n])
            jmap = {t: j for j, t in enumerate(mtypes)}
            is_mod = np.zeros(N, bool)
            is_mod[mod_n] = True
            e = is_mod[cc.pre]
            pairs = np.unique(np.stack([np.array([jmap[t] for t in tid[cc.pre[e]]]), tid[cc.post[e]]], 1), axis=0)
            mj = np.array([jmap[t] for t in tid[mod_n]])
            self.register_buffer("mod_idx", torch.as_tensor(mod_n))
            self.register_buffer("mod_j", torch.as_tensor(mj))
            self.register_buffer("mod_cnt", torch.bincount(torch.as_tensor(mj), minlength=len(mtypes)).float().clamp(min=1))
            self.register_buffer("modp_j", torch.as_tensor(pairs[:, 0]))
            self.register_buffer("modp_k", torch.as_tensor(pairs[:, 1]))
            self.mod_beta = nn.Parameter(torch.zeros(len(pairs)), requires_grad=train)   # özdeşlik: β = 0
            self.n_mod_types = len(mtypes)
        lk = math.log(cfg.init_leak / (1 - cfg.init_leak))
        self.leak_raw = nn.Parameter(torch.full((c.n_types,), lk), requires_grad=train)

        # --- fotoreseptör girdisi
        pi, pc, pk = c.photoreceptor_inputs()
        self.n_in_kinds = 3
        if cfg.input_lamina:                                       # G8: lamina monopolar hücreleri (L1–L3)
            tt = c.nodes.type.to_numpy()
            colv = c.neuron_column
            extra = [(i, colv[i], 3 + ("L1", "L2", "L3").index(tt[i])) for i in np.flatnonzero(colv >= 0)
                     if tt[i] in ("L1", "L2", "L3")]
            pi = np.r_[pi, [e[0] for e in extra]].astype(np.int64)
            pc = np.r_[pc, [e[1] for e in extra]].astype(np.int64)
            pk = np.r_[pk, [e[2] for e in extra]].astype(np.int64)
            self.n_in_kinds = 6
        self.register_buffer("pr_idx", torch.as_tensor(pi))
        self.register_buffer("pr_col", torch.as_tensor(pc))
        self.register_buffer("pr_kind", torch.as_tensor(pk))
        if cfg.superposition and cfg.lamina_weights:                   # §71 H57: [L1, L2, L3] × [bakış 1..6], başta 1
            self.lam_w = nn.Parameter(torch.ones(3, 6), requires_grad=train)
        if cfg.superposition:                                          # H44: bakış 0 merkez, 1–6 kaydırılmış, 7 = 1–6 toplamı
            pv = np.zeros(len(pi), np.int64)
            rank = {}
            for i in np.flatnonzero(pk == 0):                          # kolon içi sıra → bakış 1..6
                cnum = int(pc[i])
                pv[i] = 1 + rank.get(cnum, 0) % 6
                rank[cnum] = rank.get(cnum, 0) + 1
            pv[pk >= 3] = 7                                            # L1–L3: altı bakışın toplamı
            self.register_buffer("pr_view", torch.as_tensor(pv))

        # --- ommatidyum kodlayıcı (tüm kolonlarda paylaşılan)
        if cfg.adapt not in ("none", "local", "column"):
            raise ValueError("adapt: none | local | column")
        k = cfg.in_channels * (1 if cfg.adapt == "none" else 2) * cfg.patch * cfg.patch
        if cfg.adapt != "none":
            self.adapt_k = 4 * cfg.patch + 1                           # local: uyum penceresi ~4 kolon
            self.adapt_c = nn.Parameter(torch.tensor(-2.0))           # softplus → kontrast tabanı (öğrenilir)
        if cfg.encoder_act not in ("relu", "none"):
            raise ValueError("encoder_act: relu | none")
        self.encoder = nn.Sequential(nn.Linear(k, cfg.encoder_hidden),
                                     *([nn.ReLU()] if cfg.encoder_act == "relu" else []))
        self.to_photo = nn.Linear(cfg.encoder_hidden, self.n_in_kinds * cfg.channels)
        self.bridge = nn.Linear(cfg.encoder_hidden, cfg.bridge_dim) if cfg.bridge_dim > 0 else None
        if not cfg.core:
            self.to_photo.requires_grad_(False)                          # çekirdeksizde kullanılmıyor

        # --- okuma 1: kolon × kolonsal tip ortalaması
        self.columnar_types = c.columnar_types
        r, n, v = c.columnar_readout()
        if not cfg.readout_photo:                                  # §44: fotoreseptörler okunmaz
            from .connectome import PHOTORECEPTOR_KINDS
            skip = set(PHOTORECEPTOR_KINDS) | ({"L1", "L2", "L3"} if cfg.input_lamina else set())
            self.columnar_types = [t for t in c.columnar_types if t not in skip]
            tmap = {t: i for i, t in enumerate(self.columnar_types)}
            colv = c.neuron_column
            types = c.nodes.type.to_numpy()
            n = np.array([i for i in np.flatnonzero(colv >= 0) if types[i] in tmap], dtype=np.int64)
            r = colv[n] * len(tmap) + np.array([tmap[types[i]] for i in n])
            cnt = np.bincount(r, minlength=C * len(tmap))
            v = (1.0 / cnt[r]).astype(np.float32)
        o = np.lexsort((n, r))
        n_ct = len(self.columnar_types)
        self.register_buffer("rc_crow", _csr(r[o], n[o], C * n_ct))
        self.register_buffer("rc_col", torch.as_tensor(n[o]))
        self.register_buffer("rc_val", torch.as_tensor(v[o]))

        # --- okuma 2: çıkış (VPN) tipleri, alıcı alan ağırlıklı ortalama
        self.output_types = c.output_types
        out = c.output_neurons
        R = c.receptive_fields(cfg.rf_hops)[out]                      # [n_out, C]
        R[R < cfg.rf_min] = 0
        tmap = {t: i for i, t in enumerate(self.output_types)}
        ot = np.array([tmap[t] for t in c.nodes.type.to_numpy()[out]])
        nz_n, nz_c = np.nonzero(R)
        w = R[nz_n, nz_c]
        rows = nz_c * len(tmap) + ot[nz_n]                            # satır = kolon·n_otype + tip
        den = np.bincount(rows, weights=w, minlength=C * len(tmap))
        w = w / np.maximum(den[rows], 1e-12)
        o = np.lexsort((nz_n, rows))
        self.register_buffer("ro_crow", _csr(rows[o], nz_n[o], C * len(tmap)))
        self.register_buffer("ro_col", torch.as_tensor(out[nz_n[o]]))  # doğrudan nöron indeksi
        self.register_buffer("ro_val", torch.as_tensor(w[o].astype(np.float32)))
        # Okuma matrislerinin transpozları (geri geçiş için; türetilmiş veri → kontrol noktasına yazılmaz)
        for k, n_rows in (("rc", C * n_ct), ("ro", C * len(tmap))):
            ct, cc, cv = _transpose_csr(getattr(self, f"{k}_crow"), getattr(self, f"{k}_col"),
                                        getattr(self, f"{k}_val"), N)
            self.register_buffer(f"{k}_crow_t", ct, persistent=False)
            self.register_buffer(f"{k}_col_t", cc, persistent=False)
            self.register_buffer(f"{k}_val_t", cv, persistent=False)

        # --- tespit başlığı
        feat = ((n_ct + len(self.output_types)) * cfg.channels + cfg.bridge_dim) if cfg.core else cfg.bridge_dim
        self.n_features = feat
        H = cfg.head_hidden
        n_out = 6 if cfg.quality else 5
        if cfg.readout_norm not in ("layer", "batch"):
            raise ValueError("readout_norm: layer | batch")
        norm = (lambda f: FeatureNorm(f)) if cfg.readout_norm == "batch" else (lambda f: nn.LayerNorm(f))
        self.head_nb = None
        if cfg.head_context == "hex7d":                         # §71 H56b: doğrusal okuma + komşu kolonların kolon
            self.head_in = None                                 # tipleri (köprü yok, ara katman yok; komşu kolu 0'dan)
            self.head = nn.Sequential(norm(feat), nn.Linear(feat, n_out))
            nb = torch.as_tensor(self.eye.neighbors())                          # [C_göz, 6]
            if self.n_eyes == 2:
                Ce = self.n_eye_columns
                nb = torch.cat([nb, torch.where(nb >= 0, nb + Ce, nb)])
            self.register_buffer("nb_idx", torch.where(nb >= 0, nb, torch.full_like(nb, C)), persistent=False)
            self.head_nb_norm = nn.LayerNorm(n_ct * cfg.channels, elementwise_affine=False)
            self.head_nb = nn.Linear(6 * n_ct * cfg.channels, n_out, bias=False)
            nn.init.zeros_(self.head_nb.weight)
        elif cfg.head_context == "none" and H == 0:              # doğrusal okuma (flyvis gibi): bütün doğrusal
            self.head_in = None                                 # olmayan iş çekirdekte (B5/B6)
            self.head = nn.Sequential(norm(feat), nn.Linear(feat, n_out))
        elif cfg.head_context == "none":
            self.head_in = None
            self.head = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, H), nn.ReLU(), nn.Linear(H, n_out))
        elif cfg.head_context in ("hex7", "hex19"):
            nb = torch.as_tensor(self.eye.neighbors() if cfg.head_context == "hex7" else self.eye.neighbors2())
            self.head_in = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, H), nn.ReLU())
            self.head = nn.Sequential(nn.Linear((nb.shape[1] + 1) * H, H), nn.ReLU(), nn.Linear(H, n_out))
            self.register_buffer("nb_idx", torch.where(nb >= 0, nb, torch.full_like(nb, C)), persistent=False)
        else:
            raise ValueError("head_context: none | hex7 | hex19 | hex7d")
        nn.init.constant_(self.head[-1].bias, 0.0)
        with torch.no_grad():
            self.head[-1].bias[0] = -4.0                              # başlangıçta "insan yok" (odak kaybı için)
        if self._csr_int32:
            self._apply_csr_int32()

    def _apply_csr_int32(self):
        """§46: CSR satır/sütun indekslerini 32 bit yap (değerler ve sonuçlar aynı)."""
        for name in ("crow", "col", "crow_t", "col_t", "rc_crow", "rc_col", "rc_crow_t", "rc_col_t",
                     "ro_crow", "ro_col", "ro_crow_t", "ro_col_t"):
            if hasattr(self, name):
                setattr(self, name, getattr(self, name).to(torch.int32))

    # ------------------------------------------------------------------ yardımcılar
    def _sel(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """x[idx] (ilk eksende); §75: HIZ75 açıkken index_select (aynı ileri sonuç, sıralamasız geri yayılım)."""
        return x.index_select(0, idx) if HIZ75 else x[idx]

    def weights(self) -> torch.Tensor:
        """[E] (d = 1) ya da [E, d]."""
        a = self._sel(nn.functional.softplus(self.alpha_raw), self.pair)
        return self.wbase * a if a.dim() == 1 else self.wbase[:, None] * a

    def head_forward(self, feats: torch.Tensor) -> torch.Tensor:
        """[B, C, F] → [B, C, 5|6]. hex7/hex19: kolonun ve 6/18 komşusunun ara temsilleri birlikte değerlendirilir
        (göz kenarında olmayan komşu = 0)."""
        if self.head_nb is not None:                                         # §71 H56b
            k = self.head_nb_norm.normalized_shape[0]
            fc = self.head_nb_norm(feats[..., :k])                               # kolon tipleri, kolon başına normlu
            fp = torch.cat([fc, fc.new_zeros(fc.shape[0], 1, k)], dim=1)       # "komşu yok" satırı
            ctx = fp[:, self.nb_idx].reshape(fc.shape[0], fc.shape[1], -1)      # [B, C, 6k]
            return self.head(feats) + self.head_nb(ctx)
        if self.head_in is None:
            return self.head(feats)
        h = self.head_in(feats)                                               # [B, C, H]
        hp = torch.cat([h, h.new_zeros(h.shape[0], 1, h.shape[2])], dim=1)    # sona "komşu yok" satırı
        ctx = hp[:, self.nb_idx].reshape(h.shape[0], h.shape[1], -1)          # [B, C, 6H]
        return self.head(torch.cat([h, ctx], dim=-1))

    def _readout_mm(self, k: str, a: torch.Tensor) -> torch.Tensor:
        crow = getattr(self, f"{k}_crow")
        return _FixedSpMM.apply(a, crow, getattr(self, f"{k}_col"), getattr(self, f"{k}_val"),
                                getattr(self, f"{k}_crow_t"), getattr(self, f"{k}_col_t"),
                                getattr(self, f"{k}_val_t"), crow.numel() - 1)

    def _spmm(self, crow, col, val, x):
        n_rows = crow.numel() - 1
        return torch.sparse_csr_tensor(crow, col, val, (n_rows, x.shape[0]), check_invariants=False) @ x

    def connectome_parameters(self):
        return [self.alpha_raw, self.bias, self.leak_raw] + ([self.mix] if self.cfg.channels > 1 else []) + \
            ([self.gain_raw] if self.cfg.type_gain else [])

    def n_parameters(self) -> dict:
        count = lambda ps: sum(p.numel() for p in ps)
        core = count(self.connectome_parameters()) if self.cfg.core else 0
        enc = count(self.encoder.parameters()) + (count(self.to_photo.parameters()) if self.cfg.core else 0) + \
            (count(self.bridge.parameters()) if self.bridge is not None else 0)
        head = count(self.head.parameters()) + (count(self.head_in.parameters()) if self.head_in is not None else 0)
        return {"çekirdek (α, bias, λ)": core, "kodlayıcı": enc, "başlık": head,
                "toplam": core + enc + head,
                "eğitilen": count(p for p in self.parameters() if p.requires_grad)}

    # ------------------------------------------------------------------ ileri geçiş
    def encode(self, tiles: torch.Tensor):
        """tiles [B, Cin, H, W] → fotoreseptör sürüşü [B, C, 3], köprü [B, C, bridge] ya da None."""
        B = tiles.shape[0]
        if self.cfg.adapt == "local":
            tiles = torch.cat([tiles, self._adapt(tiles)], dim=1)
        patches = self.eye.sample(tiles)                                    # [B, C, Cin, p, p]
        if self.n_eyes == 2:                                                # H35: sol göz aynalanmış karoyu görür
            patches = torch.cat([patches, self.eye.sample(torch.flip(tiles, dims=[3]))], dim=1)
        if self.cfg.adapt == "column":                                      # kolon içi uyum: yalnız kendi yaması
            mu = patches.mean((-1, -2), keepdim=True)
            sd = patches.std((-1, -2), keepdim=True, unbiased=False)
            z = (patches - mu) / (sd + nn.functional.softplus(self.adapt_c) + 1e-3)
            patches = torch.cat([patches, z], dim=2)
        patches = patches.reshape(B, self.n_columns, -1)
        h = self.encoder(patches)
        drive = self.to_photo(h)
        if self.eye_visible is not None:                                    # görüş alanı B: görmediği yarıya girdi yok
            drive = drive * self.eye_visible[None, :, None]
        return drive, (self.bridge(h) if self.bridge is not None else None)

    def encode_views(self, tiles: torch.Tensor) -> torch.Tensor:
        """H44: [8, B, C, k] — bakış 0 merkez, 1–6 altıgenin 6 yönünde yarım kolon kaydırılmış, 7 = 1–6 toplamı."""
        if HIZ80 and self.cfg.adapt != "local":                            # §80: tek toplama
            v = self._encode_views_fused(tiles)                               # [7, B, C, k]
            st = v[1:]
            if getattr(self, "lam_w", None) is not None:
                w = torch.ones(6, st.shape[-1], device=st.device, dtype=st.dtype)
                w = torch.cat([w[:, :3], self.lam_w.T], 1) if st.shape[-1] == 6 else w
                return torch.cat([v, (st * w[:, None, None, :]).sum(0)[None]])
            return torch.cat([v, st.sum(0)[None]])
        from .microsaccade import shift
        s = max(1, self.cfg.patch // 2)
        dirs = self._view_dirs()
        views = [self.encode(shift(tiles, dy, dx, s))[0] for dy, dx in dirs]
        if getattr(self, "lam_w", None) is not None:                     # §71 H57: L türleri (3..5) ağırlıklı toplam
            st = torch.stack(views[1:])                                       # [6, B, C, k]
            w = torch.ones(6, st.shape[-1], device=st.device, dtype=st.dtype)
            w = torch.cat([w[:, :3], self.lam_w.T], 1) if st.shape[-1] == 6 else w
            views.append((st * w[:, None, None, :]).sum(0))
        else:
            views.append(torch.stack(views[1:]).sum(0))
        return torch.stack(views)

    def _view_dirs(self) -> list:
        s = max(1, self.cfg.patch // 2)
        return [(0, 0), (0, s), (s, s // 2), (s, -(s // 2)), (0, -s), (-s, -(s // 2)), (-s, s // 2)]

    def _views_index(self, H: int, W: int, device) -> torch.Tensor:
        """§80: [7 · C · p · p] piksel indeksi. Bakış (dy, dx) için shift() kenar pikselini uzatır → kırpılmış indeks;
        sol göz aynalanmış karoyu görür → x yerine W − 1 − x. Cihaz başına bir kez (durum sözlüğüne girmez)."""
        cache = self.__dict__.setdefault("_vidx_cache", {})
        key = (H, W, str(device))
        if key not in cache:
            pix = self.eye._pixel_index(device)                                 # [n, p, p] = y·W + x
            y, x = pix // W, pix % W
            out = []
            for dy, dx in self._view_dirs():
                yy = (y - dy).clamp(0, H - 1) * W
                eyes = [yy + (x - dx).clamp(0, W - 1)]
                if self.n_eyes == 2:
                    eyes.append(yy + (W - 1 - x - dx).clamp(0, W - 1))
                out.append(torch.cat(eyes, 0))
            cache[key] = torch.stack(out).reshape(-1)
        return cache[key]

    def _encode_views_fused(self, tiles: torch.Tensor) -> torch.Tensor:
        """§80: encode(shift(tiles, bakış)) × 7 ile aynı sürüş [7, B, C, k], tek toplama ve tek kodlayıcı çağrısıyla."""
        B, Cin, H, W = tiles.shape
        if (H, W) != (self.eye.height, self.eye.width):
            raise ValueError(f"karo boyutu {(H, W)} ≠ göz boyutu {(self.eye.height, self.eye.width)}")
        p = self.cfg.patch
        C = self.n_columns
        bmax = max(1, FUSED_ELEMS // (7 * C * Cin * p * p))
        if B > bmax:                                                            # §80b: bellek sınırı
            return torch.cat([self._encode_views_fused(tiles[i:i + bmax]) for i in range(0, B, bmax)], 1)
        g = tiles.reshape(B, Cin, H * W).index_select(2, self._views_index(H, W, tiles.device))
        g = g.reshape(B, Cin, 7, C, p, p).permute(2, 0, 3, 1, 4, 5)           # [7, B, C, Cin, p, p]
        if self.cfg.adapt == "column":                                          # kolon içi uyum (encode ile aynı)
            mu = g.mean((-1, -2), keepdim=True)
            sd = g.std((-1, -2), keepdim=True, unbiased=False)
            z = (g - mu) / (sd + nn.functional.softplus(self.adapt_c) + 1e-3)
            g = torch.cat([g, z], dim=3)
        drive = self.to_photo(self.encoder(g.reshape(7, B, C, -1)))
        if self.eye_visible is not None:
            drive = drive * self.eye_visible[None, None, :, None]
        return drive

    def _act(self, v: torch.Tensor) -> torch.Tensor:
        """Nöron etkinliği (presinaptik çıkış): relu ya da doymalı relu (G6)."""
        if self.cfg.act_soft:                                                # H52: dereceli (eşik altında da çıktı)
            bb = (1 + nn.functional.softplus(self.beta_raw))[self.type_id][:, None]
            r = nn.functional.softplus(bb * v) / bb
        else:
            r = torch.relu(v)
        return torch.tanh(r) if self.cfg.activation == "sat" else r

    def _adapt(self, x: torch.Tensor) -> torch.Tensor:
        """Fotoreseptör adaptasyonu: (x − yerel ortalama) / (yerel std + c). Pencere ~4 kolon; c öğrenilir."""
        k = self.adapt_k
        if FAST_ADAPT:      # dikdörtgen ortalama = satır ortalaması × sütun ortalaması (kenarda sayımlar da çarpılır)
            box = lambda t: nn.functional.avg_pool2d(nn.functional.avg_pool2d(t, (1, k), 1, (0, k // 2),
                                                                              count_include_pad=False),
                                                     (k, 1), 1, (k // 2, 0), count_include_pad=False)
        else:
            box = lambda t: nn.functional.avg_pool2d(t, k, 1, k // 2, count_include_pad=False)
        mu = box(x)
        var = (box(x * x) - mu * mu).clamp(min=0)
        return (x - mu) / (var.sqrt() + nn.functional.softplus(self.adapt_c) + 1e-3)

    def core(self, drive: torch.Tensor, state: torch.Tensor | None = None, steps: int | None = None,
             record: bool = False, stp_state: torch.Tensor | None = None, return_stp: bool = False):
        """drive [B, C, 3] → son durum v [N, B] (ve istenirse her adımın relu(v)'si).
        H51 (§61) mikrosakkad: drive [S, B, C, k] verilirse adım t'de drive[min(t, S−1)] kullanılır (girdi adım adım değişir).
        §89 video (H61): state + stp_state verilirse önceki karenin durumundan devam edilir (sıfırlama yok, yalnız sızıntı);
        return_stp=True → (normal çıktı, son STP durumu) döner. Varsayılanlar eski davranışın aynısı."""
        if self.cfg.channels > 1:
            return self._core_d(drive, state, steps, record)
        views = self.cfg.superposition and drive.dim() in (4, 5)
        seq = (drive.dim() == 4 and not views) or drive.dim() == 5     # 5 boyut: [S, 8, B, C, k] (§71)
        B = drive.shape[-3]
        N = self.n_neurons

        def inp(d):
            I_ = d.new_zeros(N, B)
            if views:                                                   # H44: her girdi nöronu kendi bakışından
                I_[self.pr_idx] = d[self.pr_view, :, self.pr_col, self.pr_kind]
            else:
                I_[self.pr_idx] = d[:, self.pr_col, self.pr_kind].T
            return I_
        I = inp(drive[0] if seq else drive)
        v = drive.new_zeros(N, B) if state is None else state
        w = self.weights()
        lam = self._sel(torch.sigmoid(self.leak_raw), self.type_id)[:, None]
        b = self._sel(self.bias, self.type_id)[:, None]
        g = self._sel(torch.exp(self.gain_raw), self.type_id)[:, None] if self.cfg.type_gain else None
        if self.cfg.homeostasis:                                             # H41b: homeostatik eşik ve kazanç
            b = b + self._sel(self.homeo_bias, self.type_id)[:, None]
            gh = self._sel(torch.exp(self.homeo_gain), self.type_id)[:, None]
            g = gh if g is None else g * gh
        cfg = self.cfg
        spmm = lambda vals, x: _SparseMatmul.apply(vals, x, self.crow, self.col, self.crow_t, self.col_t, self.perm_t,
                                                    self.post, self.pre, vals.detach()[self.perm_t])
        if cfg.shunt:                                                   # H42: uyarıcı ve baskılayıcı iletkenlik
            w_pos, w_neg = torch.relu(w), torch.relu(-w)
        else:
            w_t = w.detach()[self.perm_t]                          # transpoz değerleri, ileri geçiş boyunca sabit
        if cfg.fatigue:                                                 # H40: kaynak x (1'den), adaptasyon ad (0'dan)
            U = self._sel(torch.sigmoid(self.fat_u), self.type_id)[:, None]
            tau = self._sel(1 + nn.functional.softplus(self.fat_tau), self.type_id)[:, None]
            ab = self._sel(nn.functional.softplus(self.ad_beta), self.type_id)[:, None]
            taua = self._sel(1 + nn.functional.softplus(self.ad_tau), self.type_id)[:, None]
            x_res = torch.ones_like(v)
            ad = torch.zeros_like(v)
        if cfg.stp:                                                     # H53: s = yakın geçmiş etkinlik (0'dan)
            phi = self._sel(torch.tanh(self.stp_phi), self.type_id)[:, None]
            tau_s = self._sel(1 + nn.functional.softplus(self.stp_tau), self.type_id)[:, None]
            s_stp = torch.zeros_like(v) if stp_state is None else stp_state
        hist = []
        for t in range(steps or cfg.steps):
            if seq and 0 < t < drive.shape[0]:
                I = inp(drive[t])
            a = self._act(v)
            out = a * x_res if cfg.fatigue else a
            if cfg.stp:                                                 # + kolaylaşma (≤ ×2), − yorulma (≥ ×0)
                out = out * (1 + phi * s_stp).clamp(min=0)
                s_stp = s_stp + (a - s_stp) / tau_s
            if cfg.shunt:
                rec = spmm(w_pos, out) * (1.0 - v) + spmm(w_neg, out) * (cfg.shunt_e_inh - v)
            else:
                rec = _SparseMatmul.apply(w, out, self.crow, self.col, self.crow_t, self.col_t, self.perm_t,
                                          self.post, self.pre, w_t)
            if g is not None:
                rec = g * rec
            if cfg.modulators:                                          # H45: hacim iletimi, karo başına
                M = a.new_zeros(self.n_mod_types, B).index_add_(0, self.mod_j, self._sel(a, self.mod_idx)) / self.mod_cnt[:, None]
                lg = a.new_zeros(self._n_types, B).index_add_(0, self.modp_k,
                                                         self.mod_beta[:, None] * self._sel(M, self.modp_j))
                rec = rec * torch.exp(self._sel(lg, self.type_id))
            if cfg.fatigue:
                rec = rec - ab * ad
                x_res = x_res + (1 - x_res) / tau - U * x_res * a
                ad = ad + (a - ad) / taua
            v = (1 - lam) * v + lam * (rec + b + I)
            if cfg.noise_sigma > 0 and self.training:                   # H43: stokastik rezonans
                v = v + cfg.noise_sigma * torch.randn_like(v)
            if record:
                hist.append(torch.relu(v).detach())
        if return_stp:
            return ((v, hist) if record else v), (s_stp if cfg.stp else None)
        return (v, hist) if record else v

    def _core_d(self, drive, state=None, steps=None, record=False):
        """d > 1 (H14): durum [N, d, B]. Her kanal aynı kablolamada kendi α'sıyla yayılır; sonra tip başına d×d
        karıştırma (kanallar arası etkileşim). Fotoreseptör sürüşü [B, C, 3·d]."""
        B, d, N = drive.shape[0], self.cfg.channels, self.n_neurons
        I = drive.new_zeros(N, d, B)
        dr = drive.reshape(B, self.n_columns, self.n_in_kinds, d)
        I[self.pr_idx] = dr[:, self.pr_col, self.pr_kind].permute(1, 2, 0)
        v = drive.new_zeros(N, d, B) if state is None else state
        w = self.weights()                                                    # [E, d]
        lam = torch.sigmoid(self.leak_raw)[self.type_id][:, None, None]
        b = self.bias[self.type_id][:, :, None]
        M = self.mix[self.type_id]                                            # [N, d, d]
        hist = []
        for _ in range(steps or self.cfg.steps):
            a = torch.relu(v)
            rec = torch.stack([_SparseMatmul.apply(w[:, k].contiguous(), a[:, k].contiguous(), self.crow, self.col,
                                                   self.crow_t, self.col_t, self.perm_t, self.post, self.pre)
                               for k in range(d)], 1)                         # [N, d, B]
            rec = torch.einsum("nij,njb->nib", M, rec)
            v = (1 - lam) * v + lam * (rec + b + I)
            if record:
                hist.append(torch.relu(v).detach())
        return (v, hist) if record else v

    def readout(self, v: torch.Tensor, bridge: torch.Tensor | None = None) -> torch.Tensor:
        """v [N, B] (ya da d > 1 için [N, d, B]) → özellikler [B, C, F]."""
        a = torch.relu(v) if self.cfg.readout == "relu" else v
        if a.dim() == 3:
            N, d, B = a.shape
            flat = a.permute(0, 2, 1).reshape(N, B * d)                       # sütun = b·d + k
            fc = self._readout_mm("rc", flat).reshape(self.n_columns, -1, B, d).permute(2, 0, 1, 3)
            fo = self._readout_mm("ro", flat).reshape(self.n_columns, -1, B, d).permute(2, 0, 1, 3)
            parts = [fc.reshape(B, self.n_columns, -1), fo.reshape(B, self.n_columns, -1)]
            return torch.cat(parts + ([bridge] if bridge is not None else []), dim=-1)
        B = a.shape[1]
        fc = self._readout_mm("rc", a)                                        # [C·n_ct, B]
        fo = self._readout_mm("ro", a)                                        # [C·n_ot, B]
        fc = fc.reshape(self.n_columns, -1, B).permute(2, 0, 1)
        fo = fo.reshape(self.n_columns, -1, B).permute(2, 0, 1)
        parts = [fc, fo] + ([bridge] if bridge is not None else [])
        return torch.cat(parts, dim=-1)

    def forward(self, tiles: torch.Tensor, state: torch.Tensor | None = None) -> dict:
        if HIZ80 and self.cfg.core and self.bridge is None and (self.cfg.superposition or self.cfg.microsaccade > 0):
            bridge = None                                                    # §80: sürüşü aşağıda drive() hesaplıyor
        else:
            drive, bridge = self.encode(tiles)
        if not self.cfg.core:                                                # çekirdeksiz kontrol
            return self._pack(self.head_forward(bridge), None)
        if bridge is not None and self.training and self.cfg.bridge_drop > 0:
            keep = (torch.rand(bridge.shape[0], 1, 1, device=bridge.device) >= self.cfg.bridge_drop).to(bridge.dtype)
            bridge = bridge * keep / (1 - self.cfg.bridge_drop)
        if self.cfg.superposition or self.cfg.microsaccade > 0:
            drive = self.drive(tiles)
        v = self.core(drive, state)
        return self._pack(self.head_forward(self.readout(v, bridge)), v)     # [B, C, 5|6]

    def drive(self, tiles: torch.Tensor) -> torch.Tensor:
        """İleri geçişin çekirdeğe verdiği sürüş (§71: homeostaz ölçümü, teşhis araçları ve görselleştirici de bunu
        kullanır; §68'deki tek bakış hatası tekrarlanmasın). Süperpozisyon: [8, B, C, k]; mikrosakkad: [S, B, C, k];
        ikisi birlikte ve microsaccade_views: [S, 8, B, C, k]; hiçbiri: [B, C, k]."""
        if self.cfg.superposition and self.cfg.microsaccade > 0 and self.cfg.microsaccade_views:
            from .microsaccade import offsets, shift
            amp = self.cfg.microsaccade
            return torch.stack([self.encode_views(shift(tiles, dy, dx, amp)) for dy, dx in offsets(self.cfg.steps, amp)])
        if self.cfg.superposition:                                           # H44 (eski davranış: mikrosakkad yok)
            return self.encode_views(tiles)
        if self.cfg.microsaccade > 0:                                        # H51: adım adım titreşen girdi
            from .microsaccade import jitter_drive
            return jitter_drive(self, tiles, self.cfg.microsaccade)
        return self.encode(tiles)[0]

    def _pack(self, out: torch.Tensor, v) -> dict:
        d = {"heat": out[..., 0], "offset": out[..., 1:3], "logsize": out[..., 3:5], "state": v}
        if self.n_eyes == 2:                                                # decode iki gözü ayırıp birleştirir
            d["eyes"] = (2, self.cfg.view, self.cfg.view_overlap)
        if self.cfg.quality:
            d["quality"] = out[..., 5]
        return d
