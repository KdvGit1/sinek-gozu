# Sinek Gözü: Drosophila konnektomu topolojili drone insan tespiti

Drone görüntülerinde (termal ve RGB) insan arayan, **yinelemeli ağırlık topolojisi gerçek bir sinek beyninden
alınmış** küçük bir sinir ağı. Ağın iskeleti FlyWire v783 yetişkin *Drosophila* konnektomundan gelir: iki optik lob
ve onlardan girdi alan merkez beyin nöronları, **100.473 nöron ve 1.693.309 gerçek bağlantı**. Bağlantıların varlığı,
yönü ve işareti (uyarıcı / baskılayıcı) sabittir. Öğrenilen yalnız hücre tipi düzeyindeki ölçekler, zaman sabitleri
ve küçük bir okuma başlığıdır (termal modelde **43.975 parametre**; karşılaştırma için YOLOv8n ~3 milyon).

Omurgada evrişimli ağ (CNN) yoktur. Görüntü, altıgen kolon ızgarasıyla (göz başına 796 kolon, kolon başına 8×8 piksel)
iki göze ayrılır, lamina ve fotoreseptörlere girer, sinyal konnektom üzerinde 10 zaman adımı yayılır ve her kolonda
"burada insan var mı, kutu nerede" okunur.

> **English summary.** A person detector for drone imagery (thermal and RGB) whose recurrent weight graph is the
> FlyWire v783 *Drosophila* connectome (both optic lobes plus downstream central-brain neurons: 100,473 neurons,
> 1.69 M connections). Connectivity, direction and sign are fixed; only cell-type-level gains, time constants and a
> small readout are trained (~44 k parameters, no CNN backbone). The fly wiring is decisive: with the same recipe,
> a degree-preserving random topology scores 0.000 mAP50. The model is still far behind YOLOv8n (see table below).
> Code comments and docs are in Turkish; `§` numbers in comments refer to the project's research log (not included).

## Sonuçlar

Aynı ölçüm koduyla (`tools/final_eval.py`), mAP50 (IoU ≥ 0,5). Val kümesi model seçimi için, test kümesi yalnız son
modellerde bir kez kullanıldı.

| Model | Modalite | Val mAP50 | Test mAP50 | Eğitilen parametre |
|---|---|---|---|---|
| **Sinek gözü `uzun_75`** | termal | 0,124 | 0,055 | 43.975 |
| YOLOv8n (ön eğitimli, aynı veri) | termal | 0,518 | 0,272 | ~3,0 M |
| Aynı tarif, rastgele topoloji (16k it) | termal | 0,000 | 0,000 | 43.975 |
| Aynı tarif, sinek topolojisi (16k it) | termal | 0,099 | 0,050 | 43.975 |
| **Sinek gözü `uzun_80_rgb`** (termalden aktarım) | RGB (uzun kenar 640) | 0,165 | 0,034 | 52.167 |
| YOLOv8n (ön eğitimli, aynı veri) | RGB (uzun kenar 640) | 0,500 | 0,178 | ~3,0 M |

Termal sette alt kırılım (val / test mAP50): HIT-UAV 0,409 / 0,426 (YOLO 0,916 / 0,933), RGBTDronePerson val 0,175
(YOLO 0,653), BIRDSAI val 0,024 (YOLO 0,019), WiSARD val 0,000 / test 0,035 (YOLO 0,069 / 0,094). Gevşek ölçütte
(AP@IoU 0,3) termal test 0,222 (YOLO 0,461): model insanların yerini çoğu zaman buluyor, kutuyu oturtmakta zorlanıyor.

**Dürüst özet:** Sinek gözü YOLOv8n'in termalde val'de ~%24'ü, testte ~%20'si düzeyinde; RGB'de %33 / %19. Güçlü
yanı, ~70 kat az parametreyle ve sabit biyolojik kablolamayla bu düzeye çıkması ve aynı tarifle rastgele kablolamanın
hiç öğrenememesi. Zayıf yanları: kutu yerleştirme, alan kayması (val → test düşüşü büyük), hız (RTX 3060'ta görüntü
başına ~0,10 sn; YOLOv8n ~0,01 sn). Tek tohum; val mAP50'de tohumlar arası gürültü yaklaşık ±0,02.

## Depo içeriği

```
flybrain/            model: konnektom yükleme, altıgen göz, sinek RNN'i, tespit / kod çözme, video çekirdeği
tools/
  tespit.py          görüntü / klasör üzerinde tespit (sıkıştırılmış ağırlıkla)
  agirlik.py         sıkıştırılmış ağırlık: dışa aktar, yükle, doğrula
  train_flybrain.py  eğitim
  final_eval.py      val / test değerlendirmesi (sinek ve YOLO aynı ölçümle)
  train_baseline.py  YOLOv8n karşılaştırma modeli
  convert_to_yolo.py, rgb640_hazirla.py           veri setlerini ortak YOLO biçimine çevirme
  download_connectome.py, build_connectome.py, build_bilateral.py   konnektomu FlyWire'dan baştan kurma
  train_video.py, video_bolme.py, visdrone_vid_hazirla.py, video_val_yolo.py   video uyum eğitimi (deneysel)
  olcum_95_son_model.py   son modelin iç ölçümleri (etkinlik dalgası, ablasyon, adım taraması, hız)
tests/               birim ve eşdeğerlik testleri (betik olarak çalışır)
configs/datasets.yaml   veri dönüştürme ayarları (sınıf eşleme, bölme, lisans politikası)
data/connectome/processed/   işlenmiş konnektom (bi_ol_merkez_thr5: son model; right_ol_thr5: tek lob, testler)
agirliklar/          son modellerin sıkıştırılmış ağırlıkları (uzun_75_termal.pt, uzun_80_rgb.pt)
veri_setlerini_indir.bat     veri setlerini indirip açan Windows betiği
```

## Kurulum

Python 3.10+ (geliştirme: Python 3.12, PyTorch 2.6 + CUDA 12.4, RTX 3060 12 GB, Windows 11).

```bash
pip install -r requirements.txt
```

PyTorch'u GPU'nuza uygun komutla https://pytorch.org adresinden kurun. Çıkarım CPU'da da çalışır (yavaş).

## Hızlı başlangıç: tespit

Eğitim verisi gerekmez; işlenmiş konnektom depoda hazır.

```bash
py -3 tools/tespit.py agirliklar/uzun_75_termal.pt termal_goruntu.jpg --cikis tespitler --esik 0.3
py -3 tools/tespit.py agirliklar/uzun_80_rgb.pt rgb_klasoru/ --cikis tespitler
```

Her görüntü için kutuları çizilmiş bir `.jpg` ve YOLO biçiminde `.txt` (`0 xc yc w h skor`) yazılır. Termal model gri
tonlu termal görüntü bekler; RGB modeli uzun kenarı 640 piksele küçültülmüş görüntülerle eğitildiği için büyük
görüntüler 640'a küçültülerek işlenir.

Python'dan:

```python
from tools.agirlik import yukle
model = yukle("agirliklar/uzun_75_termal.pt", "cuda")    # değerlendirme kipinde FlyEyeDetector
```

### Sıkıştırılmış ağırlıklar

Eğitim kontrol noktası (`best.pt`) konnektomdan türetilen sabit tamponlar yüzünden ~111 MB'tır. `agirliklar/*.pt`
yalnız öğrenilen parametreleri ve eğitim ayarlarını tutar (~0,2 MB); tamponlar yüklemede işlenmiş konnektomdan
yeniden kurulur. İki yolla kurulan modelin çıktıları birebir aynıdır:

```bash
py -3 tools/agirlik.py disa-aktar results/flybrain/<koşu>/best.pt agirliklar/<ad>.pt
py -3 tools/agirlik.py dogrula agirliklar/<ad>.pt results/flybrain/<koşu>/best.pt     # en büyük çıktı farkı: 0
```

## Veri

| Set | Modalite | Lisans | Rol |
|---|---|---|---|
| [WiSARD](https://sites.google.com/uw.edu/wisard/) | RGB + termal | MIT | eğitim / val / test |
| [RGBTDronePerson](https://nnnnerd.github.io/RGBTDronePerson/) | RGB + termal | CC BY 4.0 | eğitim / val / test |
| [HIT-UAV](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset) | termal | CC0 1.0 | eğitim / val / test |
| [BIRDSAI](https://lila.science/datasets/conservationdrones) | termal | CDLA-Permissive 1.0 | eğitim / val / test |
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | RGB | CC BY-NC-SA 3.0 | yalnız RGB test |

Veri setleri depoda yoktur. İndirme ve dönüştürme:

```bash
veri_setlerini_indir.bat                                   # data/raw/<set> (~50 GB; Windows)
py -3 tools/convert_to_yolo.py convert                     # data/yolo/thermal, data/yolo/rgb
py -3 tools/convert_to_yolo.py convert --only visdrone --allow-any-license   # (isteğe bağlı) RGB test
py -3 tools/rgb640_hazirla.py                              # data/yolo/rgb640 (uzun kenar 640)
```

Bölme kaynak set, uçuş ya da video bazındadır (aynı uçuştan kareler iki kümeye düşmez); ayrıntılar
`configs/datasets.yaml` içinde. Bilinen veri sorunları da orada not edilmiştir (ör. BIRDSAI'de bir test videosunda 87
karelik etiket kayması düzeltilir).

## Değerlendirme

```bash
py -3 tools/final_eval.py --fly agirliklar/uzun_75_termal.pt --yolo thermal --tag son_termal
py -3 tools/final_eval.py --modality rgb640 --yolo rgb640 --fly agirliklar/uzun_80_rgb.pt --tag son_rgb
```

Çıktı `results/rapor/<tag>.{md,json,png}`: mAP50, AP@IoU 0,3 ve görüntü başına 0,1 / 0,5 / 1 yanlış alarmdaki
duyarlılık (R@FPPI), toplam ve set bazında. YOLO karşılaştırması için önce
`py -3 tools/train_baseline.py --modality thermal rgb640` (ağırlıklar `results/baseline/yolov8n_<modalite>/`); YOLO'suz
değerlendirme için `--yolo ""`. Standart kod çözme ayarı (`FLYBRAIN_DECODE=merkez7p2`, `FLYBRAIN_OVERLAP=0.5`)
`final_eval.py` ve `tespit.py` içinde varsayılandır.

## Eğitim

Termal son model (`uzun_75`, sıfırdan, ~9,75 saat saf eğitim süresi, RTX 3060):

```bash
py -3 tools/train_flybrain.py --modality thermal --patch 8 --threshold 5 --readout voltage --reg-mode center \
  --pad replicate --adapt column --photo-aug --hard-neg 0.2 --bridge 0 --head-hidden 0 --merkez \
  --okuma-fotoreseptorsuz --lamina-girisi --agirlik-norm l2 --aktivasyon sat --init-gain 2.0 \
  --iki-lob --gorus A --superpozisyon --mikrosakkad 4 --modulator --stp --head-context hex7d --set-us 0.5 \
  --lr 3e-4 --warmup 1000 --alpha-lr-mult 3 --plato 3 --plato-taban 0.0625 --ema 0.9995 \
  --eval-every 4000 --eval-images 800 --mine-every 5000 --workers 3 \
  --iters 1000000000 --toplam-saat 9.75 --final-splits= --seed 3 --name uzun_75
```

RGB son model (`uzun_80_rgb`): aynı komut, `--modality rgb640 --init-from agirliklar/uzun_75_termal.pt --name
uzun_80_rgb` ile (termal kodlayıcının ilk katmanı üç kanala /3 ile genişletilir). En iyi RGB kontrol noktası 80.000.
iterasyondaydı; tarif ~16.000 iterasyondan sonra büyük ölçüde doyuyor.

Kayıtlar `results/flybrain/<ad>/` altına yazılır (`egitim.log`, `log.csv`, `best.pt`, `last.pt`); kesintide aynı
komut `--resume` ile kaldığı yerden sürer. Rastgele topoloji kontrolü için `--topology random`.

Bayrakların kısa anlamı: `--iki-lob --gorus A` iki optik lob ve iki göz; `--superpozisyon` R1–6 fotoreseptörlerinin
nöral süperpozisyonu; `--lamina-girisi` FlyWire'da eksik kalan retina → lamina sinapslarının yerine kodlayıcının L1–L3'e
de girmesi; `--stp` işaretli kısa süreli sinaptik plastisite; `--modulator` modülatör nöronlar; `--mikrosakkad 4` eğitimde
küçük göz kaymaları; `--head-context hex7d` okumada altıgen komşu kolonlar; `--set-us 0.5` veri setleri arası örnekleme
dengesi.

## Testler

```bash
py -3 tests/test_model.py
py -3 tests/test_eye.py
py -3 tests/test_bilateral.py
py -3 tests/test_faz1.py
py -3 tests/test_stp.py
py -3 tests/test_rgb79.py
```

Veri seti gerektirmezler (işlenmiş konnektom ve `agirliklar/` yeterli).

## Konnektomu baştan kurmak (isteğe bağlı)

`data/connectome/processed/` depoda hazırdır. Baştan üretmek için:

```bash
py -3 tools/download_connectome.py                 # FlyWire v783 (Codex) tabloları, data/connectome/flywire_783/
py -3 tools/build_connectome.py --thresholds 5     # right_ol_thr5 (tek optik lob)
py -3 tools/build_bilateral.py --thresholds 5      # bi_ol_merkez_thr5 (iki optik lob + merkez beyin; son model)
```

Sinaps eşiği 5; ağırlıklar hücre tipi çifti başına paylaşılan tek bir ölçekle (α) öğrenilir, işaret hücre tipinin
çoğunluk nörotransmiterinden (Dale ilkesi) gelir.

## Lisans ve atıf

- **Kod:** MIT (`LICENSE`).
- **İşlenmiş konnektom** (`data/connectome/processed/`): FlyWire v783'ten türetilmiştir, **CC BY 4.0**
  ([Zenodo 10676866](https://zenodo.org/records/10676866)). Kullanırken FlyWire'a atıf yapın:
  - S. Dorkenwald et al., "Neuronal wiring diagram of an adult brain," *Nature* 634, 2024.
  - P. Schlegel et al., "Whole-brain annotation and multi-connectome cell typing of *Drosophila*," *Nature* 634, 2024.
  - A. Matsliah et al., "Neuronal parts list and wiring diagram for a visual system," *Nature* 634, 2024.
- **Model ağırlıkları** (`agirliklar/`) yalnız ticari kullanıma da izin veren lisanslı verilerle (MIT, CC BY 4.0, CC0,
  CDLA-Permissive) eğitildi; eğitim verilerinin atıf koşulları geçerlidir:
  - D. Broyles, C. R. Hayner, K. Leung, "WiSARD: A Labeled Visual and Thermal Image Dataset for Wilderness Search and
    Rescue," IEEE/RSJ IROS, 2022.
  - Y. Zhang et al., "Drone-based RGBT tiny person detection," *ISPRS J. Photogramm. Remote Sens.* 204, 2023.
  - J. Suo et al., "HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object
    detection," *Scientific Data* 10, 227, 2023.
  - E. Bondi et al., "BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos," WACV, 2020.
- VisDrone yalnız test için kullanıldı (CC BY-NC-SA 3.0, akademik kullanım). Video uyum eğitimi VisDrone-VID'i de
  kullandığı için video modelinin ağırlıkları bu sürüme konmadı.

Bu çalışma Erciyes Üniversitesi Bilgisayar Mühendisliği Design Project kapsamında yapılmıştır.
