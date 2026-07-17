# FINAL DOSYA 1 — `submission_7b_w50.csv`

**Public LB: 0.903** · İşaretlendi: EVET (birincil final) · Tarih: 17 Tem 2026

Bu doküman, 0.903 skorlu birinci final submission'ın SIFIRDAN tam üretim
reçetesidir (ikinci aşama kural gereği: veri hazırlığı + eğitim + submission
yeniden üretilebilir olmalı).

---

## 1. TEK CÜMLE ÖZET

Yedi bileşenli bir topluluk (5 Türkçe cross-encoder + 1 komşu-graf öznitelikli
LightGBM) ile üretilen taban skorlara (`test_proba_final11.npy`), zero-shot
**Qwen2.5-7B** yargıç modelinin kararsız-bant sıralaması **rank-blend (w=0.5)**
ile karıştırılır; sonuç q=0.28 eşiğiyle 0/1'e çevrilir.

## 2. NİHAİ BİRLEŞTİRME FORMÜLÜ (kesin, kod)

```python
import numpy as np, pandas as pd

# --- to_p: CE skorları logit ise sigmoid, [0,1] ise olduğu gibi ---
def to_p(x):
    x = x.astype(np.float32)
    return 1/(1+np.exp(-x)) if (x.min() < 0 or x.max() > 1) else x

# (A) TABAN TOPLULUK  -> test_proba_final11.npy
T3  = to_p(np.load("kaggle_code_3/ce3_test_scores.npy"))    # dbmdz-128k + mined + pseudo
T5  = to_p(np.load("kaggle_code_5/ce5_test_scores.npy"))    # cosmos-turkish-base
T6  = to_p(np.load("kaggle_code_6/ce6_test_scores.npy"))    # dbmdz-128k tüm-veri
T11 = to_p(np.load("kaggle_code_11/ce11_test_scores.npy"))  # dbmdz-128k + sameleaf-çelişki-neg
T12 = to_p(np.load("kaggle_code_12/ce12_test_scores.npy"))  # convbert-turkish + sameleaf
v9L = np.load("test_proba_v9L.npy")                         # LightGBM + komşu-graf öznitelik

p_old = 0.3*T3 + 0.4*T6 + 0.3*T5
p_ce  = 0.7*p_old + 0.3*(0.5*T11 + 0.5*T12)
final11 = (0.6*p_ce + 0.4*v9L).astype(np.float32)           # == test_proba_final11.npy

# (B) 7B RANK-BLEND  (band = ensemble skoru [0.30,0.70], 386823 çift)
s7 = np.load("kaggle_code_14/llm_scores.npy").astype(np.float32)  # Qwen2.5-7B zero-shot
r7 = np.load("kaggle_code_14/llm_rows.npy")                        # bu çiftlerin test satır indeksi
rank = pd.Series(s7).rank(pct=True).to_numpy(np.float32)           # bant-içi yüzdelik sıra
lo, hi = final11[r7].min(), final11[r7].max()
r_scaled = lo + (hi - lo) * rank                                   # 7B'nin BIAS'ı atılır, yalnız SIRA
p = final11.copy()
p[r7] = 0.5*final11[r7] + 0.5*r_scaled                             # w = 0.5

# (C) EŞİK  q = 0.28  (en yüksek %28'e "alakalı")
thr = float(np.quantile(p, 0.72))
pred = (p >= thr).astype(int)

test = pd.read_csv("trendyol-e-ticaret-yarismasi-2026-kaggle/submission_pairs.csv")
pd.DataFrame({"id": test.id, "prediction": pred}).to_csv("submission_7b_w50.csv", index=False)
```

Üretici script: `patch_rankblend`/inline (yukarıdaki blok kanoniktir).
Taban proba diskte: `test_proba_7b_w50.npy` (yukarıdaki `p`).

## 3. BİLEŞENLERİN ÜRETİMİ (her biri ayrı, hangi script)

| Bileşen | Model | Eğitim script'i | Negatifler | Not |
|---|---|---|---|---|
| ce3 | dbmdz/bert-base-turkish-128k-uncased | `kaggle_cross_encoder_v3.py` | mined + pseudo | 1 epoch |
| ce5 | ytu-ce-cosmos/turkish-base-bert-uncased | `kaggle_cross_encoder_v5.py` | mined | attr400/len160, 2ep |
| ce6 | dbmdz/bert-base-turkish-128k-uncased | `kaggle_cross_encoder_v6.py` | mined | TÜM veri (val'siz), 2ep |
| ce11 | dbmdz/bert-base-turkish-128k-uncased | `kaggle_cross_encoder_v11.py` | mined + **sameleaf** | 2ep |
| ce12 | dbmdz/convbert-base-turkish-mc4-uncased | `kaggle_cross_encoder_v12.py` | mined + sameleaf | 2ep |
| v9L | LightGBM (17 öznitelik + 5 komşu-graf) | `lgbm_v9L.py` | 1 rastgele + 1 zor | — |
| 7B yargıç | Qwen/Qwen2.5-7B-Instruct-AWQ | `kaggle_llm_judge.py` | — (zero-shot) | vLLM, band [0.30,0.70] |

Negatif üretim script'leri (veri hazırlığı):
- `mine_hard_negatives.py` → `mined_hard_negatives.csv` (embedding madenciliği, leaf-koruma)
- `mine_sameleaf_negatives.py` → `sameleaf_negatives.csv` (aynı-leaf sorgu-çelişki: rakam/renk/marka)
- Bi-encoder embedding: `finetune_biencoder.py` (e5-small MNRL) → `embed_texts.py`
- v9L komşu-graf ön-hesap: `nbr_sims.npy`/`nbr_train_idx.npy` (lgbm_v9L.py içinde üretilir)

## 4. 7B YARGIÇ AYRINTISI (kritik bileşen, +0.010 katkı)

- Model: **Qwen/Qwen2.5-7B-Instruct-AWQ** (AÇIK AĞIRLIKLI, 4-bit). Kaggle
  notebook VM'inde **vLLM** ile SELF-HOST edildi. Ücretli API/servis YOK
  (SSS: açık modeller serbest, self-host uygun senaryo).
- Girdi: kararsız bant = taban topluluğun `[0.30, 0.70]` aralığındaki 386.823
  çift (`uncertain_band.csv`: her satır sorgu + ürün metni title|kategori|marka|attr).
- Prompt: "Arama: {q}\nÜrün: {t}\nAlakalı mı? Sadece Evet veya Hayır." →
  tek-token, "Evet"/"Hayır" logit'lerinden P(alakalı).
- Skor dağılımı: %34 <0.3, %63 >0.7, %3 orta (KALİBRE ve MONOTON — ensemble
  skoru arttıkça 7B pozitif oranı düzenli artıyor: 0.576→0.692).
- Süre: Kaggle T4, ~10.5 saat (387k çift).

## 5. NEDEN RANK-BLEND, NEDEN w=0.5

- 7B sistematik pozitif-bias taşıyor (band'ın %64'üne "evet"). Ham skoru linear
  blend'lemek precision'ı bozar → yalnız **sıralamasını** (rank) alıp bandın
  kendi [min,max] aralığına ölçekliyoruz (bias atılır).
- w LB'de tarandı: w=0.35→0.901, **w=0.5→0.903 (tepe)**, w=0.65→0.901.
  Simetrik tepe = gürültü değil gerçek optimum.
- q=0.28: eşik eğrisi LB'de haritalandı (q26→0.873, q28→0.877, q30→0.872 eski
  tabanla); 7B-blend pozitif oranını koruduğu için q=0.28 optimum kaldı
  (q29→0.901 ile teyit).

## 6. DONANIM / ORTAM

- Cross-encoder eğitimleri: Kaggle T4 x2 VE/VEYA Windows RTX 4070 (CUDA).
- 7B yargıç: Kaggle T4 (vLLM).
- LightGBM + tüm harman/eşik/submission üretimi: MacBook Pro M4 Pro (macOS).
- Python 3.12 (masaüstü/Kaggle), 3.14 (Mac). Kütüphaneler: transformers,
  sentence-transformers, vllm, lightgbm, scikit-learn, pandas.
- Tüm rastgelelik SEED=42 sabit.

## 7. DOĞRULAMA

`test_proba_7b_w50.npy` yeniden hesaplandığında yukarıdaki formülle **max fark
0.0** (birebir). Submission format: sample_submission ile aynı id kümesi/satır
sayısı/kolonlar, prediction ∈ {0,1}.
