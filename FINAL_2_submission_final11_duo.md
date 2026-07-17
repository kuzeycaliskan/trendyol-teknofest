# FINAL DOSYA 2 — `submission_final11_duo_q28.csv`

**Public LB: 0.893** · İşaretlendi: EVET (ikinci final / sigorta) · Tarih: 17 Tem 2026

Bu doküman, sigorta amaçlı ikinci final submission'ın SIFIRDAN tam üretim
reçetesidir. Dosya-1'den (`submission_7b_w50.csv`, 0.903) **bilinçli olarak
farklı eksende**: LLM yargıç YOK, yalnızca klasik ML topluluğu. Amaç: 7B-blend
public banda overfit ederse, bu temiz topluluk private'ta öne geçebilsin.

---

## 1. TEK CÜMLE ÖZET

5 Türkçe cross-encoder + 1 komşu-graf öznitelikli LightGBM'in ağırlıklı
topluluğu (`test_proba_final11.npy`), q=0.28 eşiğiyle doğrudan 0/1'e çevrilir.
7B yargıç KULLANILMAZ.

## 2. NİHAİ FORMÜL (kesin, kod)

```python
import numpy as np, pandas as pd

def to_p(x):
    x = x.astype(np.float32)
    return 1/(1+np.exp(-x)) if (x.min() < 0 or x.max() > 1) else x

T3  = to_p(np.load("kaggle_code_3/ce3_test_scores.npy"))    # dbmdz-128k + mined + pseudo
T5  = to_p(np.load("kaggle_code_5/ce5_test_scores.npy"))    # cosmos-turkish-base
T6  = to_p(np.load("kaggle_code_6/ce6_test_scores.npy"))    # dbmdz-128k tüm-veri
T11 = to_p(np.load("kaggle_code_11/ce11_test_scores.npy"))  # dbmdz-128k + sameleaf-çelişki
T12 = to_p(np.load("kaggle_code_12/ce12_test_scores.npy"))  # convbert-turkish + sameleaf
v9L = np.load("test_proba_v9L.npy")                         # LightGBM + komşu-graf öznitelik

p_old = 0.3*T3 + 0.4*T6 + 0.3*T5
p_ce  = 0.7*p_old + 0.3*(0.5*T11 + 0.5*T12)
final11 = (0.6*p_ce + 0.4*v9L).astype(np.float32)          # == test_proba_final11.npy

thr = float(np.quantile(final11, 0.72))                     # q = 0.28
pred = (final11 >= thr).astype(int)

test = pd.read_csv("trendyol-e-ticaret-yarismasi-2026-kaggle/submission_pairs.csv")
pd.DataFrame({"id": test.id, "prediction": pred}).to_csv("submission_final11_duo_q28.csv", index=False)
```

Taban proba diskte: `test_proba_final11.npy`.
**Dosya-1 ile TEK fark:** Dosya-1 bu `final11`'e 7B rank-blend ekler; Dosya-2
ekler EKLEMEZ (ham `final11`). İkisi de aynı q=0.28 eşiğini kullanır.

## 3. BİLEŞENLER (Dosya-1 ile AYNI; 7B hariç)

| Bileşen | Model | Script | Negatifler |
|---|---|---|---|
| ce3 | dbmdz/bert-base-turkish-128k-uncased | `kaggle_cross_encoder_v3.py` | mined + pseudo |
| ce5 | ytu-ce-cosmos/turkish-base-bert-uncased | `kaggle_cross_encoder_v5.py` | mined |
| ce6 | dbmdz/bert-base-turkish-128k-uncased | `kaggle_cross_encoder_v6.py` | mined (tüm veri) |
| ce11 | dbmdz/bert-base-turkish-128k-uncased | `kaggle_cross_encoder_v11.py` | mined + sameleaf |
| ce12 | dbmdz/convbert-base-turkish-mc4-uncased | `kaggle_cross_encoder_v12.py` | mined + sameleaf |
| v9L | LightGBM (17 + 5 komşu-graf öznitelik) | `lgbm_v9L.py` | 1 rastgele + 1 zor |

Veri hazırlığı script'leri: `mine_hard_negatives.py`, `mine_sameleaf_negatives.py`,
`finetune_biencoder.py`, `embed_texts.py` (Dosya-1 ile ortak).

## 4. NEDEN SİGORTA OLARAK BU SEÇİLDİ

- Dosya-1 (0.903), 7B yargıcın kararsız bant SIRALAMASINA dayanır — zero-shot
  bir LLM'in public banda hafif overfit etme riski VAR.
- Dosya-2 (0.893) hiçbir LLM içermez; tamamen fine-tuned Türkçe cross-encoder +
  LightGBM. **Model ekseni tamamen farklı** → iki dosya private'ta birlikte
  düşmez (r yüksek değil, farklı hata profilleri).
- Kaggle nihai skoru = işaretli 2 dosyanın private'ta DAHA İYİSİ. Public 0.903
  private'ta bir miktar kayarsa, temiz 0.893 tavan görevi görür.

## 5. DONANIM / ORTAM / DOĞRULAMA

Dosya-1 ile aynı (bkz. `FINAL_1_submission_7b_w50.md` §6-7). SEED=42 sabit.
`test_proba_final11.npy` yeniden üretildiğinde formül birebir tutar. Submission
format sample_submission ile doğrulandı (id kümesi/satır/kolon, pred ∈ {0,1}).
