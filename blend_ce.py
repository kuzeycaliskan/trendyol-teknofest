"""
LightGBM + Cross-Encoder harmanı.

Yöntem (stacking değil, ağırlıklı harman — bilinçli tercih):
  final = alpha * lgbm_proba + (1 - alpha) * ce_proba
alpha ve karar eşiği, HER İKİ modelin de eğitiminde görmediği validasyon
çiftleri üzerinde seçilir. CE'nin val çiftleri (ce*_val_pairs.csv) aynı seed
prosedürüyle ayrılmış val terimlerinden gelir; LGBM da (aynı split) bu
terimleri eğitimde görmemiştir → iki taraf da out-of-sample, seçim temiz.

Neden stacking değil: CE'nin kendi train çiftlerindeki skorları in-sample
(iyimser) olur; onları LGBM özniteliği yapmak dağılım kayması yaratır.
Ağırlıklı harman bu sorunu taşımaz.

Kullanım: blend_ce.py [ce_prefix] [lgbm_tag]   (vars: ce, v7)
Girdi: {ce_prefix}_val_scores.npy, _val_labels.npy, _val_pairs.csv,
       {ce_prefix}_test_scores.npy, model_{lgbm_tag}.txt,
       test_proba_{lgbm_tag}.npy, embft3_*.npy
Çıktı: submission_blend_{ce_prefix}.csv
"""

import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ.setdefault("EMB_PREFIX", "embft3")
from lgbm_pipeline import DATA, SEED, load_catalog, load_terms, pairwise_cosine
from lgbm_pipeline_v3 import (
    EMB_PREFIX, TEST_POS_RATE, build_features, emb_cosine, resample_to_prior,
)

CE = sys.argv[1] if len(sys.argv) > 1 else "ce"
TAG = sys.argv[2] if len(sys.argv) > 2 else "v7"


def to_proba(x):
    x = x.astype(np.float32)
    if x.min() < 0 or x.max() > 1:  # logit ise sigmoid'e geçir
        x = 1.0 / (1.0 + np.exp(-x))
    return x


rng = np.random.default_rng(SEED)

print("[1/4] Girdiler yükleniyor...")
va = pd.read_csv(f"{CE}_val_pairs.csv")
ce_va = to_proba(np.load(f"{CE}_val_scores.npy"))
ce_te = to_proba(np.load(f"{CE}_test_scores.npy"))
lgbm_te = np.load(f"test_proba_{TAG}.npy")
booster = lgb.Booster(model_file=f"model_{TAG}.txt")
test = pd.read_csv(f"{DATA}/submission_pairs.csv")
assert len(ce_te) == len(test) and len(ce_va) == len(va)

print("[2/4] LGBM, CE'nin val çiftleri üzerinde skorlanıyor...")
items, item_meta, _ = load_catalog()
term_meta = load_terms()
E_q = np.load(f"{EMB_PREFIX}_terms.npy")
E_i = np.load(f"{EMB_PREFIX}_items.npy")
q_row = {t: k for k, t in enumerate(np.load(f"{EMB_PREFIX}_terms_ids.npy", allow_pickle=True))}
i_row = {i: k for k, i in enumerate(np.load(f"{EMB_PREFIX}_items_ids.npy", allow_pickle=True))}

item_ids_all = items.item_id.to_numpy()
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                      max_features=200_000, sublinear_tf=True, dtype=np.float32)
I_mat = vec.fit_transform([item_meta[i][0] for i in item_ids_all])
term_ids_all = list(term_meta.keys())
Q_mat = vec.transform([term_meta[t][1] for t in term_ids_all])
tf_item_row = {i: k for k, i in enumerate(item_ids_all)}
tf_term_row = {t: k for k, t in enumerate(term_ids_all)}

tf = pairwise_cosine(Q_mat, I_mat,
                     va.term_id.map(tf_term_row).to_numpy(),
                     va.item_id.map(tf_item_row).to_numpy())
em = emb_cosine(E_q, E_i,
                va.term_id.map(q_row).to_numpy(),
                va.item_id.map(i_row).to_numpy())
X_va = build_features(va.term_id.to_numpy(), va.item_id.to_numpy(),
                      term_meta, item_meta, tf, em)
lgbm_va = booster.predict(X_va)

print("[3/4] alpha + eşik, 26% dengeli validasyonda taranıyor...")
va = va.reset_index(drop=True)
best = (0.0, 0.5, 0.5)  # f1, alpha, thr
for alpha in np.arange(0.0, 1.01, 0.05):
    va["proba"] = alpha * lgbm_va + (1 - alpha) * ce_va
    va_p = resample_to_prior(va.rename(columns={"label": "label"}), TEST_POS_RATE, rng)
    for thr in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(va_p.label, (va_p.proba >= thr).astype(int), average="macro")
        if f1 > best[0]:
            best = (f1, float(alpha), float(thr))
f1, alpha, thr = best
print(f"   en iyi: macro-F1 {f1:.4f} @ alpha={alpha:.2f} (lgbm payı), eşik {thr:.2f}")
for a_chk in (0.0, 1.0):
    va["proba"] = a_chk * lgbm_va + (1 - a_chk) * ce_va
    va_p = resample_to_prior(va, TEST_POS_RATE, rng)
    f1s = max(f1_score(va_p.label, (va_p.proba >= t).astype(int), average="macro")
              for t in np.arange(0.05, 0.96, 0.01))
    kim = "yalnız CE" if a_chk == 0 else "yalnız LGBM"
    print(f"   kıyas ({kim}): {f1s:.4f}")

print("[4/4] Test harmanlanıyor...")
p = alpha * lgbm_te + (1 - alpha) * ce_te
np.save(f"test_proba_blend_{CE}.npy", p)
pred = (p >= thr).astype(int)
print(f"   test pozitif oranı: {pred.mean():.3f}")
pd.DataFrame({"id": test.id, "prediction": pred}).to_csv(
    f"submission_blend_{CE}.csv", index=False)
print(f"Yazıldı: submission_blend_{CE}.csv")
