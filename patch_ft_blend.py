"""
Fine-tuned hakem bant yaması (sabah, Mac). Bandta model skorunu FT-hakem
skoruyla harmanlar, bandın dışına dokunmaz, q=0.28.

Kullanım: patch_ft_blend.py <llm_agirligi>   (örn. 0.4 / 0.6)
Girdi: llm_ft_band_scores.npy, llm_ft_band_rows.npy (masaüstünden push)
Çıktı: submission_ftjudge_w<pct>.csv
Önce göreve karar: llm_ft_val_scores.npy isabeti raporlanır.
"""
import sys

import numpy as np
import pandas as pd

W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"

# go/no-go teyidi
try:
    vs = np.load("llm_ft_val_scores.npy"); vl = np.load("llm_ft_val_labels.npy")
    acc = max(((vs >= t).astype(int) == vl).mean() for t in np.arange(0.3, 0.71, 0.05))
    print(f"FT-hakem doğrulama isabeti: {acc:.3f}")
except Exception:
    print("uyarı: val skoru bulunamadı")

p = np.load("test_proba_final11.npy").astype(np.float32).copy()
llm = np.load("llm_ft_band_scores.npy").astype(np.float32)
rows = np.load("llm_ft_band_rows.npy")
scored = llm != 0.5
print(f"bant {len(rows)} çift, skorlanmış {int(scored.sum())} | "
      f"FT Evet oranı {float((llm[scored]>0.5).mean()):.3f}")
p[rows[scored]] = (1 - W) * p[rows[scored]] + W * llm[scored]

test = pd.read_csv(f"{DATA}/submission_pairs.csv")
sample = pd.read_csv(f"{DATA}/sample_submission.csv")
thr = float(np.quantile(p, 0.72))
pred = (p >= thr).astype(int)
f = f"submission_ftjudge_w{int(W*100)}.csv"
pd.DataFrame({"id": test.id, "prediction": pred}).to_csv(f, index=False)
sub = pd.read_csv(f)
assert list(sub.columns) == list(sample.columns) and len(sub) == len(sample)
assert set(sub.id) == set(sample.id) and sub.prediction.isin([0, 1]).all()
base = np.load("test_proba_final11.npy")
flip = int((pred != (base >= np.quantile(base.astype(np.float32), 0.72))).sum())
print(f"{f}: pozitif {pred.mean():.3f} | 0.893'e göre değişen karar {flip} — OK")
