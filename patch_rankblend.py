"""
Zero-shot hakem RANK-blend (bugünkü test). Hakem kalibrasyonsuz (skorlar ~0.007)
ama AUC 0.72 → SIRALAMASI iyi. Bu yüzden ham skoru değil, bant-içi yüzdelik
sırasını kullanıp banda katıyoruz. Bandın dışına dokunmuyoruz, q=0.28.

Kullanım: patch_rankblend.py <w>   (örn. 0.3 / 0.5)
Girdi: llm_core_scores.npy, llm_core_rows.npy (masaüstü zero-shot çıktısı)
Çıktı: submission_rank_w<pct>.csv
"""
import sys

import numpy as np
import pandas as pd

W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"

p = np.load("test_proba_final11.npy").astype(np.float32).copy()
j = np.load("llm_core_scores.npy").astype(np.float32)
rows = np.load("llm_core_rows.npy")
scored = j != 0.5
print(f"bant {len(rows)} çift, skorlanmış {int(scored.sum())}")

# bant-içi yüzdelik sıra → bandın [min,max] aralığına ölçekle (kalibrasyondan bağımsız)
r = rows[scored]
jj = j[scored]
rank = pd.Series(jj).rank(pct=True).to_numpy(dtype=np.float32)
lo, hi = p[r].min(), p[r].max()
j_scaled = lo + (hi - lo) * rank
p[r] = (1 - W) * p[r] + W * j_scaled

test = pd.read_csv(f"{DATA}/submission_pairs.csv")
sample = pd.read_csv(f"{DATA}/sample_submission.csv")
thr = float(np.quantile(p, 0.72))
pred = (p >= thr).astype(int)
f = f"submission_rank_w{int(W*100)}.csv"
pd.DataFrame({"id": test.id, "prediction": pred}).to_csv(f, index=False)
sub = pd.read_csv(f)
assert list(sub.columns) == list(sample.columns) and len(sub) == len(sample)
assert set(sub.id) == set(sample.id) and sub.prediction.isin([0, 1]).all()
base = np.load("test_proba_final11.npy")
flip = int((pred != (base >= np.quantile(base.astype(np.float32), 0.72))).sum())
print(f"{f}: pozitif {pred.mean():.3f} | 0.893'e göre değişen {flip} çift — OK")
