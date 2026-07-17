"""
LLM bant yaması: çekirdek banttaki (p 0.40-0.60) çiftlerde model skorunu
LLM hakem skoruyla harmanlar, bandın dışına DOKUNMAZ, q=0.28 ile keser.

Kullanım: patch_llm_blend.py <llm_agirligi>   (örn. 0.5 veya 0.8)
Girdi: llm_core_scores.npy, llm_core_rows.npy (Colab çıktısı, proje kökünde)
Çıktı: submission_llm_w<pct>.csv
"""

import sys

import numpy as np
import pandas as pd

W_LLM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"

p = np.load("test_proba_final11.npy").astype(np.float32).copy()
llm = np.load("llm_core_scores.npy").astype(np.float32)
rows = np.load("llm_core_rows.npy")
assert len(llm) == len(rows)

# yalnız skorlanmış satırlar (kesinti olduysa done'a kadar); 0.5 = skorlanmamış
scored = llm != 0.5
print(f"bant: {len(rows)} çift, skorlanmış: {int(scored.sum())} | "
      f"LLM Evet oranı: {float((llm[scored] > 0.5).mean()):.3f}")

p[rows[scored]] = (1 - W_LLM) * p[rows[scored]] + W_LLM * llm[scored]

test = pd.read_csv(f"{DATA}/submission_pairs.csv")
sample = pd.read_csv(f"{DATA}/sample_submission.csv")
thr = float(np.quantile(p, 0.72))
pred = (p >= thr).astype(int)
fname = f"submission_llm_w{int(W_LLM*100)}.csv"
pd.DataFrame({"id": test.id, "prediction": pred}).to_csv(fname, index=False)
sub = pd.read_csv(fname)
assert list(sub.columns) == list(sample.columns) and len(sub) == len(sample)
assert set(sub.id) == set(sample.id) and sub.prediction.isin([0, 1]).all()
flipped = int((pred != (np.load('test_proba_final11.npy') >= np.quantile(np.load('test_proba_final11.npy').astype(np.float32), 0.72))).sum())
print(f"{fname}: pozitif {pred.mean():.3f} | final11'e göre değişen karar: {flipped} — format OK")
