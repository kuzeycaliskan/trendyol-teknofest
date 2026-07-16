"""
Zero-shot hakem UÇ-güncelleme yaması (bugünkü test). Hakem kalibrasyonsuz
(skorlar ~0.007) ama AUC~0.72 → SIRALAMASI iyi. Kritik nüans (denetim): bu
0.72 hard-labeled sette; gerçek band [0.40,0.60] daha zor, hakemin band-AUC'si
muhtemelen ~0.55-0.63. Bu yüzden ham skoru DEĞİL, yalnız hakemin EN EMİN
olduğu uçları kullanıyoruz: judge-rank üst %EDGE → yukarı it, alt %EDGE →
aşağı it, belirsiz ortaya (düşük-AUC) DOKUNMA. Bandın dışına dokunmuyoruz, q=0.28.

Kullanım: patch_rankblend.py <w> [edge_frac] [band]
          (varsayılan w=0.3, edge=0.15, band=tight)
Girdi: llm_<band>_scores.npy, llm_<band>_rows.npy
       band=tight -> fast_judge.py çıktısı (1.5B, 66k dar bant)  [BUGÜNKÜ]
       band=core  -> desktop_llm_judge_core.py çıktısı (3B, 170k)
Çıktı: submission_rank_w<pct>.csv

NOT (masaüstü, 17 Tem): fast_judge.py çıktıyı llm_tight_* yazıyor ama bu yama
llm_core_* okuyordu -> girdi parametreleştirildi, varsayılan tight. 3B core
koşusu 5248/170574'te durduruldu (ETA 4 saat, dar banda geçildi); o yarım
checkpoint .stale/ altında -- kullanılmamalı.
"""
import sys

import numpy as np
import pandas as pd

W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
EDGE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
# hangi hakem çıktısı: "tight" (1.5B, 66k dar bant) | "core" (3B, 170k)
BAND = sys.argv[3] if len(sys.argv) > 3 else "tight"
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"

p = np.load("test_proba_final11.npy").astype(np.float32).copy()
j = np.load(f"llm_{BAND}_scores.npy").astype(np.float32)
rows = np.load(f"llm_{BAND}_rows.npy")
scored = j != 0.5
r = rows[scored]
rank = pd.Series(j[scored]).rank(pct=True).to_numpy(dtype=np.float32)
lo, hi = p[r].min(), p[r].max()

top = rank >= (1 - EDGE)     # hakemin "kesin alakalı" dediği uç
bot = rank <= EDGE           # hakemin "kesin alakasız" dediği uç
p[r[top]] = (1 - W) * p[r[top]] + W * hi
p[r[bot]] = (1 - W) * p[r[bot]] + W * lo
print(f"bant {int(scored.sum())} çift | uç güncellenen: üst {int(top.sum())} + alt {int(bot.sum())} "
      f"(orta {int((~top & ~bot).sum())} dokunulmadı)")

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
