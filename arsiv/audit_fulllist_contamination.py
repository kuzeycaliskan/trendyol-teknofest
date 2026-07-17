"""
ce14 tam-liste negatif kirlilik denetimi: fulllist_negatives.csv'nin TAMAMI
v9L LGBM ile skorlanir; "negatif" etiketli satirlarin ne kadarinin muhtemelen
gercek pozitif oldugu (false-negative kirliligi) rank dilimlerine gore olculur.
Ayrica p>0.7 adaylar cikarilarak fulllist_negatives_filtered.csv uretilir.
"""

import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ.setdefault("EMB_PREFIX", "embpl")
from lgbm_pipeline import DATA, SEED, VAL_TERM_FRACTION, load_catalog, load_terms, pairwise_cosine
from lgbm_pipeline_v3 import EMB_PREFIX, build_features, emb_cosine

K = 10

print("[1/6] fulllist + rank...")
fl = pd.read_csv("fulllist_negatives.csv")
fl["rank"] = fl.groupby("term_id").cumcount()  # terim ici retrieval sirasi (0-bazli)

print("[2/6] katalog/terim/embedding...")
items, item_meta, _ = load_catalog()
term_meta = load_terms()
train_pos = pd.read_csv(f"{DATA}/training_pairs.csv")
E_q = np.load(f"{EMB_PREFIX}_terms.npy").astype(np.float32)
E_i = np.load(f"{EMB_PREFIX}_items.npy").astype(np.float32)
q_row = {t: k for k, t in enumerate(np.load(f"{EMB_PREFIX}_terms_ids.npy", allow_pickle=True))}
i_row = {i: k for k, i in enumerate(np.load(f"{EMB_PREFIX}_items_ids.npy", allow_pickle=True))}

print("[3/6] komsu altyapisi (lgbm_v9L ile birebir ayni)...")
rng = np.random.default_rng(SEED)
terms_unique = np.asarray(train_pos.term_id.unique(), dtype=object)
perm = rng.permutation(len(terms_unique))
n_val = int(len(terms_unique) * VAL_TERM_FRACTION)
val_terms = set(terms_unique[perm[:n_val]])
pos_tr = train_pos[~train_pos.term_id.isin(val_terms)]

nbr_S = np.load("nbr_sims.npy")
nbr_I = np.load("nbr_train_idx.npy")
nbr_terms = np.load("nbr_train_terms.npy", allow_pickle=True)
trainsplit = set(pos_tr.term_id.unique())
pool_ok = np.array([t in trainsplit for t in nbr_terms])
pos_sets = pos_tr.groupby("term_id")["item_id"].agg(set).to_dict()
cent = np.zeros((len(nbr_terms), E_i.shape[1]), dtype=np.float32)
for k, t in enumerate(nbr_terms):
    if not pool_ok[k]:
        continue
    rows = [i_row[i] for i in pos_sets[t] if i in i_row]
    if rows:
        c = E_i[rows].mean(axis=0)
        n = np.linalg.norm(c)
        cent[k] = c / n if n > 0 else 0


def neighbor_feats(term_ids, item_ids):
    out = np.zeros((len(term_ids), 5), dtype=np.float32)
    t_rows = np.array([q_row[t] for t in term_ids])
    i_rows = np.array([i_row[i] for i in item_ids])
    for r in range(len(term_ids)):
        t = term_ids[r]
        sel_idx, sel_sim = [], []
        for ci, cs in zip(nbr_I[t_rows[r]], nbr_S[t_rows[r]]):
            ct = nbr_terms[ci]
            if not pool_ok[ci] or ct == t:
                continue
            sel_idx.append(ci)
            sel_sim.append(cs)
            if len(sel_idx) == K:
                break
        if not sel_idx:
            continue
        item = item_ids[r]
        hits = [sim for ci, sim in zip(sel_idx, sel_sim) if item in pos_sets[nbr_terms[ci]]]
        out[r, 0] = max(hits) if hits else 0.0
        out[r, 1] = len(hits)
        csims = cent[sel_idx] @ E_i[i_rows[r]]
        out[r, 2] = float(np.max(np.array(sel_sim) * csims))
        out[r, 3] = float(np.mean(csims[:3]))
        out[r, 4] = sel_sim[0]
    return out


print("[4/6] TF-IDF fit...")
item_ids_all = items.item_id.to_numpy()
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                      max_features=200_000, sublinear_tf=True, dtype=np.float32)
I_mat = vec.fit_transform([item_meta[i][0] for i in item_ids_all])
term_ids_all = list(term_meta.keys())
Q_mat = vec.transform([term_meta[t][1] for t in term_ids_all])
tf_i = {i: k for k, i in enumerate(item_ids_all)}
tf_t = {t: k for k, t in enumerate(term_ids_all)}

print("[5/6] 1.155M satir skorlaniyor...")
booster = lgb.Booster(model_file="model_v9L.txt")
p = np.empty(len(fl), dtype=np.float32)
for s in range(0, len(fl), 300_000):
    e = min(s + 300_000, len(fl))
    part = fl.iloc[s:e]
    t_ids, i_ids = part.term_id.to_numpy(), part.item_id.to_numpy()
    tf = pairwise_cosine(Q_mat, I_mat,
                         part.term_id.map(tf_t).to_numpy(),
                         part.item_id.map(tf_i).to_numpy())
    em = emb_cosine(E_q, E_i,
                    part.term_id.map(q_row).to_numpy(),
                    part.item_id.map(i_row).to_numpy())
    X = build_features(t_ids, i_ids, term_meta, item_meta, tf, em)
    X = np.hstack([X, neighbor_feats(t_ids, i_ids)])
    p[s:e] = booster.predict(X)
    print(f"   {e}/{len(fl)}")
fl["p"] = p

print("[6/6] RAPOR")
for thr in (0.5, 0.7, 0.8, 0.95):
    print(f"  p>{thr}: {float((p > thr).mean()) * 100:.2f}%")
for lo, hi, name in ((0, 10, "rank 1-10"), (10, 30, "rank 11-30"), (30, 999, "rank 31-80")):
    m = (fl["rank"] >= lo) & (fl["rank"] < hi)
    sub = fl[m]
    print(f"  {name}: n={len(sub)} | p>0.5 {float((sub.p>0.5).mean())*100:.2f}% | "
          f"p>0.8 {float((sub.p>0.8).mean())*100:.2f}% | p>0.95 {float((sub.p>0.95).mean())*100:.2f}%")

keep = fl[fl.p <= 0.7]
keep[["term_id", "item_id", "label"]].to_csv("fulllist_negatives_filtered.csv", index=False)
print(f"filtreli CSV: fulllist_negatives_filtered.csv — {len(keep)} satir "
      f"(atilan: {len(fl)-len(keep)} = %{(len(fl)-len(keep))/len(fl)*100:.1f})")
# 100k orneklem istatistigi (seed 42) — direktifteki tanimla uyum icin
smp = fl.sample(100_000, random_state=42)
print(f"100k orneklem: p>0.5 {float((smp.p>0.5).mean())*100:.2f}% | "
      f"p>0.8 {float((smp.p>0.8).mean())*100:.2f}% | p>0.95 {float((smp.p>0.95).mean())*100:.2f}%")
