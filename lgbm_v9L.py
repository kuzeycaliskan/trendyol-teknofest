"""
LGBM v9L: v3 pipeline + 5 komşu-sorgu graf özniteliği (denetim-onaylı spec).

Öznitelikler — (q,i) çifti için, q'nun en benzer K=10 TRAIN-SPLIT sorgusu
(n_1..n_K, benzerlikleri s_k) üzerinden:
  f1 = max_k [ s_k · 1(i, n_k'nın pozitifi) ]     (doğrudan etiket kanıtı)
  f2 = Σ_k 1(i, n_k'nın pozitifi)                  (kanıt sayısı)
  f3 = max_k [ s_k · cos(i, centroid(n_k)) ]       (dolaylı benzerlik kanıtı)
  f4 = ilk 3 komşu için cos(i, centroid) ortalaması
  f5 = s_1                                          (kanıt-var-mı kapısı)

Sızıntı önlemleri (kritik):
- Komşu havuzu YALNIZ train-split terimleri (val terimleri havuzda değil —
  val çiftlerinin etiketi hiçbir özniteliğe sızamaz).
- Self-exclusion: train çiftinde terimin kendisi komşu olamaz (aksi halde
  f1 = etiketin kendisi olurdu).
- Test terimleri trainde yok (%0) → test koşulu, self-exclusion'lı train
  koşuluyla simetrik. M2 ölçümü (denetim): test→train ve train→train(self
  hariç) benzerlik dağılımları birebir aynı → near-dup şişmesi yok.
"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ.setdefault("EMB_PREFIX", "embpl")
from lgbm_pipeline import DATA, SEED, VAL_TERM_FRACTION, load_catalog, load_terms, pairwise_cosine, sample_negatives
from lgbm_pipeline_v3 import EMB_PREFIX, FEATURES as BASE_FEATURES, build_features, emb_cosine

K = 10
NBR_FEATURES = ["nbr_f1_labelmax", "nbr_f2_labelcnt", "nbr_f3_centmax",
                "nbr_f4_centtop3", "nbr_f5_sim1"]
FEATURES = BASE_FEATURES + NBR_FEATURES


def main():
    rng = np.random.default_rng(SEED)

    print("[1/8] Katalog/terim/embedding...")
    items, item_meta, item_cat = load_catalog()
    term_meta = load_terms()
    train_pos = pd.read_csv(f"{DATA}/training_pairs.csv")
    test = pd.read_csv(f"{DATA}/submission_pairs.csv")
    E_q = np.load(f"{EMB_PREFIX}_terms.npy").astype(np.float32)
    E_i = np.load(f"{EMB_PREFIX}_items.npy").astype(np.float32)
    q_row = {t: k for k, t in enumerate(np.load(f"{EMB_PREFIX}_terms_ids.npy", allow_pickle=True))}
    i_row = {i: k for k, i in enumerate(np.load(f"{EMB_PREFIX}_items_ids.npy", allow_pickle=True))}

    print("[2/8] Terim ayrımı (v3 ile birebir aynı)...")
    terms_unique = np.asarray(train_pos.term_id.unique(), dtype=object)
    perm = rng.permutation(len(terms_unique))
    n_val = int(len(terms_unique) * VAL_TERM_FRACTION)
    val_terms = set(terms_unique[perm[:n_val]])
    is_val = train_pos.term_id.isin(val_terms)
    pos_tr, pos_va = train_pos[~is_val], train_pos[is_val]

    print("[3/8] Komşu altyapısı (havuz = train-split)...")
    nbr_S = np.load("nbr_sims.npy")            # 50k x 31 (tüm train terimlerine göre)
    nbr_I = np.load("nbr_train_idx.npy")
    nbr_terms = np.load("nbr_train_terms.npy", allow_pickle=True)  # 18k
    trainsplit = set(pos_tr.term_id.unique())
    pool_ok = np.array([t in trainsplit for t in nbr_terms])       # val terimleri havuz dışı
    pos_sets = pos_tr.groupby("term_id")["item_id"].agg(set).to_dict()

    # centroidler (yalnız train-split terimleri; L2-normalize)
    cent = np.zeros((len(nbr_terms), E_i.shape[1]), dtype=np.float32)
    for k, t in enumerate(nbr_terms):
        if not pool_ok[k]:
            continue
        rows = [i_row[i] for i in pos_sets[t] if i in i_row]
        if rows:
            c = E_i[rows].mean(axis=0)
            n = np.linalg.norm(c)
            cent[k] = c / n if n > 0 else 0

    def neighbor_feats(term_ids, item_ids, chunk=100_000):
        out = np.zeros((len(term_ids), 5), dtype=np.float32)
        t_rows = np.array([q_row[t] for t in term_ids])
        i_rows = np.array([i_row[i] for i in item_ids])
        for s in range(0, len(term_ids), chunk):
            e = min(s + chunk, len(term_ids))
            for r in range(s, e):
                t = term_ids[r]
                cands_i = nbr_I[t_rows[r]]
                cands_s = nbr_S[t_rows[r]]
                sel_idx, sel_sim = [], []
                for ci, cs in zip(cands_i, cands_s):
                    ct = nbr_terms[ci]
                    if not pool_ok[ci] or ct == t:   # havuz + self-exclusion
                        continue
                    sel_idx.append(ci)
                    sel_sim.append(cs)
                    if len(sel_idx) == K:
                        break
                if not sel_idx:
                    continue
                item = item_ids[r]
                hits = [sim for ci, sim in zip(sel_idx, sel_sim)
                        if item in pos_sets[nbr_terms[ci]]]
                out[r, 0] = max(hits) if hits else 0.0
                out[r, 1] = len(hits)
                cvecs = cent[sel_idx]                        # k x 384
                csims = cvecs @ E_i[i_rows[r]]
                out[r, 2] = float(np.max(np.array(sel_sim) * csims))
                out[r, 3] = float(np.mean(csims[:3]))
                out[r, 4] = sel_sim[0]
        return out

    print("[4/8] Negatifler + TF-IDF (v3 akışı)...")
    neg_tr = sample_negatives(pos_tr, items, item_cat, rng)
    neg_va = sample_negatives(pos_va, items, item_cat, rng)
    df_tr = pd.concat([pos_tr[["term_id", "item_id", "label"]], neg_tr],
                      ignore_index=True).sample(frac=1, random_state=SEED)
    df_va = pd.concat([pos_va[["term_id", "item_id", "label"]], neg_va],
                      ignore_index=True)

    item_ids_all = items.item_id.to_numpy()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                          max_features=200_000, sublinear_tf=True, dtype=np.float32)
    I_mat = vec.fit_transform([item_meta[i][0] for i in item_ids_all])
    term_ids_all = list(term_meta.keys())
    Q_mat = vec.transform([term_meta[t][1] for t in term_ids_all])
    tf_i = {i: k for k, i in enumerate(item_ids_all)}
    tf_t = {t: k for k, t in enumerate(term_ids_all)}

    def feats(df):
        t_ids, i_ids = df.term_id.to_numpy(), df.item_id.to_numpy()
        tf = pairwise_cosine(Q_mat, I_mat,
                             df.term_id.map(tf_t).to_numpy(),
                             df.item_id.map(tf_i).to_numpy())
        em = emb_cosine(E_q, E_i,
                        df.term_id.map(q_row).to_numpy(),
                        df.item_id.map(i_row).to_numpy())
        X = build_features(t_ids, i_ids, term_meta, item_meta, tf, em)
        return np.hstack([X, neighbor_feats(t_ids, i_ids)])

    print("[5/8] Öznitelikler...")
    X_tr, y_tr = feats(df_tr), df_tr.label.to_numpy()
    X_va, y_va = feats(df_va), df_va.label.to_numpy()

    print("[6/8] LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.05, num_leaves=127,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        min_child_samples=50, random_state=SEED, n_jobs=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    p_va = model.predict_proba(X_va)[:, 1]
    print(f"   val AUC: {roc_auc_score(y_va, p_va):.4f} (v8 referans: 0.9971)")
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("   komşu özniteliklerinin sırası:",
          [f"{n}#{k+1}" for k, (n, _) in enumerate(imp) if n.startswith("nbr_")])

    print("[7/8] Test skorlanıyor...")
    p_te = np.empty(len(test), dtype=np.float32)
    for s in range(0, len(test), 500_000):
        e = min(s + 500_000, len(test))
        p_te[s:e] = model.predict_proba(feats(test.iloc[s:e]))[:, 1]
    np.save("test_proba_v9L.npy", p_te)
    model.booster_.save_model("model_v9L.txt")
    print("[8/8] Yazıldı: test_proba_v9L.npy, model_v9L.txt")


if __name__ == "__main__":
    main()
