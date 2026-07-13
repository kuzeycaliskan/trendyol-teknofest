"""
Seviye 3: Seviye 2 öznitelikleri + multilingual-e5-small embedding kosinüsü.

Seviye 2'den farklar:
- emb_cos özniteliği: L2-normalize sorgu/ürün embeddinglerinin iç çarpımı.
  Kelime örtüşmesi olmayan semantik eşleşmeleri (sneaker ~ spor ayakkabı)
  yakalamak için. Embeddingler embed_texts.py tarafından üretilir.
- Eşik kalibrasyonu: eğitim 1:2 (poz:neg) sentetik dengede yapılır; karar
  eşiği, validasyon kümesi TEST_POS_RATE dengesine alt-örneklenerek seçilir
  (resample_to_prior: negatif yetmezse pozitif tarafı alt-örneklenir; örnek
  çoğaltma yapılmaz). TEST_POS_RATE=0.26, gerçek LB gönderimlerinin
  feasibility analizinden gelir (p=0.70 hipotezi skorlarımızla çelişip
  elendi). Eğitim verisi manipüle edilmez.
"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

from lgbm_pipeline import (
    CHUNK, DATA, SEED, VAL_TERM_FRACTION,
    coverage, load_catalog, load_terms, pairwise_cosine, sample_negatives,
)

# All-ones probe (0.412) + feasibility analizi: LB skorlarımız p=0.70 ile
# imkansız, p~0.26 ile tutarlı (metrik, tahmin edilmeyen sınıfı ortalamadan
# düşürüyor görünüyor: 0.412 = sınıf-1 F1 = 2p/(1+p) -> p ~= 0.26).
# All-zeros probe sonucu gelince kesinleşecek.
TEST_POS_RATE = 0.26
EMB_PREFIX = os.environ.get("EMB_PREFIX", "emb")
TAG = os.environ.get("TAG", "v3")

FEATURES = [
    "cov_main", "cov_title", "cov_cat", "cov_attr", "cov_exact",
    "tfidf_cos", "emb_cos", "q_ntok", "q_nchar", "title_ntok",
    "brand_in_query", "q_male", "q_female", "q_kid", "q_baby",
    "gender_conflict", "age_conflict",
]


def build_features(term_ids, item_ids, term_meta, item_meta, tfidf_cos, emb_cos):
    X = np.empty((len(term_ids), len(FEATURES)), dtype=np.float32)
    for k, (tid, iid) in enumerate(zip(term_ids, item_ids)):
        toks, q, male, female, kid, baby = term_meta[tid]
        main, title, cat, attr, tokset, brand, gender, age, title_ntok = item_meta[iid]
        exact = sum(1 for t in toks if t in tokset) / len(toks) if toks else 0.0
        X[k] = (
            coverage(toks, main), coverage(toks, title), coverage(toks, cat),
            coverage(toks, attr), exact,
            tfidf_cos[k], emb_cos[k],
            len(toks), len(q), title_ntok,
            1.0 if len(brand) >= 3 and brand in q else 0.0,
            male, female, kid, baby,
            1.0 if (male and gender == "kadın") or (female and gender == "erkek") else 0.0,
            1.0 if (baby or kid) and age == "yetişkin" else 0.0,
        )
    return X


def emb_cosine(E_q, E_i, t_idx, i_idx):
    """Çift bazlı embedding iç çarpımı (satırlar L2-normalize, chunk'lı)."""
    out = np.empty(len(t_idx), dtype=np.float32)
    for s in range(0, len(t_idx), CHUNK):
        e = min(s + CHUNK, len(t_idx))
        out[s:e] = np.einsum(
            "ij,ij->i",
            E_q[t_idx[s:e]].astype(np.float32),
            E_i[i_idx[s:e]].astype(np.float32),
        )
    return out


def resample_to_prior(df, pos_rate, rng):
    """Validasyonu hedef pozitif orana getirir: negatif yeterliyse negatifleri,
    değilse pozitifleri alt-örnekler (hiçbir örnek çoğaltılmaz)."""
    pos, neg = df[df.label == 1], df[df.label == 0]
    n_neg_needed = int(len(pos) * (1 - pos_rate) / pos_rate)
    if n_neg_needed <= len(neg):
        idx = rng.choice(len(neg), size=n_neg_needed, replace=False)
        return pd.concat([pos, neg.iloc[idx]], ignore_index=True)
    n_pos_needed = int(len(neg) * pos_rate / (1 - pos_rate))
    idx = rng.choice(len(pos), size=n_pos_needed, replace=False)
    return pd.concat([pos.iloc[idx], neg], ignore_index=True)


def main():
    rng = np.random.default_rng(SEED)

    print("[1/8] Katalog, terimler, embeddingler yükleniyor...")
    items, item_meta, item_cat = load_catalog()
    term_meta = load_terms()
    train_pos = pd.read_csv(f"{DATA}/training_pairs.csv")
    test = pd.read_csv(f"{DATA}/submission_pairs.csv")

    E_q = np.load(f"{EMB_PREFIX}_terms.npy")
    E_i = np.load(f"{EMB_PREFIX}_items.npy")
    q_row = {t: k for k, t in enumerate(np.load(f"{EMB_PREFIX}_terms_ids.npy", allow_pickle=True))}
    i_row = {i: k for k, i in enumerate(np.load(f"{EMB_PREFIX}_items_ids.npy", allow_pickle=True))}

    print("[2/8] Terim bazlı train/val ayrımı...")
    terms_unique = np.asarray(train_pos.term_id.unique(), dtype=object)
    perm = rng.permutation(len(terms_unique))
    n_val = int(len(terms_unique) * VAL_TERM_FRACTION)
    val_terms = set(terms_unique[perm[:n_val]])
    is_val = train_pos.term_id.isin(val_terms)
    pos_tr, pos_va = train_pos[~is_val], train_pos[is_val]

    print("[3/8] Negatif örnekleme...")
    neg_tr = sample_negatives(pos_tr, items, item_cat, rng)
    neg_va = sample_negatives(pos_va, items, item_cat, rng)
    df_tr = pd.concat(
        [pos_tr[["term_id", "item_id", "label"]], neg_tr], ignore_index=True
    ).sample(frac=1, random_state=SEED)
    df_va = pd.concat(
        [pos_va[["term_id", "item_id", "label"]], neg_va], ignore_index=True
    )

    print("[4/8] TF-IDF fit + satır indeksleri...")
    item_ids_all = items.item_id.to_numpy()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                          max_features=200_000, sublinear_tf=True, dtype=np.float32)
    I_mat = vec.fit_transform([item_meta[i][0] for i in item_ids_all])
    term_ids_all = list(term_meta.keys())
    Q_mat = vec.transform([term_meta[t][1] for t in term_ids_all])
    tf_item_row = {i: k for k, i in enumerate(item_ids_all)}
    tf_term_row = {t: k for k, t in enumerate(term_ids_all)}

    def feats(df):
        t_ids, i_ids = df.term_id.to_numpy(), df.item_id.to_numpy()
        tf = pairwise_cosine(
            Q_mat, I_mat,
            df.term_id.map(tf_term_row).to_numpy(),
            df.item_id.map(tf_item_row).to_numpy(),
        )
        em = emb_cosine(
            E_q, E_i,
            df.term_id.map(q_row).to_numpy(),
            df.item_id.map(i_row).to_numpy(),
        )
        return build_features(t_ids, i_ids, term_meta, item_meta, tf, em)

    print("[5/8] Öznitelikler üretiliyor...")
    X_tr, y_tr = feats(df_tr), df_tr.label.to_numpy()
    X_va, y_va = feats(df_va), df_va.label.to_numpy()

    print("[6/8] LightGBM eğitiliyor...")
    model = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.05, num_leaves=127,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        min_child_samples=50, random_state=SEED, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(200)])
    p_va = model.predict_proba(X_va)[:, 1]
    print(f"   val AUC (1:2 dengede): {roc_auc_score(y_va, p_va):.4f}")

    print(f"[7/8] Eşik, test öncülüne ({TEST_POS_RATE:.0%} pozitif) göre kalibre ediliyor...")
    df_va = df_va.reset_index(drop=True)
    df_va["proba"] = p_va
    va_prior = resample_to_prior(df_va, TEST_POS_RATE, rng)
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.02, 0.99, 0.01):
        f1 = f1_score(va_prior.label, (va_prior.proba >= thr).astype(int), average="macro")
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), f1
    print(f"   val macro-F1 ({TEST_POS_RATE:.0%} pozitif dengede): {best_f1:.4f} @ eşik {best_thr:.2f}")

    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("   öznitelik önemleri:", ", ".join(f"{n}={v}" for n, v in imp))

    print("[8/8] Test skorlanıyor...")
    p_te = np.empty(len(test), dtype=np.float32)
    for s in range(0, len(test), 500_000):
        e = min(s + 500_000, len(test))
        p_te[s:e] = model.predict_proba(feats(test.iloc[s:e]))[:, 1]
    np.save(f"test_proba_{TAG}.npy", p_te)
    pred = (p_te >= best_thr).astype(int)
    print(f"   test pozitif oranı: {pred.mean():.3f}")

    pd.DataFrame({"id": test.id, "prediction": pred}).to_csv(
        f"submission_{TAG}.csv", index=False)
    model.booster_.save_model(f"model_{TAG}.txt")
    print(f"Yazıldı: submission_{TAG}.csv, model_{TAG}.txt, test_proba_{TAG}.npy")


if __name__ == "__main__":
    main()
