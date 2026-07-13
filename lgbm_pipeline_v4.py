"""
KULLANILMIYOR — LB'de başarısız oldu (0.648/0.558 vs v3'ün 0.767'si); ders
kaydı olarak saklanıyor: grup-içi rank öznitelikleri sentetik grup yapısını
(rastgele negatifleri tanımayı) öğrenip gerçek test gruplarına transfer
olmadı. Grup-bazlı öznitelik, ancak gerçekçi grup dağılımıyla eğitilebilir.

Seviye 4: v3 öznitelikleri + terim-içi sıralama (rank) öznitelikleri.

v3'ten farklar:
- Rank öznitelikleri: emb_cos, tfidf_cos ve cov_main için her terimin kendi
  aday kümesi içinde yüzdelik sıra (pct), z-skoru ve gruptaki maksimumdan fark.
  Gerekçe: test her terim için ~104 aday içeriyor; bir adayın MUTLAK skoru
  kadar, aynı aramanın diğer adaylarına GÖRE konumu da bilgi taşır.
  Not: eğitim gruplarının aday dağılımı sentetik olduğundan test gruplarıyla
  birebir aynı değildir; pct/z gibi ölçekten bağımsız istatistikler bu
  kaymaya görece dayanıklıdır (bilinçli tercih).
- Negatif oranı 3:1 (1 rastgele + 2 zor): eğitim/validasyon dengesi (%25
  pozitif), gerçek gönderimlerden çıkarılan test öncülüne (~%26) hizalanır.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

from lgbm_pipeline import (
    CHUNK, DATA, SEED, VAL_TERM_FRACTION,
    load_catalog, load_terms, pairwise_cosine, sample_negatives,
)
from lgbm_pipeline_v3 import (
    FEATURES as BASE_FEATURES,
    build_features, emb_cosine, resample_to_prior,
)

TEST_POS_RATE = 0.26  # gerçek gönderim skorlarının feasibility analizinden
RANK_ON = ["emb_cos", "tfidf_cos", "cov_main"]
FEATURES = BASE_FEATURES + [f"{c}_{s}" for c in RANK_ON for s in ("pct", "z", "dmax")]


def add_rank_features(X_base: np.ndarray, term_ids: np.ndarray) -> np.ndarray:
    """Terim grubu içi pct/z/dmax kolonlarını üretip X'e ekler."""
    df = pd.DataFrame(
        {c: X_base[:, BASE_FEATURES.index(c)] for c in RANK_ON},
        copy=False,
    )
    df["term_id"] = term_ids
    g = df.groupby("term_id", sort=False)
    extras = []
    for c in RANK_ON:
        pct = g[c].rank(pct=True).to_numpy(dtype=np.float32)
        mean = g[c].transform("mean").to_numpy(dtype=np.float32)
        std = g[c].transform("std").fillna(0.0).to_numpy(dtype=np.float32)
        gmax = g[c].transform("max").to_numpy(dtype=np.float32)
        x = df[c].to_numpy(dtype=np.float32)
        z = np.where(std > 0, (x - mean) / np.where(std > 0, std, 1.0), 0.0)
        extras += [pct, z.astype(np.float32), (x - gmax).astype(np.float32)]
    return np.column_stack([X_base] + extras)


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("[1/8] Katalog, terimler, embeddingler yükleniyor...")
    items, item_meta, item_cat = load_catalog()
    term_meta = load_terms()
    train_pos = pd.read_csv(f"{DATA}/training_pairs.csv")
    test = pd.read_csv(f"{DATA}/submission_pairs.csv")

    E_q = np.load("emb_terms.npy")
    E_i = np.load("emb_items.npy")
    q_row = {t: k for k, t in enumerate(np.load("emb_terms_ids.npy", allow_pickle=True))}
    i_row = {i: k for k, i in enumerate(np.load("emb_items_ids.npy", allow_pickle=True))}

    print("[2/8] Terim bazlı train/val ayrımı...")
    terms_unique = np.asarray(train_pos.term_id.unique(), dtype=object)
    perm = rng.permutation(len(terms_unique))
    n_val = int(len(terms_unique) * VAL_TERM_FRACTION)
    val_terms = set(terms_unique[perm[:n_val]])
    is_val = train_pos.term_id.isin(val_terms)
    pos_tr, pos_va = train_pos[~is_val], train_pos[is_val]

    print("[3/8] Negatif örnekleme (1 rastgele + 2 zor, %25 pozitif denge)...")
    def negatives(pos_df):
        n1 = sample_negatives(pos_df, items, item_cat, rng)   # 1 rastgele + 1 zor
        n2 = sample_negatives(pos_df, items, item_cat, rng)   # ikinci tur
        n2 = n2.iloc[len(pos_df):]                            # sadece zor yarısı
        return pd.concat([n1, n2], ignore_index=True)

    df_tr = pd.concat(
        [pos_tr[["term_id", "item_id", "label"]], negatives(pos_tr)],
        ignore_index=True,
    ).sample(frac=1, random_state=SEED)
    df_va = pd.concat(
        [pos_va[["term_id", "item_id", "label"]], negatives(pos_va)],
        ignore_index=True,
    )
    print(f"   train: {len(df_tr)} satır (%{df_tr.label.mean()*100:.1f} pozitif) | "
          f"val: {len(df_va)} satır")

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
        X_base = build_features(
            df.term_id.to_numpy(), df.item_id.to_numpy(),
            term_meta, item_meta, tf, em,
        )
        return add_rank_features(X_base, df.term_id.to_numpy())

    print("[5/8] Öznitelikler üretiliyor (train/val)...")
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
    print(f"   val AUC: {roc_auc_score(y_va, p_va):.4f}")

    print(f"[7/8] Eşik kalibrasyonu (%{TEST_POS_RATE*100:.0f} pozitif denge)...")
    df_va = df_va.reset_index(drop=True)
    df_va["proba"] = p_va
    va_prior = resample_to_prior(df_va, TEST_POS_RATE, rng)
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.02, 0.99, 0.01):
        f1 = f1_score(va_prior.label, (va_prior.proba >= thr).astype(int),
                      average="macro")
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), f1
    print(f"   val macro-F1 ({TEST_POS_RATE:.0%} dengede): {best_f1:.4f} @ eşik {best_thr:.2f}")

    # Duyarlılık: öncül varsayımı sapsa eşik/skor ne kadar oynuyor?
    for p_alt in (0.20, 0.30, 0.35):
        va_alt = resample_to_prior(df_va, p_alt, rng)
        f1_alt = f1_score(va_alt.label, (va_alt.proba >= best_thr).astype(int),
                          average="macro")
        print(f"   duyarlılık p={p_alt:.2f}: aynı eşikle macro-F1 {f1_alt:.4f}")

    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("   öznitelik önemleri (ilk 12):",
          ", ".join(f"{n}={v}" for n, v in imp[:12]))

    print("[8/8] Test skorlanıyor...")
    # Rank öznitelikleri grup bütünlüğü gerektirir: önce tüm öznitelikler,
    # tahmin sonra chunk'lı.
    X_te = feats(test)
    p_te = np.empty(len(test), dtype=np.float32)
    for s in range(0, len(test), 500_000):
        e = min(s + 500_000, len(test))
        p_te[s:e] = model.predict_proba(X_te[s:e])[:, 1]
    np.save("test_proba_v4.npy", p_te)
    pred = (p_te >= best_thr).astype(int)
    print(f"   test pozitif oranı: {pred.mean():.3f}")

    pd.DataFrame({"id": test.id, "prediction": pred}).to_csv(
        "submission_v4.csv", index=False)
    model.booster_.save_model("model_v4.txt")
    print("Yazıldı: submission_v4.csv, model_v4.txt, test_proba_v4.npy")


if __name__ == "__main__":
    main()
