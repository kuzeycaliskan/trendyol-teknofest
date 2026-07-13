"""
Seviye 2: öznitelik mühendisliği + LightGBM ile (sorgu, ürün) alaka sınıflandırması.

Metodoloji notları (bilinçli kararlar):
- Train/val ayrımı TERM bazlıdır (GroupSplit): validasyon terimleri eğitimde hiç
  görünmez. Test terimlerinin de eğitimde hiç görünmediği doğrulanmıştır (%0
  örtüşme); bu ayrım test koşulunu taklit eder.
- Eğitim verisi yalnızca pozitif içerdiğinden negatifler sentetiktir:
  pozitif başına 1 uniform-rastgele + 1 zor negatif (aynı üst kategori, farklı
  alt kategori). SINIRLILIK: sentetik negatif dağılımı, testin gerçek negatif
  dağılımıyla birebir aynı olmayabilir; validasyon skoru bir vekildir (proxy).
- TF-IDF yalnızca etiketsiz ürün kataloğu metni üzerinde fit edilir. Etiket
  bilgisi hiçbir ön-işleme adımına girmez.
- Karar eşiği yalnızca validasyon kümesi üzerinde, macro-F1 maksimize edilerek
  seçilir.
- Tüm rastgelelik SEED ile sabittir; sonuçlar tekrarlanabilir.
"""

from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
VAL_TERM_FRACTION = 0.15
CHUNK = 200_000

TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+")

Q_MALE = ("erkek",)
Q_FEMALE = ("kadın", "bayan", "kadin")
Q_KID = ("çocuk", "cocuk", "kız çocuk", "erkek çocuk")
Q_BABY = ("bebek",)

FEATURES = [
    "cov_main", "cov_title", "cov_cat", "cov_attr", "cov_exact",
    "tfidf_cos", "q_ntok", "q_nchar", "title_ntok", "brand_in_query",
    "q_male", "q_female", "q_kid", "q_baby",
    "gender_conflict", "age_conflict",
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------
# Veri yükleme ve ürün/terim sözlükleri
# --------------------------------------------------------------------------

def load_catalog() -> tuple[pd.DataFrame, dict, dict]:
    items = pd.read_csv(f"{DATA}/items.csv").fillna("")
    for col in ("title", "category", "brand", "attributes"):
        items[col] = items[col].str.lower()

    item_meta = {}
    for row in items.itertuples(index=False):
        title = row.title
        cat = row.category.replace("/", " ")
        main = f" {title} {cat} {row.brand}"
        item_meta[row.item_id] = (
            main,                          # 0: başlık+kategori+marka metni
            f" {title}",                   # 1: başlık
            f" {cat}",                     # 2: kategori
            f" {row.attributes}",          # 3: özellikler
            frozenset(tokenize(main)),     # 4: ana metin kelime kümesi
            row.brand,                     # 5: marka
            row.gender,                    # 6
            row.age_group,                 # 7
            len(tokenize(title)),          # 8: başlık kelime sayısı
        )
    # Zor negatif örnekleme için: üst kategori -> item dizisi
    items["top_cat"] = items.category.str.split("/").str[0]
    return items, item_meta, dict(zip(items.item_id, items.category))


def load_terms() -> dict:
    terms = pd.read_csv(f"{DATA}/terms.csv")
    term_meta = {}
    for tid, query in zip(terms.term_id, terms["query"]):
        q = str(query).lower()
        toks = tokenize(q)
        term_meta[tid] = (
            toks,
            q,
            any(w in toks for w in Q_MALE),
            any(w in toks for w in Q_FEMALE),
            any(w in q for w in Q_KID),
            any(w in toks for w in Q_BABY),
        )
    return term_meta


# --------------------------------------------------------------------------
# Negatif örnekleme
# --------------------------------------------------------------------------

def sample_negatives(
    pairs: pd.DataFrame,
    items: pd.DataFrame,
    item_cat: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Pozitif başına 1 rastgele + 1 zor (aynı üst kategori, farklı alt
    kategori) negatif üretir. Terimin bilinen pozitifleri asla negatif
    olarak etiketlenmez.

    Not: Aynı alt kategoriden (leaf) negatif üretmiyoruz çünkü etiketsiz bir
    aynı-leaf ürünü gerçekte alakalı olma ihtimali yüksek (false negative
    riski). Üst kategori aynı / leaf farklı seçimi "yakın ama alakasız"
    bölgesini hedefler.
    """
    term_pos: dict[str, set] = pairs.groupby("term_id")["item_id"].agg(set).to_dict()
    all_ids = items.item_id.to_numpy()
    by_topcat = {tc: g.item_id.to_numpy() for tc, g in items.groupby("top_cat")}

    neg_terms, neg_items = [], []

    # 1) Uniform rastgele negatifler (vektörize, çakışanlar yeniden denenir)
    cand = rng.choice(all_ids, size=len(pairs))
    for t, c in zip(pairs.term_id, cand):
        while c in term_pos[t]:
            c = all_ids[rng.integers(len(all_ids))]
        neg_terms.append(t)
        neg_items.append(c)

    # 2) Zor negatifler
    pos_cats = [item_cat[i] for i in pairs.item_id]
    for t, pos_cat in zip(pairs.term_id, pos_cats):
        pool = by_topcat.get(pos_cat.split("/")[0], all_ids)
        c = None
        for _ in range(10):  # farklı leaf bulmak için sınırlı deneme
            cand_i = pool[rng.integers(len(pool))]
            if item_cat[cand_i] != pos_cat and cand_i not in term_pos[t]:
                c = cand_i
                break
        if c is None:  # üst kategori tek leaf'ten ibaretse rastgeleye düş
            c = all_ids[rng.integers(len(all_ids))]
            while c in term_pos[t]:
                c = all_ids[rng.integers(len(all_ids))]
        neg_terms.append(t)
        neg_items.append(c)

    return pd.DataFrame({"term_id": neg_terms, "item_id": neg_items, "label": 0})


# --------------------------------------------------------------------------
# Öznitelikler
# --------------------------------------------------------------------------

def coverage(tokens: list[str], text: str) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in text) / len(tokens)


def build_features(
    term_ids: np.ndarray,
    item_ids: np.ndarray,
    term_meta: dict,
    item_meta: dict,
    tfidf_cos: np.ndarray,
) -> np.ndarray:
    X = np.empty((len(term_ids), len(FEATURES)), dtype=np.float32)
    for k, (tid, iid) in enumerate(zip(term_ids, item_ids)):
        toks, q, male, female, kid, baby = term_meta[tid]
        main, title, cat, attr, tokset, brand, gender, age, title_ntok = item_meta[iid]
        exact = sum(1 for t in toks if t in tokset) / len(toks) if toks else 0.0
        X[k] = (
            coverage(toks, main),
            coverage(toks, title),
            coverage(toks, cat),
            coverage(toks, attr),
            exact,
            tfidf_cos[k],
            len(toks),
            len(q),
            title_ntok,
            1.0 if len(brand) >= 3 and brand in q else 0.0,
            male, female, kid, baby,
            1.0 if (male and gender == "kadın") or (female and gender == "erkek") else 0.0,
            1.0 if (baby or kid) and age == "yetişkin" else 0.0,
        )
    return X


def pairwise_cosine(
    Q: sparse.csr_matrix, I: sparse.csr_matrix,
    t_idx: np.ndarray, i_idx: np.ndarray,
) -> np.ndarray:
    """L2-normalize edilmiş satırlar için çift bazlı kosinüs (chunk'lı)."""
    out = np.empty(len(t_idx), dtype=np.float32)
    for s in range(0, len(t_idx), CHUNK):
        e = min(s + CHUNK, len(t_idx))
        out[s:e] = np.asarray(
            Q[t_idx[s:e]].multiply(I[i_idx[s:e]]).sum(axis=1)
        ).ravel()
    return out


# --------------------------------------------------------------------------
# Ana akış
# --------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)

    print("[1/7] Katalog ve terimler yükleniyor...")
    items, item_meta, item_cat = load_catalog()
    term_meta = load_terms()
    train_pos = pd.read_csv(f"{DATA}/training_pairs.csv")
    test = pd.read_csv(f"{DATA}/submission_pairs.csv")

    print("[2/7] Terim bazlı train/val ayrımı...")
    terms_unique = np.asarray(train_pos.term_id.unique(), dtype=object)
    perm = rng.permutation(len(terms_unique))
    n_val = int(len(terms_unique) * VAL_TERM_FRACTION)
    val_terms = set(terms_unique[perm[:n_val]])
    is_val = train_pos.term_id.isin(val_terms)
    pos_tr, pos_va = train_pos[~is_val], train_pos[is_val]
    print(f"   train pozitif: {len(pos_tr)} | val pozitif: {len(pos_va)}")

    print("[3/7] Negatif örnekleme (train ve val için ayrı ayrı)...")
    neg_tr = sample_negatives(pos_tr, items, item_cat, rng)
    neg_va = sample_negatives(pos_va, items, item_cat, rng)
    df_tr = pd.concat(
        [pos_tr[["term_id", "item_id", "label"]], neg_tr], ignore_index=True
    ).sample(frac=1, random_state=SEED)
    df_va = pd.concat(
        [pos_va[["term_id", "item_id", "label"]], neg_va], ignore_index=True
    )

    print("[4/7] TF-IDF (char 3-4) katalog üzerinde fit ediliyor...")
    item_ids_all = items.item_id.to_numpy()
    item_texts = [item_meta[i][0] for i in item_ids_all]
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4),
        max_features=200_000, sublinear_tf=True, dtype=np.float32,
    )
    I_mat = vec.fit_transform(item_texts)  # etiketsiz katalog: sızıntı yok
    term_ids_all = list(term_meta.keys())
    Q_mat = vec.transform([term_meta[t][1] for t in term_ids_all])
    item_row = {i: k for k, i in enumerate(item_ids_all)}
    term_row = {t: k for k, t in enumerate(term_ids_all)}

    def tfidf_for(df: pd.DataFrame) -> np.ndarray:
        return pairwise_cosine(
            Q_mat, I_mat,
            df.term_id.map(term_row).to_numpy(),
            df.item_id.map(item_row).to_numpy(),
        )

    print("[5/7] Öznitelikler üretiliyor...")
    X_tr = build_features(
        df_tr.term_id.to_numpy(), df_tr.item_id.to_numpy(),
        term_meta, item_meta, tfidf_for(df_tr),
    )
    y_tr = df_tr.label.to_numpy()
    X_va = build_features(
        df_va.term_id.to_numpy(), df_va.item_id.to_numpy(),
        term_meta, item_meta, tfidf_for(df_va),
    )
    y_va = df_va.label.to_numpy()

    print("[6/7] LightGBM eğitiliyor...")
    model = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.05, num_leaves=127,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        min_child_samples=50, random_state=SEED, n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)], eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )
    p_va = model.predict_proba(X_va)[:, 1]
    print(f"   val AUC: {roc_auc_score(y_va, p_va):.4f}")

    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.05, 0.96, 0.025):
        f1 = f1_score(y_va, (p_va >= thr).astype(int), average="macro")
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), f1
    print(f"   val macro-F1: {best_f1:.4f} @ eşik {best_thr:.3f}")
    print("   (sentetik negatiflerle proxy skor — LB ile birebir kıyaslanamaz)")

    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("   öznitelik önemleri:", ", ".join(f"{n}={v}" for n, v in imp))

    print("[7/7] Test skorlanıyor (3.36M çift)...")
    X_te = build_features(
        test.term_id.to_numpy(), test.item_id.to_numpy(),
        term_meta, item_meta, tfidf_for(test),
    )
    p_te = model.predict_proba(X_te)[:, 1]
    np.save("test_proba_lgbm.npy", p_te)  # eşik değişikliği için sakla
    pred = (p_te >= best_thr).astype(int)
    print(f"   test pozitif oranı: {pred.mean():.3f}")

    pd.DataFrame({"id": test.id, "prediction": pred}).to_csv(
        "submission_lgbm.csv", index=False
    )
    model.booster_.save_model("model_lgbm.txt")
    print("Yazıldı: submission_lgbm.csv, model_lgbm.txt, test_proba_lgbm.npy")


if __name__ == "__main__":
    main()
