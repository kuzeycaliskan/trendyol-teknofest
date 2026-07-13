"""
Cross-encoder eğitimi + test skorlama — KAGGLE NOTEBOOK'TA (GPU ile) çalıştırılır.

Mimari: sorgu ve ürün metni TEK modele birlikte girer; model dikkat mekanizmasıyla
ikisini kelime kelime karşılaştırır. Bi-encoder'dan (ayrı embeddingler) daha
güçlüdür çünkü etkileşimi görür; bedeli, her çift için ayrı ileri geçiş.

Veri/metodoloji yerel pipeline ile tutarlı:
- Aynı seed ve prosedürle terim bazlı %15 validasyon ayrımı (sızıntı yok).
- Negatifler: pozitif başına 1 rastgele + 1 zor (aynı üst kategori, farklı leaf).
- Çıktılar: ce_val_scores.npy + ce_val_labels.npy (yerel harman/eşik analizi
  için), ce_test_scores.npy (3.36M skor), ce_model/ (model).

Kaggle'da beklenen süre (T4 GPU): eğitim ~1 saat, test skorlama ~30-40 dk.
"""

import os

# sentence-transformers CrossEncoder.fit, DataParallel (çoklu GPU) ile uyumsuz
# (AttributeError: 'DataParallel' object has no attribute 'preprocess').
# Tek GPU'ya sabitle — torch import edilmeden ÖNCE ayarlanmalı.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

SEED = 42
VAL_TERM_FRACTION = 0.15
BASE_MODEL = "distilbert-base-multilingual-cased"
# Veri klasörünü otomatik bul: /kaggle/input altında (her derinlikte) items.csv ara
import glob

DATA, OUT = None, "/kaggle/working"
hits = glob.glob("/kaggle/input/**/items.csv", recursive=True)
if hits:
    DATA = os.path.dirname(hits[0])
elif os.path.isdir("/kaggle/input"):
    print("HATA: /kaggle/input altında items.csv yok. Mevcut içerik:")
    for root, dirs, files in os.walk("/kaggle/input"):
        print(" ", root, "->", files[:5])
    raise SystemExit(
        "Yarışma verisi oturuma bağlı değil. Sağ panel > Input > Add Input ile "
        "yarışmayı ekleyin; eklediyseniz oturumu yeniden başlatın "
        "(sağ üstteki güç düğmesi) veya Save & Run All (Commit) kullanın."
    )
else:  # yerel çalıştırma
    DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
    OUT = "."
print(f"Veri klasörü: {DATA}")

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

print("[1/5] Veri yükleniyor...")
items = pd.read_csv(f"{DATA}/items.csv",
                    usecols=["item_id", "title", "category", "brand"]).fillna("")
terms = pd.read_csv(f"{DATA}/terms.csv")
train_pos = pd.read_csv(f"{DATA}/training_pairs.csv")
test = pd.read_csv(f"{DATA}/submission_pairs.csv")

q_of = dict(zip(terms.term_id, terms["query"].astype(str).str.lower()))
items["text"] = (items.title.str.lower() + " | "
                 + items.category.str.replace("/", " ").str.lower() + " | "
                 + items.brand.str.lower())
t_of = dict(zip(items.item_id, items.text))
item_cat = dict(zip(items.item_id, items.category))
items["top_cat"] = items.category.str.split("/").str[0]

print("[2/5] Terim bazlı ayrım + negatif örnekleme...")
terms_unique = np.asarray(train_pos.term_id.unique(), dtype=object)
perm = rng.permutation(len(terms_unique))
n_val = int(len(terms_unique) * VAL_TERM_FRACTION)
val_terms = set(terms_unique[perm[:n_val]])
is_val = train_pos.term_id.isin(val_terms)
pos_tr, pos_va = train_pos[~is_val], train_pos[is_val]

all_ids = items.item_id.to_numpy()
by_topcat = {tc: g.item_id.to_numpy() for tc, g in items.groupby("top_cat")}


def sample_negatives(pairs):
    term_pos = pairs.groupby("term_id")["item_id"].agg(set).to_dict()
    neg_t, neg_i = [], []
    for t, i in zip(pairs.term_id, pairs.item_id):
        c = all_ids[rng.integers(len(all_ids))]          # rastgele
        while c in term_pos[t]:
            c = all_ids[rng.integers(len(all_ids))]
        neg_t.append(t); neg_i.append(c)
        pool = by_topcat.get(item_cat[i].split("/")[0], all_ids)  # zor
        c = None
        for _ in range(10):
            cand = pool[rng.integers(len(pool))]
            if item_cat[cand] != item_cat[i] and cand not in term_pos[t]:
                c = cand; break
        if c is None:
            c = all_ids[rng.integers(len(all_ids))]
            while c in term_pos[t]:
                c = all_ids[rng.integers(len(all_ids))]
        neg_t.append(t); neg_i.append(c)
    return pd.DataFrame({"term_id": neg_t, "item_id": neg_i, "label": 0})


df_tr = pd.concat([pos_tr[["term_id", "item_id", "label"]],
                   sample_negatives(pos_tr)], ignore_index=True)
df_tr = df_tr.sample(frac=1, random_state=SEED)
df_va = pd.concat([pos_va[["term_id", "item_id", "label"]],
                   sample_negatives(pos_va)], ignore_index=True)
print(f"   train: {len(df_tr)} | val: {len(df_va)}")

print("[3/5] Cross-encoder eğitiliyor...")
model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=96)
train_examples = [
    InputExample(texts=[q_of[t], t_of[i]], label=float(y))
    for t, i, y in zip(df_tr.term_id, df_tr.item_id, df_tr.label)
]
loader = DataLoader(train_examples, shuffle=True, batch_size=64, drop_last=True)
model.fit(train_dataloader=loader, epochs=1, warmup_steps=1000,
          show_progress_bar=True)
model.save(f"{OUT}/ce_model")

print("[4/5] Validasyon skorlanıyor...")
va_pairs = [[q_of[t], t_of[i]] for t, i in zip(df_va.term_id, df_va.item_id)]
va_scores = model.predict(va_pairs, batch_size=512, show_progress_bar=True,
                          apply_softmax=False)
np.save(f"{OUT}/ce_val_scores.npy", va_scores.astype(np.float16))
np.save(f"{OUT}/ce_val_labels.npy", df_va.label.to_numpy())
df_va[["term_id", "item_id", "label"]].to_csv(f"{OUT}/ce_val_pairs.csv", index=False)

print("[5/5] Test skorlanıyor (3.36M çift)...")
te_pairs = [[q_of[t], t_of[i]] for t, i in zip(test.term_id, test.item_id)]
te_scores = model.predict(te_pairs, batch_size=512, show_progress_bar=True,
                          apply_softmax=False)
np.save(f"{OUT}/ce_test_scores.npy", te_scores.astype(np.float16))
print("Bitti. İndirilecekler: ce_test_scores.npy, ce_val_scores.npy, "
      "ce_val_labels.npy, ce_val_pairs.csv (ce_model/ opsiyonel)")
