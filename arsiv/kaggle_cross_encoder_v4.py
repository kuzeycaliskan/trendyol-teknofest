"""
Cross-encoder v4 — KAGGLE NOTEBOOK'TA (GPU ile) çalıştırılır.

v2'den TEK fark: epochs=2 (pseudo YOK — o hat nötr çıktı ve kapandı).
Gerekçe: v2 reçetesi (pozitif + rastgele + madenci negatif) LB 0.856 ile
kanıtlandı; bi-encoder tarafında 2. epoch'un belirgin kazanç getirdiğini
gördük (v5→v6 LB +0.014). Validasyon kurulumu yine birebir aynı —
ce2/ce3 ile karşılaştırılabilir. Beklenen süre (T4): ~4.5-5 saat.

v1'den (distilbert) farklar — denetim raporu doğrultusunda:
- Taban model: dbmdz/bert-base-turkish-128k-uncased (Türkçe'ye özel; metinler
  zaten lowercase kullanılıyor, uncased uyumlu).
- Negatifler: pozitif başına 1 rastgele + embedding madenciliğiyle bulunmuş
  zor negatifler (mined_hard_negatives.csv — leaf-kategori false-negative
  korumalı, yerelde üretildi, Kaggle'a dataset olarak yüklenir).
- Ürün metnine attributes eklendi (renk/materyal/beden sorguları için);
  max_length 128.
- Test tahmini chunk'lı (RAM güvenliği), use_amp=True (T4 tensor core hızı).

Beklenen süre (tek T4): eğitim ~2 saat, test skorlama ~1 saat.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # DataParallel bug'ından kaçın

import glob

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

SEED = 42
VAL_TERM_FRACTION = 0.15
BASE_MODEL = "dbmdz/bert-base-turkish-128k-uncased"
MAX_LEN = 128
ATTR_CHARS = 200  # attributes'ın ilk N karakteri (tokenizer zaten 128 tokene kırpar)

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def find_input(fname: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
    if hits:
        return hits[0]
    if os.path.isfile(fname):  # yerel çalıştırma
        return fname
    local = os.path.join("trendyol-e-ticaret-yarismasi-2026-kaggle", fname)
    if os.path.isfile(local):
        return local
    raise SystemExit(
        f"{fname} bulunamadı. Yarışma verisi VE mined_hard_negatives.csv "
        "dataset'i notebook'a Input olarak ekli mi? Eklediyseniz oturumu "
        "yeniden başlatın."
    )


OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

print("[1/5] Veri yükleniyor...")
items = pd.read_csv(find_input("items.csv"),
                    usecols=["item_id", "title", "category", "brand",
                             "attributes"]).fillna("")
terms = pd.read_csv(find_input("terms.csv"))
train_pos = pd.read_csv(find_input("training_pairs.csv"))
test = pd.read_csv(find_input("submission_pairs.csv"))
mined = pd.read_csv(find_input("mined_hard_negatives.csv"))

q_of = dict(zip(terms.term_id, terms["query"].astype(str).str.lower()))
items["text"] = (items.title.str.lower() + " | "
                 + items.category.str.replace("/", " ").str.lower() + " | "
                 + items.brand.str.lower() + " | "
                 + items.attributes.str.lower().str.slice(0, ATTR_CHARS))
t_of = dict(zip(items.item_id, items.text))
all_ids = items.item_id.to_numpy()

print("[2/5] Terim bazlı ayrım + negatif kurulumu...")
terms_unique = np.asarray(train_pos.term_id.unique(), dtype=object)
perm = rng.permutation(len(terms_unique))
n_val = int(len(terms_unique) * VAL_TERM_FRACTION)
val_terms = set(terms_unique[perm[:n_val]])
is_val = train_pos.term_id.isin(val_terms)
pos_tr, pos_va = train_pos[~is_val], train_pos[is_val]


def random_negatives(pairs):
    term_pos = pairs.groupby("term_id")["item_id"].agg(set).to_dict()
    neg_t, neg_i = [], []
    for t in pairs.term_id:
        c = all_ids[rng.integers(len(all_ids))]
        while c in term_pos[t]:
            c = all_ids[rng.integers(len(all_ids))]
        neg_t.append(t)
        neg_i.append(c)
    return pd.DataFrame({"term_id": neg_t, "item_id": neg_i, "label": 0})


def build_split(pos_df):
    m = mined[mined.term_id.isin(set(pos_df.term_id))]
    df = pd.concat(
        [pos_df[["term_id", "item_id", "label"]], random_negatives(pos_df), m],
        ignore_index=True,
    )
    return df.sample(frac=1, random_state=SEED)


df_tr, df_va = build_split(pos_tr), build_split(pos_va)
print(f"   train: {len(df_tr)} (%{df_tr.label.mean()*100:.0f} pozitif) | "
      f"val: {len(df_va)}")

print("[3/5] Cross-encoder eğitiliyor...")
model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=MAX_LEN)
train_examples = [
    InputExample(texts=[q_of[t], t_of[i]], label=float(y))
    for t, i, y in zip(df_tr.term_id, df_tr.item_id, df_tr.label)
]
loader = DataLoader(train_examples, shuffle=True, batch_size=64, drop_last=True)
model.fit(train_dataloader=loader, epochs=2, warmup_steps=1000,
          use_amp=True, show_progress_bar=True)
model.save(f"{OUT}/ce4_model")


def predict_chunked(pairs_df, chunk=250_000):
    scores = np.empty(len(pairs_df), dtype=np.float16)
    for s in range(0, len(pairs_df), chunk):
        part = pairs_df.iloc[s:s + chunk]
        batch_pairs = [[q_of[t], t_of[i]]
                       for t, i in zip(part.term_id, part.item_id)]
        scores[s:s + chunk] = model.predict(
            batch_pairs, batch_size=512, show_progress_bar=True,
        ).astype(np.float16)
        print(f"   {min(s + chunk, len(pairs_df))}/{len(pairs_df)}")
    return scores


print("[4/5] Validasyon skorlanıyor...")
np.save(f"{OUT}/ce4_val_scores.npy", predict_chunked(df_va))
np.save(f"{OUT}/ce4_val_labels.npy", df_va.label.to_numpy())
df_va[["term_id", "item_id", "label"]].to_csv(f"{OUT}/ce4_val_pairs.csv",
                                              index=False)

print("[5/5] Test skorlanıyor (3.36M çift)...")
np.save(f"{OUT}/ce4_test_scores.npy", predict_chunked(test))
print("Bitti. İndirilecekler: ce4_test_scores.npy, ce4_val_scores.npy, "
      "ce4_val_labels.npy, ce4_val_pairs.csv")
