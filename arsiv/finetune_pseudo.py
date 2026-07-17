"""
Pseudo-labeling turu: bi-encoder'ı testin KENDİ sorgularına uyarlama.

Gerekçe: bi-encoder test terimlerini hiç görmedi (train/test terim örtüşmesi
%0). En iyi modelin (v7 LGBM, LB 0.813) test üzerindeki ÇOK EMİN pozitif
tahminleri (p >= 0.99) sözde-pozitif çift olarak alınır ve MNRL ile bir tur
daha eğitilir. In-batch negatives kullanıldığı için sentetik negatif
varsayımı yine yoktur; tek varsayım sözde-pozitiflerin doğruluğudur —
eşik bu yüzden çok muhafazakâr (0.99) ve terim başına üst sınırlı.

Dengeleme: terim başına en fazla 25 sözde-pozitif (skora göre en yüksekler);
orijinal train pozitifleri (val terimleri hariç, her zamanki split) karışıma
aynen dahil — model eski görevi unutmasın.

Çıktı: e5-small-pl/ (embed_texts.py ile embpl_* üretilecek, sonra TAG=v8)
"""

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

SEED = 42
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
VAL_TERM_FRACTION = 0.15
BASE = "e5-small-ft3"          # v7'nin embedding modeli
OUT = "e5-small-pl"
P_MIN = 0.99                   # sözde-pozitif güven eşiği
CAP_PER_TERM = 25

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# --- Orijinal pozitifler (val terimleri hariç; lgbm ile birebir aynı split) ---
train_pos = pd.read_csv(f"{DATA}/training_pairs.csv")
terms_unique = np.asarray(train_pos.term_id.unique(), dtype=object)
perm = rng.permutation(len(terms_unique))
n_val = int(len(terms_unique) * VAL_TERM_FRACTION)
val_terms = set(terms_unique[perm[:n_val]])
df_orig = train_pos[~train_pos.term_id.isin(val_terms)]
df_orig = df_orig.drop_duplicates(subset=["term_id", "item_id"])

# --- Sözde-pozitifler: v7'nin çok emin test tahminleri ---
test = pd.read_csv(f"{DATA}/submission_pairs.csv")
proba = np.load("test_proba_v7.npy")
conf = test[proba >= P_MIN].copy()
conf["proba"] = proba[proba >= P_MIN]
conf = (conf.sort_values("proba", ascending=False)
        .groupby("term_id").head(CAP_PER_TERM))
print(f"orijinal pozitif: {len(df_orig)} | sözde-pozitif: {len(conf)} "
      f"({conf.term_id.nunique()} test terimi)")

terms = pd.read_csv(f"{DATA}/terms.csv")
items = pd.read_csv(f"{DATA}/items.csv",
                    usecols=["item_id", "title", "category", "brand"]).fillna("")
q_of = dict(zip(terms.term_id, terms["query"].astype(str).str.lower()))
t_of = dict(zip(
    items.item_id,
    (items.title.str.lower() + " | "
     + items.category.str.replace("/", " ").str.lower() + " | "
     + items.brand.str.lower()),
))

pairs = pd.concat(
    [df_orig[["term_id", "item_id"]], conf[["term_id", "item_id"]]],
    ignore_index=True,
)
examples = [
    InputExample(texts=[f"query: {q_of[t]}", f"passage: {t_of[i]}"])
    for t, i in zip(pairs.term_id, pairs.item_id)
]

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(BASE, device=device)
model.max_seq_length = 128

loader = DataLoader(examples, shuffle=True, batch_size=96, drop_last=True)
loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(loader, loss)], epochs=1, warmup_steps=500,
          output_path=OUT, show_progress_bar=True)
print(f"Bitti: {OUT}/")
