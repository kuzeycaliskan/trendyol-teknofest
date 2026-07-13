"""
Bi-encoder fine-tune: multilingual-e5-small, MultipleNegativesRankingLoss.

Yöntem: 250k gerçek pozitif (sorgu, ürün) çiftiyle contrastive öğrenme.
Her batch'te bir sorgunun pozitif ürünü, batch'teki DİĞER sorguların
ürünlerinden daha yakın olmaya itilir (in-batch negatives). Sentetik negatif
üretimi gerektirmez — v4'teki dağılım kayması sınıfı burada yoktur.

Sızıntı önlemi: LightGBM pipeline'larıyla AYNI seed ve AYNI prosedürle
türetilen validasyon terimlerinin çiftleri fine-tune verisinden çıkarılır;
aksi halde downstream validasyon iyimser ölçerdi.
"""

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

SEED = 42
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
VAL_TERM_FRACTION = 0.15  # lgbm_pipeline ile birebir aynı
import sys
BASE = sys.argv[1] if len(sys.argv) > 1 else "intfloat/multilingual-e5-small"
OUT = sys.argv[2] if len(sys.argv) > 2 else "e5-small-ft"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 1

train_pos = pd.read_csv(f"{DATA}/training_pairs.csv")

# lgbm_pipeline'daki ayrımın birebir kopyası (aynı seed, aynı sıra)
terms_unique = np.asarray(train_pos.term_id.unique(), dtype=object)
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(terms_unique))
n_val = int(len(terms_unique) * VAL_TERM_FRACTION)
val_terms = set(terms_unique[perm[:n_val]])
df = train_pos[~train_pos.term_id.isin(val_terms)]
df = df.drop_duplicates(subset=["term_id", "item_id"])
print(f"fine-tune çifti: {len(df)} (val terimleri hariç)")

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

examples = [
    InputExample(texts=[f"query: {q_of[t]}", f"passage: {t_of[i]}"])
    for t, i in zip(df.term_id, df.item_id)
]

torch.manual_seed(SEED)  # DataLoader shuffle + ağırlık güncellemeleri tekrarlanabilir olsun
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(BASE, device=device)
model.max_seq_length = 128  # başlıklar kısa; hız için yeterli

loader = DataLoader(examples, shuffle=True, batch_size=96, drop_last=True)
loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=EPOCHS,
    warmup_steps=500,
    output_path=OUT,
    show_progress_bar=True,
)
print(f"Bitti: {OUT}/")
