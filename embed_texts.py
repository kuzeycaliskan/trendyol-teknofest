"""
Seviye 3 - adım 1: sorgu ve ürün metinlerini çok dilli cümle embeddinglerine çevirir.

Model: intfloat/multilingual-e5-small (384 boyut). E5 ailesi asimetrik arama
için eğitilmiştir: sorgulara "query: ", belgelere "passage: " öneki eklenmesi
model kartının gereğidir. Embeddingler L2-normalize kaydedilir; kosinüs = dot.

Çıktılar:
  emb_terms.npy  (50k x 384, float16)  + emb_terms_ids.npy
  emb_items.npy  (962k x 384, float16) + emb_items_ids.npy
"""

import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
# kullanım: embed_texts.py [model_yolu] [çıktı_öneki]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "intfloat/multilingual-e5-small"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "emb"
BATCH = 512

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Cihaz: {device}")
model = SentenceTransformer(MODEL, device=device)

terms = pd.read_csv(f"{DATA}/terms.csv")
items = pd.read_csv(f"{DATA}/items.csv", usecols=["item_id", "title", "category", "brand"]).fillna("")

print(f"{len(terms)} sorgu embed ediliyor...")
q_texts = ("query: " + terms["query"].astype(str).str.lower()).tolist()
q_emb = model.encode(q_texts, batch_size=BATCH, normalize_embeddings=True,
                     show_progress_bar=True).astype(np.float16)
np.save(f"{PREFIX}_terms.npy", q_emb)
np.save(f"{PREFIX}_terms_ids.npy", terms.term_id.to_numpy())

print(f"{len(items)} ürün embed ediliyor (uzun sürer)...")
i_texts = ("passage: " + items.title.str.lower() + " | "
           + items.category.str.replace("/", " ").str.lower() + " | "
           + items.brand.str.lower()).tolist()
i_emb = model.encode(i_texts, batch_size=BATCH, normalize_embeddings=True,
                     show_progress_bar=True).astype(np.float16)
np.save(f"{PREFIX}_items.npy", i_emb)
np.save(f"{PREFIX}_items_ids.npy", items.item_id.to_numpy())
print(f"Bitti: {PREFIX}_terms.npy, {PREFIX}_items.npy")
