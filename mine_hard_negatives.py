"""
Zor negatif madenciliği: fine-tuned bi-encoder ile her eğitim terimi için
embedding uzayında en benzer ürünleri bul; pozitif olmayanları "zor negatif"
olarak dışa aktar. Cross-encoder v2 eğitiminde kullanılacak.

False-negative koruması (iki katman — denetim bulgusu R1 gereği):
1. Leaf-kategori dışlaması: terimin pozitiflerinin leaf kategorilerindeki
   adaylar havuza ALINMAZ. Geniş sorgularda ("elbise") 11-100. sıradaki
   aynı-leaf ürünler büyük olasılıkla gerçekte alakalıdır; onları negatif
   etiketlemek CE'yi sistematik yanlışa iter (v4 dersinin CE karşılığı).
   Bedeli: aynı-leaf zor negatifler (iphone 11 vs 13 kılıfı) kaybedilir;
   bu bölge etiketsiz veride güvenle ayrıştırılamıyor, bilinçli feragat.
2. Tepe tamponu: kalan adayların ilk 5'i yine atlanır (kategori alanı
   gürültülü olabilir).
"""

import numpy as np
import pandas as pd
import torch

SEED = 42
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
EMB = "embft2"          # v6 embeddingleri (LB 0.809'un tabanı)
SKIP_TOP = 5            # tepe tamponu (leaf dışlamasına ek güvenlik)
POOL_TOP = 150          # aday havuzu (leaf dışlaması havuzu daralttığı için geniş)
NEG_PER_POS = 2
OUT = "mined_hard_negatives.csv"

rng = np.random.default_rng(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"

train_pos = pd.read_csv(f"{DATA}/training_pairs.csv")
items_cat = pd.read_csv(f"{DATA}/items.csv", usecols=["item_id", "category"]).fillna("")
cat_of = dict(zip(items_cat.item_id, items_cat.category))
E_q = np.load(f"{EMB}_terms.npy")
E_i = np.load(f"{EMB}_items.npy")
q_ids = np.load(f"{EMB}_terms_ids.npy", allow_pickle=True)
i_ids = np.load(f"{EMB}_items_ids.npy", allow_pickle=True)
q_row = {t: k for k, t in enumerate(q_ids)}

term_pos = train_pos.groupby("term_id")["item_id"].agg(set).to_dict()
terms = list(term_pos.keys())
pos_count = train_pos.term_id.value_counts().to_dict()

I_t = torch.from_numpy(E_i.astype(np.float32)).to(device)  # 962k x 384
rows_out_t, rows_out_i = [], []

CHUNK = 512
for s in range(0, len(terms), CHUNK):
    batch = terms[s:s + CHUNK]
    Q_t = torch.from_numpy(
        E_q[[q_row[t] for t in batch]].astype(np.float32)
    ).to(device)
    sims = Q_t @ I_t.T                                   # chunk x 962k
    top = torch.topk(sims, POOL_TOP, dim=1).indices.cpu().numpy()
    for r, t in enumerate(batch):
        pos = term_pos[t]
        pos_cats = {cat_of[i] for i in pos if i in cat_of}
        pool = [
            i_ids[j] for j in top[r, SKIP_TOP:]
            if i_ids[j] not in pos and cat_of.get(i_ids[j]) not in pos_cats
        ]
        n_need = min(NEG_PER_POS * pos_count[t], len(pool))
        picked = rng.choice(len(pool), size=n_need, replace=False)
        for k in picked:
            rows_out_t.append(t)
            rows_out_i.append(pool[k])
    if (s // CHUNK) % 5 == 0:
        print(f"{s + len(batch)}/{len(terms)} terim")

out = pd.DataFrame({"term_id": rows_out_t, "item_id": rows_out_i, "label": 0})
out.to_csv(OUT, index=False)
print(f"Yazıldı: {OUT} ({len(out)} zor negatif, "
      f"{out.term_id.nunique()} terim)")
