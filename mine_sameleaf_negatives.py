"""
Aynı-leaf ÇELİŞKİ negatifleri (T1 körlüğüne saldırı — ce11'in girdisi).

Mantık: mevcut CE'ler leaf-koruması nedeniyle aynı-leaf negatifi hiç görmedi;
"iphone 13 kılıfı" vs "iphone 11 kılıfı" ayrımını öğrenemediler. Etiketsiz
aynı-leaf ürünü negatif saymak false-negative riskiydi; burada negatifliği
KANIT belirliyor: sorgunun spesifik token'ı (rakamlı token / beden) pozitif
ürünlerin çoğunda geçiyorsa, o token'ı TAŞIMAYAN aynı-leaf benzer ürünler
yüksek kesinlikle alakasızdır (yanlış varyant).

Kanıt sınıfları: (1) rakam/beden token çelişkisi, (2) renk çelişkisi
(sorgu rengi üründe yok + ürün başka renk beyan ediyor), (3) marka çelişkisi
(sorgu, pozitiflerin markasını içeriyor; aday farklı markalı).

Çıktı: sameleaf_negatives.csv (term_id, item_id, label=0)
"""

import re

import numpy as np
import pandas as pd
import torch

SEED = 42
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
EMB = "embpl"
MIN_POS_SHARE = 0.6   # spec token, pozitiflerin en az %60'ında geçmeli
TOP_M = 5             # terim başına en fazla negatif (kanıt sınıfı başına)
COLORS = {"siyah","beyaz","kırmızı","mavi","lacivert","yeşil","sarı","mor",
          "pembe","gri","kahverengi","bej","turuncu","bordo","haki","ekru",
          "krem","altın","gümüş","fuşya","turkuaz"}
SIZE_TOKENS = {"xs", "xxs", "s̶", "xl", "xxl", "3xl", "4xl", "5xl"}
TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+")

rng = np.random.default_rng(SEED)

print("[1/4] Veri yükleniyor...")
items = pd.read_csv(f"{DATA}/items.csv",
                    usecols=["item_id", "title", "category", "brand", "attributes"]).fillna("")
terms = pd.read_csv(f"{DATA}/terms.csv")
train_pos = pd.read_csv(f"{DATA}/training_pairs.csv")

q_of = dict(zip(terms.term_id, terms["query"].astype(str).str.lower()))
item_tokens = {}
for iid, title, attr in zip(items.item_id, items.title, items.attributes):
    item_tokens[iid] = frozenset(TOKEN_RE.findall(f"{title} {attr}".lower()))
cat_of = dict(zip(items.item_id, items.category))
brand_of = dict(zip(items.item_id, items.brand.str.lower()))

print("[2/4] Embedding ve leaf indeksleri...")
E_i = np.load(f"{EMB}_items.npy").astype(np.float32)
E_q = np.load(f"{EMB}_terms.npy").astype(np.float32)
i_row = {i: k for k, i in enumerate(np.load(f"{EMB}_items_ids.npy", allow_pickle=True))}
q_row = {t: k for k, t in enumerate(np.load(f"{EMB}_terms_ids.npy", allow_pickle=True))}
ids_arr = items.item_id.to_numpy()
leaf_items: dict[str, list] = {}
for iid, cat in zip(items.item_id, items.category):
    leaf_items.setdefault(cat, []).append(iid)

device = "mps" if torch.backends.mps.is_available() else "cpu"
E_i_t = torch.from_numpy(E_i).to(device)

def spec_tokens(query: str) -> set[str]:
    toks = TOKEN_RE.findall(query)
    return {t for t in toks
            if (any(c.isdigit() for c in t) and len(t) >= 2) or t in SIZE_TOKENS}

print("[3/4] Terim bazlı madencilik...")
term_pos = train_pos.groupby("term_id")["item_id"].agg(list).to_dict()
out_t, out_i = [], []
n_spec_terms = 0
for t, pos_items in term_pos.items():
    q = q_of[t]
    q_toks = set(TOKEN_RE.findall(q))
    pos_set = set(pos_items)
    leafs = {cat_of[i] for i in pos_items if i in cat_of}
    leaf_pool = [i for lf in leafs for i in leaf_items.get(lf, []) if i not in pos_set]
    if not leaf_pool:
        continue

    pool = []
    # (1) rakam/beden çelişkisi
    st = spec_tokens(q)
    required = {tok for tok in st
                if np.mean([tok in item_tokens.get(i, frozenset()) for i in pos_items]) >= MIN_POS_SHARE}
    if required:
        pool += [i for i in leaf_pool if required - item_tokens.get(i, frozenset())][:400]
    # (2) renk çelişkisi: sorgu rengi pozitiflerde tutuyor; aday o rengi taşımıyor AMA başka renk beyan ediyor
    q_colors = {c for c in (q_toks & COLORS)
                if np.mean([c in item_tokens.get(i, frozenset()) for i in pos_items]) >= MIN_POS_SHARE}
    if q_colors:
        pool += [i for i in leaf_pool
                 if not (q_colors & item_tokens.get(i, frozenset()))
                 and (item_tokens.get(i, frozenset()) & COLORS)][:400]
    # (3) marka çelişkisi: sorgu, pozitiflerin markasını içeriyor; aday farklı markalı
    pos_brands = {brand_of.get(i, "") for i in pos_items} - {""}
    q_brands = {b for b in pos_brands if len(b) >= 3 and b in q}
    if q_brands:
        pool += [i for i in leaf_pool
                 if brand_of.get(i, "") and brand_of.get(i, "") not in q_brands][:400]
    if not pool:
        continue
    n_spec_terms += 1
    pool = list(dict.fromkeys(pool))  # sınıflar arası tekrarları at
    # embedding benzerliğine göre en yakın (en aldatıcı) M çelişkili ürün
    pool_idx = torch.tensor([i_row[i] for i in pool], device=device)
    q_vec = torch.from_numpy(E_q[q_row[t]]).to(device)
    sims = E_i_t[pool_idx] @ q_vec
    top = torch.topk(sims, min(3 * TOP_M, len(pool))).indices.cpu().numpy()
    for j in top:
        out_t.append(t)
        out_i.append(pool[j])

out = pd.DataFrame({"term_id": out_t, "item_id": out_i, "label": 0})
out.to_csv("sameleaf_negatives.csv", index=False)
print(f"[4/4] Yazıldı: sameleaf_negatives.csv — {len(out)} negatif, "
      f"{out.term_id.nunique()} terim (spec-token'lı terim: {n_spec_terms})")
