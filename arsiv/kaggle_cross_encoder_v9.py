"""
Cross-encoder v9 — masaüstü (RTX 4070), GECE İŞİ: XLM-RoBERTa-large (550M).
Amaç KOPYA DEĞİL FARKLI SES: mevcut CE'ler birbirine r=0.91-0.96 korelasyonlu
(aynı aile); large + farklı mimari, karışıma düşük korelasyonlu güç katmalı.
Ayarlar 12GB VRAM için: batch 16, max_len 128, lr 1e-5 (large modeller yüksek
lr'de patlar), 1 epoch, tüm veri. Süre: ~6-7 saat.
İZLEME: ilk 30 dk loss'a bak — NaN/patlama görürsen durdur, haber ver.

ce4 reçetesinin (dbmdz, 2 epoch, mined+random negatif) TÜM VERİYLE eğitimi:
val ayrımı YOK — eşik/ağırlık kararları artık LB'den geldiği için val'in
tuttuğu %15 terim eğitime geri katılıyor. Bu modelin val skoru ÜRETİLMEZ
(kendi eğitim verisi olurdu, anlamsız); yalnız test skorlanır. Karışımdaki
rolü LB'de ce4'ün yerine geçerek test edilir.
Masaüstü (RTX 4070) için: epochs=2, süre ~2.5-3 saat (kota derdi yok).

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
BASE_MODEL = "xlm-roberta-large"
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
pos_tr = train_pos  # endgame: tüm terimler eğitimde, val yok


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


df_tr = build_split(pos_tr)
print(f"   train: {len(df_tr)} (%{df_tr.label.mean()*100:.0f} pozitif) | val: YOK (endgame)")

print("[3/5] Cross-encoder eğitiliyor...")
model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=MAX_LEN)
train_examples = [
    InputExample(texts=[q_of[t], t_of[i]], label=float(y))
    for t, i, y in zip(df_tr.term_id, df_tr.item_id, df_tr.label)
]
loader = DataLoader(train_examples, shuffle=True, batch_size=16, drop_last=True)
model.fit(train_dataloader=loader, epochs=1, warmup_steps=1000,
          optimizer_params={"lr": 1e-5}, use_amp=True, show_progress_bar=True)
model.save(f"{OUT}/ce9_model")


def predict_chunked(pairs_df, chunk=250_000):
    scores = np.empty(len(pairs_df), dtype=np.float16)
    for s in range(0, len(pairs_df), chunk):
        part = pairs_df.iloc[s:s + chunk]
        batch_pairs = [[q_of[t], t_of[i]]
                       for t, i in zip(part.term_id, part.item_id)]
        scores[s:s + chunk] = model.predict(
            batch_pairs, batch_size=256, show_progress_bar=True,  # 512: 66 çift/s, 256: 565 çift/s (4070 ölçümü)
        ).astype(np.float16)
        print(f"   {min(s + chunk, len(pairs_df))}/{len(pairs_df)}")
    return scores


print("[4/4] Test skorlanıyor (3.36M çift)...")
np.save(f"{OUT}/ce9_test_scores.npy", predict_chunked(test))
print("Bitti. İndirilecek TEK dosya: ce9_test_scores.npy")
