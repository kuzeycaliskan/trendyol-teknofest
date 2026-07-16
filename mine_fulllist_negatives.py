"""
Tam-liste (retrieval-dağılımlı) negatifler — sektör-standardı reçete denemesi.

Her train terimi için fine-tuned bi-encoder ile katalogdan top-N aday çekilir
(testin aday listelerinin üretimini taklit eder); terimin pozitifleri hariç
TÜMÜ negatif etiketlenir. Önceki stratejilerin aksine filtre YOK (leaf
koruması yok, tepe atlama yok): amaç belirli bir zorluk dilimini değil,
testin negatif DAĞILIMINI olduğu gibi kapsamak. Bilinen risk: etiketsiz ama
gerçekte alakalı adaylar (false negative) gürültü olarak girer; hacmin
(terim başına ~65 neg) bu gürültüyü bastırması beklenir — bu, denemenin
test ettiği hipotezin ta kendisidir.

Çıktı: fulllist_negatives.csv (term_id, item_id, label=0)
"""

import numpy as np
import pandas as pd
import torch

DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
EMB = "embpl"
TOP_N = 80          # terim başına aday
CAP_NEG = 65        # pozitifler düşüldükten sonra tutulacak negatif sayısı

train_pos = pd.read_csv(f"{DATA}/training_pairs.csv")
term_pos = train_pos.groupby("term_id")["item_id"].agg(set).to_dict()
terms = list(term_pos.keys())

E_q = np.load(f"{EMB}_terms.npy").astype(np.float32)
E_i = np.load(f"{EMB}_items.npy").astype(np.float32)
q_row = {t: k for k, t in enumerate(np.load(f"{EMB}_terms_ids.npy", allow_pickle=True))}
i_ids = np.load(f"{EMB}_items_ids.npy", allow_pickle=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
I_t = torch.from_numpy(E_i).to(device)
out_t, out_i = [], []

CHUNK = 256
for s in range(0, len(terms), CHUNK):
    batch = terms[s:s + CHUNK]
    Q = torch.from_numpy(E_q[[q_row[t] for t in batch]]).to(device)
    top = torch.topk(Q @ I_t.T, TOP_N, dim=1).indices.cpu().numpy()
    for r, t in enumerate(batch):
        pos = term_pos[t]
        negs = [i_ids[j] for j in top[r] if i_ids[j] not in pos][:CAP_NEG]
        out_t.extend([t] * len(negs))
        out_i.extend(negs)
    if (s // CHUNK) % 10 == 0:
        print(f"{s + len(batch)}/{len(terms)}")

out = pd.DataFrame({"term_id": out_t, "item_id": out_i, "label": 0})
out.to_csv("fulllist_negatives.csv", index=False)
print(f"Yazıldı: fulllist_negatives.csv — {len(out)} negatif, {out.term_id.nunique()} terim "
      f"(terim başına ort. {len(out)/out.term_id.nunique():.1f})")
