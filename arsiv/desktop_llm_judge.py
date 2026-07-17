"""
LLM hakem — MASAÜSTÜ (RTX 4070, Windows) sürümü. Genişletilmiş bant (216k).

Colab'da vLLM CUDA-13/12 çatışması verdi; bu sürüm önyüklü transformers ile
çalışır. Tek-token sınıflandırma: batch'e tek forward, son pozisyonun
logit'lerinden P(Evet | Evet∪Hayır). Chunk başına diske kayıt (kesinti-güvenli).

Kullanım: T4 runtime + uncertain_core.csv sol panelde → bu hücreyi çalıştır.
İlk '5000/...' satırının süresi hız testidir: ≤2 dk ise toplam ~1 saat.
"""

import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
CSV = "uncertain_ext.csv"
BATCH = 128  # 4070 12GB, fp16 3B rahat taşır
SAVE_EVERY = 5_000

df = pd.read_csv(CSV)
print(f"{len(df)} çift skorlanacak")

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda")
model.eval()

# 'Evet'/'Hayır' hem başta hem boşluk-önekli formlarıyla (BPE varyantları)
def ids_of(word):
    out = []
    for w in (word, " " + word):
        t = tok.encode(w, add_special_tokens=False)
        if len(t) >= 1:
            out.append(t[0])
    return list(set(out))

yes_ids, no_ids = ids_of("Evet"), ids_of("Hayır")
print("yes token ids:", yes_ids, "| no token ids:", no_ids)

def make_prompt(q, t):
    msgs = [{"role": "user", "content":
             f"Arama: {q}\nÜrün: {t}\nBu ürün bu aramayla alakalı mı? Sadece Evet veya Hayır."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

prompts = [make_prompt(q, t) for q, t in zip(df["query"], df["item_text"])]

if os.path.exists("llm_ext_scores.npy"):
    scores = np.load("llm_ext_scores.npy")
    done = int(np.load("llm_ext_done.npy"))
    print(f"devam: {done} hazır")
else:
    scores = np.full(len(df), 0.5, dtype=np.float32)
    done = 0

with torch.inference_mode():
    for s in range(done, len(prompts), BATCH):
        batch = prompts[s:s + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to("cuda")
        logits = model(**enc).logits[:, -1, :]          # son pozisyon
        py = torch.softmax(logits, dim=-1)
        p_yes = py[:, yes_ids].sum(dim=1)
        p_no = py[:, no_ids].sum(dim=1)
        scores[s:s + len(batch)] = (p_yes / (p_yes + p_no + 1e-9)).float().cpu().numpy()
        if (s - done) % SAVE_EVERY < BATCH:
            np.save("llm_ext_scores.npy", scores)
            np.save("llm_ext_done.npy", np.array(s + len(batch)))
            np.save("llm_ext_rows.npy", df.row_idx.to_numpy())
            print(f"{s + len(batch)}/{len(prompts)} — kaydedildi")

np.save("llm_ext_scores.npy", scores)
np.save("llm_ext_done.npy", np.array(len(prompts)))
np.save("llm_ext_rows.npy", df.row_idx.to_numpy())
print("Bitti. İndir: llm_ext_scores.npy, llm_ext_rows.npy")
print("Evet oranı (>0.5):", round(float((scores > 0.5).mean()), 3))
