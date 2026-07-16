"""
LLM hakem v2 — GOOGLE COLAB (ücretsiz T4) için. Kaggle kota faciasının dersleriyle:
- Dar bant: p [0.40, 0.60] çekirdeği, 170.574 çift (387k değil)
- Küçük model: Qwen2.5-3B-Instruct-AWQ (7B'nin ~3 katı hız, bant işi için yeterli)
- Kısa prompt: ~100 token (uzun sistem mesajı + tam attributes yok)
- HER CHUNK'TA KAYIT: kesinti artık veri kaybettirmez; kaldığı yerden devam eder

Kullanım (Colab):
  1. Runtime -> Change runtime type -> T4 GPU
  2. Sol panel Files -> uncertain_core.csv'yi sürükle
  3. Hücre 1: !pip -q install vllm
  4. Hücre 2: bu dosyanın içeriği
  5. Bitince llm_core_scores.npy + llm_core_rows.npy dosyalarını indir
Beklenen süre: ~45-75 dk (ölçülmüş varsayım değil — İLK CHUNK'IN süresine bak:
>6 dk ise bana bildir, model küçültülür.)
"""

import os

import numpy as np
import pandas as pd

MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"
CSV = "uncertain_core.csv"
CHUNK = 10_000

df = pd.read_csv(CSV)
print(f"{len(df)} çift skorlanacak")

from vllm import LLM, SamplingParams

llm = LLM(model=MODEL, max_model_len=256, gpu_memory_utilization=0.90, dtype="half")
tok = llm.get_tokenizer()
yes_id = tok.encode("Evet", add_special_tokens=False)[0]
no_id = tok.encode("Hayır", add_special_tokens=False)[0]

def make_prompt(q, t):
    msgs = [{"role": "user", "content":
             f"Arama: {q}\nÜrün: {t}\nBu ürün bu aramayla alakalı mı? Sadece Evet veya Hayır."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

prompts = [make_prompt(q, t) for q, t in zip(df["query"], df["item_text"])]
sp = SamplingParams(max_tokens=1, logprobs=20, temperature=0.0)

# kaldığı yerden devam
if os.path.exists("llm_core_scores.npy"):
    scores = np.load("llm_core_scores.npy")
    done = int(np.load("llm_core_done.npy"))
    print(f"devam: {done} çift hazır")
else:
    scores = np.full(len(df), 0.5, dtype=np.float32)
    done = 0

for s in range(done, len(prompts), CHUNK):
    outs = llm.generate(prompts[s:s + CHUNK], sp)
    for j, o in enumerate(outs):
        lp = o.outputs[0].logprobs[0]
        py = np.exp(lp[yes_id].logprob) if yes_id in lp else 0.0
        pn = np.exp(lp[no_id].logprob) if no_id in lp else 0.0
        if py + pn > 0:
            scores[s + j] = py / (py + pn)
    np.save("llm_core_scores.npy", scores)           # HER chunk'ta kayıt
    np.save("llm_core_done.npy", np.array(min(s + CHUNK, len(prompts))))
    np.save("llm_core_rows.npy", df.row_idx.to_numpy())
    print(f"{min(s + CHUNK, len(prompts))}/{len(prompts)} — kaydedildi")

print("Bitti. İndir: llm_core_scores.npy, llm_core_rows.npy")
print("Evet oranı (>0.5):", float((scores > 0.5).mean()).__round__(3))
