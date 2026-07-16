"""
LLM hakem — KAGGLE NOTEBOOK'TA (GPU: T4 x2 SEÇME, tek T4 veya L4; vLLM
compute-capability >= 7.5 ister, P100 ÇALIŞMAZ; T4 tek GPU yeterli).

Kurallara uygunluk: Qwen2.5-7B-Instruct-AWQ AÇIK AĞIRLIKLI modeldir ve
Kaggle VM'inde self-host edilir (SSS: 'open-source modeller için kısıtımız
bulunmuyor', 'self-host uygun kullanım senaryosudur'). Ücretli API yok.

Yöntem: kararsız bant çiftleri (uncertain_band.csv, Kaggle dataset'i olarak
yüklenir) için tek-token sınıflandırma: prompt sonrası 'Evet'/'Hayır' token
olasılıklarından P(alakalı) türetilir. Üretim yok → hızlı (prefill-ağırlıklı).

Çıktı: llm_scores.npy (bant satırlarıyla hizalı P(alakalı)) + llm_rows.npy
"""

import glob
import os

import numpy as np
import pandas as pd

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

hits = glob.glob("/kaggle/input/**/uncertain_band.csv", recursive=True)
if not hits:
    raise SystemExit("uncertain_band.csv dataset'i Input'a ekli değil!")
df = pd.read_csv(hits[0])
OUT = "/kaggle/working"
print(f"{len(df)} çift skorlanacak")

from vllm import LLM, SamplingParams

llm = LLM(model=MODEL, max_model_len=512, gpu_memory_utilization=0.90,
          enforce_eager=False, dtype="half")
tok = llm.get_tokenizer()
yes_id = tok.encode("Evet", add_special_tokens=False)[0]
no_id = tok.encode("Hayır", add_special_tokens=False)[0]

SYS = ("Sen bir e-ticaret arama alaka uzmanısın. Sorgu ile ürünün alakalı "
       "olup olmadığına karar ver. Ürün, sorgudaki niyeti karşılıyorsa "
       "(doğru tür, doğru model/beden/renk/marka kısıtları) alakalıdır. "
       "Sadece 'Evet' veya 'Hayır' yaz.")

def make_prompt(q, t):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Sorgu: {q}\nÜrün: {t}\nAlakalı mı?"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

prompts = [make_prompt(q, t) for q, t in zip(df["query"], df["item_text"])]
sp = SamplingParams(max_tokens=1, logprobs=20, temperature=0.0)

scores = np.full(len(df), 0.5, dtype=np.float32)
CHUNK = 20_000
for s in range(0, len(prompts), CHUNK):
    outs = llm.generate(prompts[s:s + CHUNK], sp)
    for j, o in enumerate(outs):
        lp = o.outputs[0].logprobs[0]  # ilk üretilen tokenın logprob tablosu
        py = np.exp(lp[yes_id].logprob) if yes_id in lp else 0.0
        pn = np.exp(lp[no_id].logprob) if no_id in lp else 0.0
        if py + pn > 0:
            scores[s + j] = py / (py + pn)
    print(f"{min(s + CHUNK, len(prompts))}/{len(prompts)}")

np.save(f"{OUT}/llm_scores.npy", scores)
np.save(f"{OUT}/llm_rows.npy", df.row_idx.to_numpy())
print("Bitti. İndirilecekler: llm_scores.npy, llm_rows.npy")
print("skor dağılımı: <0.3:", (scores < 0.3).mean().round(3),
      "| >0.7:", (scores > 0.7).mean().round(3))
