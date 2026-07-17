"""KURTARMA — eğitim bitti (5000/5000, loss 0.187), skorlama OOM'da çöktü.
Eğitimi TEKRARLAMAZ: judge_ckpt/checkpoint-5000 LoRA'sını yükler, gate + band skorlar.

OOM kök sebebi (train_judge.py satır 123):
    torch.softmax(model(**enc).logits[:, -1, :], dim=-1)
`.logits` TÜM pozisyonlar için vocab dağılımı üretir:
    128 batch x 256 token x 151k vocab x 4 byte = 12.3 GB
`[:, -1, :]` dilimi o dev tensör OLUŞTUKTAN SONRA çalışır -> çok geç.

Düzeltme: logits_to_keep=1 -> model yalnız SON pozisyonu hesaplar (128x1x151k = 77 MB).
Ek olarak: batch 128->32, fp16 softmax yerine float32'ye son adımda çevirme.
Mantık train_judge.py ile BİREBİR aynı (prompt, yes/no id, eşik taraması, gate).
"""
import gc
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-3B-Instruct"
CKPT = "judge_ckpt/checkpoint-5000"
MAXLEN = 160          # train_judge.py ile aynı
B = 32                # 128 -> 32 (OOM emniyeti)
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"

assert os.path.isdir(CKPT), f"{CKPT} yok!"
print(f">>> LoRA: {CKPT}  (eğitim tekrarlanmıyor)")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"   # inference: son pozisyon gerçek token olsun

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda")
model = PeftModel.from_pretrained(base, CKPT)
model = model.merge_and_unload()
model.eval()
gc.collect(); torch.cuda.empty_cache()
print(f"model hazır | VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB")

yes_ids = list({tok.encode(w, add_special_tokens=False)[0] for w in ("Evet", " Evet")})
no_ids = list({tok.encode(w, add_special_tokens=False)[0] for w in ("Hayır", " Hayır")})
print("yes:", yes_ids, "| no:", no_ids)


def prompt_of(q, t):
    m = [{"role": "user", "content":
          f"Arama: {q}\nÜrün: {t}\nBu ürün bu aramayla alakalı mı? Sadece Evet veya Hayır."}]
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def score(qs, ts, tag, fs, fr, rows):
    prompts = [prompt_of(q, t) for q, t in zip(qs, ts)]
    out = np.full(len(prompts), 0.5, dtype=np.float32)
    done = 0
    if os.path.exists(fs):                      # kesinti-güvenli devam
        prev = np.load(fs)
        if len(prev) == len(out):
            out = prev
            nz = np.nonzero(out != 0.5)[0]
            done = (int(nz[-1]) + 1) // B * B if len(nz) else 0
            if done:
                print(f"  {tag}: devam {done}/{len(prompts)}")
    for s in range(done, len(prompts), B):
        enc = tok(prompts[s:s + B], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to("cuda")
        # logits_to_keep=1 -> yalnız son pozisyon (12.3 GB yerine 77 MB)
        lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        pr = torch.softmax(lg, dim=-1)
        py = pr[:, yes_ids].sum(1)
        pn = pr[:, no_ids].sum(1)
        out[s:s + enc["input_ids"].shape[0]] = (py / (py + pn + 1e-9)).cpu().numpy()
        if s % (B * 50) == 0:
            np.save(fs, out); np.save(fr, rows)
            print(f"  {tag}: {min(s + B, len(prompts))}/{len(prompts)}", flush=True)
    np.save(fs, out); np.save(fr, rows)
    return out


# ---- GATE (held-out, train_judge.py ile birebir) ----
val = pd.read_csv("judge_val.csv")
lab = val["label"].to_numpy()
vs = score(val["query"], val["item_text"], "val",
           "llm_ft_val_scores.npy", "llm_ft_val_labels.npy", lab)
np.save("llm_ft_val_labels.npy", lab)

thr = max(np.arange(0.3, 0.71, 0.05),
          key=lambda t: ((vs >= t).astype(int) == lab).mean())
print(f"\n=== GATE (eşik {thr:.2f}): genel {((vs>=thr).astype(int)==lab).mean():.3f} ===")
for s in ("pos", "sameleaf", "mined"):
    m = val["src"].to_numpy() == s
    if m.sum():
        print(f"    {s:9s}: {((vs[m]>=thr).astype(int)==lab[m]).mean():.3f}")
print("    >>> sameleaf >=0.72 GO | <0.68 DUR (0.893 cepte)")

# eşikten bağımsız sıralama gücü (zero-shot: 3B 0.721 / 1.5B 0.628)
order = np.argsort(vs); r = np.empty(len(vs)); r[order] = np.arange(1, len(vs) + 1)
u, inv, c = np.unique(vs, return_inverse=True, return_counts=True)
r = (np.bincount(inv, weights=r) / c)[inv]
n1, n0 = lab.sum(), (1 - lab).sum()
print(f"    AUC: {(r[lab==1].sum() - n1*(n1+1)/2)/(n1*n0):.4f}   [zero-shot 3B: 0.721]\n")

# ---- BAND ----
try:
    band = pd.read_csv("uncertain_band.csv")
    print(f"band skorlama: {len(band)} çift")
    score(band["query"], band["item_text"], "band",
          "llm_ft_band_scores.npy", "llm_ft_band_rows.npy", band["row_idx"].to_numpy())
    print("Bitti. push: llm_ft_band_scores.npy llm_ft_band_rows.npy "
          "llm_ft_val_scores.npy llm_ft_val_labels.npy")
except Exception as e:
    print("band hata (eğitim+gate sağlam):", type(e).__name__, e)
