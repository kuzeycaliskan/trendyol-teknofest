"""
FINAL judge fine-tune — MASAÜSTÜ RTX 4070 (gözetimli, ~2.5 saat).
Generatif Evet/Hayır LoRA judge. Denetim kararı: generatif (classification-head
DEĞİL) — LLM'in dünya-bilgisini korur; kalibrasyonu %50 Evet dengesiyle eğitim
düzeltir.

Model: Qwen2.5-3B QLoRA(4bit). bitsandbytes yüklenemezse OTOMATİK Qwen2.5-1.5B
bf16'ya düşer (tek komut, fallback dahili). Duman-gate: ilk 200 adımda loss
düşmüyorsa Ctrl+C, Mac'e bildir.

Sonra: judge_val.csv (held-out, sızıntı yok) gate + uncertain_band.csv skorla.
Önkoşul: pip install -q peft transformers accelerate bitsandbytes
Çıktı: llm_ft_band_scores.npy, llm_ft_band_rows.npy, llm_ft_val_scores.npy, llm_ft_val_labels.npy
"""
import os

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

SEED = 42
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
MAXLEN = 192
CKPT = "judge_ckpt"
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ---- model seçimi: 3B QLoRA, olmazsa 1.5B bf16 ----
try:
    import bitsandbytes  # noqa
    from transformers import BitsAndBytesConfig
    MODEL, QUANT = "Qwen/Qwen2.5-3B-Instruct", True
    print(">>> 3B QLoRA (4bit)")
except Exception as e:
    MODEL, QUANT = "Qwen/Qwen2.5-1.5B-Instruct", False
    print(f">>> bitsandbytes yok ({e}); 1.5B bf16 fallback")

# ---- veri (train_judge.py ile judge_val üretimi BİREBİR aynı seed) ----
terms = pd.read_csv(f"{DATA}/terms.csv")
items = pd.read_csv(f"{DATA}/items.csv",
                    usecols=["item_id", "title", "category", "brand", "attributes"]).fillna("")
q_of = dict(zip(terms.term_id, terms["query"].astype(str)))
items["text"] = (items.title.str.slice(0, 90) + " | "
                 + items.category.str.split("/").str[-1] + " | " + items.brand
                 + " | " + items.attributes.str.slice(0, 100))
t_of = dict(zip(items.item_id, items.text))

pos = pd.read_csv(f"{DATA}/training_pairs.csv").sample(90_000, random_state=SEED)[["term_id", "item_id"]].assign(y="Evet")
mined = pd.read_csv("mined_hard_negatives.csv").sample(40_000, random_state=SEED)[["term_id", "item_id"]]
same = pd.read_csv("sameleaf_negatives.csv").sample(40_000, random_state=SEED)[["term_id", "item_id"]]
ri = items.item_id.sample(10_000, replace=True, random_state=SEED).to_numpy()
rt = terms.term_id.sample(10_000, replace=True, random_state=SEED).to_numpy()
rand = pd.DataFrame({"term_id": rt, "item_id": ri})
neg = pd.concat([mined, same, rand], ignore_index=True).assign(y="Hayır")
df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=SEED)
print(f"eğitim: {len(df)} (%{(df.y=='Evet').mean()*100:.0f} pozitif) | zor-neg: mined40k+sameleaf40k+rand10k")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

def prompt_of(q, t):
    m = [{"role": "user", "content":
          f"Arama: {q}\nÜrün: {t}\nBu ürün bu aramayla alakalı mı? Sadece Evet veya Hayır."}]
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

def encode(tid, iid, y):
    p = tok(prompt_of(q_of.get(tid, ""), t_of.get(iid, "")), truncation=True, max_length=MAXLEN - 2)["input_ids"]
    a = tok(y, add_special_tokens=False)["input_ids"][:1] + [tok.eos_token_id]
    return {"input_ids": p + a, "labels": [-100] * len(p) + a}

print("tokenizasyon...")
recs = [encode(r.term_id, r.item_id, r.y) for r in df.itertuples(index=False)]

class DS(torch.utils.data.Dataset):
    def __len__(self): return len(recs)
    def __getitem__(self, i): return recs[i]

def collate(b):
    m = max(len(x["input_ids"]) for x in b); pad = tok.pad_token_id
    return {"input_ids": torch.tensor([x["input_ids"]+[pad]*(m-len(x["input_ids"])) for x in b]),
            "labels": torch.tensor([x["labels"]+[-100]*(m-len(x["labels"])) for x in b]),
            "attention_mask": torch.tensor([[1]*len(x["input_ids"])+[0]*(m-len(x["input_ids"])) for x in b])}

print("model yükleniyor...")
if QUANT:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda")
    model = prepare_model_for_kbit_training(model)
    lr = 1e-4
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    lr = 5e-5
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(output_dir=CKPT, max_steps=5000, per_device_train_batch_size=8,  # denetim: tam cosine anneal (~3.1s)
                         gradient_accumulation_steps=2, learning_rate=lr, warmup_steps=200,
                         lr_scheduler_type="cosine", logging_steps=50, save_steps=250,
                         save_total_limit=2, bf16=True, report_to=[], seed=SEED)
trainer = Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate)
resume = os.path.isdir(CKPT) and any(d.startswith("checkpoint") for d in os.listdir(CKPT))
print(f"eğitim (resume={resume})... duman-gate: 200 adımda loss düşmezse Ctrl+C")
trainer.train(resume_from_checkpoint=resume)
model = model.merge_and_unload(); model.eval()
tok.padding_side = "left"   # inference: son pozisyon (logits[:,-1]) gerçek token olsun

yes_ids = list({tok.encode(w, add_special_tokens=False)[0] for w in ("Evet", " Evet")})
no_ids = list({tok.encode(w, add_special_tokens=False)[0] for w in ("Hayır", " Hayır")})

@torch.inference_mode()
def score(qs, ts, tag, fs, fr, rows, B=128):
    prompts = [prompt_of(q, t) for q, t in zip(qs, ts)]
    out = np.full(len(prompts), 0.5, dtype=np.float32)
    for s in range(0, len(prompts), B):
        enc = tok(prompts[s:s+B], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to("cuda")
        pr = torch.softmax(model(**enc).logits[:, -1, :], dim=-1)
        py = pr[:, yes_ids].sum(1); pn = pr[:, no_ids].sum(1)
        out[s:s+enc["input_ids"].shape[0]] = (py/(py+pn+1e-9)).float().cpu().numpy()
        if s % (B*40) == 0:
            np.save(fs, out); np.save(fr, rows); print(f"  {tag}: {min(s+B,len(prompts))}/{len(prompts)}")
    np.save(fs, out); np.save(fr, rows); return out

# GATE (held-out)
val = pd.read_csv("judge_val.csv"); lab = val["label"].to_numpy()
vs = score(val["query"], val["item_text"], "val", "llm_ft_val_scores.npy", "llm_ft_val_labels.npy", lab)
thr = max(np.arange(0.3, 0.71, 0.05), key=lambda t: ((vs >= t).astype(int) == lab).mean())
print(f"\n=== GATE (eşik {thr:.2f}): genel {((vs>=thr).astype(int)==lab).mean():.3f} ===")
for s in ("pos", "sameleaf", "mined"):
    m = val["src"].to_numpy() == s
    print(f"    {s:9s}: {((vs[m]>=thr).astype(int)==lab[m]).mean():.3f}")
print("    >>> sameleaf >=0.72 GO | <0.68 DUR (0.893 cepte)\n")

# BAND
try:
    band = pd.read_csv("uncertain_band.csv")
    score(band["query"], band["item_text"], "band", "llm_ft_band_scores.npy", "llm_ft_band_rows.npy",
          band["row_idx"].to_numpy())
    print("Bitti. push: llm_ft_band_scores.npy llm_ft_band_rows.npy llm_ft_val_scores.npy llm_ft_val_labels.npy")
except Exception as e:
    print("band hata (eğitim+gate sağlam):", e)
