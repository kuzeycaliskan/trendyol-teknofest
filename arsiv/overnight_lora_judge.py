"""
GECE İŞİ (tek komut, unattended-güvenli) — MASAÜSTÜ RTX 4070.
LoRA-fine-tuned LLM hakem: eğit -> bilinen etiketlerde doğrula -> bandı skorla.

Bu geceki 4 çöküşün dersleriyle sağlamlaştırıldı:
- bf16 LoRA (bitsandbytes YOK — Windows QLoRA riskini tamamen kaldırır)
- Otomatik resume (restart olursa checkpoint'ten devam)
- Tek script: train+val+score bir arada, zincir kırılması yok
- Qwen2.5-1.5B: 4070'te rahat, hızlı, güvenilir
- Band skorlama try/except içinde: orası patlasa bile eğitilmiş model + val
  isabeti elde kalır

Önkoşul (pip, saf-python, Windows'ta sorunsuz):
  pip install -q peft transformers accelerate
Çıktı: llm_ft_band_scores.npy + llm_ft_band_rows.npy (harman için)
       ve konsola: HAKEM İSABETİ (go/no-go)
"""

import os

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

SEED = 42
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAXLEN = 160          # attributes'lı item_text + sameleaf çelişkileri için
CKPT = "lora_judge_ckpt"
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------- veri
terms = pd.read_csv(f"{DATA}/terms.csv")
items = pd.read_csv(f"{DATA}/items.csv",
                    usecols=["item_id", "title", "category", "brand", "attributes"]).fillna("")
q_of = dict(zip(terms.term_id, terms["query"].astype(str)))
# item_text CSV'lerle BİREBİR aynı format (attributes dahil — kritik tutarlılık)
items["text"] = (items.title.str.slice(0, 90) + " | "
                 + items.category.str.split("/").str[-1] + " | " + items.brand
                 + " | " + items.attributes.str.slice(0, 60))
t_of = dict(zip(items.item_id, items.text))

# 50/50 denge (90k poz / 90k neg) — sıralama kalitesi için
pos = pd.read_csv(f"{DATA}/training_pairs.csv").sample(90_000, random_state=SEED)
pos = pos[["term_id", "item_id"]].assign(y="Evet")
mined = pd.read_csv("mined_hard_negatives.csv").sample(45_000, random_state=SEED)[["term_id", "item_id"]]
same = pd.read_csv("sameleaf_negatives.csv")
same = same.sample(min(30_000, len(same)), random_state=SEED)[["term_id", "item_id"]]
ri = items.item_id.sample(15_000, replace=True, random_state=SEED).to_numpy()
rt = terms.term_id.sample(15_000, replace=True, random_state=SEED).to_numpy()
rand = pd.DataFrame({"term_id": rt, "item_id": ri})
neg = pd.concat([mined, same, rand], ignore_index=True).assign(y="Hayır")
df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=SEED)
print(f"eğitim: {len(df)} örnek (%{(df.y=='Evet').mean()*100:.0f} pozitif)")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

def prompt_of(tid, iid):
    msgs = [{"role": "user", "content":
             f"Arama: {q_of.get(tid,'')}\nÜrün: {t_of.get(iid,'')}\n"
             "Bu ürün bu aramayla alakalı mı? Sadece Evet veya Hayır."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def encode(tid, iid, y):
    p = tok(prompt_of(tid, iid), truncation=True, max_length=MAXLEN - 2)["input_ids"]
    a = tok(y, add_special_tokens=False)["input_ids"][:1] + [tok.eos_token_id]
    return {"input_ids": p + a, "labels": [-100] * len(p) + a}

print("tokenizasyon...")
recs = [encode(r.term_id, r.item_id, r.y) for r in df.itertuples(index=False)]

class DS(torch.utils.data.Dataset):
    def __len__(self): return len(recs)
    def __getitem__(self, i): return recs[i]

def collate(b):
    m = max(len(x["input_ids"]) for x in b)
    pad = tok.pad_token_id
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad]*(m-len(x["input_ids"])) for x in b]),
        "labels": torch.tensor([x["labels"] + [-100]*(m-len(x["labels"])) for x in b]),
        "attention_mask": torch.tensor([[1]*len(x["input_ids"]) + [0]*(m-len(x["input_ids"])) for x in b]),
    }

# ---------------------------------------------------------------- model + LoRA (bf16, bitsandbytes YOK)
print("model yükleniyor (bf16)...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda")
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=CKPT, num_train_epochs=1, per_device_train_batch_size=8,
    gradient_accumulation_steps=2, learning_rate=1e-4, warmup_steps=200,
    lr_scheduler_type="cosine", logging_steps=50, save_steps=500,
    save_total_limit=2, bf16=True, report_to=[], seed=SEED)
trainer = Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate)

resume = os.path.isdir(CKPT) and any(d.startswith("checkpoint") for d in os.listdir(CKPT))
print(f"eğitim başlıyor (resume={resume})...")
trainer.train(resume_from_checkpoint=resume)
model = model.merge_and_unload()
model.eval()
# 🔴 KRİTİK: inference'te SOL padding — score() son pozisyonu (logits[:,-1]) okuyor;
# sağ-padding'de o pozisyon PAD olur ve skorlar çöp çıkar. Eğitim manuel collate
# ile sağ-pad kullandı (doğru), yalnız çıkarım sol-pad olmalı.
tok.padding_side = "left"

# ---------------------------------------------------------------- skorlama yardımcısı
yes_ids = list({tok.encode(w, add_special_tokens=False)[0] for w in ("Evet", " Evet")})
no_ids = list({tok.encode(w, add_special_tokens=False)[0] for w in ("Hayır", " Hayır")})

@torch.inference_mode()
def score(queries, texts, tag, save_scores, save_rows, rows, BATCH=128):
    prompts = [prompt_of(None, None) for _ in range(0)]  # placeholder
    prompts = []
    for q, t in zip(queries, texts):
        msgs = [{"role": "user", "content":
                 f"Arama: {q}\nÜrün: {t}\nBu ürün bu aramayla alakalı mı? Sadece Evet veya Hayır."}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    out = np.full(len(prompts), 0.5, dtype=np.float32)
    for s in range(0, len(prompts), BATCH):
        enc = tok(prompts[s:s+BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to("cuda")
        pr = torch.softmax(model(**enc).logits[:, -1, :], dim=-1)
        py = pr[:, yes_ids].sum(1); pn = pr[:, no_ids].sum(1)
        out[s:s+enc["input_ids"].shape[0]] = (py/(py+pn+1e-9)).float().cpu().numpy()
        if s % (BATCH*40) == 0:
            np.save(save_scores, out); np.save(save_rows, rows)
            print(f"  {tag}: {min(s+BATCH,len(prompts))}/{len(prompts)}")
    np.save(save_scores, out); np.save(save_rows, rows)
    return out

# ---------------------------------------------------------------- 1) DOĞRULAMA (go/no-go)
val = pd.read_csv("judge_val.csv")
lab = val["label"].to_numpy()
vs = score(val["query"], val["item_text"], "val", "llm_ft_val_scores.npy",
           "llm_ft_val_labels.npy", lab)
# en iyi tek eşik (genel), sonra alt-küme bazlı isabet
thr_star = max(np.arange(0.3, 0.71, 0.05),
               key=lambda t: ((vs >= t).astype(int) == lab).mean())
overall = ((vs >= thr_star).astype(int) == lab).mean()
print(f"\n=== FINE-TUNED HAKEM İSABETİ (eşik {thr_star:.2f}): genel {overall:.3f} ===")
if "src" in val.columns:
    src = val["src"].to_numpy()
    for s in ("pos", "sameleaf", "mined"):
        m = src == s
        if m.any():
            acc = ((vs[m] >= thr_star).astype(int) == lab[m]).mean()
            print(f"    {s:9s}: {acc:.3f} ({int(m.sum())} çift)")
    print("    >>> KARAR: sameleaf isabeti banda en yakın sinyaldir.")
print(">=0.75 gönder | 0.68-0.75 tek atış | <0.68 mühürle (0.893 cepte).\n")

# ---------------------------------------------------------------- 2) BANDI SKORLA (harman için)
try:
    band = pd.read_csv("uncertain_core.csv")
    score(band["query"], band["item_text"], "band", "llm_ft_band_scores.npy",
          "llm_ft_band_rows.npy", band["row_idx"].to_numpy())
    print("Bitti. İndir/push: llm_ft_band_scores.npy, llm_ft_band_rows.npy, "
          "llm_ft_val_scores.npy, llm_ft_val_labels.npy")
except Exception as e:
    print("Band skorlama hatası (eğitim+val yine de sağlam):", e)
