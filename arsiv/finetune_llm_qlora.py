"""
Qwen2.5-3B hakem fine-tune (QLoRA) — MASAÜSTÜ (RTX 4070, Windows).

Zero-shot hakemi Trendyol verisiyle uzmanlaştırır: prompt -> tek token
("Evet"/"Hayır") SFT. Kayıp yalnız cevap tokenında (prompt maskeli).

Veri: 150k pozitif + 250k negatif karışımı (mined 100k + sameleaf 100k
[tekrarlı örnekleme yok, mevcutsa] + rastgele 50k) = 400k örnek, 1 epoch.
Süre (4070, 4-bit taban + LoRA r16, bs16 x ga2): ~2-3 saat. Chunk'lı
checkpoint: her 2000 adımda adapter kaydedilir.

Önkoşul: pip install -q peft bitsandbytes datasets
Çıktı: qlora_judge/ (LoRA adapter)
"""

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

SEED = 42
DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_POS, N_MINED, N_SAME, N_RAND = 150_000, 100_000, 100_000, 50_000
MAXLEN = 160

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

print("[1/4] Veri hazırlanıyor...")
terms = pd.read_csv(f"{DATA}/terms.csv")
items = pd.read_csv(f"{DATA}/items.csv",
                    usecols=["item_id", "title", "category", "brand"]).fillna("")
q_of = dict(zip(terms.term_id, terms["query"].astype(str)))
items["text"] = (items.title.str.slice(0, 90) + " | "
                 + items.category.str.split("/").str[-1] + " | " + items.brand)
t_of = dict(zip(items.item_id, items.text))

pos = pd.read_csv(f"{DATA}/training_pairs.csv").sample(N_POS, random_state=SEED)
pos = pos[["term_id", "item_id"]].assign(y="Evet")
mined = pd.read_csv("mined_hard_negatives.csv").sample(N_MINED, random_state=SEED)
same = pd.read_csv("sameleaf_negatives.csv")
same = same.sample(min(N_SAME, len(same)), random_state=SEED)
rand_items = items.item_id.sample(N_RAND, replace=True, random_state=SEED).to_numpy()
rand_terms = terms.term_id.sample(N_RAND, replace=True, random_state=SEED).to_numpy()
rand = pd.DataFrame({"term_id": rand_terms, "item_id": rand_items})
neg = pd.concat([mined[["term_id", "item_id"]], same[["term_id", "item_id"]], rand],
                ignore_index=True).assign(y="Hayır")
df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=SEED)
print(f"   {len(df)} örnek (%{(df.y == 'Evet').mean()*100:.0f} pozitif)")

tok = AutoTokenizer.from_pretrained(MODEL)

def encode(row):
    msgs = [{"role": "user", "content":
             f"Arama: {q_of[row.term_id]}\nÜrün: {t_of.get(row.item_id, '')}\n"
             "Bu ürün bu aramayla alakalı mı? Sadece Evet veya Hayır."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, truncation=True, max_length=MAXLEN - 2)["input_ids"]
    a_ids = tok(row.y, add_special_tokens=False)["input_ids"][:1] + [tok.eos_token_id]
    return {"input_ids": p_ids + a_ids,
            "labels": [-100] * len(p_ids) + a_ids}

print("[2/4] Tokenizasyon (birkaç dk)...")
records = [encode(r) for r in df.itertuples(index=False)]

class DS(torch.utils.data.Dataset):
    def __len__(self): return len(records)
    def __getitem__(self, i): return records[i]

def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id or tok.eos_token_id
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]),
    }

print("[3/4] Model (4-bit) + LoRA...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map="cuda")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

print("[4/4] Eğitim...")
args = TrainingArguments(
    output_dir="qlora_judge", num_train_epochs=1,
    per_device_train_batch_size=16, gradient_accumulation_steps=2,
    learning_rate=1e-4, warmup_steps=300, lr_scheduler_type="cosine",
    logging_steps=100, save_steps=2000, save_total_limit=2,
    fp16=True, report_to=[], seed=SEED)
Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate).train()
model.save_pretrained("qlora_judge/final")
print("Bitti: qlora_judge/final")
