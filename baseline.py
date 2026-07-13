"""
Seviye 1 baseline: sorgu kelimelerinin ürün metninde geçme oranı (coverage).

- Ürün metni = başlık + kategori + marka (küçük harfe çevrilmiş).
- Bir sorgu kelimesi, ürün metninde substring olarak geçiyorsa "eşleşti" sayılır
  (Türkçe ekleri kabaca tolere etmek için: "ayakkabı" -> "ayakkabısı" eşleşir).
- Skor = eşleşen sorgu kelimesi / toplam sorgu kelimesi.
- Eşik, eğitim pozitifleri + rastgele üretilmiş negatiflerle macro-F1'e göre seçilir.
"""
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA = "trendyol-e-ticaret-yarismasi-2026-kaggle"
RNG = np.random.default_rng(42)

TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+")

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

print("Veriler okunuyor...")
items = pd.read_csv(f"{DATA}/items.csv", usecols=["item_id", "title", "category", "brand"])
terms = pd.read_csv(f"{DATA}/terms.csv")
train = pd.read_csv(f"{DATA}/training_pairs.csv")
test = pd.read_csv(f"{DATA}/submission_pairs.csv")

print("Ürün metinleri hazırlanıyor...")
item_text = {}
for iid, title, cat, brand in zip(items.item_id, items.title, items.category, items.brand):
    parts = [str(title), str(cat).replace("/", " "), str(brand)]
    item_text[iid] = " " + " ".join(p.lower() for p in parts if p != "nan")

term_tokens = {tid: tokenize(q) for tid, q in zip(terms.term_id, terms["query"])}

def coverage(tid: str, iid: str) -> float:
    toks = term_tokens.get(tid)
    text = item_text.get(iid)
    if not toks or text is None:
        return 0.0
    hit = sum(1 for t in toks if t in text)
    return hit / len(toks)

# --- Eşik seçimi: pozitifler + rastgele negatifler ---
print("Yerel validasyon kuruluyor...")
pos = train.sample(50_000, random_state=42)
all_items = items.item_id.to_numpy()
neg_items = RNG.choice(all_items, size=len(pos))
neg_terms = pos.term_id.to_numpy()

pos_scores = np.array([coverage(t, i) for t, i in zip(pos.term_id, pos.item_id)])
neg_scores = np.array([coverage(t, i) for t, i in zip(neg_terms, neg_items)])

y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
scores = np.concatenate([pos_scores, neg_scores])

print(f"pozitif ort. coverage: {pos_scores.mean():.3f} | negatif ort. coverage: {neg_scores.mean():.3f}")

best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.15, 1.01, 0.05):
    f1 = f1_score(y_true, (scores >= thr).astype(int), average="macro")
    print(f"  eşik {thr:.2f} -> yerel macro-F1 {f1:.4f}")
    if f1 > best_f1:
        best_thr, best_f1 = thr, f1
print(f"Seçilen eşik: {best_thr:.2f} (yerel macro-F1 {best_f1:.4f})")
print("Not: negatifler rastgele olduğu için bu skor iyimserdir; gerçek LB daha düşük çıkar.")

# --- Test tahminleri ---
print("Test çiftleri skorlanıyor (3.36M satır)...")
test_scores = np.fromiter(
    (coverage(t, i) for t, i in zip(test.term_id, test.item_id)),
    dtype=np.float32, count=len(test),
)
pred = (test_scores >= best_thr).astype(int)
print(f"Test pozitif oranı: {pred.mean():.3f}")

sub = pd.DataFrame({"id": test.id, "prediction": pred})
sub.to_csv("submission_baseline.csv", index=False)
print(f"Yazıldı: submission_baseline.csv ({len(sub)} satır)")
