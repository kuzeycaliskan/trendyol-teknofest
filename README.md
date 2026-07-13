# Trendyol Datathon 2026 — Arama Alaka Modeli

(Sorgu, ürün) çiftleri için binary relevance tahmini. Metrik: macro-F1.
Güncel LB: **0.889** (dörtlü cross-encoder ensemble + %40 LightGBM, q=0.28).

## Masaüstünde ce6 eğitimi (öncelikli iş)

Kurulum: `SETUP_WINDOWS.md` (Python 3.12, CUDA'lı PyTorch, veri indirme).
`mined_hard_negatives.csv` bu repoda hazır; yarışma CSV'leri Kaggle'dan indirilir.

```powershell
cd C:\trendyol
.\venv\Scripts\Activate.ps1
python kaggle_cross_encoder_v6.py
```

Süre (RTX 4070): ~2.5-3 saat. Çıktı: `ce6_test_scores.npy` → Mac'e taşınacak.

## Dosya haritası

| Dosya | Ne |
|---|---|
| `baseline.py` | Seviye 1: kelime örtüşmesi (LB 0.697) |
| `lgbm_pipeline.py` | Seviye 2: 16 öznitelik + LightGBM (LB 0.746); ortak yardımcılar |
| `lgbm_pipeline_v3.py` | Seviye 3+: + embedding kosinüsü; EMB_PREFIX/TAG env ile parametrik (v5/v6/v7/v8 bununla eğitildi) |
| `lgbm_pipeline_v4.py` | KULLANILMIYOR — rank öznitelikleri LB'de battı, ders kaydı |
| `embed_texts.py` | Sorgu+ürün embeddingleri (argv: model, çıktı öneki) |
| `finetune_biencoder.py` | Bi-encoder MNRL fine-tune (argv: taban, çıktı, epoch) |
| `finetune_pseudo.py` | Pseudo-label MNRL turu (nötr çıktı, kapatıldı) |
| `mine_hard_negatives.py` | Embedding'le zor negatif madenciliği (leaf-kategori korumalı) |
| `mined_hard_negatives.csv` | Madencilik çıktısı: 176k zor negatif — CE eğitimlerinin girdisi |
| `kaggle_cross_encoder.py` | CE v1: distilbert (LB 0.799, sentetik negatif dersi) |
| `kaggle_cross_encoder_v2.py` | CE v2: dbmdz Türkçe BERT + mined (LB 0.856) |
| `kaggle_cross_encoder_v3.py` | CE v3: + pseudo etiketler (nötr) |
| `kaggle_cross_encoder_v4.py` | CE v4: v2 + 2 epoch |
| `kaggle_cross_encoder_v5.py` | CE v5: cosmos Türkçe BERT, zengin girdi |
| `kaggle_cross_encoder_v6.py` | **CE v6: endgame — tüm veri, val'siz, 2 epoch (masaüstünde koşulacak)** |
| `blend_ce.py` | LGBM + CE ağırlıklı harman + eşik kalibrasyonu |
| `SETUP_WINDOWS.md` | Masaüstü kurulum rehberi |

## Metodoloji özeti

- Terim bazlı train/val ayrımı (test terimleri trainde yok — %0 örtüşme)
- Negatifler: rastgele + embedding-madenciliği (leaf-kategori false-negative koruması)
- Eşik (q) ve LGBM ağırlığı (w) LB'den kalibre: q eğrisi tepesi 0.28, w tepesi ≥0.4 (aranıyor)
- Sentetik validasyon doygun (0.92 bandı) — küçük farklar için pusula LB
- Final seçim: 1 dosya tepede (q=0.28), 1 dosya muhafazakâr (q=0.26)

> Not: Bu repo yarışma süresince PRIVATE kalmalı (kod paylaşımı kurallara aykırı).
