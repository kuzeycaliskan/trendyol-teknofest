# Trendyol Datathon 2026 — Arama Alaka Modeli

TEKNOFEST 2026 E-Ticaret Yarışması (Kaggle aşaması, 26 Haziran – 17 Temmuz 2026).
Görev: bir (arama terimi, ürün) çifti için ürünün terimle **alakalı (1)** veya
**alakasız (0)** olduğunu tahmin eden binary model. Metrik: **macro-F1**.
Eğitim verisi yalnızca pozitif çiftler içerir; negatifler yarışmacı tarafından üretilir.

## Sonuç

| | |
|---|---|
| **Public LB** | **0.903** |
| **Private LB / sıra** | **68. / 364 takım** |
| Başlangıç (baseline) | 0.697 |
| Toplam iyileşme | **+0.206** |
| Benchmark Level 4 (0.900) | aşıldı |

İlk ML yarışması; ML'e bu projeyle başlandı.

## Nihai çözüm (iki final submission)

Kaggle'da işaretlenen 2 submission — tam reçeteleri ayrı dokümanlarda:

1. **`submission_7b_w50.csv` → 0.903** — bkz. [`FINAL_1_submission_7b_w50.md`](FINAL_1_submission_7b_w50.md)
   5 Türkçe cross-encoder + komşu-graf öznitelikli LightGBM topluluğu, üzerine
   **zero-shot Qwen2.5-7B** yargıcın kararsız-bant sıralaması rank-blend (w=0.5), q=0.28.
2. **`submission_final11_duo_q28.csv` → 0.893** — bkz. [`FINAL_2_submission_final11_duo.md`](FINAL_2_submission_final11_duo.md)
   Aynı topluluk, LLM yargıç YOK (temiz sigorta; farklı model ekseni).

İkinci-aşama tekrarüretilebilirlik & kural uyumu: [`STAGE2_REPRODUCIBILITY.md`](STAGE2_REPRODUCIBILITY.md).
Tam günlük: [`PROJECT_LOG.md`](PROJECT_LOG.md) · Skor özeti: [`FINAL_RESULT.md`](FINAL_RESULT.md).

## Yaklaşım — skor merdiveni

| Aşama | Yöntem | LB |
|---|---|---|
| Baseline | Sorgu kelimelerinin ürün metninde geçme oranı (coverage) | 0.697 |
| Seviye 2 | 16 öznitelik + LightGBM; sentetik negatif (rastgele + kategori-zor) | 0.746 |
| Seviye 3 | + çok dilli embedding (multilingual-e5) kosinüsü | 0.767 |
| Fine-tune | e5-small bi-encoder MNRL fine-tune (in-batch negatives) | 0.813 |
| Cross-encoder | Türkçe BERT (dbmdz) + **embedding-madenciliği zor negatifler** | 0.856 |
| Kalibrasyon | Eşik (q) ve ağırlık (w) LB'den haritalandı; q=0.28, w=0.40 | 0.889 |
| Topluluk | 5 CE (dbmdz/cosmos/convbert) + komşu-graf LightGBM | 0.893 |
| **LLM yargıç** | **zero-shot Qwen2.5-7B rank-blend** (kararsız bant) | **0.903** |

## Pipeline dosyaları (kök dizin)

**Veri hazırlığı:** `mine_hard_negatives.py` (embedding zor negatif, leaf-koruma),
`mine_sameleaf_negatives.py` (aynı-leaf sorgu-çelişki negatifi), `finetune_biencoder.py`,
`embed_texts.py`.
**Cross-encoder eğitimi:** `kaggle_cross_encoder_v{2,3,5,6,11,12}.py`.
**LLM yargıç:** `kaggle_llm_judge.py` (Qwen2.5-7B-AWQ, vLLM, self-host).
**LightGBM + topluluk:** `lgbm_pipeline.py`, `lgbm_pipeline_v3.py`, `lgbm_v9L.py`, `blend_ce.py`.
**Arşiv:** `arsiv/` — tüm ölü denemeler ve ara çıktılar (silinmedi, geri alınabilir).

## Metodoloji ilkeleri

- **Sızıntı yok:** train/val ayrımı hep terim (term_id) bazlı; test terimleri trainde %0
  görülüyor, validasyon bunu taklit eder. Ön-işleme yalnız etiketsiz veriyle fit edilir.
- **Ölçmeden gitme:** her yapısal fikir ölçümle sınandı; başarısızlar ucuza elendi —
  XLM ailesi, pseudo-labeling, grup-içi rank öznitelikleri, LightGBM bagging,
  per-term normalizasyon, tam-liste retrieval negatifleri (hepsi ölçümle reddedildi).
- **LB-overfit kontrolü:** yalnız global parametreler (q, w) LB'den kalibre edildi
  (tek-parametre, pürüzsüz eğri); çok sayıda mikro-karar LB'den öğrenilmedi.
- **Sabit seed (42); tüm ara ürünler script'lerle yeniden üretilebilir.**

## Öğrenilen ders (post-mortem)

En pahalı ders: **yarışmanın tam kural/kısıt uzayı (hangi modeller serbest, donanım
sınırı, ne skorlanıyor) veriye dokunmadan ÖNCE netleştirilmeli.** Açık kaynak
LLM'lerin tahminde serbest olduğunu (SSS) geç öğrendik; 7B yargıç yalnız son gün
devreye girdi ve +0.010 getirdi. Erken bilinseydi *fine-tuned büyük LLM yargıç*
(üst takımların reçetesi) haftalarca olgunlaştırılıp ilk-20 (~0.92) realistik
hedeflenebilirdi. Reçete artık hazır — bir sonraki yarışmada ilk günden başlangıç noktası.

## Donanım

MacBook Pro M4 Pro (veri/LightGBM/harman) + Windows RTX 4070 (CUDA, CE eğitimi) +
Kaggle T4 (7B yargıç, vLLM). Python 3.12/3.14.
