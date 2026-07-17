# İKİNCİ AŞAMA — Tekrarüretilebilirlik & Kural Uyumu

Bu doküman, ikinci aşama (finalist / ilk-20 inceleme) taleplerine hazırlıktır.
Kaggle kuralı: *"İkinci aşamaya hak kazanan takımların; veri hazırlığı, eğitim
ve submission'larını yeniden üretebildikleri çözüm kaynak kodlarını iletmeleri
gerekmektedir"* ve *"İlk 20 takımın çalışmaları talep edilip tutarlılığı
incelenecek."*

---

## A. İKİ FINAL SUBMISSION (ayrı dokümanlar)

- **Dosya 1 — `submission_7b_w50.csv` (0.903):** bkz. `FINAL_1_submission_7b_w50.md`
- **Dosya 2 — `submission_final11_duo_q28.csv` (0.893):** bkz. `FINAL_2_submission_final11_duo.md`

Her iki dosya da Kaggle Submissions ekranında "final" olarak işaretlendi.

## B. UÇTAN UCA ÜRETİM SIRASI (sıfırdan)

1. **Veri hazırlığı**
   - Yarışma CSV'leri: `trendyol-e-ticaret-yarismasi-2026-kaggle/` (items, terms,
     training_pairs, submission_pairs).
   - Bi-encoder fine-tune (e5-small, MNRL): `finetune_biencoder.py` → embedding
     üretimi `embed_texts.py` (`embpl_*.npy` vb.).
   - Zor negatif madenciliği: `mine_hard_negatives.py` → `mined_hard_negatives.csv`
     (embedding top-K, leaf-kategori false-negative koruması).
   - Aynı-leaf çelişki negatifleri: `mine_sameleaf_negatives.py` →
     `sameleaf_negatives.csv` (sorgu rakam/renk/marka token'ı pozitiflerde tutup
     adayda çeliştiği durumlar).
2. **Cross-encoder eğitimleri** (Kaggle T4 / RTX 4070): `kaggle_cross_encoder_v3,
   _v5, _v6, _v11, _v12.py` → her biri `ce{k}_test_scores.npy`.
3. **LightGBM (komşu-graf öznitelikli):** `lgbm_v9L.py` → `test_proba_v9L.npy`.
4. **Taban topluluk:** yukarıdakileri ağırlıklı birleştir → `test_proba_final11.npy`
   (formül: FINAL_2 §2).
5. **7B yargıç (zero-shot):** `kaggle_llm_judge.py` (Qwen2.5-7B-AWQ, vLLM, band
   [0.30,0.70]) → `llm_scores.npy` + `llm_rows.npy`.
6. **Nihai birleştirme + submission:** FINAL_1 §2 (rank-blend w=0.5, q=0.28).

Tüm kod tek repoda; tüm rastgelelik `SEED=42`; ara ürünler `.npy` olarak diskte.

## C. KURAL UYUMU (madde madde)

| Kural | Durum |
|---|---|
| Submission verisine ücretli/kapalı LLM ile tahmin YOK | ✅ Yalnız açık ağırlıklı Qwen2.5-7B, Kaggle VM'inde vLLM ile SELF-HOST. Ücretli API/servis kullanılmadı. |
| Açık kaynak modeller (Qwen, e5, dbmdz, cosmos, convbert) serbest | ✅ Hepsi açık ağırlıklı; HuggingFace'ten indirilip self-host edildi. |
| Scraping / üçüncü-taraf platform verisi YOK | ✅ Yalnızca yarışma verisi + açık-kaynak önceden-eğitilmiş modeller. Trendyol/başka site scrape edilmedi. |
| Yarışma verisi üçüncü taraflarla paylaşılmadı | ✅ Repo PRIVATE; veri yalnızca kendi ortamlarımızda (Kaggle/Mac/PC). |
| Takım içi kod/model transferi (private repo) | ✅ Kurallara uygun (takım içi). |
| Tek Kaggle hesabı, kayıtlı takım | ✅ (Takım yapısı başvuruyla aynı tutulmalı — KULLANICI SORUMLULUĞU, aşağıda ⚠️). |
| Günlük ≤5 submission, ≤2 final | ✅ Uyuldu; 2 final işaretlendi. |
| Reproducibility | ✅ Tüm script + seed + ara .npy repoda. |

**Şeffaflık notu (inceleme için):** Yarışmanın erken bir aşamasında, testin
sınıf dengesini tahmin etmek için BİR KEZ "all-ones" (hepsi 1) sabit tahmin
gönderildi (LB 0.412); sonuç yorumlanıp bu tür probe gönderimleri BIRAKILDI.
Sonraki tüm kararlar (eşik, ağırlık) gerçek model gönderimlerinin skorlarından
çıkarıldı. Manipülasyon amacı yoktur; kayıt için belirtilmiştir.

## D. ⚠️ KULLANICININ KONTROL ETMESİ GEREKENLER (kurallar)

1. **Takım yapısı = başvuru:** Kaggle'daki takım üyeleri, yarışmaya başvururken
   tanımlanan kişilerle BİREBİR aynı olmalı. Farklıysa Kaggle aşaması sonunda
   elenme sebebi. (Kural: "Başvurudakinden farklı kişilerle takımlar elenecek.")
2. **Tek hesap:** Hiçbir üye birden fazla Kaggle hesabıyla giriş yapmamalı.
3. **Submission bütçesi:** Takım üyeleri toplamda bir takımın günlük 5 hakkını
   aşmamış olmalı (sonradan takım olmada uyumsuzluk riski).
4. **Repo private kalmalı** — yarışma verisi/çözüm üçüncü taraflarla paylaşılamaz.
5. **İlk 10'a girilirse (18 Tem+):** yeniden-üretilebilir kaynak kod teslimi
   istenecek → bu repo + bu 3 doküman doğrudan cevap.

## E. DOSYA ENVANTERİ (repo)

Eğitim/veri script'leri: `kaggle_cross_encoder_v{2,3,5,6,11,12}.py`,
`kaggle_llm_judge.py`, `lgbm_pipeline.py`, `lgbm_pipeline_v3.py`, `lgbm_v9L.py`,
`finetune_biencoder.py`, `embed_texts.py`, `mine_hard_negatives.py`,
`mine_sameleaf_negatives.py`.
Harman/submission: FINAL_1 & FINAL_2 §2 blokları (kanonik).
Kayıtlar: `PROJECT_LOG.md`, `FINAL_RESULT.md`, bu doküman.
Büyük ara ürünler (.npy, .csv model çıktıları) repo dışı ama script'lerle
yeniden üretilebilir; kritik `ce*_test_scores.npy` ve `llm_scores.npy`
`kaggle_code_*/` altında saklıdır.

## F. NİHAİ SKOR YOLCULUĞU

0.697 (kelime örtüşmesi) → 0.746 (LightGBM) → 0.767 (+embedding) → 0.813
(fine-tuned bi-encoder) → 0.856 (Türkçe BERT CE + madenci negatif) → 0.889
(eşik/ağırlık kalibrasyonu) → 0.893 (CE topluluk + komşu-graf LGBM) → **0.903
(zero-shot Qwen2.5-7B rank-blend)**. Benchmark Level 4 (0.900) aşıldı.
