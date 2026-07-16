# Trendyol Datathon 2026 — Proje Günlüğü ve Devir Dokümanı

> **KAPANIŞ (16 Tem):** Zirve LB 0.893 (final11 = 0.6·[0.7·(0.3ce3+0.4ce6+0.3ce5) + 0.3·(ce11+ce12)/2] + 0.4·v9L, q=0.28).
> Son denemeler: v9L (komşu-graf LGBM) +0.001; ce11 (sameleaf çelişki) yarı dozda nötr, ce12 (ConvBERT) ile çift olarak +0.002;
> ce13 (max-config) nötr; per-term rank hibriti −0.060 (heterojenlik GERÇEK, global eşik doğru);
> ce14 (tam-liste negatif) EĞİTİLMEDEN İPTAL — kirlilik ölçümü %15-20 false-negative gösterdi (taban-oranı aritmetiği).
> **FİNAL ÇİFT: submission_final11_duo_q28.csv (0.893) + submission_final3_q27.csv (0.890) — 17 Tem'de Kaggle'da ELLE işaretlenecek.**

> Bu doküman, başka bir Claude oturumunun (veya bir insanın) projeyi sıfır bağlamla
> devralabilmesi için yazıldı. Son güncelleme: 14 Temmuz 2026 (yarışmanın bitimine 3 gün).

---

## 1. Yarışma ve bağlam

- **Görev:** (arama terimi, ürün) çifti için binary relevance tahmini (1=alakalı, 0=alakasız).
- **Metrik:** macro-averaged F1 (iki sınıfın F1 ortalaması). Eşik/denge seçimi kritik.
- **Veri:** Kaggle — "Trendyol E-Ticaret Yarışması 2026". Public LB canlı; nihai sıralama private subset.
- **Takvim:** Kaggle aşaması 26 Haziran – **17 Temmuz 2026**. İlk 10 takım fiziksel hackathona gider.
  Kaggle skoru nihai puanın %40'ı. Hackathon'da ayrıca: model hızı %10, açıklanabilirlik %10, rapor %10.
- **Kritik kurallar:** Finalist çözümleri kod incelemesinden geçer (tekrarlanabilirlik şart).
  Organizatörler skorlamada bazı satırları yok sayarak probe/manipülasyonu engelliyor
  → probe gönderimi YAPMIYORUZ (bir kez all-ones denendi, sonra bırakıldı; raporda şeffaf belirtilecek).
  Kod yarışma süresince paylaşılamaz → **bu repo PRIVATE kalmalı.**
- **Kullanıcı (Kuzey):** backend/frontend developer + SDET; ML'e bu yarışmayla girdi.
  Açıklamalar Türkçe ve jargonsuz olmalı; senior-ML disiplini istiyor (sızıntı yok,
  LB-overfit yok, best practice). Günde 5 submission hakkı; haklar ertesi güne DEVRETMEZ
  (UTC gece yarısı sıfırlanır).

## 2. Donanım ve altyapı

| Makine | Özellik | Rol |
|---|---|---|
| MacBook Pro M4 Pro | 12 CPU, 16 GPU çekirdek (MPS), 24 GB | Veri işleme, LGBM, bi-encoder fine-tune, harman/eşik, submission üretimi |
| Windows masaüstü | i7-13700KF, 32 GB DDR5, **RTX 4070 12GB** | Cross-encoder eğitimleri (T4'ün ~2.5-3 katı hız). Kurulum: `SETUP_WINDOWS.md`, `C:\trendyol` |
| Kaggle | T4 GPU, haftada 30 saat (≈5 saat kaldı) | Eski CE eğitimleri; artık yedek |

**Akış:** Kod bu repoda (git@github.com:kuzeycaliskan/trendyol-teknofest.git). Masaüstü `git pull`
ile script alır, eğitir, `ce*_test_scores.npy` çıktısını `git add -f` ile pushlar. Mac pull'layıp
harmanlar, `submission_*.csv` üretir; kullanıcı Kaggle web'den yükler. Yarışma CSV'leri repoda YOK
(Mac: `trendyol-e-ticaret-yarismasi-2026-kaggle/`, masaüstü: `C:\trendyol\...` altında lokal).

## 3. Veri gerçekleri (ölçülmüş)

- items.csv 962.873 ürün; terms.csv 50.154 terim; training_pairs.csv 250k **yalnız pozitif**;
  submission_pairs.csv 3.359.679 test çifti.
- Test terimlerinin train ile örtüşmesi **%0** (term_id VE normalize sorgu metni düzeyinde — ikisi de ölçüldü).
  → Genelleme terim bazlı; tüm ayrımlar term_id ile yapılır.
- Test itemlarının %21'i trainde görülür; test terim başına ~104 aday çift.
- Testin gerçek pozitif oranı ~%28 (LB eşik eğrisinden; eski %26 tahmini feasibility analizindendi).

## 4. Kronoloji ve LB sonuçları (tümü)

| Sürüm | İçerik | LB |
|---|---|---|
| baseline | kelime örtüşme oranı (coverage), eşik 0.45 | 0.697 |
| v2 lgbm | 16 öznitelik + LightGBM, sentetik negatif (1 rastgele + 1 zor/kategori) | 0.746 |
| thr0176 | v2, eşik 0.176 (yanlış %70 hipotezi) | 0.683 |
| all-ones probe | (bir kez; sonra probe bırakıldı) | 0.412 |
| v3 | + multilingual-e5-small kosinüsü | 0.767 |
| v4 | + terim-içi rank öznitelikleri | **0.648/0.558 — FİYASKO** |
| v5 | e5 fine-tune (MNRL in-batch, 1 epoch) | 0.795 |
| v6 | 2. epoch | 0.809 |
| v7 | 3. epoch (bi-encoder hattı mühürlendi) | 0.813 |
| v8 | + pseudo-label MNRL turu | 0.814 (nötr) |
| ce1 distilbert harmanı | sentetik negatifli CE + LGBM | 0.799 (zararlı) |
| **ce2** | dbmdz bert-base-turkish-128k-uncased + MINED negatifler, α=0.2 LGBM, q=0.242 | **0.856** |
| q33 | ce2 probası q=0.33 | 0.838 |
| ce3 | ce2 + 571k pseudo etiket | 0.853 (nötr) |
| ce4 | ce2 reçetesi 2 epoch | 0.855 |
| ens ce2+ce3 / ce234 | skor ortalamaları | 0.854 |
| final1 | ens(.3/.3/.4) + 0.2 LGBM, q=0.234 | 0.856 (plato) |
| final1 q=0.26 / **q=0.28** / q=0.30 | eşik eğrisi | 0.873 / **0.877** / 0.872 |
| final2 q28 w=0.2 / w=0.3 / **w=0.4** / w=0.5 | LGBM ağırlık eğrisi | 0.880 / 0.885 / **0.889** / 0.887 |
| w40dual (LGBM=v7+v8 ort.) | | 0.888 (elendi) |
| **final3** | CE karışımı ce3/ce6/ce5 (.3/.4/.3) + 0.4·LGBM(v8), q=0.28 | **0.890 — GÜNCEL EN İYİ** |
| final3b (5'li CE) | ce4'ü de ekle | 0.890 (fark yok) |
| final4 | ce6→(ce6+ce7)/2 takası | GÖNDERİLMEDİ (ce6-ce7 r=0.959, ölü) |
| final5 | LGBM 5'li bagging takası | GÖNDERİLMEDİ (bag-v8 r=0.998, ölü; final dosyalara ölçümsüz dahil edilecek) |
| final3 q=0.27 | final seçim omuz ölçümü | **BUGÜNÜN SON HAKKI — gönderildi, skor bekleniyor** |

CE modelleri: ce5 = ytu-ce-cosmos/turkish-base-bert-uncased (attr 400, len 160, 2ep, LB katkısı çeşitlilik);
ce6 = dbmdz, TÜM veri (val'siz), 2ep (masaüstünde eğitildi, 1s21dk);
ce7 = ce6 + zengin girdi (attr 400, len 160; 1s42dk) — **ce6'nın kopyası:** r=0.959, Spearman 0.932,
%96.9 aynı karar (masaüstünde ölçüldü) → ensemble çeşitliliği yok, bkz. §9.

## 5. Öğrenilen dersler (sırayla, en önemliler)

1. **Sentetik validasyon vekildir, mutlak değil.** Pointwise özniteliklerde (v2→v3) LB'ye transfer etti;
   grup-bazlı özniteliklerde (v4) tamamen yanılttı (val AUC 0.9979 → LB çöktü: model, gruplardaki
   sentetik negatifleri tanımayı öğrenmişti). Grup-bazlı öznitelik ancak gerçekçi grup dağılımıyla eğitilebilir.
2. **Sentetik negatifle eğitilen CE zararlı olabilir** (ce1 distilbert 0.799 < LGBM 0.813).
   Çözüm: embedding-madenciliğiyle zor negatif + **leaf-kategori false-negative koruması**
   (terimin pozitiflerinin leaf'indeki adaylar negatif havuzuna giremez) → ce2 0.856.
3. **Bi-encoder MNRL (in-batch negatives) sentetik negatif varsayımı taşımaz** → fine-tune hattı
   güvenle çalıştı (+0.046 toplam). Doygunluk 3. epoch'ta.
4. **Yerel val ~0.92'de doygun**: ≥+0.01'lik LB kazançlarını bile göremiyor (v6 yerel yatay, LB +0.014).
   Küçük farkların tek pusulası LB; haklar ona göre bütçelenir.
5. **Pseudo-labeling bu problemde nötr** (v8 +0.001, ce3 −0.003). Kapatıldı.
6. **Eşik (q) ve LGBM ağırlığı (w) yerelden seçilemez** — mined-val eşiği sola, LGBM ağırlığını 0'a çeker
   (mined negatifler LGBM'in zayıf olduğu bölgeyi abartır). İkisi de LB'den haritalandı:
   **q tepesi 0.28, w tepesi 0.40.** Eğriler pürüzsüz çıktı (gürültü değil sinyal).
7. **CE'ler birbirine r=0.91-0.96 korelasyonlu; gerçek çeşitlilik LGBM'de (r≈0.79).**
   Aynı aileden yeni model ~+0.001'lik kopya üretir. ce8 (seed ikizi) bu gerekçeyle İPTAL edildi.
8. Teknik tuzaklar: pandas yeni string tipi + `rng.shuffle` sessiz bozulma (permütasyon indeksi kullan);
   kalibrasyon kodunu kopyalama — **her eşik seçimi `resample_to_prior`dan geçmeli** (bir kez `min()`
   kısayolu %33'te kalibre etti); sentence-transformers fit için `datasets`+`accelerate` gerekir;
   LightGBM macOS'ta `brew install libomp` ister; CrossEncoder.fit çoklu-GPU DataParallel ile çalışmaz
   (`CUDA_VISIBLE_DEVICES=0`); mDeBERTa fp16 kararsız (bu yüzden seçilmedi).

## 6. Güncel mimari (LB 0.890)

```
p_final = 0.6 · [0.3·ce3 + 0.4·ce6 + 0.3·ce5] + 0.4 · LGBM_v8
tahmin  = 1  eğer p_final ≥ quantile(p_final, 1-0.28)   # q=0.28
```
- ce* skorları: `ce{k}_test_scores.npy` (3.36M float16, submission_pairs sırasında; logit ise sigmoid uygula
  — `to_p()` deseni: min<0 veya max>1 ise sigmoid).
- LGBM_v8: `test_proba_v8.npy` (17 öznitelik + pseudo-adapte embedding; `model_v8.txt`).
- Harman probaları: `test_proba_final3.npy` (=0.890 tabanı), `test_proba_final4.npy` (ce7'li, ölçülüyor).
- `lgbm_v8_on_ceval.npy`: v8'in ce val çiftlerindeki probaları (harman denemeleri için önbellek).

## 7. Dosya envanteri

Repo içindekiler README'de tablo halinde. Repo DIŞI kritik dosyalar (Mac'te proje kökünde):
- `emb*/embft*/embpl*` .npy'leri: bi-encoder embeddingleri (üretici: `embed_texts.py <model> <önek>`)
- `e5-small-ft/ft2/ft3/pl/`: fine-tuned bi-encoder modelleri
- `test_proba_*.npy`: tüm model/harman probaları; `submission_*.csv`: tüm gönderimler
- `kaggle_code_N/`: Kaggle/masaüstü çıktılarının indiği klasörler (ce_N skorları)
- Submission üretiminde HER ZAMAN: sample_submission ile kolon/satır/id-kümesi/0-1 doğrulaması yapılır.

## 8. Karar kuralları (aktif disiplin)

- Farkı ≤0.002 beklenen varyanta hak harcanmaz (public gürültü tabanı ~±0.001-0.002).
- Haklar yalnızca yapısal yeniliklere: farklı model ailesi, bagging, bileşen dahil/hariç kararları.
- q/w yeniden taranmaz (tepe biliniyor); istisna: final seçim öncesi TEK q-omuz ölçümü (q27, dosya hazır).
- **Final 2 dosya şablonu:** (1) en iyi karışım @ q=0.28; (2) AYNI karışım @ q=0.26-0.27 (muhafazakâr omuz,
  private öncül-kayması sigortası). İkisi de Kaggle'da elle işaretlenecek — 17 Temmuz'dan önce!

## 9. BUGÜNÜN DURUMU ve açık işler (14 Temmuz öğlen)

**final4 ASKIDA — gönderilmedi, gönderilmemeli.** Gerekçe (masaüstünde ölçüldü, 14 Tem):
ce6 ile ce7 **r=0.959 (Pearson), Spearman 0.932, q=0.28 eşiğinde çiftlerin %96.9'unda AYNI karar.**
`(ce6+ce7)/2` takası bu yüzden ce6'nın kendisinden ayırt edilemez; beklenen fark ±0.001 —
tam da §8.1'in "hak harcanmaz" bandı. Karar §8'e uygun olarak iptal edildi.
(Dosya duruyor; ce7 bir gün tek başına takas denenirse o zaman kullanılabilir.)

**Ders:** ce7 (= ce6 + zengin girdi) bir ÇEŞİTLİLİK üyesi değil, ce6'nın kopyası. Aynı seed,
aynı model, aynı veri, aynı negatifler → tek fark girdi uzunluğu, bu yeterli çeşitlilik üretmiyor.
§5.7'nin ("aynı aileden yeni model ~+0.001'lik kopya üretir") ölçülmüş kanıtı.

**Şu an çalışanlar:**
1. **Mac:** LGBM bagging — seed 43-46 ile `PIPE_SEED=$s EMB_PREFIX=embpl TAG=v8s$s python lgbm_pipeline_v3.py`
   döngüsü (loglar `v8s*.log`). Bitince: `test_proba_v8s43..46.npy` + mevcut v8 → 5'li ortalama al,
   `test_proba_lgbmbag.npy` üret. **BUGÜNKÜ HAKKIN ANA ADAYI** — LGBM karışımdaki tek gerçek
   çeşitlilik kaynağı (r≈0.79) ve ağırlığı yüksek (w=0.40); bagging onun varyansını düşürür.
   Yapısal yenilik → §8'e uygun.
2. **Masaüstü (gece):** `kaggle_cross_encoder_v9.py` = **XLM-RoBERTa-large** (550M, 1 epoch, tüm veri).
   Stratejik gerekçe sağlam (düşük korelasyonlu güçlü ses; hedef 0.890 → 0.90+), AMA script
   ÇALIŞTIRILMADAN ÖNCE 3 DÜZELTME ŞART (masaüstü denetimi, 14 Tem):
   - `predict(batch_size=512)` (satır 133, v6'dan kopya) → **large'da kesin OOM**, üstelik 6 saatlik
     eğitim BİTTİKTEN sonra patlar. → 64'e indir.
   - `use_amp=True` (fp16) + large → NaN riski; §5.8 zaten "mDeBERTa fp16 kararsız" diyor.
     → **bf16** kullan (4070 destekliyor, taşma yok, hız aynı). "İlk 30 dk loss izle" planı
     gece 3'te uyanmayı gerektirir; gereksiz.
   - batch 16 sığmayabilir: 550M × (ağırlık 2.2 + gradyan 2.2 + Adam momentleri 4.4) = **8.8 GB,
     aktivasyonlar daha başlamadan**. (ce6/base 8.3 GB kullanmıştı.) → batch 8 + grad-accum 2.
   **Önce ~15 dk duman testi** (gerçek batch, birkaç yüz adım): OOM/NaN var mı, gerçek it/s ne
   → geceye yetişip yetişmeyeceği KESİN bilinir. Süre riski gerçek: eğitim ~6s + large inference
   ~1.5-2s = sabah 08'i bulur; yarının ana hamlesi buna bağlı, tek deneme hakkı var.
   **Yedek:** yetişmezse XLM-R-**base** (270M) — aynı mimari çeşitliliği (farklı tokenizer/ön-eğitim)
   ~2 saatte verir, kesin yetişir. 3 gün kala "kesin yetişir + yeterince farklı" > "güçlü ama riskli".

**Yarın (15 Tem, 5 hak) planı:**
1. XLM'li karışım (ce9'u CE karışımına ekle; ağırlık başlangıcı ~0.25-0.3, ce9 tekil güçlüyse artır) @ q28
2. LGBM-bag'li karışım @ q28
3. En iyi ikisinin birleşimi
4. `submission_final3_q27.csv` (hazır) — final seçim omuz ölçümü
5. Yedek (sonuçlara göre)

**15 TEM SONUÇ — XLM HATTI KAPANDI:** final7_l30 (large %30) LB 0.886; final6_x25 (base %25) LB 0.887.
İki ölçüm de zarar bandında → XLM ailesi bu problemde yerli Türkçe BERT'lerin gerisinde (muhtemel:
çok-dilli tokenizer dezavantajı + large'ın tek epoch'u). Kalan 3 hak bilinçli BOŞ bırakıldı (denetim
kuralı: 0.890'ı geçme ihtimali olan ölçüm kalmadı; boş hak > gürültü ölçümü). **FİNAL 2 DOSYA KESİNLEŞTİ:**
Dosya 1 = `submission_final3_q28.csv` (LB 0.890), Dosya 2 = `submission_final3_q27.csv` (LB 0.890,
eşik sigortası). İkisi de ölçülmüş. 16-17 Tem: Kaggle'da ELLE işaretle + rapor düzeni. Deney YOK.

**KIRMIZI TAKIM KARARI (14 Tem): final 2 dosya yalnız eşik ekseninde AYRIŞAMAZ** — dosya 2, model
bileşimi farklı bir karışım olmalı (örn. LGBM ağırlığı farklı veya XLM dahil/hariç); aksi halde iki
dosya aynı model riskini taşır. Ayrıca: Kaggle final işaretleme ELLE yapılır, 17 Tem'den önce;
eşik hesapları float32'de; ce9/ce10 npy satır hizası submission_pairs ile ayrıca doğrulanacak.
Yarın (15 Tem) 5 hak: (1) ce10'lu karışım [ön kontrol sonrası], (2) ce9'lu karışım, (3) ikisi birlikte,
(4) ağırlık iterasyonu, (5) yedek. ce10/ce9 val üretmez → ağırlıklar LB'den, q=0.28 sabit.

**16-17 Tem:** Deney YOK. Final 2 dosyanın üretimi + Kaggle'da işaretlenmesi + finalist olasılığına karşı
kod/rapor düzeni (bu doküman + README güncel tutulacak). Kaggle'da final seçimi yapmayı UNUTMA.

## 10. İletişim notları (devralan oturum için)

- Kullanıcıyla Türkçe konuş; ML kavramlarını developer analojileriyle açıkla (API, cache, diff...).
- Her submission önerisinde: dosya adı + Kaggle notu + beklenti + neyi ölçtüğü söylenir.
- Kullanıcı sorgulayıcıdır ve bu iyi bir şey — "neden bunu yapıyoruz" sorusuna her zaman
  beklenen-değer gerekçesiyle cevap ver; gerekirse kararı revize et (ce8 iptali örneği).
- Uzun eğitimler: Mac'te `nohup ... &` + Monitor ile izlenir; masaüstünde kullanıcı elle çalıştırır.
- Kalıcı hafıza dosyaları: `~/.claude/projects/-Users-kuzey-teknofest-trendyol/memory/` (aynı bilgilerin özeti).
