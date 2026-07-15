# Masaüstü (Windows/RTX 4070) — Ölçümler ve Öneriler

> Bu doküman masaüstü oturumundan Mac oturumuna devirdir. **Nihai kararı Mac verir**
> (LB geçmişi ve harman kodu orada). Burada iki şey ayrı tutuldu:
> **§1-2 ÖLÇÜLDÜ** (tartışılmaz, sayılar), **§3-4 KANAAT** (tartışılır, Mac reddedebilir).
> Tarih: 14 Temmuz 2026, 12:30. Donanım: RTX 4070 12 GB, i7-13700KF, 32 GB.

---

## 0. TAMAMLANDI — v9 (XLM-R-large) skorları hazır (15 Tem, güncelleme)

v9 gece koşusu **bitti** (PC sonrasında kapanmış, işi bozmadı). Eğitim + inference eksiksiz:
- `ce9_model/model.safetensors` (2.24 GB) kaydedildi — 14 Tem 23:04.
- `ce9_test_scores.npy` yazıldı — 15 Tem 00:32 (script'in son adımı `[4/4]`).
- Doğrulama: shape `(3359679,)` = submission satır sayısıyla **birebir**; NaN **0/3.36M**;
  aralık 0.0004–0.998. fp16 large patlamadı, temiz.
- **Harman için hazır:** `kaggle_code_9/ce9_test_scores.npy` (push edildi). Mac tarafı alabilir.
  v9'un amacı düşük-korelasyonlu "farklı ses" katmaktı; korelasyon ölçümü Mac'te yapılacak.

---

## 1. ÖLÇÜLDÜ — ce6 ve ce7 aynı model (final4'ü öldüren bulgu)

3.36M test skoru üzerinden, iki `.npy` elde tutularak hesaplandı:

| Metrik | Değer |
|---|---|
| Pearson r (ce6, ce7) | **0.959** |
| Spearman (200k örnek) | 0.932 |
| 0.5 eşiğinde aynı karar oranı | **%96.9** |
| ce6 >0.5 oranı / ce7 | %26.9 / %26.2 |

**Sonuç:** ce7, ce6'nın kopyası. Aynı seed + model + veri + negatifler; tek fark girdi
uzunluğu (attr 200→400, len 128→160) ve bu yeterli çeşitlilik üretmiyor.
`(ce6+ce7)/2` takası (=final4) ce6'nın kendisinden ayırt edilemez → beklenen fark ±0.001,
tam da PROJECT_LOG §8.1'in "hak harcanmaz" bandı. **final4 gönderilmedi.**
(Log'da "gönderildi, skor bekleniyor" yazıyordu — yanlıştı, düzeltildi.)

Bu, §5.7'nin ("aynı aileden yeni model ~+0.001'lik kopya üretir") ölçülmüş kanıtı.
ce8'i iptal eden mantık ce7 için de geçerliymiş — sadece ce7 zaten eğitilmişti.

## 2. ÖLÇÜLDÜ — v9 (XLM-R-large) gerçek rakamları

`kaggle_cross_encoder_v9.py`'nin **birebir kendi ayarlarıyla** (batch 16, lr 1e-5, fp16,
max_len 128, gerçekçi uzunlukta metinler) 60 adımlık duman testi + inference taraması:

### Eğitim (XLM-R-large, 550M)
| | Ölçüm |
|---|---|
| Hız | **1.30 it/s** |
| Tepe VRAM | **11.23 / 12 GB** (sığıyor, ama kıl payı) |
| fp16 NaN | **YOK** (loss 0.696, sağlıklı) |
| Tam eğitim (676k çift, 1 epoch, 42266 adım) | **9.0 saat** |

### Inference (3.36M çift) — batch taraması
| batch | tepe VRAM | hız | 3.36M süre |
|---|---|---|---|
| 512 (script'in mevcut ayarı) | 9.13 GB | 66 çift/s | **14.2 saat** ⚠️ |
| **256** | 8.24 GB | **565 çift/s** | **1.65 saat** ✅ |
| 128 | 7.80 GB | 561 çift/s | 1.66 saat |
| 64 | 7.57 GB | 537 çift/s | 1.74 saat |

**TOPLAM (batch 256 ile): 9.0 + 1.65 = ~10.7 saat.**
12:30'da başlarsa → **bugün ~23:10'da biter.** Hak sıfırlanmasına (03:00) 4 saat pay var.
**Süre engeli YOK — v9 large yetişiyor.**

### Denetim iddialarımın sonucu (dürüstlük için)
Script'e üç itirazım vardı; ölçüm **ikisini çürüttü**:
- ❌ "batch 512 inference OOM verir" → **vermiyor**, sığıyor.
- ❌ "batch 16 eğitim 12 GB'a sığmaz (optimizer state 8.8 GB)" → **sığıyor** (11.23 GB).
- ⚠️ "fp16 + large NaN riski" → bu testte NaN yok. Risk sıfır değil (§5.8: mDeBERTa fp16
  kararsızdı) ama 60 adımda sorun çıkmadı; script'in "ilk 30 dk izle" uyarısı yeterli.
- ✅ **AMA asıl sorunu buldu:** batch 512 inference OOM vermiyor ama **8x yavaşlıyor**
  (66 vs 565 çift/s) — 14 saat vs 1.65 saat. Bu tek satır, gecenin sonucunu değiştirir.

### v9'da yapılması gereken TEK değişiklik
```python
# satır 133:
scores[s:s + chunk] = model.predict(
    batch_pairs, batch_size=256, show_progress_bar=True,   # 512 -> 256
).astype(np.float16)
```
Başka hiçbir şeye dokunmaya gerek yok. (Docstring hâlâ v6'dan kalma "dbmdz / 2 epoch /
T4" diyor — kozmetik, sonucu etkilemez.)

## 3. ÖLÇÜLDÜ — yedek: XLM-R-base (270M)

Aynı test, base model, batch 64:

| | Ölçüm |
|---|---|
| Hız | **5.18 it/s** |
| Tepe VRAM | 6.53 GB (rahat) |
| Eğitim (2 epoch, 21132 adım) | **1.13 saat** |
| Inference (batch 256) | 1821 çift/s → **0.51 saat** |
| **TOPLAM** | **~1.6 saat** |

Base, large'ın **6.5 katı hızlı**. Aynı mimari çeşitliliği getirir (XLM-R ailesi:
SentencePiece tokenizer, çok dilli ön-eğitim — dbmdz-BERT'ten gerçekten farklı ses),
sadece kapasitesi düşük.

---

## 4. KANAAT — önerim (Mac karar versin)

**Elimizdeki sayılarla ikisini birden koşabiliriz.** Bu, "large mı base mi" ikilemini ortadan kaldırıyor:

| Ne | Ne zaman | Biter | Ne işe yarar |
|---|---|---|---|
| **XLM-R-base (ce10)** | ŞİMDİ (12:30) | **~14:10** | **Bugünkü hakka yetişir.** XLM ailesinin karışıma katkısını BUGÜN ölçeriz. |
| **XLM-R-large (ce9)** | base bitince (~14:15) | **~01:00** | Yarınki 5 hakkın ana hamlesi. Gece boyu koşar. |

Sıra önemli: base önce, çünkü **bilgi değeri var**. Eğer XLM ailesi karışıma hiç katkı
vermiyorsa (base'li karışım ≤0.890), bunu bugün öğreniriz ve large'ın 11 saatlik gecesine
girmeden karar gözden geçirilir. Katkı verirse, large'ın beklenen değeri doğrulanmış olur.

Eğer Mac "tek koşu" derse: **large**. Süre yetiyor (23:10 < 03:00), beklenen değer yüksek.

### Bugünkü hak için (Mac karar verecek)
1. **LGBM bagging** (Mac'te çalışıyor) — PROJECT_LOG §9'un ana adayı. LGBM karışımdaki tek
   gerçek çeşitlilik (r≈0.79) ve ağırlığı yüksek (w=0.40). Bitmişse bu.
2. **XLM-R-base'li karışım** — bagging bitmediyse ve base ~14:10'da hazırsa. Yapısal yenilik
   (yeni model ailesi), §8'e uygun.
3. **final4'ü GÖNDERME** — §1'deki ölçüm gereği.

### Bir uyarı
ce9/ce10 de ce6/ce7 gibi **val skoru üretmiyor** (tüm terimler eğitimde). `blend_ce.py`
doğrudan çalıştırılamaz; ağırlık/eşik ce4'ün val'inden veya LB'den gelmeli. Karışıma
eklerken ce9'un ağırlığı LB'den aranmalı (PROJECT_LOG §9 ~0.25-0.3 diyor, makul başlangıç).

---

## 5. Masaüstü operasyonel notlar

- **Ortam:** Python 3.12 (venv YOK, global), torch 2.6.0+cu124, sentence-transformers 5.6.0,
  transformers 5.13.1. Script'ler repo kökünden çalıştırılıyor (`find_input()` veriyi orada bulur).
- **ST 5.6'da `CrossEncoder.fit()` HÂLÂ ÇALIŞIYOR** — deprecated ama `CrossEncoderTrainer`'a
  delege eden shim. Script'leri yeni API'ye taşımaya gerek yok (bunu önce yanlış sanmıştım,
  duman testiyle doğrulandı).
- **Gerçek süreler** (script başlıklarındaki tahminler şişkin çıktı):
  | Model | Tahmin | GERÇEK |
  |---|---|---|
  | ce6 (dbmdz base, 2ep) | "2.5-3 saat" | **1s21dk** eğitim + 23dk skorlama |
  | ce7 (a.a. + zengin girdi) | "3.5 saat" | **1s15dk** eğitim + 27dk skorlama |
  | ce9 (XLM-R-large, 1ep) | "6-7 saat" | **~9 saat** eğitim + 1.65s skorlama (ölçüldü) |
- **Uyku modu:** eğitim sırasında PC uyursa süreç ölmez (RAM korunur, uyanınca devam eder),
  ama gece koşusu için kapatmak lazım: `powercfg /change standby-timeout-ac 0`
- **Git:** `.npy` ve `kaggle_code_*/` ignore'da; her CE skoru için `.gitignore`'a açık istisna
  eklendi (`!kaggle_code_N/ceN_test_scores.npy`). ce9/ce10 için de aynısı yapılacak.
- **Disk:** `ce6_model/`, `ce7_model/` (~440 MB × 2) diskte duruyor, ignore'da. Silinebilir.
