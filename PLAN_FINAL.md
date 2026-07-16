# SON PLAN — Masaüstü (Windows/RTX 4070) Görev Talimatı

> Bu dosya masaüstündeki oturum içindir. Mac oturumu kararları verir; burada
> **sadece ne çalıştırılacağı, ne push'lanacağı ve sonucun nasıl raporlanacağı**
> yazılı. Zirve LB **0.893** (final11_duo_q28) cepte; hiçbir adım onu riske atmaz.
> İki denetimden geçti: gate sızıntısı düzeltildi, rank-blend uç-güncellemeye çevrildi.
> Tarih: 17 Temmuz 2026 (son gün). Repo private kalmalı.

---

## DURUM ÖZETİ (masaüstü Claude'u bunu bilmeli)

- Zero-shot Qwen-3B doğrulaması ölçüldü: AUC 0.721 (sinyal var) ama kalibrasyonsuz
  (skorların %98'i ~0.007). Threshold-gate NO-GO'ydu; ama **sıralama** kullanılabilir.
- İki iş kaldı: (A) bugün zero-shot band skorlama → rank-blend yön testi (1 submission),
  (B) gece fine-tune (kalibrasyonu düzeltir) → yarın asıl bahis.
- Model kararı: fine-tune **1.5B**'de kalıyor (4 çöküşten sonra reliability > capability).
- Beklenti dürüst: fine-tune ~%30 ihtimalle 0.90; ~%70 ihtimalle 0.893'te kalır.
  Her adım gate'li → işe yaramazsa submission harcanmaz.

---

## ADIM 1 — ŞİMDİ: zero-shot band skorlama (~35 dk)

```powershell
cd C:\trendyol
git pull
python desktop_llm_judge_core.py
```

Doğru çalıştığının işareti: `170574 çift skorlanacak` + ilerleyen `X/170574` satırları.
padding zaten `left` (satır 27) — skorlar sağlıklı çıkar.

Bitince push:
```powershell
git add -f llm_core_scores.npy llm_core_rows.npy
git commit -m "core band zero-shot scores"
git push
```
Sonra **Mac oturumuna "core pushlandı" de.** Mac `patch_rankblend.py 0.3` ile bugünün
TEK submission dosyasını üretir (uç-güncelleme: yalnız hakemin en emin uçları, belirsiz
ortaya dokunulmaz). Kullanıcı 1 submission atar. w=0.5 KULLANILMAZ.

---

## ADIM 2 — ADIM 1 BİTTİKTEN SONRA: gece fine-tune (~2-3 saat, yat)

```powershell
pip install -q peft transformers accelerate
powercfg /change standby-timeout-ac 0
python overnight_lora_judge.py
```

Bu script tek başına: eğit → **held-out** doğrulama (judge_val.csv, sızıntı yok) →
band skorla. bf16 LoRA, bitsandbytes YOK, auto-resume (restart'ı atlatır).

Yatmadan önce tek kontrol (~3 dk): `loss` düşüyor mu (ilk 50-100 adımda ~0.7 → 0.3-0.4)?
Düşüyorsa sağlıklı, yat. Model yüklenemezse hemen görürsün (4 saat bekleme).

Bitince ekranda şu blok çıkar:
```
=== FINE-TUNED HAKEM İSABETİ (eşik X): genel 0.XXX ===
    pos:      0.XXX
    sameleaf: 0.XXX   <-- EN ÖNEMLİ sayı
    mined:    0.XXX
```

Push:
```powershell
git add -f llm_ft_band_scores.npy llm_ft_band_rows.npy llm_ft_val_scores.npy llm_ft_val_labels.npy
git commit -m "finetuned judge outputs"
git push
```
Sonra **Mac oturumuna `sameleaf isabeti` sayısını raporla.**

---

## ADIM 3 — YARIN (son gün, 5 submission): Mac karar verir

Karar `sameleaf isabeti`ne bağlı (Mac uygular, masaüstü sadece raporlar):
- **≥0.75:** ft-blend gönder, w=0.4'ten başla, geçerse üstüne oyna.
- **0.68-0.75:** tek atış w=0.3.
- **<0.68:** hakemi HİÇ gönderme; final = 0.893 + q27.

**Final 2 dosya (kör-dosya YASAK, ikisi de LB-ölçülmüş):**
- Hakem kazanırsa: judge-blend@q28 (ölçülü) + temiz final11_duo_q28 (0.893).
- Kazanmazsa: final11_duo_q28 (0.893) + final3_q27 (0.890).
- **17 Temmuz kapanışından ÖNCE Kaggle'da bu iki dosya ELLE işaretlenmeli.**

---

## KIRMIZI ÇİZGİLER

1. 0.893 (final11_duo_q28) ve 0.890 (final3_q27) her senaryoda korunur; üstüne çıkamazsak
   final bunlar.
2. Hiçbir dosya ÖLÇÜLMEDEN final işaretlenmez.
3. Bugün en fazla 1 (gerekirse 2) submission; gürültü kovalamak için 3.'yü harcama.
4. Gate <0.68 ise submission HARCANMAZ — bilinen etiketlerde öğrenmek bedava.
5. Adım sırası: ÖNCE band skorlama (Adım 1), SONRA fine-tune (Adım 2). Aynı anda değil.
