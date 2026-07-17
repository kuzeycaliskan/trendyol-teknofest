# SON PLAN (17 Tem, iki-uzman denetimli) — Masaüstü Görev Talimatı

> Zirve LB 0.893 (final11_duo_q28) + sigorta 0.890 (final3_q27), ikisi ölçülü/hazır.
> 5 submission, ~10 saat kapanışa, RTX 4070 CUDA çalışıyor.
> Ana bahis: fine-tuned GENERATİF LLM judge (train_judge.py). Denetim: 3B QLoRA,
> max_steps=5000 (tam cosine anneal), veri/judge_val DEĞİŞMEZ (leak riski).
> Dürüst: P(final>0.893)~%30, P(>=0.90)~%15-20, EN OLASI SONUÇ 0.893.
> Repo private. Kapanıştan ÖNCE final 2 dosya Kaggle'da ELLE işaretlenmeli.

## H0 — ŞİMDİ: eğitimi başlat (sıfırdan, ~3.1 saat)
   git pull
   pip install -q peft transformers accelerate bitsandbytes
   python train_judge.py
   - İlk satır: ">>> 3B QLoRA" ideal (">>> 1.5B bf16 fallback" da OK).
   - DUMAN-GATE: ilk ~200 adımda loss 0.7->~0.3 düşmeli. Düşmezse Ctrl+C -> Mac'e bildir.
   - ETA kontrolü: 5000 adım için ~3.1s göstermeli. >4s ise Mac'e söyle.
   - NOT: zero-shot tight'ı GÖNDERME (AUC 0.63 < gate, submission israfı).

## H3.1 — EĞİTİM BİTİNCE: GATE (held-out sameleaf isabeti)
   Ekranda GATE bloğu çıkar. sameleaf satırına bak:
   - <0.68  : DUR. judge'ı HİÇ gönderme. Final = 0.893 + q27. Bitti (0 sub).
   - 0.68-0.72: tek temkinli atış (Sub1 = ft-blend w=0.3); <=0.893 -> mühürle.
   - >=0.72 : GO. Band zaten skorlandı, pushla:
       git add -f llm_ft_band_scores.npy llm_ft_band_rows.npy llm_ft_val_scores.npy llm_ft_val_labels.npy
       git commit -m "ft judge"; git push
     -> Mac patch_ft_blend ile Sub1 = ft-blend linear w=0.4 @q28 üretir (ana bahis).

## SUBMISSION DAĞITIMI (gate GO ise)
   Sub1 = ft-blend linear w=0.4 @q28   (ana)
   Sub1>0.893 -> Sub2=w0.6, Sub3=komşu (w0.5 veya band[0.35,0.65]),
                 Sub4=kazananın q27'si (final dosya-2 ölçümü), Sub5=rezerv
   Sub1<=0.893 -> Sub2=edge-update w0.3; hâlâ <= -> DUR

## FINAL 2 DOSYA (kör-dosya YASAK, ikisi ÖLÇÜLÜ)
   Judge kazanırsa: ft-blend@q28 (ölçülü) + temiz final11_duo_q28 (0.893, sigorta)
   Kazanmazsa:      final11_duo_q28 (0.893) + final3_q27 (0.890)
   >>> Kapanıştan ÖNCE Kaggle'da ELLE işaretle. EN KRİTİK ADIM.

## KIRMIZI ÇİZGİLER
   - 0.893 ve 0.890 her senaryoda korunur; en olası final bunlar.
   - Ölçülmemiş dosya final işaretlenmez.
   - Gate <0.68 -> submission HARCANMAZ.
   - Veri/judge_val/max_len DEĞİŞMEZ (denetim: leak + tek-değişken).
