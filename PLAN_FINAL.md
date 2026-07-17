# SON PLAN (17 Tem, güncel) — Masaüstü Görev Talimatı

> Zirve LB 0.893 (final11_duo_q28) + sigorta final3_q27 (0.890), ikisi ölçülü/hazır.
> 5 submission, ~12 saat, RTX 4070 CUDA çalışıyor. Ana bahis: fine-tuned generatif
> LLM judge (denetim: classification-head DEĞİL — LLM'in muhakemesini korur).
> Repo private kalmalı. Kapanıştan ÖNCE final 2 dosya Kaggle'da ELLE işaretlenmeli.

## ŞİMDİ — 2 iş paralel
1) Eğitim (ana bahis, ~2.5s):
   git pull
   pip install -q peft transformers accelerate bitsandbytes
   python train_judge.py
   - İlk satır: ">>> 3B QLoRA" veya ">>> 1.5B bf16 fallback" (ikisi de OK)
   - DUMAN-GATE: ilk ~200 adımda loss 0.7->0.3-0.4 düşmeli. Düşmezse Ctrl+C -> Mac'e bildir.
2) Bugünün bedava Sub1: dün biten tight skorunu pushla:
   git add -f llm_tight_scores.npy llm_tight_rows.npy; git commit -m "tight"; git push
   -> Mac patch_rankblend ile submission_rank_w30 üretir -> Sub1 gönderilir.

## EĞİTİM BİTİNCE (~H2.5)
Ekranda GATE bloğu çıkar; "sameleaf" isabetine bak:
  - >=0.72 GO: band zaten skorlandı, pushla:
      git add -f llm_ft_band_scores.npy llm_ft_band_rows.npy llm_ft_val_scores.npy llm_ft_val_labels.npy
      git commit -m "ft judge"; git push
    -> Mac patch_ft_blend ile Sub2=w0.4 üretir (ana bahis).
  - <0.68 DUR: judge'ı HİÇ gönderme. Final = 0.893 + q27. Bitti.

## SUBMISSION DAĞITIMI (gate GO ise)
  Sub1 = zero-shot tight rank-blend (yön testi, ŞİMDİ)
  Sub2 = ft-blend linear w=0.4 @q28 (ana)
  Sub3 = Sub2>0.893 ? w=0.6 : ft edge-update w=0.3
  Sub4 = kazananın komşusu
  Sub5 = kazanan ft-blend'in q27'si (final dosya-2 ölçümü) veya boş

## FINAL 2 DOSYA (kör-dosya YASAK, ikisi de ölçülü)
  Judge kazanırsa: ft-blend@q28 (ölçülü) + temiz final11_duo_q28 (0.893, sigorta)
  Kazanmazsa:      final11_duo_q28 (0.893) + final3_q27 (0.890)
  >>> Kapanıştan ÖNCE Kaggle'da ELLE işaretle.

## KIRMIZI ÇİZGİLER
  - 0.893 ve 0.890 her senaryoda korunur.
  - Ölçülmemiş dosya final işaretlenmez.
  - Gate <0.68 -> submission HARCANMAZ.
  - Dürüst: P(>0.893)~%35, P(>=0.90)~%20, en olası final 0.893.
