# NİHAİ SONUÇ — 17 Tem 2026

## Kaggle public LB: 0.903 | Private LB: 68. / 364 takım (başlangıç 0.697, +0.206)
## İlk 20'ye girilemedi; finalist değiliz. Post-mortem: README.md "Öğrenilen ders".

## FINAL İŞARETLENECEK 2 DOSYA (Kaggle Submissions -> checkbox):
1. submission_7b_w50.csv          -> 0.903  (EN İYİ: 7B rank-blend w=0.5, q=0.28)
2. submission_final11_duo_q28.csv -> 0.893  (SİGORTA: judge'sız temiz ensemble, farklı eksen)

## Bugünün 7B atışları (hepsi ölçüldü):
  7b_w35 q28        0.901
  7b_w50 q28        0.903  <-- TEPE
  7b_w65 q28        0.901
  soft consensus    0.901  (3B sinyali gürültü ekledi, eklenmedi)
  7b_w50 q29        0.901
  => w-tepesi 0.5, q-tepesi 0.28, konsensüs faydasız. 0.903 kesin en iyi.

## Yolculuk: baseline 0.697 -> lgbm 0.746 -> +embed 0.767 -> ft-biencoder 0.813
   -> Türkçe-BERT CE + mined-neg 0.856 -> eşik/w kalibrasyon 0.889 -> ce ensemble+v9L 0.893
   -> zero-shot Qwen2.5-7B rank-blend 0.903
