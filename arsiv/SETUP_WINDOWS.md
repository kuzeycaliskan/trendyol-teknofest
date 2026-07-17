# Windows Masaüstü — ML Eğitim Ortamı Kurulumu

Hedef: RTX 4070'li masaüstünü cross-encoder eğitim makinesi yapmak.
Tüm komutlar **PowerShell**'de çalıştırılır (Başlat → "PowerShell" → normal kullanıcı yeterli, admin gerekmez).

---

## 1. NVIDIA sürücüsünü doğrula

```powershell
nvidia-smi
```

- Tablo görüyorsan (GPU adı: RTX 4070, Driver Version, CUDA Version ≥ 12.x) → devam.
- "not recognized" hatası alırsan: https://www.nvidia.com/drivers adresinden RTX 4070 için güncel sürücüyü kur, bilgisayarı yeniden başlat, tekrar dene.

> Not: Ayrıca CUDA Toolkit kurmana GEREK YOK — PyTorch'un pip paketi kendi CUDA kütüphanelerini getiriyor. Güncel sürücü yeterli.

## 2. Python 3.12 kur

https://www.python.org/downloads/ → **Python 3.12.x** (3.13/3.14 değil — kütüphane uyumluluğu için) → indirirken:

- ✅ **"Add python.exe to PATH"** kutusunu MUTLAKA işaretle
- Install Now

Doğrula (yeni bir PowerShell penceresi aç):

```powershell
python --version    # Python 3.12.x yazmalı
```

## 3. Proje klasörü + sanal ortam

```powershell
mkdir C:\trendyol
cd C:\trendyol
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> `Activate.ps1` "execution policy" hatası verirse:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` çalıştırıp `Y` de, sonra tekrar dene.
> Satır başında `(venv)` görünce ortam aktif demektir. (Her yeni PowerShell penceresinde `cd C:\trendyol; .\venv\Scripts\Activate.ps1` tekrarlanır.)

## 4. Kütüphaneleri kur

```powershell
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install sentence-transformers datasets accelerate pandas scikit-learn lightgbm
```

(İlk satır ~2.5 GB indirir, sabır.)

GPU'yu doğrula:

```powershell
python -c "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

Beklenen çıktı: `... | cuda: True | NVIDIA GeForce RTX 4070` — bunu görmeden devam etme.

## 5. Veriyi indir

Her şey zaten Kaggle'da olduğu için Mac'ten dosya taşımaya gerek yok:

**a) Yarışma verisi** — tarayıcıdan: yarışma sayfası → Data → **Download All** → `C:\trendyol\` içine aç. Sonuç şu klasör olmalı:
`C:\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle\` (içinde 5 CSV)

**b) Kendi dataset'lerin** — kaggle.com → profil → Your Work → Datasets:
- `trendyol-mined-negatives` → indir → `mined_hard_negatives.csv` dosyasını `C:\trendyol\` içine koy
- `trendyol-pseudo-labels` → indir → `pseudo_labels_ce2.csv` dosyasını `C:\trendyol\` içine koy

**c) Eğitim script'i** — Mac'teki `kaggle_cross_encoder_v3.py` dosyasını `C:\trendyol\` içine kopyala (AirDrop yok; en kolayı kendine e-posta/Drive/USB. İçeriği değişmeden çalışır — script veri klasörünü kendisi bulur).

Son kontrol:

```powershell
dir C:\trendyol
# görmen gerekenler: venv, trendyol-e-ticaret-yarismasi-2026-kaggle, 
# mined_hard_negatives.csv, pseudo_labels_ce2.csv, kaggle_cross_encoder_v3.py
```

## 6. Türkçe karakter güvenliği (bir kere)

```powershell
setx PYTHONUTF8 1
```

Sonra PowerShell'i kapatıp yeniden aç (ve venv'i tekrar aktive et). Bu, Türkçe karakterli çıktıların Windows konsolunda hata vermesini önler.

## 7. Eğitimi başlat

```powershell
cd C:\trendyol
.\venv\Scripts\Activate.ps1
python kaggle_cross_encoder_v3.py
```

Beklenenler:
- `Veri klasörü: trendyol-e-ticaret-yarismasi-2026-kaggle` satırı
- `[3/5] Cross-encoder eğitiliyor...` altında ilerleme çubuğu; hız kabaca **7-10 adım/sn** olmalı (Kaggle T4'te 3 idi — 4070 bu işte onun ~2.5-3 katı)
- Toplam süre tahmini: eğitim ~45-60 dk + test skorlama ~20-30 dk

Bittiğinde `C:\trendyol\` içinde oluşacaklar: `ce3_test_scores.npy`, `ce3_val_scores.npy`, `ce3_val_labels.npy`, `ce3_val_pairs.csv` (+ `ce3_model\` klasörü).

## 8. Sonuçları Mac'e taşı

4 `ce3_*` dosyasını (toplam ~11 MB; `ce3_model` klasörü GEREKMEZ) Drive/USB/e-posta ile Mac'teki proje klasörüne (`/Users/kuzey/teknofest-trendyol/` altına, örn. `kaggle_code_X/` klasörüne) kopyala ve Claude'a haber ver — harmanlama ve submission üretimi Mac'te.

---

## Sık karşılaşılabilecek sorunlar

| Belirti | Çözüm |
|---|---|
| `cuda: False` | `nvidia-smi` çalışıyor mu? PyTorch'u `--index-url ...cu124` ile kurduğundan emin ol (yanlışsa: `pip uninstall torch` sonra 4. adımı tekrarla) |
| `CUDA out of memory` | Script'te `batch_size=64` → `32` yap (eğitim biraz uzar, sonuç aynı) |
| Konsolda `UnicodeEncodeError` | 6. adım atlanmış — `setx PYTHONUTF8 1` + PowerShell'i yeniden aç |
| `Activate.ps1` engellendi | 3. adımdaki execution policy komutu |
| HF Hub indirme çok yavaş/kopuyor | Tekrar dene; kalıcıysa `pip install "huggingface_hub[hf_xet]"` |

## Bundan sonrası

Bu kurulum bir kere yapılıyor. Sonraki her deneyde akış:
1. Claude Mac'te yeni script/CSV üretir → sen `C:\trendyol\`'a kopyalarsın
2. `python <script>.py` → çıktı `.npy`/`.csv` dosyalarını Mac'e geri taşırsın
3. Harman + submission Mac'te üretilir

Kaggle notebook'a kıyasla: kota yok, oturum kopması yok, ~2.5-3x hız.
