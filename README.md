# Uygulama-1 — CNN Sınıflandırma Notebooks

Bu repo/çalışma alanında üç adet Jupyter notebook bulunmaktadır: `model_1.ipynb`, `model_2.ipynb`, `model_3.ipynb`.
Bu README, her bir notebook'un kısa açıklamasını, çalıştırma talimatlarını ve ortam gereksinimlerini içerir.

**Proje Özeti**
- Amaç: İkili görüntü sınıflandırması (binary classification) ile farklı CNN yaklaşımlarını kıyaslamak.
- Ortam: Notebooks Google Colab için hazır görünüyor (Colab'a özgü `drive.mount` kullanımı mevcut). Yerel çalışma için dataset yollarını güncellemeniz gerekir.

**Modeller (notebook bazlı özet)**
- **`model_1.ipynb` — Transfer Learning (VGG16)**:
  - Yaklaşım: `VGG16` (ImageNet ağırlıkları) taban modeli kullanılarak transfer learning.
  - Base model `trainable=False` olarak dondurulmuş; üstüne `Flatten` + Dense katmanlardan oluşan sınıflandırıcı eklenmiş.
  - Görüntü boyutu: `IMG_SIZE = 128`, `BATCH_SIZE = 32`.
  - Optimizasyon: `Adam(1e-4)`, loss = `binary_crossentropy`.
  - Eğitim: `epochs=8` (notebook içi).
  - Değerlendirme: Test setinde `evaluate`, tahmin, ve confusion matrix görselleştirmesi.

- **`model_2.ipynb` — Basit CNN (Custom)**:
  - Yaklaşım: Kendi sıralı (Sequential) küçük CNN mimarisi (Conv2D + MaxPooling katmanları), daha az filtre ile.
  - Görüntü boyutu: `IMG_SIZE = 128`, `BATCH_SIZE = 32`.
  - Optimizasyon: `Adam(1e-4)`, loss = `binary_crossentropy`.
  - Eğitim: `epochs=10`.
  - Değerlendirme: Eğitim grafikleri, test evaluate, confusion matrix.

- **`model_3.ipynb` — Gelişmiş CNN + Data Augmentation**:
  - Yaklaşım: `model_2`'ye göre daha derin ve daha fazla filtre (32→128), ek Conv katmanları.
  - Data augmentation: `rotation_range`, `width_shift_range`, `height_shift_range`, `horizontal_flip` gibi artırmalar kullanılmış.
  - Görüntü boyutu: `IMG_SIZE = 128`, `BATCH_SIZE = 64` (dikkat: `model_2`’den farklı).
  - Optimizasyon: `Adam(5e-4)` (LR arttırılıyor/azaltılıyor notlarına bakın), loss = `binary_crossentropy`.
  - Eğitim: `epochs=20`.
  - Değerlendirme: Eğitim/val grafikleri, test evaluate, confusion matrix.

**Dataset & Dosya Yolları**
- Notebooks içinde dataset yolları Google Drive üzerindeki şu dizinlere işaret ediyor:
  - `/content/drive/MyDrive/Dersler/Makine-Ogrenmesi/Odev/Uygulama-1/DataSet/train`
  - `/content/drive/MyDrive/Dersler/Makine-Ogrenmesi/Odev/Uygulama-1/DataSet/val`
  - `/content/drive/MyDrive/Dersler/Makine-Ogrenmesi/Odev/Uygulama-1/DataSet/test`
- Colab kullanıyorsanız `drive.mount('/content/drive')` satırını çalıştırın ve bu yolların mevcut olduğundan emin olun.
- Yerel çalıştırma yapacaksanız bu yolları kendi dataset klasörünüze göre değiştirin (ör. `C:\data\train`).

**Gereksinimler (Önerilen)**
- Python 3.8 veya daha yeni
- TensorFlow 2.x (notebook'larda `tensorflow.keras` kullanılıyor)
- matplotlib
- seaborn
- scikit-learn
- jupyter / jupyterlab

Örnek yükleme (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow matplotlib seaborn scikit-learn jupyterlab
```
(Not: GPU hızlandırma istiyorsanız Colab veya uygun CUDA/cuDNN kurulumuna sahip bir yerel ortam kullanın.)

**Hızlı Başlangıç — Colab**
- `model_1.ipynb` gibi dosyaları Google Colab'da açın.
- İlk hücrede `drive.mount('/content/drive')` çalıştırın.
- Dataset yollarının doğru olduğunu kontrol edin.
- Hücreleri sırayla çalıştırın.

**Hızlı Başlangıç — Yerel (Jupyter)**
- Ortamı hazırlayın (yukarıdaki gereksinimler).
- Dataset dizinlerinizi yerel yollar olarak güncelleyin.
- Terminal/Powershell'de:
```powershell
jupyter lab
# veya
jupyter notebook
```
- `model_1.ipynb`, `model_2.ipynb`, `model_3.ipynb` dosyalarını açıp hücreleri çalıştırın.

**Notlar ve İpuçları**
- Eğitimi hızlandırmak için GPU kullanın (Colab Pro veya yerel GPU).
- Eğer bellek/speed sorunları yaşarsanız `BATCH_SIZE` değerini küçültün.
- Model sonuçlarını karşılaştırırken aynı eğitim/validation split ve aynı ön işleme adımlarının kullanıldığından emin olun.
- Confusion matrix ve `accuracy` çıktıları notebook'larda mevcut; gerektiğinde `classification_report` (sklearn) ekleyebilirsiniz.

**Yazar / Referans**
- Notebook’larda yazar bilgisi: `Umutcan Kemahlı` (Okul No: `2212721050`).
- Orijinal GitHub bağlantısı (notebook içinde belirtilmiş): `https://github.com/umutckem/CNN_siniflandirma.git`
