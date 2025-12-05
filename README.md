# CNN Sınıflandırma Modelleri

Bu proje, makine öğrenmesi dersi kapsamında geliştirilmiş olan üç farklı CNN (Convolutional Neural Network) tabanlı görüntü sınıflandırma modelini içermektedir.

**Öğrenci:** Umutcan Kemahlı  
**Okul Numarası:** 2212721050

---

## 📋 İçindekiler

1. [Model 1 - VGG16 Transfer Learning](#model-1---vgg16-transfer-learning)
2. [Model 2 - Basit CNN](#model-2---basit-cnn)
3. [Model 3 - Geliştirilmiş CNN](#model-3---geliştirilmiş-cnn)
4. [Karşılaştırma ve Sonuçlar](#karşılaştırma-ve-sonuçlar)
5. [Dataset Yapısı](#dataset-yapısı)

---

## Model 1 - VGG16 Transfer Learning

### 📊 Model Mimarisi

Model 1, **Transfer Learning** yaklaşımı kullanarak ImageNet üzerinde önceden eğitilmiş olan VGG16 mimarisini temel almaktadır.

#### Temel Bileşenler:

```
┌─────────────────────────────────────────────┐
│   Giriş: 128x128x3 RGB Görüntüsü            │
└────────────────┬────────────────────────────┘
                 │
┌─────────────────▼────────────────────────────┐
│   VGG16 Base Model (ImageNet Weights)       │
│   - Input Şekli: (128, 128, 3)              │
│   - Include Top: False                       │
│   - Trainable: False (Donmuş)               │
│   - Çıkış: Feature Maps                      │
└────────────────┬─────────────────────────────┘
                 │
┌─────────────────▼────────────────────────────┐
│   Flatten Layer                             │
│   - Uzatılmış vektör boyutu                 │
└────────────────┬─────────────────────────────┘
                 │
┌─────────────────▼────────────────────────────┐
│   Dense Layer 1: 512 Nöron                  │
│   - Aktivasyon: ReLU                        │
│   - Batch Normalization uygulanır           │
│   - Dropout: 0.5 (Overfitting koruma)       │
└────────────────┬─────────────────────────────┘
                 │
┌─────────────────▼────────────────────────────┐
│   Dense Layer 2: 256 Nöron                  │
│   - Aktivasyon: ReLU                        │
│   - Batch Normalization uygulanır           │
│   - Dropout: 0.4                            │
└────────────────┬─────────────────────────────┘
                 │
┌─────────────────▼────────────────────────────┐
│   Dense Layer 3: 64 Nöron                   │
│   - Aktivasyon: ReLU                        │
│   - Batch Normalization uygulanır           │
│   - Dropout: 0.3                            │
└────────────────┬─────────────────────────────┘
                 │
┌─────────────────▼────────────────────────────┐
│   Output Layer: 1 Nöron                     │
│   - Aktivasyon: Sigmoid (Binary Output)     │
│   - Çıkış: 0 veya 1 arasında değer          │
└─────────────────────────────────────────────┘
```

#### Teknik Özellikler:

| Parametreler | Değer |
|---|---|
| **Input Boyutu** | 128×128×3 |
| **Batch Size** | 32 |
| **Epochs** | 8 |
| **Optimizer** | Adam (Learning Rate: 1e-4) |
| **Loss Function** | Binary Crossentropy |
| **Toplam Katman Sayısı** | Base Model + 4 Dense Katman |

#### Model Özellikleri:

- **Transfer Learning Yaklaşımı:** VGG16'nin ImageNet üzerinde öğrendiği features kullanılır
- **Donmuş Base Model:** VGG16'nın ağırlıkları güncellenmez (trainable=False)
- **Klasifikasyon Katmanları:** Transfer learning'den sonra 3 tam bağlantılı gizli katman
- **Regularizasyon:** Batch Normalization ve Dropout yapıları overfitting'i azaltır
- **Eğitim Süresi:** Relative olarak hızlı (önceden eğitilmiş features)

#### Eğitim Yapılandırması:

```python
model.compile(
    optimizer=Adam(1e-4),      # Düşük learning rate
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

## Model 2 - Basit CNN

### 📊 Model Mimarisi

Model 2, sıfırdan eğitilen basit bir CNN modelidir. Daha az kompleks yapıda olup, daha hızlı eğitilebilir.

#### Temel Bileşenler:

```
┌──────────────────────────────────────────────┐
│   Giriş: 128x128x3 RGB Görüntüsü             │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│   Conv2D Layer 1                             │
│   - Filtre: 16, Kernel: (3×3)                │
│   - Aktivasyon: ReLU, Padding: Same          │
│   - Çıkış Boyutu: 128×128×16                 │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│   MaxPooling2D Layer 1                       │
│   - Pool Size: (2×2)                         │
│   - Çıkış Boyutu: 64×64×16                   │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│   Conv2D Layer 2                             │
│   - Filtre: 32, Kernel: (3×3)                │
│   - Aktivasyon: ReLU, Padding: Same          │
│   - Çıkış Boyutu: 64×64×32                   │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│   MaxPooling2D Layer 2                       │
│   - Pool Size: (2×2)                         │
│   - Çıkış Boyutu: 32×32×32                   │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│   Conv2D Layer 3                             │
│   - Filtre: 64, Kernel: (3×3)                │
│   - Aktivasyon: ReLU, Padding: Same          │
│   - Çıkış Boyutu: 32×32×64                   │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│   MaxPooling2D Layer 3                       │
│   - Pool Size: (2×2)                         │
│   - Çıkış Boyutu: 16×16×64                   │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│   Dropout: 0.3                               │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│   Flatten Layer                              │
│   - Vektör boyutu: 16×16×64 = 16384         │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│   Dense Layer 1: 256 Nöron                   │
│   - Aktivasyon: ReLU                         │
│   - Dropout: 0.4                             │
└───────────────┬───────────────────────────────┘
                │
┌───────────────▼───────────────────────────────┐
│   Output Layer: 1 Nöron                      │
│   - Aktivasyon: Sigmoid                      │
│   - Çıkış: Binary Classification              │
└──────────────────────────────────────────────┘
```

#### Teknik Özellikler:

| Parametreler | Değer |
|---|---|
| **Input Boyutu** | 128×128×3 |
| **Batch Size** | 32 |
| **Epochs** | 15 |
| **Optimizer** | Adam (Learning Rate: 1e-4) |
| **Loss Function** | Binary Crossentropy |
| **Convolutional Katmanlar** | 3 |
| **Dense Katmanlar** | 2 (1 hidden + 1 output) |

#### Model Özellikleri:

- **Sıfırdan Eğitim:** Model hiçbir önceden eğitilmiş ağırlığa sahip değil
- **Daha Az Parametre:** Transfer Learning modelinden daha az parametre içerir
- **3 Conv Bloğu:** Her Conv katmanı MaxPooling ile takip edilir
- **Kademeli Feature Extraction:** 16 → 32 → 64 filtre sayısı
- **Eğitim Süresi:** Model 1'den daha uzun (ağırlıklar sıfırdan öğrenilir)

#### Eğitim Yapılandırması:

```python
model2.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

## Model 3 - Geliştirilmiş CNN

### 📊 Model Mimarisi

Model 3, en kompleks yapı ile tasarlanmış CNN'dir. Daha fazla katman ve parametre içerir; ayrıca veri artırma (data augmentation) uygulanır.

#### Temel Bileşenler:

```
┌──────────────────────────────────────────────────┐
│   Giriş: 128x128x3 RGB Görüntüsü                 │
│   (Data Augmentation uygulanmış)                  │
│   - Rotation: 15°                                │
│   - Width/Height Shift: 0.1                      │
│   - Horizontal Flip: Açık                        │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   Conv2D Layer 1                                 │
│   - Filtre: 32, Kernel: (3×3)                    │
│   - Aktivasyon: ReLU, Padding: Same              │
│   - Çıkış Boyutu: 128×128×32                     │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   MaxPooling2D Layer 1                           │
│   - Pool Size: (2×2)                             │
│   - Çıkış Boyutu: 64×64×32                       │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   Conv2D Layer 2                                 │
│   - Filtre: 64, Kernel: (3×3)                    │
│   - Aktivasyon: ReLU, Padding: Same              │
│   - Çıkış Boyutu: 64×64×64                       │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   MaxPooling2D Layer 2                           │
│   - Pool Size: (2×2)                             │
│   - Çıkış Boyutu: 32×32×64                       │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   Conv2D Layer 3                                 │
│   - Filtre: 128, Kernel: (3×3)                   │
│   - Aktivasyon: ReLU, Padding: Same              │
│   - Çıkış Boyutu: 32×32×128                      │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   MaxPooling2D Layer 3                           │
│   - Pool Size: (2×2)                             │
│   - Çıkış Boyutu: 16×16×128                      │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   Conv2D Layer 4 (Yeni - Model 2'ye karşı)      │
│   - Filtre: 128, Kernel: (3×3)                   │
│   - Aktivasyon: ReLU, Padding: Same              │
│   - Çıkış Boyutu: 16×16×128                      │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   MaxPooling2D Layer 4                           │
│   - Pool Size: (2×2)                             │
│   - Çıkış Boyutu: 8×8×128                        │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   Dropout: 0.4                                   │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   Flatten Layer                                  │
│   - Vektör boyutu: 8×8×128 = 8192               │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   Dense Layer 1: 256 Nöron                       │
│   - Aktivasyon: ReLU                             │
│   - Dropout: 0.5                                 │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│   Output Layer: 1 Nöron                          │
│   - Aktivasyon: Sigmoid                          │
│   - Çıkış: Binary Classification                  │
└──────────────────────────────────────────────────┘
```

#### Teknik Özellikler:

| Parametreler | Değer |
|---|---|
| **Input Boyutu** | 128×128×3 |
| **Batch Size** | 64 (Model 2'den artırıldı) |
| **Epochs** | 20 (Model 2'den artırıldı) |
| **Optimizer** | Adam (Learning Rate: 5e-4 = 0.0005) |
| **Loss Function** | Binary Crossentropy |
| **Convolutional Katmanlar** | 4 (Model 2'ye 1 katman eklendi) |
| **Dense Katmanlar** | 2 (1 hidden + 1 output) |
| **Data Augmentation** | Evet (Rotation, Shift, Flip) |

#### Model Özellikleri:

- **Derinleştirilmiş Mimari:** 4 Conv katmanı ile daha fazla feature learning
- **Arttırılmış Filtre Sayısı:** 32 → 64 → 128 → 128
- **Data Augmentation:** Model overfitting'e karşı daha dirençli
- **Daha Yüksek Batch Size:** 32'den 64'e yükseltildi (daha stabil eğitim)
- **Düşürülmüş Learning Rate:** 1e-4'ten 5e-4'e (daha hassas adımlar)
- **Uzun Eğitim:** 20 epoch ile daha fazla iterasyon

#### Data Augmentation:

```python
train_gen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=15,          # 15 derece rotasyon
    width_shift_range=0.1,      # %10 genişlik kaydırması
    height_shift_range=0.1,     # %10 yükseklik kaydırması
    horizontal_flip=True        # Yatay çevirme
)
```

#### Eğitim Yapılandırması:

```python
model3.compile(
    optimizer=Adam(5e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

## 📊 Karşılaştırma ve Sonuçlar

### Model Karşılaştırma Tablosu

| Özellik | Model 1 (VGG16) | Model 2 (Basit CNN) | Model 3 (Geliştirilmiş CNN) |
|---|---|---|---|
| **Yaklaşım** | Transfer Learning | Scratch | Scratch + Augmentation |
| **Base Katmanlar** | VGG16 (14 layer) | - | - |
| **Conv Katmanları** | 0 (donmuş) | 3 | 4 |
| **Dense Katmanları** | 4 | 2 | 2 |
| **Toplam Parametre** | ÇOK YÜKSEK | Düşük | Orta |
| **Eğitim Süresi** | Kısa | Orta | Uzun |
| **Test Doğruluğu** | YÜKSEK | Orta | YÜKSEK |
| **Data Augmentation** | Hayır | Hayır | Evet |
| **Learning Rate** | 1e-4 | 1e-4 | 5e-4 |
| **Batch Size** | 32 | 32 | 64 |
| **Epochs** | 8 | 15 | 20 |
| **Dropout Oranı** | 0.3-0.5 | 0.3-0.4 | 0.4-0.5 |

### Model Performans Analizi

#### **Model 1 - VGG16 Transfer Learning**
✅ **Avantajları:**
- ImageNet'in gücünü kullanarak yüksek doğruluk
- Hızlı eğitim (sadece classifier kısımları eğitilir)
- Düşük veri setiyle bile iyi performans
- Overfitting riski düşük (önceden eğitilmiş features)

❌ **Dezavantajları:**
- Yüksek bellek kullanımı
- Daha az "öğrenebilir" parametre (base donmuş)
- Spesifik domain'lere tam uyum sağlayamayabilir

#### **Model 2 - Basit CNN**
✅ **Avantajları:**
- Daha az parametre (hızlı eğitim)
- Bellek açısından verimli
- Anlaşılması kolay mimari

❌ **Dezavantajları:**
- Transfer Learning kadar etkili değil
- Daha fazla eğitim verisi gerekebilir
- Overfitting riski yüksek
- Model kapasitesi sınırlı

#### **Model 3 - Geliştirilmiş CNN**
✅ **Avantajları:**
- Daha fazla katman = daha karmaşık pattern öğrenme
- Data Augmentation overfitting'i azaltır
- Daha yüksek batch size = stabil eğitim
- Model kapasitesi yeterli

❌ **Dezavantajları:**
- Daha uzun eğitim süresi
- Daha fazla parametre = bellek gereksinimi
- Overfitting riski hala mevcut
- Hiperparametre ayarı kritik

---

## 📁 Dataset Yapısı

```
DataSet/
├── train/           # Eğitim Verisi
│   ├── class_0/     # Sınıf 0 görüntüleri
│   └── class_1/     # Sınıf 1 görüntüleri
├── val/             # Validasyon Verisi
│   ├── class_0/
│   └── class_1/
└── test/            # Test Verisi
    ├── class_0/
    └── class_1/
```

### Dataset Parametreleri

| Parametre | Değer |
|---|---|
| **Resim Boyutu** | 128×128 piksel (RGB) |
| **Normalizasyon** | 1/255 (0-1 aralığına) |
| **Sınıf Sayısı** | 2 (Binary Classification) |
| **Class Mode** | Binary |
| **Batch Size** | 32 (Model 1&2) / 64 (Model 3) |

---

## 🔧 Teknik Detaylar

### Aktivasyon Fonksiyonları

- **ReLU (Rectified Linear Unit):** Hidden katmanlarda, non-lineerite ekler
- **Sigmoid:** Output katmanında, 0-1 arasında probability verir

### Optimizasyon

**Adam Optimizer:**
- Momentum ve adaptif learning rate kombinasyonu
- Eğitim sırasında dynamic learning rate ayarlaması

### Loss Function

**Binary Crossentropy:**
- 2 sınıflı sınıflandırma için ideal
- Formül: -[y·log(ŷ) + (1-y)·log(1-ŷ)]

### Regularizasyon Teknikleri

1. **Batch Normalization:** Katman çıkışlarını normalize eder
2. **Dropout:** Rastgele nöronları devre dışı bırakır
3. **Data Augmentation:** Yapay veri çeşitliliği oluşturur

---

## 📈 Eğitim Metrikleri

Tüm modeller aşağıdaki metrikler izlenerek eğitilir:

- **Accuracy:** Doğru tahminlerin oranı
- **Loss:** Model hatasının ölçüsü (azalmalı)
- **Validation Accuracy:** Validasyon setinde doğruluk
- **Validation Loss:** Validasyon setinde hata

Eğitim sırasında bu metrikler her epoch'ta hesaplanır ve grafiklerde gösterilir.

---

## 💡 Sonuçlar ve Öneriler

### Seçim Kriterleri

- **Hızlı Tahmin Gerekirse:** Model 1 (Transfer Learning)
- **Bellek Kısıtlı Ortamda:** Model 2 (Basit CNN)
- **En İyi Performans:** Model 3 (Geliştirilmiş CNN)

### İyileştirme Önerileri

1. **Hiperparametre Optimization:** Grid Search veya Random Search
2. **Ensemble Methods:** Birden fazla modeli birleştirme
3. **Daha Derin Mimariler:** ResNet, Inception, EfficientNet
4. **Özel Domain Eğitimi:** Fine-tuning with transfer learning
5. **Data Balancing:** Dengesiz veri setleri için SMOTE

---

## 🔍 Not

Tüm modeller **Google Colab** ortamında eğitilmiş olup, Google Drive'dan dataset yüklenerek çalışmaktadır.

---

**Son Güncelleme:** Aralık 2025  
**Dil:** Türkçe  
**Format:** Jupyter Notebook (.ipynb)
