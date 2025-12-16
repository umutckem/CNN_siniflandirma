# CNN Sınıflandırma Projesi - Ödev Raporu

## Öğrenci Bilgileri
- **Ad Soyad:** Umutcan Kemahlı
- **Okul Numarası:** 2212721050
- **Ders:** Makine Öğrenmesi
- **Proje:** Uygulama 1 - CNN ile Görüntü Sınıflandırma

---

## Proje Amacı

Bu projenin amacı, **kendi üretilmiş bir görüntü veri seti** kullanarak farklı derin öğrenme mimarileri ile görüntü sınıflandırma problemi çözmek ve modellerin performanslarını karşılaştırmaktır. Proje kapsamında **transfer learning**, **basit CNN mimarisi** ve **hiperparametre iyileştirmeleri + veri artırımı** adımları uygulanmıştır.

Tüm modeller **Keras (TensorFlow)** kütüphanesi kullanılarak eğitilmiştir.

Data Set Link: https://drive.google.com/drive/folders/1yo-I72TUg45VFKF2xEf93EGL-oN-robv?usp=sharing

---

## Veri Seti

* Veri setim yaprakların ölü mü yoksam canlı mı ayırt ediyor. 
* Veri seti tamamen **tarafımca çekilen görüntülerden** oluşmaktadır.
* İnternetten indirilen veya hazır veri setleri kullanılmamıştır.
* En az **2 sınıf** bulunmaktadır.
* Her sınıf için **minimum 50 adet** görüntü yer almaktadır.
* Görüntüler farklı:

  * Açılardan
  * Işık koşullarından
  * Arka planlardan
    olacak şekilde çekilmiştir.
* Tüm görüntüler **128x128 piksel** boyutuna yeniden ölçeklendirilmiştir.

### Klasör Yapısı

```
dataset/
 ├─ dead/
 │   ├─ dead_1.jpg
 │   ├─ dead_2.jpg
 │   └─ ...
 ├─ live/
 │   ├─ live_1.jpg
 │   ├─ live_2.jpg
 │   └─ ...
```

---

## Model 1 – Transfer Learning (model_1.ipynb)

Bu aşamada **state-of-the-art** bir mimari kullanılarak transfer learning uygulanmıştır.

### Kullanılan Yaklaşım

* ImageNet üzerinde önceden eğitilmiş bir mimari **VGG16** Kullanılmıştır
* Önceden eğitilmiş ağırlıklar kullanılmıştır
* Son katmanlar veri setine uygun olacak şekilde yeniden düzenlenmiştir
* Fine-tuning uygulanmıştır

### Eğitim Sonrası Çıktılar

* Eğitim doğruluk (accuracy) grafiği
* Eğitim kayıp (loss) grafiği
* Test seti doğruluğu raporlanmıştır

### Amaç

Transfer learning yaklaşımının küçük veri setlerinde sağladığı avantajı gözlemlemek.

---

## Model 2 – Basit CNN Mimarisi (model_2.ipynb)

Bu aşamada **CIFAR-10 benzeri**, sıfırdan tanımlanmış basit bir CNN modeli eğitilmiştir.

### Mimari Özellikleri

* Conv2D + MaxPooling katmanları
* Fully Connected (Dense) katmanlar
* Dropout ile overfitting önleme
* Varsayılan hiperparametreler kullanılmıştır

### Eğitim Sonrası Çıktılar

* Eğitim doğruluk ve kayıp grafikleri
* Test seti doğruluğu

### Amaç

Transfer learning kullanılmadan, temel bir CNN mimarisinin performansını gözlemlemek.

---

## Model 3 – Geliştirilmiş CNN + Veri Artırımı (model_3.ipynb)

Bu aşamada **Model 2 geliştirilmiştir**.

### Yapılan Değişiklikler

Model 2’ye kıyasla aşağıdaki hiperparametrelerden **en az 3 tanesi değiştirilmiştir**:

* Filtre sayısı artırılmış (32, 64, 128)
* Batch size 64'e çıkarılmıştır
* Öğrenme oranı (learning rate) 5e-4'e yükseltilmiştir
* Dropout oranları değiştirilmiştir (0.4 ve 0.5)
* Ek Conv katmanları eklenmiştir

### Veri Artırımı (Data Augmentation)

Eğitim sırasında **ImageDataGenerator** kullanılarak online veri artırımı uygulanmıştır:

* `rotation_range=15`
* `width_shift_range=0.1`
* `height_shift_range=0.1`
* `zoom_range=0.1`
* `horizontal_flip=True`

### Eğitim Sonrası Çıktılar

* Eğitim ve doğrulama doğruluk grafikleri
* Eğitim ve doğrulama kayıp grafikleri
* Test seti doğruluğu

### Deney Tablosu

Model 3 kapsamında yapılan tüm denemeler; kullanılan hiperparametreler ve elde edilen doğruluk değerleri ile birlikte **tablo halinde** sunulmuş ve notebook dosyasının sonuna **görsel olarak eklenmiştir**.

### Amaç

Hiperparametre optimizasyonu ve veri artırımı ile model performansını artırmak ve Model 2 ile karşılaştırmak.

---

## Genel Karşılaştırma ve Yorum

* Model 1, transfer learning sayesinde genellikle daha hızlı öğrenme ve daha yüksek doğruluk sağlamıştır.
* Model 2, sıfırdan eğitildiği için daha sınırlı performans göstermiştir.
* Model 3, yapılan hiperparametre iyileştirmeleri ve veri artırımı sayesinde Model 2’ye göre daha iyi sonuçlar vermiştir.

Model 3’ün performansı Model 2’den yüksek olup, yapılan değişikliklerin modele olumlu katkı sağladığı gözlemlenmiştir.

---

## Kullanılan Teknolojiler

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Google Colab

---
*Bu rapor, Makine Öğrenmesi dersi Uygulama 1 ödevi kapsamında hazırlanmıştır.*
