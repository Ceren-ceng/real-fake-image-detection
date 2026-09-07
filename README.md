# Gerçek / Yapay Görüntü Tespiti

Bu proje, **gerçek fotoğraflar ile yapay zekâ tarafından üretilmiş görüntüleri ayırt etmek** amacıyla geliştirilmiş bir derin öğrenme projesidir. Çalışmada hazır bir sınıflandırma modeli doğrudan kullanılmamış; bunun yerine CNN tabanlı mimariler adım adım geliştirilmiş, farklı mimari kararların başarıya ve genelleme yeteneğine etkisi incelenmiştir.

Proje boyunca yalnızca genel doğruluk değil; **5-fold cross-validation, REAL/FAKE sınıf dengesi, paired/unpaired veri yapısı, insan testi, bağımsız dış test ve sentetik görüntülere gürültü eklenmiş testler** birlikte değerlendirilmiştir.

---

## 1. Problem Tanımı

Yapay zekâ ile üretilen görüntüler günümüzde giderek daha gerçekçi hâle gelmektedir. Bu nedenle bir görüntünün gerçek bir kamera fotoğrafı mı yoksa üretici yapay zekâ tarafından oluşturulmuş sentetik bir görüntü mü olduğunu ayırt etmek zorlaşmaktadır.

Bu projede amaç:

- Bir görüntüyü **REAL** veya **FAKE** olarak sınıflandırmak,
- Farklı CNN mimarilerini karşılaştırmak,
- Modelin yalnızca eğitim/veri havuzunda değil, bağımsız görüntülerde de nasıl davrandığını görmek,
- İnsanların aynı görüntüler üzerindeki başarısıyla modeli karşılaştırmak,
- Gürültü eklenmiş sentetik görüntülerde model dayanıklılığını incelemektir.

---

## 2. Veri Kümesinin Geliştirilmesi

### İlk veri kümesi

Projenin ilk aşamasında toplam **216 görüntü** kullanıldı:

- 108 REAL
- 108 FAKE

FAKE görüntüler ChatGPT ve Gemini ile üretildi. Veri kümesinde üç farklı dosya formatı kullanıldı:

- HEIC
- JPEG
- PNG

İlk deneylerde temel RGB-CNN ile 5-fold doğrulukları yaklaşık %51–58 aralığında kaldı. Ortalama başarı yaklaşık **%54.7** oldu. Bu sonuç, veri kümesinin ve model kapasitesinin geliştirilmesi gerektiğini gösterdi.

### Veri kümesinin genişletilmesi

Vize sonrası veri kümesi büyütüldü ve yapı daha kontrollü hâle getirildi.

Ana geliştirme havuzu:

- **408 görüntü**
- 204 REAL
- 204 FAKE

FAKE sınıfı kendi içinde dengelendi:

- 102 ChatGPT
- 102 Gemini

Format dağılımı da dengelendi:

- REAL: 68 HEIC + 68 JPEG + 68 PNG
- FAKE: 68 HEIC + 68 JPEG + 68 PNG

### Paired / Unpaired yapı

Veri kümesinde iki farklı yapı birlikte kullanıldı:

**Paired veri:**  
Aynı görsel içeriğin gerçek ve yapay karşılığının bulunduğu eşli görüntüler.

- 120 REAL
- 120 FAKE
- Toplam 240 görüntü

**Unpaired veri:**  
Doğrudan eşlenmiş karşılığı bulunmayan görüntüler.

- 84 REAL
- 84 FAKE
- Toplam 168 görüntü

Böylece ana veri havuzu:

**240 paired + 168 unpaired = 408 görüntü**

olarak oluşturuldu.

### Bağımsız dış test seti

Ana 408 görüntünün dışında ayrıca:

- 12 REAL
- 12 FAKE
- Toplam **24 bağımsız test görüntüsü**

ayrıldı.

Bu test görüntüleri model eğitimi sırasında kullanılmadı.

---

## 3. Veri Ön İşleme

Model girişleri için aşağıdaki işlemler uygulandı:

- Görüntüler RGB olarak okundu.
- Görüntüler **320×320** boyutuna getirildi.
- Piksel değerleri normalize edildi.
- Eğitim sırasında sınırlı veri artırma uygulandı:
  - Yatay çevirme: `p = 0.50`
  - Küçük konum kaydırma: `±8 piksel`
- Aşırı agresif augmentation kullanılmadı.
- HEIC, JPEG ve PNG formatları birlikte desteklendi.

Modelin içerik yerine eşli görüntü ilişkisini ezberlememesi için 5-fold ayrımında **group-aware** yaklaşım kullanıldı. Aynı paired gruba ait REAL ve FAKE görüntüler aynı fold içinde tutuldu.

---

## 4. Model Eğitimi

Tüm ana modeller:

- PyTorch
- CUDA destekli NVIDIA GPU
- 5-fold cross-validation
- AdamW optimizer
- Cosine learning-rate schedule
- TTA (Test-Time Augmentation)

ile eğitildi.

Ana eğitim ayarları:

| Ayar | Değer |
|---|---:|
| Giriş boyutu | 320×320 |
| Batch size | 8 |
| Başlangıç LR | 1e-4 |
| Minimum LR | 1e-6 |
| Weight decay | 3e-4 |
| Label smoothing | 0.05 |
| Optimizer | AdamW |
| TTA | Normal + horizontal flip |

---

# 5. CNN-v3 Stronger

V3, projenin temel başarılı mimarisi oldu.

### Mimari

```text
320×320 RGB
↓
Conv + BatchNorm + ReLU
↓
Residual + ECA (64 kanal)
↓
Downsample
↓
Residual + ECA (128 kanal)
↓
Downsample
↓
Residual + ECA (256 kanal)
↓
Adaptive AvgPool 2×2
+
Adaptive MaxPool 2×2
↓
2048 özellik
↓
128
↓
Dropout 0.50
↓
32
↓
Dropout 0.25
↓
1 çıktı
```

### Kullanılan temel yapılar

**Residual bağlantı:**  
Katmanların öğrendiği özellikler doğrudan önceki özelliklerle birleştirilir. Bu yapı daha derin ağların daha kararlı öğrenmesine yardımcı olur.

**ECA (Efficient Channel Attention):**  
Modelin hangi özellik kanallarının daha önemli olduğunu öğrenmesini sağlar.

**AvgPool + MaxPool:**  
AvgPool genel özellikleri, MaxPool ise en güçlü aktivasyonları toplar. İki bilgi birlikte classifier'a aktarılır.

### Sonuç

- Parametre: yaklaşık **2.19 milyon**
- OOF TTA doğruluk: **%71.81**
- OOF NO-TTA doğruluk: **%69.12**
- ROC-AUC: **0.7095**

V3, yüksek başarı ve düşük model karmaşıklığı arasında en dengeli model oldu.

---

# 6. CNN-v4 MultiScale512

V4'te daha karmaşık bir yapı denendi.

V3 gövdesi korunurken:

- 512 kanallı hafif bottleneck,
- depthwise convolution,
- ECA,
- spatial attention,
- multi-scale feature fusion

eklendi.

Amaç, farklı ölçeklerden gelen özellikleri aynı anda kullanabilmekti.

### Sonuç

- Parametre: yaklaşık **2.63 milyon**
- OOF TTA doğruluk: **%70.83**
- OOF NO-TTA doğruluk: **%69.12**
- ROC-AUC: **0.7182**

V4 daha karmaşık olmasına rağmen V3'ü geçemedi.

Bu deney, mimari karmaşıklığın artırılmasının her zaman daha yüksek sınıflandırma doğruluğu getirmediğini gösterdi.

---

# 7. CNN-v5 Extended384

V5'te V3 yapısından kopmadan doğal bir genişletme yapıldı.

V3'ün:

`64 → 128 → 256`

kanal yapısına yeni bir:

`256 → 384`

stage eklendi.

### Mimari

```text
320×320 RGB
↓
Residual + ECA 64
↓
Residual + ECA 128
↓
Residual + ECA 256
↓
Residual + ECA 384
↓
Adaptive AvgPool 2×2
+
Adaptive MaxPool 2×2
↓
3072 özellik
↓
256
↓
Dropout 0.50
↓
64
↓
Dropout 0.25
↓
1 çıktı
```

Eğitim:

- Maximum epoch: 230
- Early stopping patience: 80
- Label smoothing: 0.05
- Diğer temel ayarlar V3 ile aynı tutuldu.

### Sonuç

- Parametre: **6,268,301**
- OOF TTA doğruluk: **%72.06**
- OOF NO-TTA doğruluk: **%69.85**
- ROC-AUC: **0.7246**

V5, ana 408 OOF testinde en yüksek sonucu aldı.

Ancak V3:

- %71.81
- 293/408 doğru

V5:

- %72.06
- 294/408 doğru

oldu.

Yani V5, yaklaşık 3 kat daha büyük olmasına rağmen V3'ü yalnızca **1 görüntü** ile geçti.

---

# 8. V3 – V4 – V5 Karşılaştırması

| Model | Parametre | OOF TTA | OOF NO-TTA | ROC-AUC |
|---|---:|---:|---:|---:|
| V3 Stronger | ~2.19M | **%71.81** | %69.12 | 0.7095 |
| V4 MultiScale512 | ~2.63M | %70.83 | %69.12 | 0.7182 |
| V5 Extended384 | **6.27M** | **%72.06** | **%69.85** | **0.7246** |

V5 en yüksek OOF skorunu vermiş olsa da, model boyutuna göre sağlanan ek kazanç oldukça sınırlı kaldı.

Bu nedenle proje sonucunda yalnızca tek bir accuracy değerine göre değil; model boyutu, bağımsız test, insan testi ve gürültü dayanıklılığı birlikte değerlendirildi.

---

# 9. İnsan Testi

Model performansını insanlarla karşılaştırmak amacıyla Google Form üzerinden bir insan testi yapıldı.

Testte:

- 24 katılımcı
- 24 görüntü
- eşli ve tekil karar görevleri

kullanıldı.

### İnsan sonuçları

- Pairwise görev: **%58.3**
- Tekil görev: **%54.5**
- Genel 18 karar: **%55.8**

### Model sonuçları — aynı görev

| Model | 18 karar |
|---|---:|
| V3 | **17/18 = %94.4** |
| V4 | 16/18 = %88.9 |
| V5 | **17/18 = %94.4** |

Ancak 24 görüntü tek tek bağımsız sınıflandırıldığında:

| Model | 24 görüntü |
|---|---:|
| V3 | **19/24 = %79.2** |
| V4 | 18/24 = %75.0 |
| V5 | 17/24 = %70.8 |

Bu sonuç, pairwise sıralama başarısı ile tek görüntü sınıflandırma başarısının aynı şey olmadığını gösterdi.

---

# 10. Bağımsız 24 Görüntü Testi

Ana 408 veri havuzunun dışında tutulan 24 görüntü ile dış test yapıldı.

V5 5-fold ensemble + TTA sonucu:

- Genel: **13/24 = %54.17**
- REAL: **11/12**
- FAKE: **2/12**
- Pairwise: **10/12**

V4 dış test:

- Genel: **13/24 = %54.17**
- REAL: 10/12
- FAKE: 3/12
- Pairwise: 11/12

Bu testte modeller özellikle FAKE görüntüleri REAL olarak sınıflandırmaya eğilim gösterdi.

Bu durum, ana veri havuzunda elde edilen OOF başarılarının bağımsız dış veri üzerinde aynı seviyede korunamadığını ve **genelleme / domain shift** probleminin bulunduğunu gösterdi.

---

# 11. Sentetik Görüntülere Gürültü Testi

Bağımsız test setindeki 12 FAKE görüntüye Gaussian noise eklendi.

Her üç model:

- Orijinal
- Hafif noise
- Orta noise

koşullarında, aynı seed ve aynı gürültü seviyeleriyle test edildi.

| Model | Orijinal | Hafif Noise | Orta Noise |
|---|---:|---:|---:|
| V3 | **3/12 (%25.0)** | **3/12 (%25.0)** | **3/12 (%25.0)** |
| V4 | 3/12 (%25.0) | 3/12 (%25.0) | 3/12 (%25.0) |
| V5 | 2/12 (%16.67) | 2/12 (%16.67) | **1/12 (%8.33)** |

Ortalama FAKE olasılıkları:

| Model | Orijinal | Hafif | Orta |
|---|---:|---:|---:|
| V3 | %38.85 | %38.48 | **%37.65** |
| V4 | %39.56 | %37.98 | %35.45 |
| V5 | %37.59 | %36.74 | %33.84 |

Sonuç:

- Üç model de bağımsız sentetik görüntülerde başlangıçta zorlandı.
- Noise arttıkça özellikle V4 ve V5'in FAKE güveni düştü.
- **V3, test edilen gürültü seviyelerinde en kararlı model oldu.**

---

# 12. Sonuç ve Tartışma

Bu projede elde edilen temel bulgular:

1. Veri kümesini büyütmek ve paired/unpaired yapıyı kullanmak, ilk temel CNN'e göre performansı belirgin şekilde artırdı.
2. Group-aware 5-fold yaklaşımı, aynı içerik ailesinin train ve validation'a sızmasını önledi.
3. V3 Stronger, yaklaşık 2.19M parametre ile güçlü ve dengeli bir temel model oldu.
4. V4'te eklenen multi-scale ve spatial attention yapıları genel doğruluğu artırmadı.
5. V5, %72.06 ile en yüksek OOF doğruluğuna ulaştı.
6. Ancak V5 yaklaşık 3 kat daha büyük olmasına rağmen V3'ü yalnızca 1 görüntüyle geçti.
7. İnsan testinde modeller, özellikle pairwise görevde insan ortalamasından daha yüksek başarı gösterdi.
8. Bağımsız dış testte modellerin FAKE görüntülere karşı genelleme problemi olduğu görüldü.
9. Gürültü testi sonucunda V3 en kararlı model oldu.
10. Daha büyük ve karmaşık modelin her durumda daha iyi olmadığı deneysel olarak görüldü.

### Genel değerlendirme

**En yüksek OOF başarı: V5 Extended384 (%72.06)**

Ancak:

**Genel denge, model boyutu ve noise kararlılığı açısından V3 Stronger en dengeli model olarak değerlendirildi.**

Bu nedenle proje yalnızca “en yüksek accuracy” değerine göre değil; genelleme, model karmaşıklığı, dış test ve dayanıklılık açısından birlikte yorumlandı.

---

# 13. Canlı Test Uygulaması

Proje için Streamlit tabanlı canlı test arayüzü geliştirildi.

Uygulama:

- JPEG
- PNG
- HEIC / HEIF

görüntülerini kabul eder.

Bir görüntü yüklendiğinde:

- REAL olasılığı
- FAKE olasılığı
- final tahmin
- 5 fold'un ayrı tahminleri

gösterilir.

Çalıştırmak için:

```bash
pip install streamlit
streamlit run live_test_REAL_FAKE_FIXED.py
```

Canlı testte varsayılan olarak **V3 Stronger** kullanılmaktadır. İstenirse V4 ve V5 de arayüz üzerinden seçilebilir.

---

# 14. Proje Dosyaları

Önerilen repository yapısı:

```text
gercek-sahte-goruntu-tespiti/
│
├── README.md
│
├── dataset/
│   ├── master_dataset_408.csv
│   ├── REAL/
│   └── FAKE/
│
├── independent_test_24/
│   ├── test_dataset_pairli.csv
│   └── images/
│
├── human_test/
│   ├── images/
│   └── results/
│
├── models/
│   ├── cnn_v3_*.py
│   ├── cnn_v4_*.py
│   └── cnn_v5_*.py
│
├── tests/
│   ├── gürültü_testi_V3_V4_V5.py
│   └── live_test_REAL_FAKE_FIXED.py
│
└── results/
    ├── V3/
    ├── V4/
    ├── V5/
    ├── human_test/
    ├── independent_test/
    └── noise_test/
```

---

# 15. Kullanılan Teknolojiler

- Python
- PyTorch
- CUDA
- NumPy
- Pandas
- Matplotlib
- Pillow
- scikit-learn
- Streamlit
- Git / GitHub

---

## Not

Bu çalışma eğitim amaçlı hazırlanmıştır. Projenin temel amacı, gerçek ve yapay görüntü ayrımı probleminde farklı CNN mimarilerini karşılaştırmak ve sonuçları yalnızca tek bir doğruluk metriği ile değil, farklı test senaryoları üzerinden değerlendirmektir.
