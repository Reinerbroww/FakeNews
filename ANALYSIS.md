# Analisis Lengkap Fake News Detection

Dokumen ini menjelaskan secara lengkap alur kerja (*pipeline*) proyek Fake News Detection, mulai dari pemahaman data hingga deployment model.

---

## Alur Kerja (Pipeline)

```
┌─────────────────────────────────────────────────────────────────┐
│                        CRISP-DM PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Business Understanding                                      │
│     │                                                           │
│     ▼                                                           │
│  2. Data Understanding ──► Dataset 6.000 artikel berita         │
│     │                     (Fake: 3.000 / Real: 3.000)           │
│     ▼                                                           │
│  3. Data Preparation                                            │
│     ├── Missing values handling                                 │
│     ├── Empty values handling                                   │
│     ├── Duplicate removal                                       │
│     └── Feature selection (title + text → full_text)            │
│     │                                                           │
│     ▼                                                           │
│  4. Text Preprocessing                                          │
│     ├── Case Folding                                            │
│     ├── Text Cleaning                                           │
│     ├── Tokenization                                            │
│     ├── Stopword Removal                                        │
│     ├── Lemmatization                                           │
│     └── Final text assembly                                     │
│     │                                                           │
│     ▼                                                           │
│  5. Feature Extraction (TF-IDF)                                 │
│     │                                                           │
│     ▼                                                           │
│  6. Modeling & Evaluation                                       │
│     ├── Logistic Regression  (96.95% accuracy)                  │
│     ├── Naive Bayes          (92.11% accuracy)                  │
│     ├── Linear SVM           (98.56% accuracy) ──► BEST MODEL   │
│     └── Feature Selection (Chi-Square) experiment               │
│     │                                                           │
│     ▼                                                           │
│  7. Deployment (Streamlit App)                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Business Understanding

**Tujuan:** Membangun sistem klasifikasi berita palsu (*fake news*) menggunakan pendekatan *text mining* dan *machine learning*.

**Label:**
- `0` = Fake News
- `1` = Real News

**Metodologi:** Proses pengembangan mengikuti framework **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

---

## 2. Data Understanding

### 2.1 Sumber Dataset

Dataset tersimpan dalam format CSV (`news_dataset.csv`) dengan **6.000 artikel berita** berbahasa Inggris.

### 2.2 Struktur Dataset

| Atribut | Tipe Data | Keterangan |
|---------|-----------|------------|
| `title` | string | Judul artikel berita |
| `text` | string | Isi artikel berita |
| `subject` | string | Kategori topik berita |
| `date` | string | Tanggal publikasi |
| `label` | integer | 0 = Fake, 1 = Real |

### 2.3 Hasil Pemeriksaan Awal

| Pemeriksaan | Hasil |
|---|---|
| Ukuran dataset | 6.000 baris x 5 kolom |
| Missing values | 0 pada seluruh kolom |
| Empty values (`text`) | 66 artikel kosong |
| Duplikat (title + text) | 108 data |
| Duplikat (seluruh baris) | 5 data |

### 2.4 Distribusi Kelas

```
Label
0 (Fake News)  : 3.000  (50%)
1 (Real News)  : 3.000  (50%)
```

Dataset seimbang (*balanced*), sehingga tidak diperlukan teknik penanganan *class imbalance*.

### 2.5 Distribusi Kategori (Subject)

```
politicsNews       1.564
worldnews          1.436
News               1.175
politics             870
left-news            573
Government News      185
Middle-east          100
US_News               97
```

---

## 3. Data Preparation

### 3.1 Penanganan Missing Values

Seluruh atribut tidak memiliki nilai `null`, sehingga tidak diperlukan penanganan tambahan.

### 3.2 Penanganan Empty Values

Terdapat 66 artikel dengan kolom `text` kosong. Karena judul (`title`) masih mengandung informasi, artikel tersebut **tidak dihapus** dan kolom `text` dibiarkan kosong.

### 3.3 Penghapusan Duplikat

Data duplikat dihapus berdasarkan kombinasi `title` + `text` (*keep first*). Setelah penanganan:

```
Ukuran dataset: (5.892, 5)
Duplikat tersisa: 0
```

### 3.4 Penggabungan Fitur Teks

Atribut `title` dan `text` digabungkan menjadi kolom baru `full_text`:

```python
df["full_text"] = (df["title"] + " " + df["text"]).str.strip()
```

---

## 4. Text Preprocessing

Tahap preprocessing bertujuan membersihkan dan menyiapkan teks agar siap diekstrak menjadi fitur numerik.

### 4.1 Case Folding

Mengubah seluruh teks menjadi huruf kecil.

```
Sebelum : WATCH: Trump Just Told All The Anti-Gay Bigots...
Sesudah : watch: trump just told all the anti gay bigots...
```

### 4.2 Text Cleaning

Menghapus elemen yang tidak diperlukan menggunakan regex:

| Langkah | Pola Regex | Fungsi |
|---------|-----------|--------|
| Hapus URL | `http\S+\|www\S+\|https\S+` | Menghapus tautan web |
| Hapus angka | `\d+` | Menghapus semua digit |
| Hapus simbol | `[^a-zA-Z\s]` | Menghapus tanda baca & karakter khusus |
| Hapus spasi berlebih | `\s+` | Merapikan spasi ganda |

### 4.3 Tokenization

Memecah teks menjadi array kata (*tokens*) menggunakan `nltk.word_tokenize()`.

```
Sebelum : "watch trump just told all the anti gay bigots"
Sesudah : ["watch", "trump", "just", "told", "all", "the", "anti", "gay", "bigots"]
```

### 4.4 Stopword Removal

Menghapus kata-kata umum (stopword) dari NLTK English corpus: *the, is, and, of, to, in, a, it*, dsb.

```
Sebelum : ["watch", "trump", "just", "told", "all", "the", "anti", "gay", "bigots"]
Sesudah : ["watch", "trump", "told", "anti", "gay", "bigots"]
```

### 4.5 Lemmatization

Mengubah kata ke bentuk dasarnya (*lemma*) menggunakan `WordNetLemmatizer`.

```
Sebelum : ["bigots", "voters", "rights", "reasons", "christians"]
Sesudah : ["bigot", "voter", "right", "reason", "christian"]
```

### 4.6 Final Text Assembly

Token hasil lemmatization digabungkan kembali menjadi teks utuh (`final_text`) sebagai input TF-IDF.

```
Sebelum (raw) : WATCH: Trump Just Told All The Anti-Gay Bigots And Mike Pence...
Sesudah (final): watch trump told anti gay bigot mike penny go f ck whole lot...
```

### 4.7 Penanganan Akhir

Satu artikel yang hanya berisi URL gambar menghasilkan `final_text` kosong dan dihapus. Dataset akhir: **5.891 artikel**.

---

## 5. Feature Extraction (TF-IDF)

### 5.1 Proses TF-IDF

`TfidfVectorizer` mengubah teks menjadi matriks numerik berdasarkan bobot *Term Frequency-Inverse Document Frequency*:

- **TF (Term Frequency):** Frekuensi kemunculan kata dalam satu dokumen.
- **IDF (Inverse Document Frequency):** Mengurangi bobot kata yang terlalu umum muncul di banyak dokumen.

### 5.2 Hasil Ekstraksi

```
Ukuran matriks TF-IDF : (5.891, 42.963)
- 5.891 = jumlah artikel
- 42.963 = jumlah fitur (kata unik)
```

### 5.3 Pembagian Data

```
Training (80%) : 4.712 artikel
Testing  (20%) : 1.179 artikel
Stratify       : proporsi label tetap seimbang
```

---

## 6. Modeling & Evaluation

Tiga algoritma klasifikasi diuji dan dibandingkan:

### 6.1 Logistic Regression (Baseline)

| Metrik | Nilai |
|--------|-------|
| Accuracy | 96.95% |
| Precision | 95.62% |
| Recall | 98.50% |
| F1-Score | 97.04% |

**Confusion Matrix:**
```
              Predicted
              Fake   Real
Actual Fake  [ 553    27 ]
       Real  [   9   590 ]
```
*36 artikel salah klasifikasi dari 1.179 total.*

### 6.2 Multinomial Naive Bayes

| Metrik | Nilai |
|--------|-------|
| Accuracy | 92.11% |
| Precision | 89.41% |
| Recall | 95.83% |
| F1-Score | 92.51% |

**Confusion Matrix:**
```
              Predicted
              Fake   Real
Actual Fake  [ 512    68 ]
       Real  [  25   574 ]
```
*93 artikel salah klasifikasi — performa paling rendah di antara ketiga model.*

### 6.3 Linear SVM (Model Terpilih)

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | **98.56%** |
| **Precision** | **97.86%** |
| **Recall** | **99.33%** |
| **F1-Score** | **98.59%** |

**Confusion Matrix:**
```
              Predicted
              Fake   Real
Actual Fake  [ 567    13 ]
       Real  [   4   595 ]
```
*Hanya 17 artikel salah klasifikasi — performa terbaik.*

### 6.4 Ringkasan Perbandingan

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 96.95% | 95.62% | 98.50% | 97.04% |
| Naive Bayes | 92.11% | 89.41% | 95.83% | 92.51% |
| **Linear SVM** | **98.56%** | **97.86%** | **99.33%** | **98.59%** |

---

## 7. Feature Selection (Chi-Square)

Pengujian dilakukan untuk melihat apakah mengurangi jumlah fitur dapat mempertahankan performa.

### 7.1 Konfigurasi

- Metode: `SelectKBest` dengan `chi2`
- Fitur dipilih: **5.000** dari 42.963

### 7.2 Hasil

| Metrik | Tanpa FS | Dengan FS (5.000 fitur) |
|--------|----------|------------------------|
| Accuracy | 98.56% | 98.05% |
| Precision | 97.86% | 97.37% |
| Recall | 99.33% | 98.83% |
| F1-Score | 98.59% | 98.09% |

Feature selection mengurangi fitur drastis (88%) dengan penurunan performa hanya ~0.5%. Namun, **model tanpa feature selection tetap dipilih** karena memberikan performa tertinggi.

---

## 8. Deployment

### 8.1 Penyimpanan Model

Model dan vectorizer disimpan menggunakan `joblib`:

```python
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model_svm, "linear_svm_model.pkl")
```

### 8.2 Streamlit Application

Aplikasi web dibangun menggunakan **Streamlit** dengan alur kerja:

```
Input Teks (user)
      │
      ▼
Text Preprocessing
(case fold → clean → tokenize → stopword → lemmatize)
      │
      ▼
TF-IDF Vectorization
(menggunakan vectorizer yang sudah dilatih)
      │
      ▼
Linear SVM Prediction
      │
      ▼
Output: "Likely Fake" / "Likely Authentic"
```

### 8.3 Fitur Aplikasi

- UI yang bersih dengan custom CSS (font Manrope, palet biru-merah)
- Tampilan hasil prediksi dengan kartu result (warna merah untuk *fake*, biru untuk *real*)
- Statistik: jumlah kata input, kata yang diproses, dan karakter
- Disclaimer bahwa prediksi berbasis model ML harus tetap diverifikasi

---

## Kesimpulan

| Aspek | Detail |
|-------|--------|
| Model Terbaik | **Linear SVM** |
| Accuracy Tertinggi | **98.56%** |
| Total Fitur TF-IDF | 42.963 |
| Data Training/Testing | 4.712 / 1.179 (80/20 split) |
| Dataset Akhir | 5.891 artikel |
| Framework | CRISP-DM |
| Deployment | Streamlit |
