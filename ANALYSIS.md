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
│  2. Data Understanding ──► Dataset WELFake 72.134 artikel       │
│     │                     (Fake: 34.790 / Real: 28.836)         │
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
│  5. Feature Extraction (TF-IDF, 50.000 fitur)                   │
│     │                                                           │
│     ▼                                                           │
│  6. Modeling & Evaluation                                       │
│     ├── Logistic Regression  (94.66% accuracy)                  │
│     ├── Naive Bayes          (86.00% accuracy)                  │
│     ├── Linear SVM           (95.65% accuracy) ──► BEST MODEL   │
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

Dataset yang digunakan adalah korpus **WELFake** (`dataset/WELFake_Dataset.csv`) yang berisi **72.134 artikel berita** berbahasa Inggris dengan topik politik, dihimpun dari sumber berita asli dan sumber berita palsu.

### 2.2 Struktur Dataset

| Atribut | Tipe Data | Keterangan |
|---------|-----------|------------|
| `Unnamed: 0` | integer | Indeks baris asli (dihapus saat ingestion) |
| `title` | string | Judul artikel berita |
| `text` | string | Isi artikel berita |
| `label` | integer | 0 = Fake, 1 = Real |

### 2.3 Hasil Pemeriksaan Awal

| Pemeriksaan | Hasil |
|---|---|
| Ukuran dataset | 72.134 baris x 4 kolom |
| Missing values (`title`) | 558 artikel |
| Missing values (`text`) | 39 artikel |
| Missing values (`label`) | 0 |
| Baris dengan `title` **dan** `text` kosong sekaligus | 0 baris |
| Sel `text` berisi spasi saja | 744 sel |
| Duplikat (seluruh baris) | 0 data |
| Duplikat (title + text) pada data mentah | 0 data |
| Duplikat (title + text) setelah normalisasi | 8.458 data |
| Duplikat (title saja) | 9.795 data → 1.337 data setelah de-duplikasi |

### 2.4 Distribusi Kelas

Setelah tahap data preparation dan text preprocessing:

```
Label
0 (Fake News)  : 34.790  (54.68%)
1 (Real News)  : 28.836  (45.32%)
```

Dataset relatif seimbang (*balanced*), sehingga tidak diperlukan teknik penanganan *class imbalance* yang signifikan.

### 2.5 Catatan Karakteristik Korpus

Korpus WELFake mengandung berita politik yang provokatif dan sesekali menggunakan bahasa yang menusuk, karena menghimpun materi dari kampanye disinformasi sesungguhnya. Karakteristik ini turut membentuk kosakata yang dipelajari model.

---

## 3. Data Preparation

### 3.1 Penanganan Missing Values

Terdapat 558 artikel tanpa judul (`title`) dan 39 artikel tanpa isi (`text`), namun **tidak ada baris yang kehilangan keduanya sekaligus**. Karena itu, aturan `dropna(subset=["title", "text"], how="all")` tidak menghapus satu pun baris. Artikel dengan hanya salah satu dari keduanya **dipertahankan**, karena judul yang masih lengkap menyediakan informasi yang dapat diekstraksi.

### 3.2 Penanganan Empty Values

Terdapat 744 sel `text` yang hanya berisi spasi (kosong secara semantik). Pada tahap persiapan, kolom `title` dan `text` dinormalisasi: setiap nilai di-*trim* dari spasi berlebih, dan string kosong diubah menjadi `pd.NA` agar konsisten diperlakukan sebagai data hilang.

```python
for col in ["title", "text"]:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)
```

### 3.3 Penghapusan Duplikat

Pada data mentah tidak ditemukan duplikat pada kombinasi `title` + `text`. Namun, setelah normalisasi whitespace dan penyamaan nilai kosong di atas, **8.458 baris** menjadi identik dan dihapus (*keep first*). Duplikat berdasarkan judul saja (1.337 data setelah de-duplikasi) **dipertahankan** karena judul yang sama belum tentu memiliki isi yang sama. Setelah penanganan:

```
Ukuran dataset: (63.676, 4)
Duplikat (title + text) tersisa: 0
```

### 3.4 Penggabungan Fitur Teks

Atribut `title` dan `text` digabungkan menjadi kolom baru `full_text`:

```python
df["full_text"] = (
    df["title"].fillna("").astype(str).str.strip()
    + " "
    + df["text"].fillna("").astype(str).str.strip()
).str.strip()
```

Pada tahap ini belum ada baris dengan `full_text` kosong.

---

## 4. Text Preprocessing

Tahap preprocessing bertujuan membersihkan dan menyiapkan teks agar siap diekstrak menjadi fitur numerik.

### 4.1 Case Folding

Mengubah seluruh teks menjadi huruf kecil.

```
Sebelum : LAW ENFORCEMENT ON HIGH ALERT Following Threats Against Cops...
Sesudah : law enforcement on high alert following threats against cops...
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
Sebelum : "law enforcement on high alert following threats against cops"
Sesudah : ["law", "enforcement", "on", "high", "alert", "following", ...]
```

### 4.4 Stopword Removal

Menghapus kata-kata umum (stopword) dari NLTK English corpus: *the, is, and, of, to, in, a, it*, dsb.

```
Sebelum : ["law", "enforcement", "on", "high", "alert", "following", "threats", ...]
Sesudah : ["law", "enforcement", "high", "alert", "following", "threats", ...]
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
Sebelum (raw) : LAW ENFORCEMENT ON HIGH ALERT Following Threats Against Cops And Whites On 9-11...
Sesudah (final): law enforcement high alert following threat cop white blacklivesmatter fyf terrorist...
```

### 4.7 Penanganan Akhir

Sebanyak **50 artikel** menghasilkan `final_text` kosong (misalnya artikel yang hanya berisi URL gambar atau teks non-Latin yang seluruh karakternya hilang saat *cleaning*). Data tersebut dihapus. Dataset akhir: **63.626 artikel**.

---

## 5. Feature Extraction (TF-IDF)

### 5.1 Proses TF-IDF

`TfidfVectorizer` mengubah teks menjadi matriks numerik berdasarkan bobot *Term Frequency-Inverse Document Frequency*:

- **TF (Term Frequency):** Frekuensi kemunculan kata dalam satu dokumen.
- **IDF (Inverse Document Frequency):** Mengurangi bobot kata yang terlalu umum muncul di banyak dokumen.

Konfigurasi yang digunakan:

```python
tfidf_vectorizer = TfidfVectorizer(max_features=50000)
```

### 5.2 Hasil Ekstraksi

```
Ukuran matriks TF-IDF (training) : (50.900, 50.000)
Ukuran matriks TF-IDF (testing)  : (12.726, 50.000)
- 50.000 = jumlah fitur maksimum (kata unik terpilih)
```

### 5.3 Pembagian Data

```
Training (80%) : 50.900 artikel
Testing  (20%) : 12.726 artikel
Stratify       : proporsi label tetap terjaga
```

Distribusi label pada data testing: Fake 6.958 artikel, Real 5.768 artikel.

---

## 6. Modeling & Evaluation

Tiga algoritma klasifikasi diuji dan dibandingkan.

> **Konvensi metrik:** precision, recall, dan F1-score yang dilaporkan di bawah dihitung untuk kelas positif (`1` = Real News, `average="binary"`), sedangkan accuracy dihitung untuk seluruh data testing. Nilai per-kelas (precision, recall, F1) untuk Fake dan Real tersedia melalui *classification report*.

### 6.1 Logistic Regression (Baseline)

| Metrik | Nilai |
|--------|-------|
| Accuracy | 94.66% |
| Precision | 94.24% |
| Recall | 93.97% |
| F1-Score | 94.11% |

**Confusion Matrix:**
```
              Predicted
              Fake   Real
Actual Fake  [ 6627   331 ]
       Real  [  348  5420 ]
```
*679 artikel salah klasifikasi dari 12.726 total.*

### 6.2 Multinomial Naive Bayes

| Metrik | Nilai |
|--------|-------|
| Accuracy | 86.00% |
| Precision | 85.36% |
| Recall | 83.41% |
| F1-Score | 84.37% |

**Confusion Matrix:**
```
              Predicted
              Fake   Real
Actual Fake  [ 6133   825 ]
       Real  [  957  4811 ]
```
*1.782 artikel salah klasifikasi — performa paling rendah di antara ketiga model.*

### 6.3 Linear SVM (Model Terpilih)

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | **95.65%** |
| **Precision** | **95.04%** |
| **Recall** | **95.37%** |
| **F1-Score** | **95.21%** |

**Confusion Matrix:**
```
              Predicted
              Fake   Real
Actual Fake  [ 6672   286 ]
       Real  [  271  5497 ]
```
*Hanya 557 artikel salah klasifikasi — performa terbaik.*

### 6.4 Ringkasan Perbandingan

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 94.66% | 94.24% | 93.97% | 94.11% |
| Naive Bayes | 86.00% | 85.36% | 83.41% | 84.37% |
| **Linear SVM** | **95.65%** | **95.04%** | **95.37%** | **95.21%** |

> **Catatan variasi antar-jalankan:** confusion matrix di atas berasal dari eksekusi ulang pipeline yang identik. Solver *liblinear* tidak deterministik secara bit-per-bit, sehingga beberapa artikel batas (557 vs. 555 salah klasifikasi, selisih ±0,02 poin akurasi) dapat berpindah. Nilai utama pada tabel merupakan hasil yang tercatat pada notebook.

---

## 7. Feature Selection (Chi-Square)

Pengujian dilakukan untuk melihat apakah mengurangi jumlah fitur dapat mempertahankan performa.

### 7.1 Konfigurasi

- Metode: `SelectKBest` dengan `chi2`
- Fitur dipilih: **5.000** dari 50.000 (penurunan 90%)

### 7.2 Hasil

| Metrik | Tanpa FS | Dengan FS (5.000 fitur) |
|--------|----------|------------------------|
| Accuracy | 95.65% | 95.27% |
| Precision | 95.04% | 94.81% |
| Recall | 95.37% | 94.75% |
| F1-Score | 95.21% | 94.78% |

Feature selection mengurangi fitur drastis (90%) dengan penurunan performa hanya ~0.38 poin persentase. Namun, **model tanpa feature selection tetap dipilih** karena memberikan performa tertinggi.

---

## 8. Deployment

### 8.1 Penyimpanan Model

Model dan vectorizer disimpan menggunakan `joblib`:

```python
joblib.dump(tfidf_vectorizer, "../models/WELF_tfidf_vectorizer.pkl")
joblib.dump(svm_model, "../models/WELF_linear_svm_model.pkl")
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

Fungsi preprocessing pada aplikasi (`app.py`) dibuat identik dengan pipeline pada notebook, sehingga inferensi konsisten dengan data training. Model, vectorizer, dan resource NLTK dimuat satu kali lalu di-*cache* (`st.cache_resource`).

### 8.3 Fitur Aplikasi

- UI yang bersih dengan custom CSS (font Manrope, palet biru-merah)
- Tampilan hasil prediksi dengan kartu result (warna merah untuk *fake*, biru untuk *real*)
- Statistik: jumlah kata input, kata yang diproses, dan karakter
- Disclaimer bahwa prediksi berbasis model ML harus tetap diverifikasi

---

## Kesimpulan

| Aspek | Detail |
|-------|--------|
| Korpus | WELFake (72.134 → 63.626 artikel) |
| Model Terbaik | **Linear SVM** |
| Accuracy Tertinggi | **95.65%** |
| Total Fitur TF-IDF | 50.000 |
| Data Training/Testing | 50.900 / 12.726 (80/20 split) |
| Distribusi Kelas | 54.68% Fake / 45.32% Real |
| Framework | CRISP-DM |
| Deployment | Streamlit |
