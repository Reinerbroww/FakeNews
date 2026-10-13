# Fake News Detection Using Text Mining and Machine Learning

*A Capstone Research Project in Applied Natural Language Processing and Supervised Learning*

---

> **Live App:** https://fakenews-ml.streamlit.app/

## Abstract

The rapid proliferation of online disinformation has created an urgent need for automated systems capable of distinguishing authentic news content from fabricated articles. This project addresses that challenge by developing a machine learning-based classification system that determines whether a given news headline or article is likely **fake** or **authentic**. The system applies a systematic text-mining pipeline—spanning text preprocessing, statistical feature extraction, and supervised classification—under the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework. Three classification algorithms were implemented and empirically evaluated: Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Machine (SVM). The **Linear SVM** model achieved the highest predictive performance, with an accuracy of **98.56%**, a precision of **97.86%**, a recall of **99.33%**, and an F1-score of **98.59%** on a held-out test set of 1,179 articles. The final model is deployed as an interactive web application built with **Streamlit**, enabling real-time classification of user-submitted text.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Project Objectives](#project-objectives)
4. [Methodology](#methodology)
5. [Dataset](#dataset)
6. [Text Preprocessing](#text-preprocessing)
7. [Feature Extraction & Selection](#feature-extraction--selection)
8. [Modeling & Evaluation](#modeling--evaluation)
9. [Web Application](#web-application)
10. [Project Structure](#project-structure)
11. [Installation & Usage](#installation--usage)
12. [Results Summary](#results-summary)
13. [Limitations](#limitations)
14. [License & Disclaimer](#license--disclaimer)

---

## 1. Introduction

In an era in which information is disseminated across digital platforms at unprecedented velocity, the spread of fraudulent news—colloquially referred to as *fake news*—poses a significant threat to public discourse, institutional trust, and democratic processes. Manual fact-checking is neither scalable nor timely enough to counteract the volume of content generated daily. Consequently, there exists a compelling need for automated, data-driven approaches to content verification.

This project proposes a computational solution grounded in **text mining** and **machine learning**. By transforming raw textual content into structured numerical representations and training supervised classifiers on a labeled corpus, the system learns to discern the linguistic patterns that discriminate between fabricated and legitimate reporting. The principal contribution of this work is an end-to-end pipeline—from data acquisition and cleaning to model deployment—that is reproducible, extensible, and immediately usable through a web-based interface.

## 2. Problem Statement

Given an arbitrary news article or headline, the system must determine whether the content is **fake** (label `0`) or **real** (label `1`). The task is formalized as a **binary text classification problem**, in which each document is represented as a high-dimensional feature vector and assigned to one of two mutually exclusive classes by a trained classifier.

## 3. Project Objectives

- Develop a robust text-preprocessing pipeline that normalizes raw news content for downstream analysis.
- Extract meaningful statistical features using **Term Frequency–Inverse Document Frequency (TF-IDF)** vectorization.
- Train and compare multiple supervised classifiers on a balanced, labled dataset.
- Identify the optimal model on the basis of standard classification metrics (accuracy, precision, recall, and F1-score).
- Deploy the selected model in an accessible, user-friendly web application.

## 4. Methodology

The project adheres to the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology, a widely adopted framework for data science projects comprising six iterative phases:

| Phase | Activities |
|-------|------------|
| Business Understanding | Define the classification objective and evaluation criteria. |
| Data Understanding | Inspect the dataset for structure, class balance, and quality issues. |
| Data Preparation | Clean the data, remove duplicates, and engineer the text feature set. |
| Modeling | Train Logistic Regression, Multinomial Naive Bayes, and Linear SVM classifiers. |
| Evaluation | Compare model performance using classification metrics and confusion matrices. |
| Deployment | Serialize the model and expose it through a Streamlit web application. |

## 5. Dataset

### 5.1 Source & Composition

The dataset comprises **6,000 English-language news articles** stored in a CSV file (`dataset/news_dataset.csv`). The class distribution is perfectly balanced, with **3,000 fake** and **3,000 real** articles, thereby obviating the need for class-imbalance correction techniques.

### 5.2 Schema

| Attribute | Data Type | Description |
|-----------|-----------|-------------|
| `title`   | string    | News article headline |
| `text`    | string    | News article body |
| `subject` | string    | Topical category of the article |
| `date`    | string    | Publication date |
| `label`   | integer   | Class label: `0` = Fake, `1` = Real |

### 5.3 Quality Checks & Preparation

| Check | Outcome |
|-------|---------|
| Missing values | None across all attributes |
| Empty `text` values | 66 articles; retained because titles still contained information |
| Duplicate rows (title + text) | 108; removed (keeping the first occurrence) |
| Post-cleaning size | 5,892 rows |
| Final dataset after preprocessing | 5,891 articles (one row discarded for empty processed text) |

The `title` and `text` attributes were concatenated into a composite feature, `full_text`, to leverage both the headline and the body as predictive signals.

## 6. Text Preprocessing

Raw text is transformed into a canonical form through a sequential normalization pipeline:

1. **Case Folding** — All characters are lowercased.
2. **Cleaning** — URLs, numerical digits, punctuation, and non-alphabetic symbols are removed via regular expressions; excess whitespace is collapsed.
3. **Tokenization** — Text is decomposed into individual word tokens using NLTK's `word_tokenize`.
4. **Stopword Removal** — High-frequency, low-information words (e.g., *the*, *is*, *and*) are filtered using the NLTK English stopword corpus.
5. **Lemmatization** — Tokens are reduced to their dictionary base forms using `WordNetLemmatizer` (e.g., *bigots* → *bigot*).
6. **Reassembly** — Lemmatized tokens are rejoined into a single processed document (`final_text`) that serves as the input to feature extraction.

> **Example:** *"WATCH: Trump Just Told All The Anti-Gay Bigots And Mike Pence..."* → `watch trump told anti gay bigot mike penny go f ck whole lot`

## 7. Feature Extraction & Selection

### 7.1 TF-IDF Vectorization

Processed documents are converted into numerical feature vectors via **TF-IDF** weighting, which captures both the frequency of a term within a document (Term Frequency) and its rarity across the corpus (Inverse Document Frequency):

- **Matrix dimensions:** `(5,891 × 42,963)` — representing 5,891 documents and 42,963 unique vocabulary terms.
- **Data split:** 80/20 stratified train/test partition (4,712 training and 1,179 testing articles), preserving class proportions.

### 7.2 Feature Selection Experiment

An auxiliary experiment evaluated `SelectKBest` feature selection with the **chi-squared (χ²)** statistic, reducing the feature space to 5,000 terms (an 88% reduction). This yielded a modest accuracy decline (98.56% → 98.05%), indicating the pipeline's robustness. The full-feature model was ultimately retained for deployment due to its superior performance.

## 8. Modeling & Evaluation

Three supervised classifiers were trained using the extracted features and evaluated on the held-out test set.

### 8.1 Baseline: Logistic Regression

| Metric | Value |
|--------|-------|
| Accuracy | 96.95% |
| Precision | 95.62% |
| Recall | 98.50% |
| F1-Score | 97.04% |

### 8.2 Multinomial Naive Bayes

| Metric | Value |
|--------|-------|
| Accuracy | 92.11% |
| Precision | 89.41% |
| Recall | 95.83% |
| F1-Score | 92.51% |

### 8.3 Selected Model: Linear SVM

| Metric | Value |
|--------|-------|
| **Accuracy** | **98.56%** |
| **Precision** | **97.86%** |
| **Recall** | **99.33%** |
| **F1-Score** | **98.59%** |

### 8.4 Comparative Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | 96.95% | 95.62% | 98.50% | 97.04% |
| Multinomial Naive Bayes | 92.11% | 89.41% | 95.83% | 92.51% |
| **Linear SVM** | **98.56%** | **97.86%** | **99.33%** | **98.59%** |

The **Linear SVM** classifier misclassified only **17 of 1,179** test articles, outperforming both baselines across every metric and is therefore designated the production model. The trained model and its corresponding TF-IDF vectorizer are persisted as serialized artifacts (`linear_svm_model.pkl` and `tfidf_vectorizer.pkl`) via `joblib`.

## 9. Web Application

A user-facing application, built with **Streamlit**, operationalizes the trained pipeline. The application:

1. Accepts a news headline or article from the user.
2. Applies the identical preprocessing and vectorization steps used in training.
3. Produces a prediction labeled **"Likely Fake"** (label `0`) or **"Likely Authentic"** (label `1`).
4. Presents the result in a styled result card, alongside descriptive statistics (input word count, processed word count, and character count).

The interface features a clean, custom-styled user experience (Manrope typography; blue/red palette) that visually distinguishes fake from authentic predictions. The model and vectorizer are loaded once and cached for efficiency.

## 10. Project Structure

```
FakeNews/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── dataset/
│   └── news_dataset.csv            # Labeled training corpus (6,000 articles)
├── models/
│   ├── linear_svm_model.pkl        # Trained Linear SVM classifier
│   └── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
├── notebook/
│   └── fake_news_detection.ipynb   # Jupyter notebook (full pipeline)
└── ANALYSIS.md                     # Detailed technical analysis (Indonesian)
```

## 11. Installation & Usage

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Reinerbroww/FakeNews.git
cd FakeNews

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the web application
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`. Paste a news headline or article into the text area and select **Analyze News** to obtain a prediction.

## 12. Results Summary

| Aspect | Detail |
|--------|--------|
| Best-performing model | **Linear SVM** |
| Highest accuracy | **98.56%** |
| TF-IDF feature space | 42,963 features |
| Train / test split | 4,712 / 1,179 (80/20, stratified) |
| Final dataset size | 5,891 articles |
| Methodology | CRISP-DM |
| Deployment | Streamlit web application |

## 13. Limitations

- The model is trained exclusively on English-language news content and may generalize poorly to other languages or domains.
- Classifications reflect statistical patterns in the training data and are **not** definitive determinations of veracity.
- Predictions should be corroborated against reliable, independent sources before being shared.

## 14. License & Disclaimer

This project is intended for **educational and research purposes** only. It is not a substitute for professional fact-checking or journalistic standards of verification.
