# Fake News Detection Using Text Mining and Machine Learning

*A Capstone Research Project in Applied Natural Language Processing and Supervised Learning*

---

> **Live App:** https://fakenews-ml.streamlit.app/

## Abstract

The rapid proliferation of online disinformation has created an urgent need for automated systems capable of distinguishing authentic news content from fabricated articles. This project addresses that challenge by developing a machine learning-based classification system that determines whether a given news headline or article is likely **fake** or **authentic**. The system applies a systematic text-mining pipeline—spanning text preprocessing, statistical feature extraction, and supervised classification—under the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework. The experiments are conducted on the **WELFake** corpus, a large-scale collection of 72,134 English-language political news articles. Three classification algorithms were implemented and empirically evaluated: Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Machine (SVM). The **Linear SVM** model achieved the highest predictive performance, with an accuracy of **95.65%**, a precision of **95.04%**, a recall of **95.37%**, and an F1-score of **95.21%** on a held-out test set of 12,726 articles. The final model is deployed as an interactive web application built with **Streamlit**, enabling real-time classification of user-submitted text.

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
- Train and compare multiple supervised classifiers on a large, labeled corpus.
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

The corpus is the **WELFake** dataset (`dataset/WELFake_Dataset.csv`), a public collection of **72,134 English-language political news articles** harvested from real and fake news sources. The distribution is mildly imbalanced toward the fake class, with **34,790 fake** and **28,836 real** articles after preprocessing — a ratio of roughly 54.7% / 45.3%. The imbalance is mild enough that no class-rebalancing strategy was required.

### 5.2 Schema

| Attribute | Data Type | Description |
|-----------|-----------|-------------|
| `Unnamed: 0` | integer   | Original row index (dropped during ingestion) |
| `title`   | string    | News article headline |
| `text`    | string    | News article body |
| `label`   | integer   | Class label: `0` = Fake, `1` = Real |

### 5.3 Quality Checks & Preparation

| Check | Outcome |
|-------|---------|
| Raw dataset size | 72,134 rows × 4 columns |
| Missing `title` | 558 articles |
| Missing `text` | 39 articles |
| Missing `label` | 0 |
| Rows with **both** `title` and `text` missing | 0 — no row is removed by the missing-value rule |
| Whitespace-only `text` cells | 744 — normalized to `pd.NA` during preparation |
| Duplicate rows (all columns) | 0 |
| Duplicate rows (title + text) | 0 on the raw read; **8,458 after** whitespace/empty-string normalization |
| Duplicate titles only | 9,795 before de-duplication; 1,337 remain afterwards and are retained, as identical headlines may carry different bodies |
| After de-duplication | 63,676 rows |
| Articles with empty processed text | 50 — discarded (non-textual or non-Latin content) |
| Final dataset after preprocessing | **63,626 articles** |

The `title` and `text` attributes were concatenated into a composite feature, `full_text`, to leverage both the headline and the body as predictive signals.

## 6. Text Preprocessing

Raw text is transformed into a canonical form through a sequential normalization pipeline:

1. **Case Folding** — All characters are lowercased.
2. **Cleaning** — URLs, numerical digits, punctuation, and non-alphabetic symbols are removed via regular expressions; excess whitespace is collapsed.
3. **Tokenization** — Text is decomposed into individual word tokens using NLTK's `word_tokenize`.
4. **Stopword Removal** — High-frequency, low-information words (e.g., *the*, *is*, *and*) are filtered using the NLTK English stopword corpus.
5. **Lemmatization** — Tokens are reduced to their dictionary base forms using `WordNetLemmatizer` (e.g., *bigots* → *bigot*).
6. **Reassembly** — Lemmatized tokens are rejoined into a single processed document (`final_text`) that serves as the input to feature extraction.

> **Example:** *"LAW ENFORCEMENT ON HIGH ALERT Following Threats Against Cops And Whites..."* → `law enforcement high alert following threat cop white blacklivesmatter fyf terrorist video comment expected barack obama member fyf fukyoflag ...`

The deployed application reproduces this pipeline **exactly**, character for character, so that inference-time text is transformed identically to training-time text.

## 7. Feature Extraction & Selection

### 7.1 TF-IDF Vectorization

Processed documents are converted into numerical feature vectors via **TF-IDF** weighting, which captures both the frequency of a term within a document (Term Frequency) and its rarity across the corpus (Inverse Document Frequency):

- **Vectorizer:** `TfidfVectorizer(max_features=50_000)` — unigrams, L2 normalization, smooth IDF.
- **Matrix dimensions:** `(50,900 × 50,000)` — representing 50,900 training documents and 50,000 selected vocabulary terms.
- **Data split:** 80/20 stratified train/test partition (50,900 training and 12,726 testing articles), preserving class proportions.

### 7.2 Feature Selection Experiment

An auxiliary experiment evaluated `SelectKBest` feature selection with the **chi-squared (χ²)** statistic, reducing the feature space from 50,000 to 5,000 terms (a 90% reduction). This yielded a modest accuracy decline (95.65% → 95.27%), indicating the pipeline's robustness. The full-feature model was ultimately retained for deployment due to its superior performance.

## 8. Modeling & Evaluation

Three supervised classifiers were trained using the extracted features and evaluated on the held-out test set of 12,726 articles.

> **Metric convention:** as implemented in the research notebook, the reported precision, recall, and F1-score are computed for the positive class (`1` = Real News, `average="binary"`); the accuracy is computed over the full test set. Per-class precision/recall/F1 for both classes is additionally reported through the classification report.

### 8.1 Baseline: Logistic Regression

| Metric | Value |
|--------|-------|
| Accuracy | 94.66% |
| Precision | 94.24% |
| Recall | 93.97% |
| F1-Score | 94.11% |

### 8.2 Multinomial Naive Bayes

| Metric | Value |
|--------|-------|
| Accuracy | 86.00% |
| Precision | 85.36% |
| Recall | 83.41% |
| F1-Score | 84.37% |

### 8.3 Selected Model: Linear SVM

| Metric | Value |
|--------|-------|
| **Accuracy** | **95.65%** |
| **Precision** | **95.04%** |
| **Recall** | **95.37%** |
| **F1-Score** | **95.21%** |

### 8.4 Comparative Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | 94.66% | 94.24% | 93.97% | 94.11% |
| Multinomial Naive Bayes | 86.00% | 85.36% | 83.41% | 84.37% |
| **Linear SVM** | **95.65%** | **95.04%** | **95.37%** | **95.21%** |

**Confusion matrices** (rows = true class, columns = predicted class):

```
Logistic Regression              Linear SVM (deployed)
              Predicted                     Predicted
              Fake   Real                  Fake   Real
Actual Fake  [ 6627   331 ]        Actual Fake  [ 6672   286 ]
       Real  [  348  5420 ]               Real  [  271  5497 ]
```

The **Linear SVM** classifier misclassified only **557 of 12,726** test articles, outperforming both baselines across every metric, and is therefore designated the production model. The trained model and its corresponding TF-IDF vectorizer are persisted as serialized artifacts (`WELF_linear_svm_model.pkl` and `WELF_tfidf_vectorizer.pkl`) via `joblib`.

> **Note on run-to-run variance:** the confusion matrices above come from an independent re-execution of the identical pipeline. The `liblinear` dual solver is not bit-deterministic, so a re-run shifts two or three borderline articles (557 vs. 555 misclassifications, i.e. ±0.02 accuracy points). The headline metrics (95.65% accuracy) are the values recorded in the research notebook.

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
├── app.py                          # Streamlit web application (deployed model)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (English)
├── ANALYSIS.md                     # Detailed technical analysis (Indonesian)
├── dataset/
│   ├── WELFake_Dataset.csv         # WELFake corpus (72,134 rows) — active
│   └── news_dataset.csv            # Legacy corpus (6,000 rows) — retained for reference
├── models/
│   ├── WELF_linear_svm_model.pkl   # Deployed Linear SVM classifier (WELFake)
│   ├── WELF_tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer (WELFake)
│   ├── linear_svm_model.pkl        # Legacy classifier (news_dataset)
│   └── tfidf_vectorizer.pkl        # Legacy vectorizer (news_dataset)
├── notebook/
│   └── fake_news_detection.ipynb   # Jupyter notebook (full pipeline)
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

> **Note on the corpus:** `WELFake_Dataset.csv` is ~245 MB and is therefore not committed to the repository. Download the WELFake dataset from Kaggle (`saurabh00007/welfake-dataset`) and place it in `dataset/`. The serialized model and vectorizer are already included, so the web application runs without the raw corpus.

The application requires the NLTK corpora `stopwords`, `wordnet`, `punkt`, and `punkt_tab`; these are fetched automatically on first launch.

## 12. Results Summary

| Aspect | Detail |
|--------|--------|
| Corpus | WELFake (72,134 articles → 63,626 after preprocessing) |
| Best-performing model | **Linear SVM** |
| Highest accuracy | **95.65%** |
| TF-IDF feature space | 50,000 features |
| Train / test split | 50,900 / 12,726 (80/20, stratified) |
| Class distribution | 54.7% Fake / 45.3% Real |
| Methodology | CRISP-DM |
| Deployment | Streamlit web application |

## 13. Limitations

- The model is trained exclusively on English-language political news content and may generalize poorly to other languages or domains.
- The WELFake corpus contains provocative and occasionally offensive language, since it aggregates material from real disinformation campaigns; this affects the vocabulary learned by the model.
- Because the classifier was trained on concatenated title-and-body documents, **headline-only** inputs are out of distribution and noticeably less reliable (roughly 68% accuracy on titles alone versus 95%+ on full articles). The application is most accurate when given a complete article.
- Classifications reflect statistical patterns in the training data and are **not** definitive determinations of veracity.
- Predictions should be corroborated against reliable, independent sources before being shared.

## 14. License & Disclaimer

This project is intended for **educational and research purposes** only. It is not a substitute for professional fact-checking or journalistic standards of verification.
