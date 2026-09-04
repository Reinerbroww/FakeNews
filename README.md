# Fake News Detection

A machine learning-powered web application for detecting fake news articles using Text Mining and Linear SVM classification.

## Overview

This project classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques and a Linear Support Vector Machine (SVM) model. It follows the **CRISP-DM** methodology for the data mining process.

### Tech Stack

- **Frontend:** Streamlit
- **ML Model:** Linear SVM (TF-IDF + Feature Selection)
- **NLP:** NLTK (tokenization, stopwords, lemmatization)
- **Language:** Python 3

## Project Structure

```
FakeNews/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── dataset/
│   └── news_dataset.csv            # Training dataset (6,000 articles)
├── models/
│   ├── linear_svm_model.pkl        # Trained Linear SVM model
│   └── tfidf_vectorizer.pkl        # TF-IDF vectorizer
└── notebook/
    └── fake_news_detection.ipynb   # Jupyter notebook (full pipeline)
```

## How It Works

1. **Text Preprocessing:** Case folding, URL/number/symbol removal, tokenization, stopword removal, and lemmatization.
2. **Feature Extraction:** TF-IDF vectorization on cleaned text.
3. **Feature Selection:** SelectKBest with chi-squared test.
4. **Classification:** Linear SVM model predicts whether the news is fake (0) or real (1).

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Reinerbroww/FakeNews.git
cd FakeNews

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Dataset

The dataset contains **6,000 news articles** with the following attributes:

| Column  | Description                      |
|---------|----------------------------------|
| title   | News article headline            |
| text    | News article body                |
| subject | Topic category                   |
| date    | Publication date                 |
| label   | 0 = Fake News, 1 = Real News    |

## License

This project is for educational and research purposes.
