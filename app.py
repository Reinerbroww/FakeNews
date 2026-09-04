import streamlit as st
import joblib
import re
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "linear_svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


# =========================================================
# NLTK RESOURCES
# =========================================================

@st.cache_resource
def load_nltk_resources():

    required_resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab")
    ]

    for resource_path, download_name in required_resources:

        try:
            nltk.data.find(resource_path)

        except LookupError:
            nltk.download(
                download_name,
                quiet=True
            )

    stop_words = set(
        stopwords.words("english")
    )

    lemmatizer = WordNetLemmatizer()

    return stop_words, lemmatizer


stop_words, lemmatizer = load_nltk_resources()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        st.error(
            f"Model tidak ditemukan:\n{MODEL_PATH}"
        )

        st.stop()

    if not VECTORIZER_PATH.exists():

        st.error(
            f"TF-IDF vectorizer tidak ditemukan:\n"
            f"{VECTORIZER_PATH}"
        )

        st.stop()

    try:

        model = joblib.load(
            MODEL_PATH
        )

        vectorizer = joblib.load(
            VECTORIZER_PATH
        )

        return model, vectorizer

    except Exception as error:

        st.error(
            "Gagal memuat model atau TF-IDF vectorizer."
        )

        st.code(str(error))

        st.stop()


model, vectorizer = load_model()


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\d+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# =========================================================
# CUSTOM CSS
# =========================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --blue-950: #102B4E;
    --blue-900: #173B6C;
    --blue-800: #1D4D88;
    --blue-700: #245EA8;
    --blue-600: #2F6DB5;
    --blue-100: #DCE8F5;
    --red-700: #9C3F2E;
    --red-100: #F3DDD7;
    --background: #F3F2ED;
    --surface: #FAF9F6;
    --surface-strong: #FFFFFF;
    --text-primary: #171716;
    --text-secondary: #676762;
    --text-muted: #92928C;
    --border: #D8D7D0;
    --border-light: #E8E7E1;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(47, 109, 181, 0.07), transparent 35%),
        var(--background);
}

.main .block-container {
    max-width: 760px;
    padding-top: 3.5rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.hero {
    margin-bottom: 2.75rem;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    margin-bottom: 1.1rem;
    background: var(--blue-100);
    color: var(--blue-700);
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.hero-dot {
    width: 7px;
    height: 7px;
    background: var(--blue-600);
    border-radius: 50%;
}

.hero-title {
    margin: 0;
    font-size: 3.05rem;
    line-height: 1.05;
    letter-spacing: -0.055em;
    font-weight: 800;
    color: var(--text-primary);
}

.hero-title span {
    color: var(--blue-700);
}

.hero-description {
    max-width: 590px;
    margin-top: 1rem;
    margin-bottom: 0;
    font-size: 0.95rem;
    line-height: 1.75;
    color: var(--text-secondary);
}

.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.7rem;
}

.section-title {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--text-secondary);
}

.section-hint {
    font-size: 0.68rem;
    color: var(--text-muted);
}

.stTextArea {
    margin-bottom: 0.8rem;
}

.stTextArea textarea {
    box-sizing: border-box;
    min-height: 220px;
    padding: 1.1rem 1.15rem !important;
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.stTextArea textarea:hover {
    border-color: var(--border);
}

.stTextArea textarea:focus {
    border-color: var(--blue-600) !important;
    box-shadow: 0 0 0 3px var(--blue-100) !important;
}

.stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
    opacity: 1;
}

.stTextArea label {
    display: none;
}

.stButton > button {
    height: 48px;
    border: none !important;
    border-radius: 11px !important;
    background: var(--blue-900) !important;
    color: #FFFFFF !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.86rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.015em;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    box-shadow: 0 5px 14px rgba(23, 59, 108, 0.16);
}

.stButton > button:hover {
    background: var(--blue-700) !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(23, 59, 108, 0.22);
}

.stButton > button:active {
    transform: translateY(0);
    box-shadow: none;
}

.result-container {
    margin-top: 2.4rem;
}

.result-card {
    padding: 1.65rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.035);
}

.result-card.fake {
    border-left: 4px solid var(--red-700);
}

.result-card.real {
    border-left: 4px solid var(--blue-700);
}

.result-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.result-label {
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.result-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 0.63rem;
    font-weight: 800;
}

.result-status.fake {
    color: var(--red-700);
    background: var(--red-100);
}

.result-status.real {
    color: var(--blue-700);
    background: var(--blue-100);
}

.result-title {
    margin: 0 0 0.55rem 0;
    font-size: 1.65rem;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.035em;
}

.result-title.fake {
    color: var(--red-700);
}

.result-title.real {
    color: var(--blue-900);
}

.result-description {
    margin: 0;
    font-size: 0.86rem;
    line-height: 1.7;
    color: var(--text-secondary);
}

.result-meta {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 1.35rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border-light);
}

.meta-item {
    padding: 0.75rem;
    background: rgba(0, 0, 0, 0.018);
    border-radius: 9px;
}

.meta-label {
    display: block;
    margin-bottom: 3px;
    font-size: 0.62rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.meta-value {
    display: block;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--text-secondary);
}

.stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 1rem;
}

.stat {
    padding: 0.85rem;
    background: var(--surface);
    border: 1px solid var(--border-light);
    border-radius: 11px;
}

.stat-value {
    display: block;
    font-size: 1rem;
    font-weight: 800;
    color: var(--text-primary);
}

.stat-label {
    display: block;
    margin-top: 2px;
    font-size: 0.62rem;
    color: var(--text-muted);
}

.disclaimer {
    margin-top: 1rem;
    padding: 0.9rem 1rem;
    background: rgba(0, 0, 0, 0.025);
    border-radius: 10px;
    font-size: 0.7rem;
    line-height: 1.6;
    color: var(--text-muted);
}

.footer {
    margin-top: 4rem;
    padding-top: 1.35rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}

.footer-brand {
    font-size: 0.72rem;
    font-weight: 800;
    color: var(--text-secondary);
}

.footer-tech {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-align: right;
}

.stAlert {
    border-radius: 11px !important;
}

.stAlert > div {
    font-family: 'Manrope', sans-serif !important;
}

.stSpinner > div {
    font-family: 'Manrope', sans-serif !important;
}

@media (max-width: 600px) {
    .main .block-container {
        padding-top: 2.25rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .hero-title {
        font-size: 2.35rem;
    }
    .hero-description {
        font-size: 0.86rem;
    }
    .result-meta {
        grid-template-columns: 1fr;
    }
    .stats {
        grid-template-columns: 1fr 1fr;
    }
    .footer {
        align-items: flex-start;
        flex-direction: column;
    }
    .footer-tech {
        text-align: left;
    }
}
</style>
"""

st.html(CSS)


# =========================================================
# HERO
# =========================================================

HERO_HTML = """
<div class="hero">
    <div class="hero-badge">
        <span class="hero-dot"></span>
        AI-POWERED ANALYSIS
    </div>
    <h1 class="hero-title">
        Fake News <span>Detection</span>
    </h1>
    <p class="hero-description">
        Analyze a news headline or article and classify
        whether it is likely to be fake or authentic
        using machine learning.
    </p>
</div>
"""

st.html(HERO_HTML)


# =========================================================
# INPUT HEADER
# =========================================================

HEADER_HTML = """
<div class="section-header">
    <div class="section-title">
        News Content
    </div>
    <div class="section-hint">
        English text recommended
    </div>
</div>
"""

st.html(HEADER_HTML)


# =========================================================
# INPUT
# =========================================================

news_text = st.text_area(
    "News Content",
    height=220,
    placeholder=(
        "Paste a news headline or article here..."
    ),
    label_visibility="collapsed"
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

predict_button = st.button(
    "Analyze News",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if not news_text.strip():

        st.warning(
            "Please enter a news headline or article first."
        )

    else:

        with st.spinner("Analyzing news content..."):

            cleaned_text = preprocess_text(
                news_text
            )

            if not cleaned_text:

                st.warning(
                    "The submitted text does not contain "
                    "enough meaningful words for analysis."
                )

                st.stop()

            text_vector = vectorizer.transform(
                [cleaned_text]
            )

            prediction = model.predict(
                text_vector
            )[0]


        original_words = re.findall(
            r"\b[a-zA-Z]+\b",
            news_text
        )

        cleaned_words = cleaned_text.split()

        original_word_count = len(
            original_words
        )

        cleaned_word_count = len(
            cleaned_words
        )

        character_count = len(
            news_text.strip()
        )


        if prediction == 0:

            result_card = """
            <div class="result-card fake">
                <div class="result-top">
                    <div class="result-label">
                        Analysis Result
                    </div>
                    <div class="result-status fake">
                        ● Fake
                    </div>
                </div>
                <h2 class="result-title fake">
                    Likely Fake
                </h2>
                <p class="result-description">
                    The model classified this content as
                    <strong>fake news</strong>.
                    Consider verifying the information
                    through reliable and independent
                    sources before sharing it.
                </p>
                <div class="result-meta">
                    <div class="meta-item">
                        <span class="meta-label">
                            Prediction
                        </span>
                        <span class="meta-value">
                            Fake News
                        </span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">
                            Model
                        </span>
                        <span class="meta-value">
                            Linear SVM
                        </span>
                    </div>
                </div>
            </div>
            """

        else:

            result_card = """
            <div class="result-card real">
                <div class="result-top">
                    <div class="result-label">
                        Analysis Result
                    </div>
                    <div class="result-status real">
                        ● Authentic
                    </div>
                </div>
                <h2 class="result-title real">
                    Likely Authentic
                </h2>
                <p class="result-description">
                    The model classified this content as
                    <strong>real news</strong>.
                    However, machine learning predictions
                    should still be verified against
                    reliable sources.
                </p>
                <div class="result-meta">
                    <div class="meta-item">
                        <span class="meta-label">
                            Prediction
                        </span>
                        <span class="meta-value">
                            Real News
                        </span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">
                            Model
                        </span>
                        <span class="meta-value">
                            Linear SVM
                        </span>
                    </div>
                </div>
            </div>
            """


        RESULT_HTML = f"""
        <div class="result-container">
            {result_card}
            <div class="stats">
                <div class="stat">
                    <span class="stat-value">
                        {original_word_count}
                    </span>
                    <span class="stat-label">
                        Input Words
                    </span>
                </div>
                <div class="stat">
                    <span class="stat-value">
                        {cleaned_word_count}
                    </span>
                    <span class="stat-label">
                        Processed Words
                    </span>
                </div>
                <div class="stat">
                    <span class="stat-value">
                        {character_count}
                    </span>
                    <span class="stat-label">
                        Characters
                    </span>
                </div>
            </div>
            <div class="disclaimer">
                This prediction is generated by a machine
                learning model and should not be considered
                definitive proof of whether a news article is
                true or false. Always verify important claims
                using trusted sources.
            </div>
        </div>
        """

        st.html(RESULT_HTML)


# =========================================================
# FOOTER
# =========================================================

FOOTER_HTML = """
<div class="footer">
    <div class="footer-brand">
        Fake News Detection
    </div>
    <div class="footer-tech">
        TF-IDF · Linear SVM
    </div>
</div>
"""

st.html(FOOTER_HTML)
