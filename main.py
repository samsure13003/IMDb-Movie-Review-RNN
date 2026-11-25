import re
import streamlit as st
from pathlib import Path

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.html("""
<style>

/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');


/* Global Styling */

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 50%,
        #1e1b4b 100%
    );
    color: #ffffff;
}


/* Main Container */

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* Hide Default Header */

header[data-testid="stHeader"] {
    background: transparent;
}


/* Hero Section */

.hero {
    background: linear-gradient(
        135deg,
        #312e81,
        #4f46e5,
        #7c3aed
    );

    padding: 45px 25px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;

    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
}

.hero h1 {
    color: #ffffff !important;
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 12px;
}

.hero p {
    color: #e0e7ff !important;
    font-size: 16px;
    line-height: 1.7;
}


/* Section Heading */

.section-heading {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 18px;
}


/* Information Cards */

.info-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;

    padding: 25px;
    margin-bottom: 20px;

    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.info-card h3 {
    color: #a5b4fc !important;
    font-size: 19px;
    margin-bottom: 18px;
}

.info-card p {
    color: #cbd5e1 !important;
    font-size: 14px;
    line-height: 1.9;
}


/* Text Area */

[data-testid="stTextArea"] textarea {
    background-color: #1e293b !important;
    color: #ffffff !important;

    border: 1px solid #475569 !important;
    border-radius: 12px !important;

    font-size: 15px !important;
    padding: 15px !important;
}

[data-testid="stTextArea"] textarea:focus {
    border: 1px solid #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129,140,248,0.2) !important;
}


/* Analyze Button */

.stButton button {
    width: 100%;
    height: 50px;

    background: linear-gradient(
        90deg,
        #6366f1,
        #8b5cf6
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 10px !important;

    font-size: 16px !important;
    font-weight: 700 !important;

    transition: all 0.3s ease;
}

.stButton button:hover {
    background: linear-gradient(
        90deg,
        #4f46e5,
        #7c3aed
    ) !important;

    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(99,102,241,0.3);
}


/* Result Cards */

.result-card {
    padding: 35px 20px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

.positive-card {
    background: rgba(34,197,94,0.10);
    border: 1px solid rgba(34,197,94,0.5);
}

.negative-card {
    background: rgba(239,68,68,0.10);
    border: 1px solid rgba(239,68,68,0.5);
}

.result-icon {
    font-size: 55px;
    margin-bottom: 10px;
}

.positive-title {
    color: #4ade80 !important;
    font-size: 30px;
    font-weight: 800;
}

.negative-title {
    color: #f87171 !important;
    font-size: 30px;
    font-weight: 800;
}

.result-description {
    color: #cbd5e1 !important;
    font-size: 14px;
    margin-top: 10px;
}


/* Metric Cards */

.metric-card {
    background: rgba(255,255,255,0.06);

    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;

    padding: 22px 10px;
    text-align: center;

    margin-top: 15px;
}

.metric-value {
    color: #a5b4fc !important;
    font-size: 26px;
    font-weight: 800;
}

.metric-label {
    color: #94a3b8 !important;
    font-size: 13px;
    margin-top: 8px;
}


/* Confidence Section */

.confidence-title {
    color: #ffffff !important;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    margin-top: 30px;
}

.confidence-value {
    color: #a5b4fc !important;
    font-size: 38px;
    font-weight: 800;
    text-align: center;
}


/* Footer */

.footer {
    text-align: center;
    color: #64748b !important;

    font-size: 13px;
    padding: 35px 0 10px 0;

    border-top: 1px solid rgba(255,255,255,0.1);
    margin-top: 50px;
}


/* Progress Bar */

[data-testid="stProgressBar"] > div {
    background-color: #334155;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(
        90deg,
        #6366f1,
        #a78bfa
    );
}


/* Warning and Success Messages */

.stAlert {
    border-radius: 10px;
}

</style>
""")


# =====================================================
# 3. LOAD MODEL AND DATASET
# =====================================================

@st.cache_resource
def load_sentiment_model():

    # Load IMDb word index
    word_index = imdb.get_word_index()

    # Load trained model
    model_path = Path("artifacts/simple_rnn_imdb.h5")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = load_model(model_path)

    return model, word_index


try:
    model, word_index = load_sentiment_model()

except Exception as e:

    st.error(f"Error loading model: {e}")
    st.stop()


# =====================================================
# 4. TEXT PREPROCESSING FUNCTION
# =====================================================

def preprocess_text(text, max_features=10000, maxlen=500):

    # Convert text to lowercase and remove punctuation
    words = re.findall(r"[a-z']+", text.lower())

    # IMDb start token
    encoded = [1]

    for word in words:

        idx = word_index.get(word)

        # Unknown words
        idx = idx + 3 if idx is not None else 2

        # Restrict vocabulary size
        encoded.append(
            idx if idx < max_features else 2
        )

    # Padding
    padded_sequence = sequence.pad_sequences(
        [encoded],
        maxlen=maxlen
    )

    return padded_sequence


# =====================================================
# 5. HERO SECTION
# =====================================================

st.html("""
<div class="hero">

    <h1>🎬 IMDb Sentiment Analyzer</h1>

    <p>
        Understand the emotions behind movie reviews
        using Artificial Intelligence and Deep Learning.

        <br>

        Enter your movie review and let our AI model
        predict whether the sentiment is Positive or Negative.
    </p>

</div>
""")


# =====================================================
# 6. MAIN APPLICATION LAYOUT
# =====================================================

col1, col2 = st.columns(
    [1.5, 1],
    gap="large"
)


# =====================================================
# 7. USER INPUT SECTION
# =====================================================

with col1:

    st.html(
        '<div class="section-heading">✍️ Enter Your Movie Review</div>',
    )

    user_input = st.text_area(
        "Movie Review",
        placeholder=(
            "Example: The movie was absolutely fantastic! "
            "The acting, storyline, and cinematography were amazing..."
        ),
        height=230,
        label_visibility="collapsed"
    )

    # Review statistics

    word_count = len(user_input.split())

    character_count = len(user_input)

    st.html(
        f"""
        <p style="color:#94a3b8;font-size:13px;">
            📝 Words: {word_count}
            &nbsp;&nbsp; | &nbsp;&nbsp;
            🔤 Characters: {character_count}
        </p>
        """
    )

    st.write("")

    analyze_button = st.button(
        "🚀 Analyze Sentiment",
        use_container_width=True,
        type="primary"
    )


# =====================================================
# 8. INFORMATION SECTION
# =====================================================

with col2:

    st.html("""
    <div class="info-card">

        <h3>🧠 How It Works</h3>

        <p>

        <b style="color:#a5b4fc;">01. Input Review</b>
        <br>
        Enter your movie review.

        <br><br>

        <b style="color:#a5b4fc;">02. Text Processing</b>
        <br>
        Convert text into numerical sequences.

        <br><br>

        <b style="color:#a5b4fc;">03. Deep Learning</b>
        <br>
        The trained Simple RNN analyzes your review.

        <br><br>

        <b style="color:#a5b4fc;">04. Prediction</b>
        <br>
        Get your predicted sentiment and confidence score.

        </p>

    </div>
    """)


    st.html("""
    <div class="info-card">

        <h3>📊 Model Information</h3>

        <p>

        🔹 <b>Model:</b> Simple RNN

        <br>

        🔹 <b>Dataset:</b> IMDb Movie Reviews

        <br>

        🔹 <b>Task:</b> Binary Classification

        <br>

        🔹 <b>Classes:</b> Positive / Negative

        <br>

        🔹 <b>Framework:</b> TensorFlow / Keras

        </p>

    </div>
    """)


# =====================================================
# 9. SENTIMENT PREDICTION
# =====================================================

if analyze_button:

    if not user_input.strip():

        st.warning(
            "⚠️ Please enter a movie review before analyzing."
        )

    else:

        with st.spinner(
            "🤖 AI is analyzing your movie review..."
        ):

            # Preprocess input
            preprocessed_input = preprocess_text(
                user_input
            )

            # Model prediction
            prediction = model.predict(
                preprocessed_input,
                verbose=0
            )

            score = float(prediction[0][0])

            # Determine sentiment
            sentiment = (
                "Positive" if score > 0.5 else "Negative"
            )

            # Calculate confidence
            confidence = (
                score if score > 0.5 else 1 - score
            )


        # =================================================
        # 10. RESULT SECTION
        # =================================================

        st.markdown("---")

        st.html(
            """
            <h2 style="
                text-align:center;
                color:white;
                font-weight:800;
            ">
                📈 Sentiment Analysis Results
            </h2>
            """
        )


        # Positive Result

        if sentiment == "Positive":

            st.html(
                """
                <div class="result-card positive-card">

                    <div class="result-icon">😊</div>

                    <div class="positive-title">
                        POSITIVE REVIEW
                    </div>

                    <div class="result-description">
                        The model predicts that this review
                        expresses a positive sentiment.
                    </div>

                </div>
                """
            )


        # Negative Result

        else:

            st.html(
                """
                <div class="result-card negative-card">

                    <div class="result-icon">😞</div>

                    <div class="negative-title">
                        NEGATIVE REVIEW
                    </div>

                    <div class="result-description">
                        The model predicts that this review
                        expresses a negative sentiment.
                    </div>

                </div>
                """
            )


        # =================================================
        # 11. CONFIDENCE SCORE
        # =================================================

        st.html(
            """
            <div class="confidence-title">
                🎯 Prediction Confidence
            </div>
            """
        )

        # Progress bar
        st.progress(confidence)

        # Percentage
        st.html(
            f"""
            <div class="confidence-value">
                {confidence * 100:.2f}%
            </div>
            """
        )


        # =================================================
        # 12. PERFORMANCE METRICS
        # =================================================

        st.html("<br>")

        col_a, col_b, col_c = st.columns(3)

        with col_a:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {score:.4f}
                    </div>

                    <div class="metric-label">
                        Raw Prediction Score
                    </div>

                </div>
                """
            )


        with col_b:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {confidence * 100:.2f}%
                    </div>

                    <div class="metric-label">
                        Prediction Confidence
                    </div>

                </div>
                """
            )


        with col_c:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {word_count}
                    </div>

                    <div class="metric-label">
                        Review Word Count
                    </div>

                </div>
                """
            )


# =====================================================
# 13. FOOTER
# =====================================================

st.html("""
<div class="footer">

    <p>
        Built with ❤️ using Streamlit, TensorFlow & Deep Learning
    </p>

    <p>
        IMDb Movie Review Sentiment Analysis | Simple RNN
    </p>

</div>
""")