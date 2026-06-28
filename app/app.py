"""
app.py
------
Hinglish Sentiment Analyzer - Streamlit app.

Two modes:
1. Single text: type/paste one sentence, get sentiment + confidence +
   the top words that pushed the prediction (explainability).
2. Bulk comments: paste multiple lines (e.g. comments copied from your
   own Reels/Shorts) and get a sentiment breakdown table + summary chart.
"""

import os
import re
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

st.set_page_config(
    page_title="Hinglish Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
)

# ---------- Styling (dark, consistent with CineMatch's cinematic theme) ----------
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #f5f5f5; }
    .badge {
        display: inline-block; padding: 4px 14px; border-radius: 8px;
        font-weight: 600; font-size: 0.85rem; letter-spacing: 0.02em;
    }
    .badge-positive { background: #1d9e75; color: #eafff5; }
    .badge-negative { background: #d85a30; color: #fff3ee; }
    .badge-neutral  { background: #444441; color: #f1efe8; }
    .word-pos { background: #1d9e75; color: #eafff5; padding: 2px 6px; border-radius: 4px; margin: 2px; display:inline-block; }
    .word-neg { background: #d85a30; color: #fff3ee; padding: 2px 6px; border-radius: 4px; margin: 2px; display:inline-block; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Hinglish Sentiment Analyzer")
st.caption(
    "Type a Hinglish/English sentence or paste a batch of comments "
    "(e.g. from your own Reels/Shorts) to see the predicted sentiment."
)


@st.cache_resource
def load_model():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
    classifier = joblib.load(os.path.join(MODEL_DIR, "classifier.joblib"))
    return vectorizer, classifier


vectorizer, classifier = load_model()
BADGE_CLASS = {"positive": "badge-positive", "negative": "badge-negative", "neutral": "badge-neutral"}


def predict_one(text):
    vec = vectorizer.transform([text])
    probs = classifier.predict_proba(vec)[0]
    classes = classifier.classes_
    pred_idx = probs.argmax()
    label = classes[pred_idx]
    confidence = probs[pred_idx]
    return label, confidence, dict(zip(classes, probs))


def explain(text, label):
    """Return tokens in `text` that have the strongest learned weight
    towards the predicted class, for a simple explainability view."""
    if label == "neutral":
        return [], []
    class_idx = list(classifier.classes_).index(label)
    coefs = classifier.coef_[class_idx]
    vocab = vectorizer.vocabulary_

    tokens = re.findall(r"\w+", text.lower())
    scored = []
    for tok in set(tokens):
        if tok in vocab:
            scored.append((tok, coefs[vocab[tok]]))
    scored.sort(key=lambda x: x[1], reverse=(label == "positive"))
    top_pos = [t for t, s in scored if s > 0][:5]
    top_neg = [t for t, s in scored if s < 0][:5]
    return top_pos, top_neg


mode = st.radio("Mode", ["Single text", "Bulk comments"], horizontal=True)

if mode == "Single text":
    text = st.text_area(
        "Your sentence",
        placeholder="Yaar yeh reel ekdum mast bani hai, dil khush ho gaya!",
        height=100,
    )
    if st.button("Analyze", type="primary") and text.strip():
        label, confidence, probs = predict_one(text)
        st.markdown(
            f'<span class="badge {BADGE_CLASS[label]}">{label.upper()}</span> '
            f'&nbsp; confidence: **{confidence:.0%}**',
            unsafe_allow_html=True,
        )
        st.write("")
        prob_df = pd.DataFrame({"sentiment": list(probs.keys()), "probability": list(probs.values())})
        st.bar_chart(prob_df.set_index("sentiment"))

        pos_words, neg_words = explain(text, label)
        if pos_words or neg_words:
            st.write("**Why the model thinks this:**")
            html = "".join(f'<span class="word-pos">{w}</span>' for w in pos_words)
            html += "".join(f'<span class="word-neg">{w}</span>' for w in neg_words)
            st.markdown(html, unsafe_allow_html=True)

else:
    bulk_text = st.text_area(
        "Paste comments, one per line",
        placeholder=(
            "bhai ye episode bohot mast tha\n"
            "itna boring video kabhi nahi dekha\n"
            "next part kab aa raha hai?"
        ),
        height=160,
    )
    if st.button("Analyze all", type="primary") and bulk_text.strip():
        lines = [l.strip() for l in bulk_text.split("\n") if l.strip()]
        results = []
        for line in lines:
            label, confidence, _ = predict_one(line)
            results.append({"comment": line, "sentiment": label, "confidence": f"{confidence:.0%}"})
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.write("**Summary**")
        counts = df["sentiment"].value_counts()
        st.bar_chart(counts)
        total = len(df)
        pos_pct = (df["sentiment"] == "positive").mean()
        neg_pct = (df["sentiment"] == "negative").mean()
        st.write(
            f"{total} comments analyzed - "
            f"{pos_pct:.0%} positive, {neg_pct:.0%} negative."
        )

st.divider()
with st.expander("About this model"):
    st.write(
        "Trained on a curated, template-generated Hinglish dataset (660 sentences, "
        "balanced across positive/negative/neutral) using TF-IDF features and "
        "Logistic Regression. See README.md for full methodology, evaluation, "
        "and known limitations."
    )
