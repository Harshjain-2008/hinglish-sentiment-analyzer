"""
train_model.py
---------------
Trains a TF-IDF + Logistic Regression sentiment classifier on the curated
Hinglish dataset, evaluates it, and saves the fitted vectorizer + model so
the Streamlit app can load them instantly (no retraining at runtime).

Why TF-IDF + Logistic Regression (not a transformer):
- Fast to train and run on CPU, easy to deploy on Streamlit Community Cloud.
- Coefficients are directly interpretable -> lets the app show WHICH words
  pushed a prediction towards positive/negative (explainability), which a
  black-box transformer would need extra tooling (e.g. SHAP) for.
- A natural "v2" upgrade path is swapping in a multilingual transformer
  (e.g. a code-mixed/Hindi-English fine-tuned model) once a larger, real
  labeled dataset is available - noted in the README.
"""

import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = "hinglish-sentiment/data/sentiment_dataset.csv"
MODEL_DIR = "hinglish-sentiment/model"

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    token_pattern=r"(?u)\b\w+\b",  # keep short Hinglish tokens
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000, C=5.0)
clf.fit(X_train_vec, y_train)

preds = clf.predict(X_test_vec)
acc = accuracy_score(y_test, preds)
report = classification_report(y_test, preds, output_dict=True)
cm = confusion_matrix(y_test, preds, labels=clf.classes_)

print(f"\nTest accuracy: {acc:.3f}\n")
print(classification_report(y_test, preds))
print("Confusion matrix (rows=true, cols=pred), labels:", list(clf.classes_))
print(cm)

# Save model artifacts
joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.joblib")
joblib.dump(clf, f"{MODEL_DIR}/classifier.joblib")

metrics = {
    "accuracy": acc,
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "labels": list(clf.classes_),
    "n_train": len(X_train),
    "n_test": len(X_test),
}
with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved vectorizer, classifier, and metrics.json to {MODEL_DIR}")
