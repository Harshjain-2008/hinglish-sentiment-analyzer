# Hinglish sentiment analyzer

A text classifier that detects positive / negative / neutral sentiment in
Hinglish (Hindi + English code-mixed) text — the kind of language used in
Instagram/YouTube comments, casual chat, and short-form content captions.

Built as a portfolio companion to **CineMatch**: same "curated + transparent
synthetic data" philosophy, but applied to NLP/text classification instead
of recommendations.

## What it does

- **Single text mode** — type one sentence, get a sentiment label, a
  confidence score, and the specific words that pushed the model toward
  that prediction (explainability).
- **Bulk comments mode** — paste a batch of comments (e.g. copied from your
  own Reels/Shorts) and get a sentiment breakdown table + summary chart.
  Useful for getting a quick read on how an audience reacted to a post.

## How it works

1. **Dataset** (`data/generate_dataset.py`) — Public, *reliably licensed*
   labeled Hinglish sentiment datasets are scarce, so the training data is
   generated from hand-written templates + word banks covering everyday
   content-creator topics (reels, videos, exams, apps, matches, food, etc).
   This produces 660 sentences, balanced across the three classes. **This is
   clearly synthetic, not scraped real-world data** — documented here for
   transparency, same as CineMatch's synthetic ratings.
2. **Model** (`model/train_model.py`) — TF-IDF vectorizer (word 1-2 grams)
   + Logistic Regression. Chosen over a transformer because it's fast to
   train/deploy on CPU, and its coefficients are directly interpretable —
   which is what powers the "why the model thinks this" explainability
   feature in the app.
3. **App** (`app/app.py`) — Streamlit UI, dark theme to match CineMatch's
   cinematic look.

## Evaluation — and an honest caveat

On a held-out 20% split of the synthetic dataset, the model scores
**100% accuracy** (132/132 test rows correct). That number looks great but
is **misleading on its own** — the template-based sentences are formulaic
enough that a simple linear model can separate them perfectly. It's not
evidence the model has learned to handle real-world Hinglish nuance
(sarcasm, slang variation, mixed sentiment in one sentence, spelling
variants like "accha"/"acha"/"achha").

To get a more honest read, I hand-wrote 8 sentences that **don't match any
training template** and ran the trained model on them:

| Sentence (gist) | Predicted | Correct? |
|---|---|---|
| "this reel was so good, made my day" | positive | ✅ |
| "nothing good about this episode, got bored" | negative | ✅ |
| "video uploads at 5pm tomorrow" | neutral | ✅ |
| "such lovely content, won my heart" | positive | ✅ |
| "this app lags a lot, irritating" | neutral | ❌ (should be negative) |
| "lecture ran for 2 hours" | neutral | ✅ |
| "what a scene, next level energy" | positive | ✅ |
| "didn't like this trailer, didn't meet expectations" | negative | ✅ |

**7/8 (87.5%) on genuinely novel sentences** — a much more trustworthy
number than the 100% on the synthetic test split, and worth quoting if
this goes in a portfolio/resume instead of the inflated figure.

## Known limitations & next steps

- Synthetic training data → doesn't capture real slang drift, sarcasm, or
  spelling inconsistency as well as real labeled data would.
- No handling of mixed sentiment within one sentence (e.g. "acting was
  great but story was bekar").
- **v2 ideas:** collect real comments from your own Reels/Shorts and
  hand-label a few hundred for fine-tuning; try a multilingual transformer
  (e.g. a Hindi-English code-mixed BERT variant) for a harder benchmark;
  add a confusion-matrix view in the app itself.

## Running it locally

```bash
pip install -r requirements.txt

# (optional) regenerate the dataset and retrain — already done, artifacts included
python data/generate_dataset.py
python model/train_model.py

# launch the app
streamlit run app/app.py
```

## Project structure

```
hinglish-sentiment/
├── data/
│   ├── generate_dataset.py     # builds the curated synthetic dataset
│   └── sentiment_dataset.csv   # 660 labeled sentences
├── model/
│   ├── train_model.py          # trains + evaluates + saves the model
│   ├── vectorizer.joblib       # fitted TF-IDF vectorizer
│   ├── classifier.joblib       # fitted Logistic Regression model
│   └── metrics.json            # accuracy, classification report, confusion matrix
├── app/
│   └── app.py                  # Streamlit app
├── requirements.txt
└── README.md
```
