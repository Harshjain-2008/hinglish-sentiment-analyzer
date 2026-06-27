"""
generate_dataset.py
--------------------
Builds a curated, template-based Hinglish (Hindi+English code-mixed) sentiment
dataset for the Hinglish Sentiment Analyzer project.

WHY SYNTHETIC/TEMPLATE-BASED:
Public labeled Hinglish sentiment corpora are scarce and inconsistent in
licensing. To keep this project fully reproducible and transparent (same
philosophy as CineMatch's curated + synthetic data approach), we generate
sentences from hand-written templates and word banks covering everyday
content-creator/social-media topics (reels, videos, exams, apps, food,
matches, etc.), then combine them programmatically for variety.

This is clearly documented (not pretending to be "real world scraped data")
so anyone reviewing the portfolio project understands exactly how the
training data was produced and can extend it with real labeled data later.
"""

import csv
import random

random.seed(42)

SUBJECTS = [
    "yeh movie", "yeh video", "tera reel", "woh episode", "naya song",
    "yeh trailer", "tumhara channel", "is content", "yeh vlog", "yeh tutorial",
    "lecture", "exam paper", "kal ka match", "team ka performance", "yeh app",
    "naya project", "is restaurant ka khana", "yeh phone", "college ka event",
    "yeh series", "uska gaana", "is product ka review", "naya update",
    "yeh assignment", "internship experience", "hostel ka room", "yeh class",
    "training session", "presentation", "interview",
]

POS_ADJ_HI = [
    "mast", "zabardast", "badhiya", "ekdum sahi", "bohot accha",
    "dhamakedaar", "full paisa vasool", "kamaal", "next level",
]
POS_ADJ_EN = [
    "amazing", "awesome", "fantastic", "brilliant", "superb",
    "incredible", "outstanding", "perfect", "top notch",
]

NEG_ADJ_HI = [
    "bekar", "bakwaas", "ganda", "faltu", "ekdum bura", "bore karne wala",
    "time waste", "third class", "bilkul bekaar",
]
NEG_ADJ_EN = [
    "terrible", "awful", "disappointing", "boring", "pathetic",
    "useless", "horrible", "underwhelming", "frustrating",
]

NEUTRAL_FACTS = [
    "{s} kal release hua tha.",
    "{s} ka duration 10 minute ka hai.",
    "{s} aaj subah upload hua.",
    "{s} next week schedule hai.",
    "{s} abhi process ho raha hai.",
    "{s} ke baare mein notification aaya.",
    "{s} ka link description mein hai.",
    "{s} 3 baje start hoga.",
    "{s} ka second part bhi available hai.",
    "{s} office ke system pe save hai.",
    "{s} group mein share kiya gaya.",
    "{s} ka format change hua hai.",
]

POS_TEMPLATES = [
    "Yaar {s} {a} hai, dil khush ho gaya!",
    "Bro {s} is absolutely {a_en}, loved it!",
    "{s} dekh ke bohot khushi hui, total {a}!",
    "Itna {a} {s} maine pehle nahi dekha, hats off!",
    "{s} ka content full {a} tha, mast laga!",
    "Honestly {s} was {a_en}, highly recommend.",
    "{s} ne expectations cross kar diye, ekdum {a}!",
    "What a {a_en} {s}, proud feeling aa raha hai.",
    "{s} bohot {a} nikla, sabko bata diya maine.",
    "Thoroughly enjoyed {s}, {a_en} effort by everyone.",
]

NEG_TEMPLATES = [
    "Yaar {s} bilkul {a} tha, time hi waste hua.",
    "Honestly {s} was {a_en}, very disappointed.",
    "{s} dekh ke mood off ho gaya, {a} experience.",
    "{s} itna {a} nikla, expect nahi kiya tha.",
    "Bro {s} is {a_en}, not worth it at all.",
    "{s} ka level {a} tha, sudhaar ki zarurat hai.",
    "What a {a_en} {s}, complete waste of time.",
    "{s} bohot {a} laga mujhe, na dekho bekar mein.",
    "Frankly {s} was {a_en}, would not recommend.",
    "{s} se bilkul bhi satisfied nahi hoon, {a} tha.",
]


def gen_rows(n_per_class=220):
    rows = []
    seen = set()

    def add(text, label):
        text = " ".join(text.split())  # collapse whitespace
        if text not in seen:
            seen.add(text)
            rows.append((text, label))

    # Positive
    while sum(1 for _, l in rows if l == "positive") < n_per_class:
        s = random.choice(SUBJECTS)
        t = random.choice(POS_TEMPLATES)
        a = random.choice(POS_ADJ_HI)
        a_en = random.choice(POS_ADJ_EN)
        add(t.format(s=s, a=a, a_en=a_en), "positive")

    # Negative
    while sum(1 for _, l in rows if l == "negative") < n_per_class:
        s = random.choice(SUBJECTS)
        t = random.choice(NEG_TEMPLATES)
        a = random.choice(NEG_ADJ_HI)
        a_en = random.choice(NEG_ADJ_EN)
        add(t.format(s=s, a=a, a_en=a_en), "negative")

    # Neutral
    while sum(1 for _, l in rows if l == "neutral") < n_per_class:
        s = random.choice(SUBJECTS)
        t = random.choice(NEUTRAL_FACTS)
        add(t.format(s=s.capitalize()), "neutral")

    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = gen_rows(n_per_class=220)
    out_path = "data/sentiment_dataset.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    from collections import Counter
    print(Counter(l for _, l in rows))
