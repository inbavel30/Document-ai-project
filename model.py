from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline

# -----------------------------
# TRAINING DATA (SIMPLE DEMO)
# -----------------------------
documents = [
    "This invoice contains billing and payment details",
    "This resume describes skills and work experience",
    "This is a legal contract agreement",
    "This email is about a meeting schedule"
]

labels = ["Invoice", "Resume", "Legal", "Email"]

# -----------------------------
# DOCUMENT CLASSIFICATION
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

classifier = MultinomialNB()
classifier.fit(X, labels)

def classify_document(text):
    vec = vectorizer.transform([text])
    prediction = classifier.predict(vec)[0]
    confidence = max(classifier.predict_proba(vec)[0])
    return prediction, round(confidence * 100, 2)

# -----------------------------
# BIG AI SUMMARIZATION
# -----------------------------
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    framework="pt"
)

def summarize_document(text):
    max_chunk_size = 800
    words = text.split()

    chunks = [
        " ".join(words[i:i + max_chunk_size])
        for i in range(0, len(words), max_chunk_size)
    ]

    summaries = []

    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=250,
            min_length=120,
            do_sample=False
        )
        summaries.append(summary[0]["summary_text"])

    return "\n\n".join(summaries)

