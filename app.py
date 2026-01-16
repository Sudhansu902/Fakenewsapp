import streamlit as st
import joblib

# Load model
import os
import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
from nltk.corpus import stopwords

MODEL_PATH = "model.pkl"

# Check if model.pkl exists
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Train model automatically
    fake = pd.read_csv("dataset/Fake.csv")
    true = pd.read_csv("dataset/True.csv")
    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true], axis=0).sample(frac=1).reset_index(drop=True)

    def clean_text(text):
        text = text.lower()
        text = "".join([c for c in text if c not in string.punctuation])
        return text

    data["text"] = data["text"].apply(clean_text)

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words=stopwords.words("english"))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    # Save model for next time
    joblib.dump(model, MODEL_PATH)

st.title("📰 Fake News Detection App")

news_text = st.text_area("Enter News Text:")

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([news_text])[0]
        confidence = max(model.predict_proba([news_text])[0]) * 100
        
        if prediction == 1:
            st.success(f"🟢 REAL NEWS\nConfidence: {confidence:.2f}%")
        else:

            st.error(f"🔴 FAKE NEWS\nConfidence: {confidence:.2f}%")
