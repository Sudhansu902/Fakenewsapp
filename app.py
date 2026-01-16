import streamlit as st
import os
import joblib
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

st.title("📰 Fake News Detection App")

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    st.info("Training model for first time... Please wait ⏳")

    # Public dataset mirrors (small & accessible)
    fake_url = "https://raw.githubusercontent.com/selva86/datasets/master/Fake.csv"
    true_url = "https://raw.githubusercontent.com/selva86/datasets/master/True.csv"

fake_url = "https://raw.githubusercontent.com/selva86/datasets/master/Fake.csv"
true_url = "https://raw.githubusercontent.com/selva86/datasets/master/True.csv"

fake = pd.read_csv(fake_url)
true = pd.read_csv(true_url)

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

    def clean_text(text):
        text = str(text).lower()
        text = "".join([c for c in text if c not in string.punctuation])
        return text

    data["text"] = data["text"].apply(clean_text)

    X = data["text"]
    y = data["label"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words=stopwords.words("english"))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    return model


model = load_or_train_model()

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

