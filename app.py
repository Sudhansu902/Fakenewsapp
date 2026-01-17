import streamlit as st
import os
import joblib
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

nltk.download('stopwords')

MODEL_PATH = "model.pkl"

st.title("📰 Fake News Detection App")

@st.cache_resource
def load_or_train_model():

    # ✅ If model already exists → load it
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    st.info("Training model for first time... Please wait ⏳")

    # ✅ Download dataset automatically (no GitHub upload needed)
   
   fake = pd.read_csv("dataset/Fake.csv")
   true = pd.read_csv("dataset/True.csv") 

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])
    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    # ✅ Save trained model so next time it loads fast
    joblib.dump(model, MODEL_PATH)

    return model


# Load model
model = load_or_train_model()

news_text = st.text_area("Enter News Text:")

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([news_text])[0]

        if prediction == 1:
            st.success("✅ This news looks REAL")
        else:
            st.error("❌ This news looks FAKE")

