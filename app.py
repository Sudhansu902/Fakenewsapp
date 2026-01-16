import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")

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