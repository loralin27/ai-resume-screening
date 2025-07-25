import streamlit as st
import pickle
import spacy
import en_core_web_sm
from spacy.cli import download


# Load model and vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

st.title("AI Resume Screening System")
st.write("Upload a resume and get the predicted job category.")

uploaded_file = st.file_uploader("Upload a resume (.txt)", type=["txt"])

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode("utf-8")
    st.text_area("Resume Text", raw_text, height=250)

    # Preprocess
    cleaned_text = clean_text(raw_text)
    vector = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(vector)[0]
    st.success(f"Predicted Job Category: **{prediction}**")
