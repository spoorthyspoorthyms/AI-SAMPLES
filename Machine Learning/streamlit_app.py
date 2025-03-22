import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib  # For saving and loading the model

# Function to train and save the model (run once)
def train_and_save_model():
    data = pd.read_csv("spam.csv", encoding='latin1')
    data = data[['v1', 'v2']]
    data = data.rename(columns={'v1': 'label', 'v2': 'text'})
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    vectorizer = TfidfVectorizer()
    X_vectors = vectorizer.fit_transform(data['text'])
    y = data['label']

    model = MultinomialNB()
    model.fit(X_vectors, y)

    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(model, 'spam_model.joblib')

# Load the trained model and vectorizer
try:
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('spam_model.joblib')
except FileNotFoundError:
    st.warning("Model files not found. Training model. This will only happen once.")
    train_and_save_model()
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('spam_model.joblib')

# Streamlit app
st.title("Spam Email Classifier")

email_text = st.text_area("Enter the email text:")

if st.button("Classify"):
    if email_text:
        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]

        if prediction == 1:
            st.error("This email is likely spam.")
        else:
            st.success("This email is likely not spam (ham).")
    else:
        st.warning("Please enter email text.")

# Add a section to retrain the model.
if st.checkbox("Retrain Model (Use with caution!)"):
    if st.button("Retrain"):
        st.warning("Retraining model. This may take some time.")
        train_and_save_model()
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        model = joblib.load('spam_model.joblib')
        st.success("Model retrained successfully.")