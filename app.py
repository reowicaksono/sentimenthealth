import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

model_filenames = {
    "Bernoulli Naive Bayes": "bernoulli_naive_bayes_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl"
}

models = {name: joblib.load(filename) for name, filename in model_filenames.items()}
lbl_enc = joblib.load("label_encoder.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'http[s]?://\S+', '', text) 
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    return text

st.title("Sentiment Analysis with Machine Learning Models")
st.write("Enter some text, and the model will predict the sentiment.")

user_input = st.text_area("Enter your text:")

if st.button("Analyze Sentiment"):
    if user_input:

        processed_text = preprocess_text(user_input)
        

        text_features = vectorizer.transform([processed_text])
        
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(text_features)
            predicted_label = lbl_enc.inverse_transform(prediction)[0]
            predictions[model_name] = predicted_label
        
        st.write("Predictions:")
        for model_name, prediction in predictions.items():
            st.write(f"**{model_name}:** {prediction}")
    else:
        st.write("Please enter some text for analysis.")
