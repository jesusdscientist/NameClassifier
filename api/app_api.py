# api/app_api.py
from fastapi import FastAPI
import pickle
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

app = FastAPI()

with open('../model/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../model/svm_vectorizer.pkl', 'rb') as v:
    vectorizer = pickle.load(v)


def feature_vector(name, vectorizer):
    # Use the CountVectorizer to create bigram features
    name = [" ".join(list(name))]  # To create bigrams of characters
    return vectorizer.transform(name)


@app.get('/predict/{name}')
def predict(name: str):
    prediction = model.predict(feature_vector(name, vectorizer))[0]
    return {"prediction": prediction}