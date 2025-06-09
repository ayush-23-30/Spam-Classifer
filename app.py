import pickle
import pandas as pd 
import numpy as np 
import streamlit as st 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'), encoding='latin1')
model = pickle.load(open('model.pkl', 'rb'), encoding='utf-8')

st.title('Email/SMS spam Classifier');

input_sms = st.text_area('Enter The Email')

if st.button('Predict'):
    # 1. Preprocess
    tranformed_sms = transform_text(input_sms)
    # 2. vectorize 
    vector_input = tfidf.transform([tranformed_sms])
    # 3. Predict 
    result = model.predict(vector_input)[0]
    # 4. Display

    if result == 1:
        st.header('This Text is a Spam')
    elif result == 0 :
        st.header('This is Not a Spam Text')