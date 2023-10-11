# frontend/app_streamlit.py
import streamlit as st
import httpx

st.title('Name Classifier by Country')

name = st.text_input('Enter a name to classify:', '')

if st.button('Classify'):
    if name:
        response = httpx.get(f'http://api:8000/predict/{name}')
        prediction = response.json()['prediction']
        st.write(f'The name likely belongs to: {prediction}')