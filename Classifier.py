import json
import streamlit as st
from transformers import pipeline

st.set_page_config(base_url='https://survey.cmix.com/8384C031/19FTK32D/en-US')


# Load pre-trained text classification model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
text_classifier = pipeline("text-classification", model=model_name)

# Create a text input box for the user to enter feedback
feedback = st.text_input('Enter your feedback:', value='The product was great, but the delivery was delayed.')

# Classify the feedback into different topics
results = text_classifier(feedback)

label = results[0]['label']
score = results[0]['score']

# Display the classification results
st.write('Classification results:')
NPSPlusVal = (f"Feedback: Category: {label}, {score}")
st.write(results)
st.write(NPSPlusVal)

st.markdown(f'<script>window.parent.postMessage({{"label": "{label}"}}, "*");</script>', unsafe_allow_html=True)


