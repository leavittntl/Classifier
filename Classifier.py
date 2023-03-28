import json
import streamlit as st
from transformers import pipeline

# Load pre-trained text classification model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
text_classifier = pipeline("text-classification", model=model_name)

# Create a text input box for the user to enter feedback
feedback = st.text_input('Enter your feedback:', value='The product was great, but the delivery was delayed.')

# Classify the feedback into different topics
results = text_classifier(feedback)

# Display the classification results
st.write('Classification results:')
NPSPlusVal = (f"Feedback: Category: {results.label}, {results.score}")
st.write(results)
st.write(NPSPlusVal)

