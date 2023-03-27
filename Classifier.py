import streamlit as st
from transformers import pipeline

# Load pre-trained text classification model
text_classifier = pipeline("text-classification", model="text-classification", revision="text-classification-en-finetuned-sentiment")

# Create a text input box for the user to enter feedback
feedback = st.text_input('Enter your feedback:', value='The product was great, but the delivery was delayed.')

# Classify the feedback into different topics
results = text_classifier(feedback)

# Display the classification results
st.write('Classification results:')
for i, result in enumerate(results):
    NPSPlusVal = (f"Feedback {i+1}: Category: {result['label']}, Score: {result['score']}")
    st.write(NPSPlusVal)
