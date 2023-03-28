import json
import streamlit as st
from transformers import pipeline
import streamlit.components.v1 as components

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

# Create a placeholder object to hold the script tag
placeholder = st.empty()

# Define the script tag
script = f"<script>var npsPlusVal = '{NPSPlusVal}';</script>"

# Cache the value of NPSPlusVal
@st.cache(allow_output_mutation=True)
def cache_nps_plus_val(val):
    return val

cached_nps_plus_val = cache_nps_plus_val(NPSPlusVal)

# Update the placeholder object with the script tag
placeholder.components.v1.html(script)

# Use cached_nps_plus_val in the HTML page
components.html(f"<p>{cached_nps_plus_val}</p>")



