import json
import streamlit as st
from streamlit.components.v1 import ComponentMixin
from streamlit.components.v1 import declare_component
from transformers import pipeline

# Load pre-trained text classification model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
text_classifier = pipeline("text-classification", model=model_name)

# Define a new custom component for passing data to a JavaScript variable
NPS_COMPONENT_ID = "nps_component"

class NPSComponent(ComponentMixin):
    def __init__(self, feedback):
        self.feedback = feedback
        self.results = text_classifier(feedback)
        self.label = self.results[0]['label']
        self.score = self.results[0]['score']
        
    def _render(self):
        return json.dumps({
            "label": self.label,
            "score": self.score
        })

nps_component = declare_component(NPS_COMPONENT_ID, NPSComponent)

# Create a text input box for the user to enter feedback
feedback = st.text_input('Enter your feedback:', value='The product was great, but the delivery was delayed.')

# Call the custom component to get the classification results
results_json = nps_component(feedback)

# Parse the results JSON and extract the label and score
results = json.loads(results_json)
label = results['label']
score = results['score']

# Create a string containing the feedback category and score
NPSPlusVal = f"Feedback: Category: {label}, {score}"

# Display the classification results
st.write('Classification results:')
st.write(results)
st.write(NPSPlusVal)


