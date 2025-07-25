import joblib
import re
import streamlit as st
from langchain.tools import tool

@st.cache_resource
def load_risk_pipeline():
    """Loads the trained risk classifier model, cached for performance."""
    return joblib.load('risk_classifier.joblib')

risk_pipeline = load_risk_pipeline()

@tool
def assess_risk(text: str) -> dict:
    """
    Assesses the mental health risk level from a text on a scale of 1 to 5.
    Use this tool for EVERY user message to determine the immediate risk.
    """
    prediction = risk_pipeline.predict([text])[0]
    print(f"Risk level - {text} : ", prediction)
    return {"risk_level": int(prediction)}

@tool
def get_tip_from_rag(topic: str) -> str:
    """
    Retrieves a pre-written, safe tip on a specific topic like 'stress', 'anxiety', or 'low_mood'.
    Use this when a user expresses mild to moderate distress (risk level 2, 3, or 4).
    """
    try:
        with open('tips.txt', 'r') as f:
            content = f.read()
        tip_key = f"[{topic.upper()}_TIP]"
        # FIX: Using a raw string (rf"...") to prevent the SyntaxWarning.
        match = re.search(rf"{re.escape(tip_key)}\n(.*?)(?=\n\\[[A-Z_]+_TIP\\]|\Z)", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "I couldn't find a specific tip for that, but remember that taking a moment for yourself is always a good idea."
    except FileNotFoundError:
        return "I'm having trouble accessing my resources right now, but I'm still here to listen."
