import joblib
import re
import streamlit as st
from langchain.tools import tool
from transformers import pipeline

@st.cache_resource
def load_risk_pipeline():
    """Loads the trained risk classifier model, cached for performance."""
    return joblib.load('risk_classifier.joblib')

@st.cache_resource
def load_emotion_pipeline():
    """Loads the Hugging Face emotion analysis model, cached for performance."""
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-emotion",
        return_all_scores=True
    )

risk_pipeline = load_risk_pipeline()
emotion_classifier = load_emotion_pipeline()

@tool
def assess_risk(text: str) -> dict:
    """
    Assesses the mental health risk level from a text on a scale of 1 to 5.
    Use this tool for EVERY user message to determine the immediate risk.
    """
    prediction = risk_pipeline.predict([text])[0]
    return {"risk_level": int(prediction)}

@tool
def analyze_emotion(text: str) -> dict:
    """
    Analyzes the emotional content of a text to understand the user's feelings.
    Use this for non-critical risk levels (1-3) to craft a more empathetic response.
    Returns a dictionary of emotions and their scores.
    """
    raw_scores = emotion_classifier(text)
    scores = {item['label']: round(item['score'], 4) for item in raw_scores[0]}
    return scores

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
        match = re.search(rf"{re.escape(tip_key)}\n(.*?)(?=\n\\[[A-Z_]+_TIP\\]|\Z)", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "I couldn't find a specific tip for that, but remember that taking a moment for yourself is always a good idea."
    except FileNotFoundError:
        return "I'm having trouble accessing my resources right now, but I'm still here to listen."
