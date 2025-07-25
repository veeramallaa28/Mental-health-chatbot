import os
import joblib
from transformers import pipeline
import google.generativeai as genai

# --- AGENT SETUP ---
# This section loads the necessary models and pipelines when the application starts.

# 1. Emotion Analysis Agent's Tool
try:
    emotion_classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-emotion",
        return_all_scores=True
    )
except Exception as e:
    print(f"Error loading emotion model: {e}")
    emotion_classifier = None

# 2. Risk Assessment Agent's Tool
try:
    risk_pipeline = joblib.load('risk_classifier.joblib')
except FileNotFoundError:
    print("Error: risk_classifier.joblib not found. Please run the training script first.")
    risk_pipeline = None

# 3. Response Generation Agent's Tool - Set up Gemini API Key
try:
    genai.configure(api_key="AIzaSyBAg00VDeZM2sSp_1m9cHK9iTnkKhzfOiE")
    
    
    # We define the generation config once...
    generation_config = genai.types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=150
    )
    
    gemini_model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        generation_config=generation_config
    )
    

except Exception as e:
    print(f"Error configuring Gemini: {e}")
    gemini_model = None

# --- AGENT FUNCTIONS ---

def analyze_emotion(text):
    """Emotion Agent: Analyzes text and returns emotion scores."""
    if not emotion_classifier:
        return {}
    try:
        raw_scores = emotion_classifier(text)
        scores = {item['label']: round(item['score'], 4) for item in raw_scores[0]}
        print(f"{text} : emotion Scores: ", scores)
        return scores
    except Exception as e:
        print(f"Error during emotion analysis: {e}")
        return {}

def assess_risk(text):
    """Risk Agent: Predicts risk level (1-5) from text."""
    if not risk_pipeline:
        return {"level": 1, "score": 1.0}
    
    prediction = risk_pipeline.predict([text])[0]
    probabilities = risk_pipeline.predict_proba([text])[0]
    score = probabilities[prediction - 1]
    print(f"{text} : risk Scores: ", score)
    return {"level": int(prediction), "score": round(score, 4)}

def decide_strategy(risk_data):
    """Contextual Advisor Agent: Decides the response strategy based on risk level."""
    risk_level = risk_data['level']
    
    if risk_level == 5:
        return {'strategy': 'Escalate_Critical'}
    elif risk_level == 4:
        return {'strategy': 'Direct_Guidance'}
    elif risk_level == 3:
        return {'strategy': 'Concerned_Inquiry'}
    elif risk_level == 2:
        return {'strategy': 'Empathetic_Inquiry'}
    else: # risk_level == 1
        return {'strategy': 'Supportive_Chat'}

def generate_response(strategy, history):
    """Response Generation Agent: Crafts a response using Gemini Flash based on the strategy."""
    if not gemini_model:
        if strategy['strategy'] == 'Escalate_Critical':
            return "It's very important that you talk to someone who can help right now. Please reach out to a crisis hotline or a medical professional immediately."
        return "I'm having a little trouble connecting right now, but please know I'm still here to listen."

    system_prompts = {
        'Supportive_Chat': "You are 'Sparky', a friendly and supportive companion. Keep the conversation light and positive.",
        'Empathetic_Inquiry': "You are 'Sparky'. The user is showing signs of mild distress. Respond with empathy, validate their feelings, and gently ask an open-ended question to help them elaborate.",
        'Concerned_Inquiry': "You are 'Sparky'. The user is expressing moderate concern. Your tone should be more serious and caring. Acknowledge their pain and ask a supportive question like 'That sounds really tough, what's been on your mind?'",
        'Direct_Guidance': "You are 'Sparky'. The user's message is alarming. Your top priority is to guide them towards help. Be direct but gentle. Strongly encourage them to talk to a professional or a trusted person. Do not ask probing questions about their feelings.",
        'Escalate_Critical': "You are 'Sparky'. The user is in a critical high-risk state. Your response must be calm, supportive, and immediately direct them to help. Do not ask questions. State clearly that it's important to talk to someone now."
    }
    print("\nHistory : ", history)
    print("\n Strategy : ", strategy)
    system_prompt = system_prompts.get(strategy['strategy'], system_prompts['Supportive_Chat'])

    # The Gemini API expects roles to be 'user' and 'model'
    gemini_history = []
    # --- FIX IS HERE: Add the system prompt to the start of the history ---
    gemini_history.append({'role': 'user', 'parts': [system_prompt]})
    gemini_history.append({'role': 'model', 'parts': ["Understood. I will act as 'Sparky' and follow these instructions."]})
    # --- END OF FIX ---

    for msg in history:
        role = 'model' if msg['role'] == 'assistant' else 'user'
        gemini_history.append({'role': role, 'parts': [msg['content']]})

    try:
        # Create a chat session with the full history
        chat_session = gemini_model.start_chat(history=gemini_history)
        
        # The last message is already in the history, so we send a simple follow-up
        # This is a bit of a workaround for how start_chat works. We just need to send something to get the next response.
        response = chat_session.send_message("...") # Sending a minimal message to trigger the response
        
        return response.text
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        if strategy['strategy'] == 'Escalate_Critical':
            return "It's very important that you talk to someone who can help right now. Please reach out to a crisis hotline or a medical professional immediately."
        return "I'm having a little trouble connecting right now, but please know I'm still here to listen."

def get_action(strategy):
    """Action Agent: Determines if a UI action is needed."""
    if strategy['strategy'] == 'Escalate_Critical':
        return {
            "type": "resource_card",
            "title": "Immediate Help is Available 24/7",
            "content": "National Crisis and Suicide Lifeline: Call or Text 988"
        }
    return None