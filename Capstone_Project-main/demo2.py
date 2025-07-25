import os
import re
import json
import joblib
import pandas as pd
import streamlit as st
from langchain.tools import tool
from transformers import pipeline
from sklearn.pipeline import Pipeline
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Step 1: Automated Setup ---
# This function creates necessary files and trains the model if they don't exist.

def setup_poc():
    """Checks for necessary files and creates them if missing."""
    
    # Define file contents as strings
    risk_data_content = """text,label
"I had a really nice day today, feeling happy and content",1
"The weather is beautiful today, I'm going for a walk",1
"I'm so excited for the weekend, planning a trip!",1
"Just feeling a bit down and out, work was tough",2
"Feeling overwhelmed and stressed out by everything on my plate",2
"I'm so tired of everything lately",2
"I feel so lonely and isolated from everyone",3
"Nothing seems to make me happy anymore, I feel empty",3
"I've been crying a lot and I don't know why",3
"I'm so tired of fighting this feeling, everything is pointless",4
"I just want all of this pain to end for good",4
"I can't take it anymore, I'm thinking about disappearing",4
"It feels like the only way out is to just end it all",5
"I have a plan to hurt myself and I don't know if I can stop",5
"I've decided to end my life and have written a note",5
"""

    tips_content = """[STRESS_TIP]
When you feel stressed, try the 4-7-8 breathing technique. It can help calm your nervous system. Here's how:
1. Breathe in quietly through your nose for 4 seconds.
2. Hold your breath for a count of 7 seconds.
3. Exhale completely through your mouth, making a whoosh sound for 8 seconds.
Repeat this cycle three to four times.

[ANXIETY_TIP]
For feelings of anxiety, try grounding yourself with the 5-4-3-2-1 method. It brings you back to the present moment. Acknowledge:
- 5 things you can see around you.
- 4 things you can touch.
- 3 things you can hear.
- 2 things you can smell.
- 1 thing you can taste.

[LOW_MOOD_TIP]
When your mood is low, sometimes a small change of scenery can help. Consider stepping outside for just five minutes of fresh air, or putting on a favorite upbeat song. It's not a cure, but it can be a gentle step in a positive direction.
"""

    # Create data and tips files if they don't exist
    if not os.path.exists('risk_data.csv'):
        with open('risk_data.csv', 'w') as f:
            f.write(risk_data_content)
        print("Created risk_data.csv")

    if not os.path.exists('tips.txt'):
        with open('tips.txt', 'w') as f:
            f.write(tips_content)
        print("Created tips.txt")

    # Train model only if it doesn't exist
    if not os.path.exists('risk_classifier.joblib'):
        print("Risk model not found. Training a new one...")
        df = pd.read_csv('risk_data.csv')
        X = df['text']
        y = df['label']
        pipeline_obj = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LogisticRegression())
        ])
        pipeline_obj.fit(X, y)
        joblib.dump(pipeline_obj, 'risk_classifier.joblib')
        print("Model training complete. Model saved to 'risk_classifier.joblib'")

# Run the setup function once at the start
setup_poc()


# --- Step 2: Define Agent Tools ---
# These are the functions the LangChain agent can use.

@st.cache_resource
def load_risk_pipeline():
    return joblib.load('risk_classifier.joblib')

risk_pipeline = load_risk_pipeline()

@tool
def assess_risk(text: str) -> dict:
    """
    Assesses the mental health risk level from a text on a scale of 1 to 5.
    Use this tool for EVERY user message to determine the immediate risk.
    """
    prediction = risk_pipeline.predict([text])[0]
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
        match = re.search(f"{re.escape(tip_key)}\n(.*?)(?=\n\\[[A-Z_]+_TIP\\]|\Z)", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "I couldn't find a specific tip for that, but remember that taking a moment for yourself is always a good idea."
    except FileNotFoundError:
        return "I'm having trouble accessing my resources right now, but I'm still here to listen."


# --- Step 3: Setup LangChain Agent ---

# --- Step 4: Streamlit UI ---

st.set_page_config(page_title="Sparky - LangChain PoC", layout="wide")
st.title("Sparky ✨ - Proactive AI Companion PoC")
st.info("Disclaimer: This is a PoC and not a substitute for a medical professional. If you are in crisis, please contact a helpline.")

# --- MANUAL API KEY SETUP ---
# Paste your Google API Key here.
api_key = "AIzaSyBAg00VDeZM2sSp_1m9cHK9iTnkKhzfOiE"

if not api_key or api_key == "AIzaSyBAg00VDeZM2sSp_1m9cHK9iTnkKhzfOiE":
    st.error("Google API Key not found. Please paste your key directly into the `api_key` variable in the code.")
    st.stop()

# Initialize the agent only if the key exists
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=api_key)
    tools = [assess_risk, get_tip_from_rag]

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are 'Sparky', a friendly and empathetic mental health companion. Your primary goal is to provide a safe, supportive, and proactive space.

        Your process for every user message is:
        1.  You MUST start by using the 'assess_risk' tool to determine the user's risk level (1-5).
        2.  Based on the risk level, formulate your response strategy.
            - Risk 1: Normal, supportive chat. Be warm and friendly.
            - Risk 2 or 3: The user is feeling down. Respond with empathy. Consider using the 'get_tip_from_rag' tool with a relevant topic like 'stress' or 'low_mood'.
            - Risk 4: The user is in significant distress. Your tone must be more serious. Strongly encourage them to talk to someone they trust or a professional. You can use the RAG tool for a grounding exercise tip.
            - Risk 5: This is a critical alert. You MUST provide the escalation message and trigger the action card. Do not ask questions.
        3.  Your final output to the user MUST be a single JSON object with two keys:
            - "response_text": Your text message to the user.
            - "action_card": A JSON object for UI actions, or null. For Risk 5, this MUST be:
              { "title": "Immediate Help is Available 24/7", "content": "National Crisis and Suicide Lifeline: Call or Text 988" }
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
except Exception as e:
    st.error(f"Failed to initialize the AI Agent. Please check your API key and dependencies. Error: {e}")
    st.stop()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "greeting_sent" not in st.session_state:
    st.session_state.greeting_sent = False

# Display chat history
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        try:
            content_data = json.loads(msg.content)
            st.markdown(content_data.get("response_text", ""))
            if content_data.get("action_card"):
                action = content_data["action_card"]
                st.warning(f"**{action['title']}**\n\n{action['content']}")
        except (json.JSONDecodeError, TypeError):
            st.markdown(msg.content)

# Simulated proactive greeting
if not st.session_state.greeting_sent:
    with st.chat_message("assistant"):
        st.markdown("Hello! I'm Sparky, your proactive companion. How are you feeling as you start your day?")
    st.session_state.chat_history.append(AIMessage(content='{"response_text": "Hello! I\'m Sparky, your proactive companion. How are you feeling as you start your day?", "action_card": null}'))
    st.session_state.greeting_sent = True

# Main chat input loop
if prompt := st.chat_input("How are you feeling?"):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Sparky is thinking..."):
        try:
            response = agent_executor.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })
            ai_response_content = response['output']
        except Exception as e:
            st.error("An error occurred while communicating with the AI. Please try again.")
            print(f"Agent execution error: {e}")
            ai_response_content = '{"response_text": "I seem to be having some trouble thinking right now. Could you try rephrasing that?", "action_card": null}'
    
    st.session_state.chat_history.append(AIMessage(content=ai_response_content))
    st.rerun()