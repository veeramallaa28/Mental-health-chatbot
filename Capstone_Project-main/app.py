import json
import re
import datetime # Import the datetime library
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import the tools from our tools.py file, now including analyze_emotion
from tools import assess_risk, get_tip_from_rag, analyze_emotion

# --- History Management Functions ---
HISTORY_FILE = "chat_history.json"

def save_history(user_id, history):
    """Saves the chat history for a specific user to the JSON file."""
    try:
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    
    history_to_save = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            history_to_save.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history_to_save.append({"role": "assistant", "content": msg.content})
            
    data[user_id] = history_to_save
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def load_history(user_id):
    """Loads the chat history for a specific user from the JSON file."""
    try:
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
        
        history_from_db = data.get(user_id, [])
        history_to_load = []
        for item in history_from_db:
            if item["role"] == "user":
                history_to_load.append(HumanMessage(content=item["content"]))
            elif item["role"] == "assistant":
                history_to_load.append(AIMessage(content=item["content"]))
        return history_to_load
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def extract_json_from_string(text):
    """Finds and loads the first valid JSON object from a string."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

# --- UI & Agent Setup ---

st.set_page_config(page_title="Sparky - LangChain PoC", layout="wide")
st.title("Sparky ✨ - Proactive AI Companion PoC")

# --- User Identification ---
user_id = st.text_input("Please enter your User ID to start or continue a session:", key="user_id_input")

if not user_id:
    st.warning("Please enter a User ID to begin.")
    st.stop()

st.info(f"Disclaimer for user **{user_id}**: This is a PoC and not a substitute for a medical professional. If you are in crisis, please contact a helpline.")

# --- MANUAL API KEY SETUP ---
api_key = "AIzaSyD5S7K8N7WfGLgOm0BNrV8fwj7rVhYjtlA" 

if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
    st.error("Google API Key not found. Please paste your key directly into the `api_key` variable in the app.py file.")
    st.stop()

# Initialize the agent only if the key exists
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=api_key)
    tools = [assess_risk, get_tip_from_rag, analyze_emotion]
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are 'Sparky', a friendly and empathetic mental health companion. Your primary goal is to provide a safe, supportive, and proactive space.

        Your process is a multi-step thought process:
        1.  **Analyze:** For every user message, you MUST start by using the 'assess_risk' tool.
        2.  **Strategize:** Based on the risk level from the tool, decide your strategy. You may need to use other tools like 'analyze_emotion' or 'get_tip_from_rag' to gather more information for your response.
        3.  **Synthesize & Respond:** After you have used all necessary tools and have their results (Observations), your final job is to synthesize all the information into a single, conversational response. **Your final answer must NOT be a tool call.** It must be a user-facing message.
        4.  **Format:** Your final output MUST be a single JSON object with two keys: "response_text" and "action_card".
            - For Risk 5, the action_card MUST be: {{ "title": "Immediate Help is Available 24/7", "content": "National Crisis and Suicide Lifeline: Call or Text 988" }}
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

# --- Chat Logic ---

if "current_user_id" not in st.session_state or st.session_state.current_user_id != user_id:
    st.session_state.current_user_id = user_id
    st.session_state.chat_history = load_history(user_id)
    st.session_state.greeting_sent = (len(st.session_state.chat_history) > 0)


# Display chat history
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        if role == "assistant":
            content_data = extract_json_from_string(msg.content)
            if content_data:
                st.markdown(content_data.get("response_text", "I'm not sure what to say, but I'm here for you."))
                if content_data.get("action_card"):
                    action = content_data["action_card"]
                    st.warning(f"**{action['title']}**\n\n{action['content']}")
            else:
                st.markdown(msg.content)
        else:
            st.markdown(msg.content)

# UPDATED: Time-aware proactive greeting for new users
if not st.session_state.greeting_sent:
    current_hour = datetime.datetime.now().hour
    greeting = "Hello!"
    if 5 <= current_hour < 12:
        greeting = "Good morning!"
    elif 12 <= current_hour < 18:
        greeting = "Good afternoon!"
    else:
        greeting = "Good evening!"
    
    full_greeting_message = f"{greeting} I'm Sparky, your proactive companion. How are you feeling as you start your day?"
    
    with st.chat_message("assistant"):
        st.markdown(full_greeting_message)
        
    greeting_message_obj = AIMessage(content=json.dumps({"response_text": full_greeting_message, "action_card": None}))
    st.session_state.chat_history.append(greeting_message_obj)
    save_history(user_id, st.session_state.chat_history)
    st.session_state.greeting_sent = True
    st.rerun()

# Main chat input loop
if prompt := st.chat_input("How are you feeling?"):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Sparky is thinking..."):
        try:
            CONVERSATION_WINDOW_SIZE = 10 
            recent_history = st.session_state.chat_history[-CONVERSATION_WINDOW_SIZE:]

            response = agent_executor.invoke({
                "input": prompt,
                "chat_history": recent_history
            })
            ai_response_content = response['output']
        except Exception as e:
            st.error("An error occurred while communicating with the AI. Please try again.")
            print(f"Agent execution error: {e}")
            ai_response_content = '{"response_text": "I seem to be having some trouble thinking right now. Could you try rephrasing that?", "action_card": null}'
    
    st.session_state.chat_history.append(AIMessage(content=ai_response_content))
    save_history(user_id, st.session_state.chat_history)
    st.rerun()