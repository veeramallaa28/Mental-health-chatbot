import json
import re
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import the tools from our tools.py file
from tools import assess_risk, get_tip_from_rag

# --- Helper Function to Extract JSON ---
def extract_json_from_string(text):
    """Finds and loads the first valid JSON object from a string."""
    # This regex finds a JSON object within a larger string
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
st.info("Disclaimer: This is a PoC and not a substitute for a medical professional. If you are in crisis, please contact a helpline.")

# --- MANUAL API KEY SETUP ---
# Paste your Google API Key here.
api_key = "AIzaSyBAg00VDeZM2sSp_1m9cHK9iTnkKhzfOiE" 

if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
    st.error("Google API Key not found. Please paste your key directly into the `api_key` variable in the app.py file.")
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
              {{ "title": "Immediate Help is Available 24/7", "content": "National Crisis and Suicide Lifeline: Call or Text 988" }}
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "greeting_sent" not in st.session_state:
    st.session_state.greeting_sent = False

# Display chat history
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        # FIX: Use the robust JSON extractor for assistant messages
        if role == "assistant":
            content_data = extract_json_from_string(msg.content)
            if content_data:
                st.markdown(content_data.get("response_text", "I'm not sure what to say, but I'm here for you."))
                if content_data.get("action_card"):
                    action = content_data["action_card"]
                    st.warning(f"**{action['title']}**\n\n{action['content']}")
            else:
                # Fallback if no JSON is found in the response
                st.markdown(msg.content)
        else:
            # For user messages, just display the content
            st.markdown(msg.content)


# Simulated proactive greeting
if not st.session_state.greeting_sent:
    with st.chat_message("assistant"):
        st.markdown("Hello! I'm Sparky, your proactive companion. How are you feeling as you start your day?")
    # We store the greeting as a JSON string to be consistent with the agent's output format
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
