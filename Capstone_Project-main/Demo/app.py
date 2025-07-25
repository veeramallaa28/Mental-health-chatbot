import streamlit as st
from agents import analyze_emotion, assess_risk, decide_strategy, generate_response, get_action

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Sparky - Your Mental Health Companion", layout="wide")

st.title("Sparky ✨ - Your AI Companion")
st.markdown("This is a safe and anonymous space to share what's on your mind. I'm here to listen without judgment.")
st.info("Disclaimer: I am an AI assistant and not a substitute for a human therapist or medical professional. If you are in crisis, please contact a helpline immediately.")
st.markdown("---")

# --- Session State Initialization ---
# This ensures that the chat history persists as the user interacts.
if "history" not in st.session_state:
    st.session_state.history = []

# --- Display Chat History ---
# This loop goes through the history and displays each message.
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If an action card was generated for a message, display it
        if "action" in message and message["action"]:
            st.warning(f"**{message['action']['title']}**\n\n{message['action']['content']}")

# --- Main Application Logic ---
# This block runs every time the user submits a new message.
if prompt := st.chat_input("How are you feeling today?"):
    # 1. Add user message to history and display it on the screen
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call Agents in sequence
    with st.spinner("Sparky is thinking..."):
        # The user's prompt is the input for the analysis agents
        emotion_data = analyze_emotion(prompt)
        risk_data = assess_risk(prompt)

        # The risk data determines the strategy
        strategy = decide_strategy(risk_data)

        # The strategy and history guide the response generation
        assistant_response_text = generate_response(strategy, st.session_state.history)

        # The strategy determines if a special action is needed
        action = get_action(strategy)

    # 3. Add assistant response to history and display it
    assistant_message = {
        "role": "assistant",
        "content": assistant_response_text,
        "action": action  # Store the action with the message
    }
    st.session_state.history.append(assistant_message)

    # We use st.rerun() to make the new message appear instantly.
    st.rerun()