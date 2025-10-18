import streamlit as st
import requests


OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def format_props(props_dict):
    return "\n".join([f"- {key}: {value}" for key, value in props_dict.items()])

def clear_session():
    st.session_state.clear()

# Deepseek
def call_deepseek_api(user_input):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        # "model": "deepseek/deepseek-chat",
        "model": "openai/gpt-4o-mini",
        "messages": st.session_state.messages + [{"role": "user", "content": user_input}],
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# chat assisstant UI
def wound_chat_assistant(wound_props):
    if not wound_props:
        return

    # Context from wound properties
    context = "\n".join([f"{k}: {v}" for k, v in wound_props.items()])
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful medical assistant. "
            "You help interpret wound analysis results based on detected image properties."
        )
    }
    user_message = {
        "role": "user",
        "content": f"The following wound properties were detected:\n{context}"
    }

    # Initialize message in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [system_message, user_message]

    # Display the UI
    st.markdown("### AI Medical Assistant")
    
    with st.expander("Chat with AI about the wound", expanded=True):
        user_input = st.chat_input("Ask a question about the wound...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("AI is generating a response..."):
                response = call_deepseek_api(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Display all messages in reversed order
        for msg in reversed(st.session_state.messages[2:]):
            st.chat_message(msg["role"]).markdown(msg["content"])

