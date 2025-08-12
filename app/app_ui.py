import streamlit as st
from app.agent_router import run_agent
from app.session_manager import get_or_create_chat_id

# Set up the Streamlit page configuration
st.set_page_config(page_title="AI Paper Assistant", layout="wide")
st.title("📑 AI Paper Assistant")

# Initialize session state variables
if "chat_id" not in st.session_state:
    # Create a unique chat ID for the user session
    st.session_state.chat_id = get_or_create_chat_id("streamlit-user")
if "history" not in st.session_state:
    # Initialize the chat history for the session
    st.session_state.history = []

# Display the chat history
for role, msg in st.session_state.history:
    # Render each message in the chat history with the appropriate role
    with st.chat_message(role):
        st.markdown(msg)

# Handle user input
user_msg = st.chat_input("Ask to search / summarize / QA / recommend…")
if user_msg:
    # Append the user's message to the chat history
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        # Display the user's message in the chat interface
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        # Display a spinner while processing the assistant's response
        with st.spinner("Thinking…"):
            # Route the user's input to the agent and get the response
            answer = run_agent(user_msg, user_identifier=st.session_state.chat_id)
            # Display the assistant's response in the chat interface
            st.markdown(answer)
    # Append the assistant's response to the chat history
    st.session_state.history.append(("assistant", answer))