import sys
import uuid
import time

# === CLI ===
from app.agent_router import run_agent  # Import the function to handle agent execution
from app.session_manager import get_or_create_chat_id  # Import the function to manage chat sessions

# === STREAMLIT GUI ===
def gui_mode_streamlit():
    """
    Launch the Streamlit GUI for the AI Paper Assistant.

    This function initializes the Streamlit interface, sets up the session state,
    and handles user interactions through the GUI.

    Streamlit is used for a web-based interface, allowing users to interact with
    the assistant in a more visual and user-friendly way.
    """
    import streamlit as st  # Import Streamlit inside the function to avoid CLI dependency
    st.set_page_config(page_title="AI Paper Assistant", page_icon="📚", layout="wide")  # Configure the Streamlit page

    # -- Status bar (placeholder pattern) --
    status_ph = st.empty()  # Create a placeholder for the status bar
    status_ph.markdown("⏳ Initializing...")  # Display an initialization message

    # -- Session setup --
    if "session_id" not in st.session_state:
        # Initialize session state variables if not already set
        st.session_state.session_id = f"web-{uuid.uuid4()}"  # Generate a unique session ID
        st.session_state.chat_id = get_or_create_chat_id(st.session_state.session_id)  # Retrieve or create a chat ID
        st.session_state.history = []  # Initialize chat history as an empty list

    # Warmup (light)
    try:
        time.sleep(0.2)  # Simulate a short delay for initialization
        status_ph.markdown("✅ Ready!")  # Update the status bar to indicate readiness
    except Exception as e:
        # Display a warning message if initialization fails
        status_ph.markdown(f"⚠️ Init warning: {e}")

    st.title("📚 AI Paper Assistant")  # Set the title of the Streamlit app
    st.caption("LangChain + RAG + Agent + Hybrid Search")  # Add a caption describing the app

    # -- Chat history render --
    for u, b in st.session_state.history:
        # Render the chat history (user and assistant messages)
        with st.chat_message("user"):
            st.markdown(u)  # Display the user's message
        with st.chat_message("assistant"):
            st.markdown(b)  # Display the assistant's response

    # -- Chat input --
    prompt = st.chat_input("Ask me to search / summarize / QA / recommend…")  # Input box for user queries
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)  # Display the user's input in the chat

        status_ph.markdown("🤖 Thinking…")  # Update the status bar to indicate processing
        try:
            reply = run_agent(prompt, user_identifier=st.session_state.chat_id)  # Get the assistant's response
        except Exception as e:
            reply = f"⚠️ Error: {e}"  # Handle errors and display an error message

        st.session_state.history.append((prompt, reply))  # Append the interaction to the chat history

        with st.chat_message("assistant"):
            st.markdown(reply)  # Display the assistant's response in the chat

        status_ph.markdown("✅ Ready!")  # Update the status bar to indicate readiness

# === CLI mode (unchanged) ===
def cli_mode():
    """
    Launch the CLI mode for the AI Paper Assistant.

    This function provides a command-line interface for interacting with the assistant.
    Users can type queries and receive responses directly in the terminal.
    """
    print("🤖 AI Paper Assistant (CLI). Type 'exit' to quit.\n")  # Display a welcome message
    user_id = "cli-" + str(uuid.uuid4())[:8]  # Generate a unique user ID for the CLI session
    chat_id = get_or_create_chat_id(user_id)  # Retrieve or create a chat ID for the user
    while True:
        try:
            user_input = input("You: ").strip()  # Prompt the user for input
        except (EOFError, KeyboardInterrupt):
            # Handle exit signals (Ctrl+C or EOF)
            print("\nBye!")
            break
        if user_input.lower() in {"exit", "quit"}:
            # Exit the loop if the user types "exit" or "quit"
            print("Bye!")
            break
        if not user_input:
            # Skip empty input
            continue
        resp = run_agent(user_input, user_identifier=chat_id)  # Get the assistant's response
        print(f"\nBot: {resp}\n")  # Display the response

if __name__ == "__main__":
    """
    Entry point for the script.

    Depending on the command-line arguments, this script either launches the
    Streamlit GUI or the CLI mode.
    """
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Launch CLI mode if the "cli" argument is provided
        cli_mode()
    else:
        try:
            # Check if Streamlit is installed and provide a helpful error message if not
            import streamlit as st  # noqa: F401
            raise SystemExit(
                "This file is a Streamlit app. Launch it with:\n\n"
                "    streamlit run main.py\n\n"
                "Or use CLI mode:\n\n"
                "    python main.py cli\n"
            )
        except ModuleNotFoundError:
            # Provide instructions for installing Streamlit if it's not installed
            raise SystemExit(
                "Streamlit is not installed.\n"
                "Install it and run the GUI with:\n\n"
                "    pip install streamlit\n"
                "    streamlit run main.py\n\n"
                "Or use CLI mode:\n\n"
                "    python main.py cli\n"
            )