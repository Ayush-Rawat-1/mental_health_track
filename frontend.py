import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage

# --- Backend Imports ---
# Ensure these files are in the same directory as your Streamlit app
from main_back import app
from quick_help import quick_help_app

# --- Page Configuration ---
st.set_page_config(page_title="Gen AI Mental Health Assistant", layout="wide")

st.title("Mental Health AI Assistant")
st.markdown("A safe space to find support and track your well-being.")

# --- Questionnaire Data ---
# Based on the DSM-5 PDF provided earlier
questionnaire_data = [
    {"id": "1", "text": "Little interest or pleasure in doing things?"},
    {"id": "2", "text": "Feeling down, depressed, or hopeless?"},
    {"id": "3", "text": "Feeling more irritated, grouchy, or angry than usual?"},
    {"id": "4", "text": "Sleeping less than usual, but still have a lot of energy?"},
    {"id": "5", "text": "Starting lots more projects than usual or doing more risky things than usual?"},
    {"id": "6", "text": "Feeling nervous, anxious, frightened, worried, or on edge?"},
    {"id": "7", "text": "Feeling panic or being frightened?"},
    {"id": "8", "text": "Avoiding situations that make you anxious?"},
    {"id": "9", "text": "Unexplained aches and pains (e.g., head, back, joints, abdomen, legs)?"},
    {"id": "10", "text": "Feeling that your illnesses are not being taken seriously enough?"},
    {"id": "11", "text": "Thoughts of actually hurting yourself?"},
    {"id": "12", "text": "Hearing things other people couldn't hear, such as voices even when no one was around?"},
    {"id": "13", "text": "Feeling that someone could hear your thoughts, or that you could hear what another person was thinking?"},
    {"id": "14", "text": "Problems with sleep that affected your sleep quality over all?"},
    {"id": "15", "text": "Problems with memory (e.g., learning new information) or with location (e.g., finding your way home)?"},
    {"id": "16", "text": "Unpleasant thoughts, urges, or images that repeatedly enter your mind?"},
    {"id": "17", "text": "Feeling driven to perform certain behaviors or mental acts over and over again?"},
    {"id": "18", "text": "Feeling detached or distant from yourself, your body, your physical surroundings, or your memories?"},
    {"id": "19", "text": "Not knowing who you really are or what you want out of life?"},
    {"id": "20", "text": "Not feeling close to other people or enjoying your relationships with them?"},
    {"id": "21", "text": "Drinking at least 4 drinks of any kind of alcohol in a single day?"},
    {"id": "22", "text": "Smoking any cigarettes, a cigar, or pipe, or using snuff or chewing tobacco?"},
    {"id": "23", "text": "Using any medicines ON YOUR OWN, that is, without a doctor's prescription, in greater amounts or longer than prescribed?"},
]
response_options = ["None (0)", "Slight (1)", "Mild (2)", "Moderate (3)", "Severe (4)"]


# --- App State Initialization ---
def initialize_session_state():
    # General app mode
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Quick Help"

    # Quick Help state
    if 'quick_help_history' not in st.session_state:
        st.session_state.quick_help_history = []
    
    # Tracking Health state
    if 'tracking_stage' not in st.session_state:
        st.session_state.tracking_stage = "questionnaire" # questionnaire -> chat
    if 'tracking_history' not in st.session_state:
        st.session_state.tracking_history = []
    if 'tracking_thread_id' not in st.session_state:
        st.session_state.tracking_thread_id = None

initialize_session_state()

# --- Sidebar for Navigation ---
with st.sidebar:
    st.header("Navigation")
    st.session_state.app_mode = st.radio(
        "Choose a feature:",
        ("Quick Help", "Track Your Health"),
        key="app_mode_selector"
    )
    st.info("Your conversations are private. We do not store personally identifiable information.")

# --- Main App Logic ---

# --- Quick Help Feature ---
if st.session_state.app_mode == "Quick Help":
    st.header("Quick Help Chat")
    st.markdown("Get immediate, supportive advice. How are you feeling right now?")

    # Display chat history
    for message in st.session_state.quick_help_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_input := st.chat_input("Share your thoughts..."):
        st.session_state.quick_help_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # The input for the non-persistent app is the full history each time
            history_for_input = [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                for msg in st.session_state.quick_help_history
            ]
            
            ai_response = quick_help_app.invoke({'messages': history_for_input}).get("messages",["Could not generate response"])[-1].content
            st.write(ai_response)

        st.session_state.quick_help_history.append({"role": "assistant", "content": ai_response})

# --- Track Your Health Feature ---
elif st.session_state.app_mode == "Track Your Health":
    
    # --- Stage 1: Questionnaire ---
    if st.session_state.tracking_stage == "questionnaire":
        st.header("Health & Well-being Questionnaire")
        st.markdown("Please answer the following questions based on your feelings over the **last two weeks**.")

        with st.form("health_questionnaire"):
            responses = {}
            for q in questionnaire_data:
                # Get the integer value from the selection
                response_str = st.radio(q["text"], options=response_options, key=q["id"], horizontal=True)
                responses[q["id"]] = response_options.index(response_str)

            submitted = st.form_submit_button("Analyze & Create My Plan")
            
            if submitted:
                # Generate a unique thread ID for this user's persistent session
                st.session_state.tracking_thread_id = f"user_{uuid.uuid4()}"
                config = {"configurable": {"thread_id": st.session_state.tracking_thread_id}}
                
                # Call the main backend app with the questionnaire responses
                initial_input = {"questionnaire_responses": responses}
                
                with st.spinner("Analyzing your responses and generating a personalized plan..."):
                    # Use .invoke() for the first call as we want the full plan at once
                    result = app.invoke(initial_input, config=config)
                    initial_plan = result.get('messages', ["Could not generate a plan."])[-1].content
                    
                # Store the initial plan and switch to chat mode
                st.session_state.tracking_history = [{"role": "assistant", "content": initial_plan}]
                st.session_state.tracking_stage = "chat"
                st.rerun()

    # --- Stage 2: Chat with the Plan ---
    elif st.session_state.tracking_stage == "chat":
        st.header("Your Personalized Plan & Chat")
        st.markdown("Here is an initial plan based on your responses. You can ask questions about it or request different exercises.")
        
        config = {"configurable": {"thread_id": st.session_state.tracking_thread_id}}

        # Display chat history
        for message in st.session_state.tracking_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input for follow-up questions
        if user_input := st.chat_input("Ask a question about your plan..."):
            st.session_state.tracking_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                # For follow-ups, we only need to send the new message.
                # The checkpointer on the backend handles loading the history.
                follow_up_input = {"messages": [HumanMessage(content=user_input)]}
                result = app.invoke(follow_up_input, config=config)
                print(result,follow_up_input)
                ai_response = result.get('messages', ["Could not generate a plan."])[-1].content

            st.session_state.tracking_history.append({"role": "assistant", "content": ai_response})
