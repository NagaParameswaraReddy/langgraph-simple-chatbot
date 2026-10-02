import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="Chitti AI Assistant", page_icon="🤖", layout="wide")

# 1. Define State
class SimpleState(TypedDict):
    messages: List

# 2. Initialize LangGraph with a Checkpointer (MemorySaver)
@st.cache_resource
def get_graph_app():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
    workflow = StateGraph(SimpleState)

    def call_model(state: SimpleState):
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [AIMessage(content=response.content)]}

    workflow.add_node("chatbot", call_model)
    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", END)
    
    # Compile with memory checkpointer to support threads
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

app = get_graph_app()

# 3. Initialize Multi-Session State in Streamlit
if "sessions" not in st.session_state:
    default_thread_id = str(uuid.uuid4())
    st.session_state.sessions = {
        default_thread_id: {
            "title": "Chat 1",
            "messages": [{"role": "assistant", "content": "Hey, I'm Chitti! 🤖 How may I help you today?"}]
        }
    }
    st.session_state.current_thread_id = default_thread_id

# --- Sidebar UI for Managing Threads ---
with st.sidebar:
    st.title("💬 Chat Sessions")
    
    # Button to create a new chat thread
    if st.button("➕ New Chat", use_container_width=True):
        new_thread_id = str(uuid.uuid4())
        chat_count = len(st.session_state.sessions) + 1
        st.session_state.sessions[new_thread_id] = {
            "title": f"Chat {chat_count}",
            "messages": [{"role": "assistant", "content": "Hey, I'm Chitti! 🤖 How may I help you today?"}]
        }
        st.session_state.current_thread_id = new_thread_id
        st.rerun()

    st.divider()
    
    # List existing chats for switching
    st.markdown("### Recent Chats")
    for thread_id, session_data in list(st.session_state.sessions.items()):
        # Highlight active chat button
        is_active = thread_id == st.session_state.current_thread_id
        button_type = "primary" if is_active else "secondary"
        
        if st.button(session_data["title"], key=thread_id, type=button_type, use_container_width=True):
            st.session_state.current_thread_id = thread_id
            st.rerun()

# --- Main Chat Interface ---
st.title("🤖 Chitti Chatbot")

# Get current active session messages
current_session = st.session_state.sessions[st.session_state.current_thread_id]

# Display historical chat messages for the current thread
for message in current_session["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Type your message here..."):
    # Append user message to current session state
    current_session["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert session history to LangChain messages for graph invocation
    lc_messages = []
    for m in current_session["messages"]:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    # Pass the thread_id in config so LangGraph tracks state per thread
    config = {"configurable": {"thread_id": st.session_state.current_thread_id}}

    # Invoke the LangGraph app
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke({"messages": lc_messages}, config=config)
            ai_reply = result["messages"][-1].content
            st.markdown(ai_reply)

    # Append assistant response to current session state
    current_session["messages"].append({"role": "assistant", "content": ai_reply})