import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Chitti", page_icon="🤖")
st.title("Hey, I'm Chitti! 🤖 How may I help you today?")

# 1. Define State
class SimpleState(TypedDict):
    messages: List

# 2. Initialize and cache the LangGraph app so it doesn't rebuild on every click
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
    return workflow.compile()

app = get_graph_app()

# 3. Initialize Streamlit session state for chat history UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display historical chat messages on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle user input from chat box
if prompt := st.chat_input("Type your message here..."):
    # Add user message to Streamlit state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert Streamlit history into LangChain BaseMessage objects for LangGraph
    lc_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    # Invoke the LangGraph app
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke({"messages": lc_messages})
            ai_reply = result["messages"][-1].content
            st.markdown(ai_reply)

    # Add assistant response to Streamlit state
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})