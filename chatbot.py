import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Load environment variables (Make sure GROQ_API_KEY is in your .env file)
load_dotenv()

# 1. Define the simple state schema
class SimpleState(TypedDict):
    messages: List[BaseMessage]

# 2. Initialize the Groq LLM using ChatGroq (NOT ChatOpenAI)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
workflow = StateGraph(SimpleState)

# 3. Define the chatbot node function
def call_model(state: SimpleState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}

# 4. Add the node and wire up the simple linear graph
workflow.add_node("chatbot", call_model)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)

# Compile the graph
app = workflow.compile()

# 5. Main CLI Loop
if __name__ == "__main__":
    print("--- Groq Chatbot Initialized ---")
    print("Type 'exit' or 'quit' to end.\n")
    
    # In-memory list for conversation turn history
    chat_history = []
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
                
            # Append user message
            chat_history.append(HumanMessage(content=user_input))
            
            # Invoke the graph
            result = app.invoke({"messages": chat_history})
            
            # Extract and display the assistant response
            ai_reply = result["messages"][-1]
            print(f"\nAI: {ai_reply.content}\n")
            
            # Keep history updated for the next turn
            chat_history = result["messages"]
            
        except KeyboardInterrupt:
            print("\nExiting...\n")
            break