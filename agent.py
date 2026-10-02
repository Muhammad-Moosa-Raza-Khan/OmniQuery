from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from rag_engine import retriever

@tool
def search_clinical_literature(query: str) -> str:
    """Searches the local medical knowledge base for clinical imaging research."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant clinical documents found in the database."
    return "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" for d in docs])

tools = [search_clinical_literature]
llm = ChatOllama(model="qwen2.5:1.5b") 
llm_with_tools = llm.bind_tools(tools)

def assistant(state: MessagesState):
    # The updated system message strictly enforces $ and $$ for math.
    sys_msg = SystemMessage(content="""You are OmniQuery, an expert clinical AI. You have access to a local database of medical research papers. 
    CRITICAL RULE: You MUST format ALL mathematical equations, variables, and Greek letters using standard LaTeX enclosed in dollar signs. Use a single dollar sign for inline math (e.g., $x_1$) and double dollar signs for standalone block equations (e.g., $$y = mx + b$$). NEVER use brackets like [...] or (...) for math.
    If the user mentions 'the paper', 'this paper', or asks a medical question, you MUST use the 'search_clinical_literature' tool to find context before answering.""")
    
    messages = [sys_msg] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
agent_executor = builder.compile(checkpointer=memory)