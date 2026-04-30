# Build workflow
from typing import Literal

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from .agent import extraction, retrieval, solveForQuiz, solveForSummary, solveForConversation
from .state import MessagesState

agent_builder = StateGraph(MessagesState)

def choosePath(state: MessagesState) -> Literal["solveForSummary", "solveForQuiz", "solveForConversation", END]:
    """Decide what task to perform"""

    operation = state["operation"]

    if operation == "summary":
        return "solveForSummary"
    elif operation == "quiz":
        return "solveForQuiz"
    elif operation == "conversation":
        return "solveForConversation"
    else:
        return END

# Add nodes
agent_builder.add_node("extraction", extraction)
agent_builder.add_node("retrieval", retrieval)
agent_builder.add_node("solveForSummary", solveForSummary)
agent_builder.add_node("solveForQuiz", solveForQuiz)
agent_builder.add_node("solveForConversation", solveForConversation)

# Add edges to connect nodes
agent_builder.add_edge(START, "extraction")
agent_builder.add_edge("extraction", "retrieval")
agent_builder.add_conditional_edges(
    "retrieval",
    choosePath,
    ["solveForSummary", "solveForQuiz", "solveForConversation", END]
)
agent_builder.add_edge("solveForSummary", END)
agent_builder.add_edge("solveForQuiz", END)
agent_builder.add_edge("solveForConversation", END)

# Compile the agent
agent = agent_builder.compile()

# Show the agent
from IPython.display import Image, display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
