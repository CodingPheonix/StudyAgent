# Build workflow
from typing import Literal

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy

from .agent import (
    extraction,
    retrieval,
    solveForConversation,
    solveForFlashCards,
    solveForQuiz,
    solveForSummary,
)
from .state import MessagesState

agent_builder = StateGraph(MessagesState)

def choosePath(state: MessagesState) -> Literal[
    "solveForSummary",
    "solveForQuiz",
    "solveForConversation",
    "solveForFlashCards",
    END,
]:
    """Decide what task to perform"""

    operation = state["operation"]

    if operation == "summary":
        return "solveForSummary"
    elif operation == "quiz":
        return "solveForQuiz"
    elif operation == "conversation":
        return "solveForConversation"
    elif operation == "flashcards":
        return "solveForFlashCards"
    else:
        return END

# Add nodes
agent_builder.add_node("extraction", extraction, retry_policy=RetryPolicy(max_attempts=1, initial_interval=1.0))
agent_builder.add_node("retrieval", retrieval, retry_policy=RetryPolicy(max_attempts=1, initial_interval=1.0))
agent_builder.add_node("solveForSummary", solveForSummary, retry_policy=RetryPolicy(max_attempts=1, initial_interval=1.0))
agent_builder.add_node("solveForQuiz", solveForQuiz, retry_policy=RetryPolicy(max_attempts=1, initial_interval=1.0))
agent_builder.add_node("solveForConversation", solveForConversation, retry_policy=RetryPolicy(max_attempts=1, initial_interval=1.0))
agent_builder.add_node("solveForFlashCards", solveForFlashCards, retry_policy=RetryPolicy(max_attempts=1, initial_interval=1.0))

# Add edges to connect nodes
agent_builder.add_edge(START, "extraction")
agent_builder.add_edge("extraction", "retrieval")
agent_builder.add_conditional_edges(
    "retrieval",
    choosePath,
    ["solveForSummary", "solveForQuiz", "solveForConversation", "solveForFlashCards", END]
)
agent_builder.add_edge("solveForSummary", END)
agent_builder.add_edge("solveForQuiz", END)
agent_builder.add_edge("solveForConversation", END)
agent_builder.add_edge("solveForFlashCards", END)

# Compile the agent
agent = agent_builder.compile()
