import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel

from pageindex.client import PageIndexClient
from pageindex.retrieve import smart_get_content

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE = BASE_DIR / "workspace"

openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
if openrouter_api_key:
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

model = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    temperature=0.3,
)


def extraction(state: dict):
    return {
        "doc_id": state.get("doc_id"),
        "operation": state["operation"],
        "query": state["query"],
    }


def _format_retrieved_pages(raw_content: str) -> str:
    data = json.loads(raw_content)
    if isinstance(data, dict) and data.get("error"):
        return data["error"]
    if not isinstance(data, list):
        return str(data)

    chunks = []
    for page in data:
        content = " ".join(str(page.get("content", "")).split())
        chunks.append(f"Page {page.get('page')}:\n{content}")
    return "\n\n".join(chunks)


def _context_text(state: dict) -> str:
    context = state.get("context", "")
    if isinstance(context, dict):
        return context.get("context", "")
    return str(context)


def _invoke_with_context(agent, context: str):
    return agent.invoke({
        "messages": [HumanMessage(content=f"Context:\n{context}")]
    })


def _get_structured_field(response: dict, field: str):
    if field in response:
        return response[field]

    structured = response.get("structured_response")
    if isinstance(structured, dict):
        return structured.get(field)
    if isinstance(structured, BaseModel):
        return getattr(structured, field, None)

    return None


def retrieval(state: dict):
    print("operation:", state.get("operation"))
    print("query:", state.get("query"))
    print("doc_id:", state.get("doc_id"))

    if state.get("operation") == "conversation" and not state.get("doc_id"):
        return {"context": {"context": state.get("query", "")}}

    doc_id = state.get("doc_id")
    if not doc_id:
        return {"context": {"context": "No document selected. Upload a PDF first and pass its doc_id."}}

    client = PageIndexClient(workspace=WORKSPACE)
    if doc_id not in client.documents:
        return {"context": {"context": f"Document not found for doc_id: {doc_id}"}}

    client._ensure_doc_loaded(doc_id)
    doc = client.documents.get(doc_id, {})

    if state.get("operation") == "summary":
        summary_lines = []
        for page in doc.get("pages", []):
            summary = page.get("summary") or page.get("content", "")
            summary_lines.append(f"Page {page.get('page')}: {' '.join(str(summary).split())}")
        return {"context": {"context": "\n\n".join(summary_lines)}}

    raw_content = smart_get_content(client.documents, doc_id, state.get("query", ""))

    if "No relevant content found" in raw_content:
        page_count = doc.get("page_count") or len(doc.get("pages", []))
        fallback_pages = ",".join(str(page) for page in range(1, min(page_count, 5) + 1))
        raw_content = client.get_page_content(doc_id, fallback_pages)

    return {
        "context": {
            "context": _format_retrieved_pages(raw_content),
        }
    }


SUMMARY_PROMPT = (
    "You are an excellent Summarization agent. "
    "You are provided with a context. "
    "Your task is to generate a summary of the context, keep in mind the following points: "
    "RULE 1: Length of the summary should be half of the total context. "
    "RULE 2: Do not miss any point. Include them no matter how precise. "
    "RULE 3: Do not add any buzz words (eg. 'Do you want more elaboration', 'I can help you on this part'). "
    "RULE 4: Summary should be complete, compact and integrated. "
)


class summaryAgentResponse(BaseModel):
    summary: str


summaryAgent = create_agent(
    model,
    system_prompt=SUMMARY_PROMPT,
    response_format=summaryAgentResponse,
)


def solveForSummary(state: dict):
    response = _invoke_with_context(summaryAgent, _context_text(state))
    summary = _get_structured_field(response, "summary")

    if not summary:
        summary = "Unable to generate summary from the retrieved context."

    return {
        "messages": [AIMessage(content=summary)],
    }


QUIZ_PROMPT = (
    "You are a Quiz Generation Agent. "
    "You are given a context. "
    "Your task is to generate a quiz based on the context. "
    "RULE 1: Generate 5 questions. "
    "RULE 2: Each question must have exactly 4 options. "
    "RULE 3: Only one option should be correct. "
    "RULE 4: Keep questions clear and based strictly on the context. "
    "RULE 5: Do not add explanations. "
    "Return strictly in the required structured format."
)


class QuizItem(BaseModel):
    question: str
    options: List[str]
    correct_ans: str


class quizAgentResponse(BaseModel):
    quiz: List[QuizItem]


quizAgent = create_agent(
    model,
    system_prompt=QUIZ_PROMPT,
    response_format=quizAgentResponse,
)


def solveForQuiz(state: dict):
    response = _invoke_with_context(quizAgent, _context_text(state))
    quiz = _get_structured_field(response, "quiz")

    if quiz is None:
        quiz = []

    return {
        "messages": [AIMessage(content=json.dumps(quiz, default=lambda item: item.model_dump()))],
    }


FLASH_PROMPT = (
    "You are an intelligent flashcard generation agent. "
    "You are given a context and optionally a user-given topic. "
    "Pick distinct, meaningful, separate points within the boundary of the context and topic if provided. "
    "Generate one concise flashcard line for each point while preserving the original meaning. "
    "Each flashcard line should be around 20-25 words, descriptive, and understandable. "
    "Return strictly in the required structured format."
)


class flashAgentResponse(BaseModel):
    flash: List[str]


flashAgent = create_agent(
    model,
    system_prompt=FLASH_PROMPT,
    response_format=flashAgentResponse,
)


def solveForFlashCards(state: dict):
    context = _context_text(state)
    topic = state.get("query", "")
    response = flashAgent.invoke({
        "messages": [
            HumanMessage(content=f"Topic: {topic}\n\nContext:\n{context}")
        ]
    })
    flashcards = _get_structured_field(response, "flash")

    if flashcards is None:
        flashcards = []

    return {
        "messages": [AIMessage(content=json.dumps(flashcards))],
    }


CONVO_PROMPT = (
    "You are a helpful conversational assistant. "
    "Respond naturally and clearly. "
    "Do not be overly verbose. "
    "Stay relevant to the user's message. "
)


class convoAgentResponse(BaseModel):
    reply: str


conversationAgent = create_agent(
    model,
    system_prompt=CONVO_PROMPT,
    response_format=convoAgentResponse,
)


def solveForConversation(state: dict):
    last_message = state["messages"][-1]
    response = conversationAgent.invoke(last_message.content)
    reply = _get_structured_field(response, "reply")

    if not reply:
        reply = "Unable to generate a response."

    return {
        "messages": [AIMessage(content=reply)],
    }
