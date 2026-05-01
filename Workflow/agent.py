import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_openrouter import ChatOpenRouter

os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

model = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    temperature=0.3
)


# model = ChatOpenRouter(
#     model="nvidia/nemotron-3-super-120b-a12b:free",
#     temperature=0.3
# )

def extraction(state: dict):
    return {
        "operation": state["operation"],
        "query": state["query"]
    }


def retrieval(state: dict):
    print("operation:", state.get("operation"))
    print("query:", state.get("query"))
    print("context:", state.get("context"))

    temporary_context = '''
            The Lion’s Lesson
            In a sun-dappled forest, a mighty lion ruled with a roar that made leaves tremble. Every creature feared him, for he hunted not just when hungry, but whenever his pride demanded it.
            One morning, the animals gathered in secret. They decided to send one animal each day to the lion, hoping to spare the rest. When it was the rabbit’s turn, he did not hurry. He hopped slowly, stopping to nibble grass and watch clouds drift lazily.
            By the time he reached the lion’s den, the sun was already leaning west. The lion’s golden eyes blazed. “Why are you late?” he growled.
            The rabbit bowed low. “On my way, I met another lion,” he said softly. “He claimed to be the true king of this forest. He even threatened to eat me before you could.”
            The lion’s pride flared hotter than his hunger. “Show me this imposter!”
            The rabbit led him to a deep, still pond. “He lives here,” the rabbit whispered.
            The lion peered into the water and saw a fierce face staring back. He roared, and the reflection roared too. Enraged, he leapt into the pond to fight his rival—only to sink beneath the water, never to return.
            The rabbit hopped away, his heart pounding with both fear and triumph. That evening, the forest was quieter than it had been in years.
            And from that day on, the animals learned that courage and wit could outmatch even the sharpest claws.
        '''

    return {
        "context": {
            "context": temporary_context
        }
    }

    # return {
    #     "context": temporary_context
    # }


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
    """
    Creates a summary of given text
    """

    response = summaryAgent.invoke(state["context"])

    content = response["messages"][-1].content  # last message = actual output
    # summary = json.loads(content)["summary"]

    # print("response is : ", content)

    return {
        "messages": [AIMessage(content=content)]
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

from typing import List, Literal


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
    """
    Generates quiz from given context
    """

    response = quizAgent.invoke(state["context"])
    answer = response["messages"][-1].content

    return {
        "messages": [AIMessage(content=answer)]
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
    """
    Handles normal conversation
    """

    last_message = state["messages"][-1]

    response = conversationAgent.invoke(last_message.content)
    answer = response["messages"][-1].content

    return {
        "messages": [AIMessage(content=answer)]
    }

FLASH_PROMPT = (
    "You are an intelligent agent. "
    "You are given a context and optionally a user given topic. "
    "Your task is to pick up distinct, meaningful, separate points within the boundary of the context and the provided topic (if any). "
    "Only after that, generate a single line context of each point, preserving the meaning and context relevance to the original text. "
    "The context lines should be around 20-25 words in length, descriptive and understandable. "
    "Return the total data as a single list of strings. "
)

class flashAgentResponse(BaseModel):
    flash: List[str]

flashAgent = create_agent(
    model,
    system_prompt=FLASH_PROMPT,
    response_format=flashAgentResponse,
)

def solveForFlashCards(state: dict):
    """
    Creates a list of flash card responses
    """

    response = flashAgent.invoke({
        "context": state["context"],
        "given_topic": state['query']
    })
    answer = response["messages"][-1].content

    return {
        "messages": [AIMessage(content=answer)]
    }