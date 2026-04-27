import os

from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter

os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

model = ChatOpenRouter("nvidia/nemotron-3-super-120b-a12b:free")
