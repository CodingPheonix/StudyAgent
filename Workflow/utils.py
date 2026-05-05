import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

from mongo.init import client

load_dotenv()

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

db = client["rag_db"]
collection = db["documents"]

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection,
    index_name="vector_index",
    relevance_score_fn="cosine",
)