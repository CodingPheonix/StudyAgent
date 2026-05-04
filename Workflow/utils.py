import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

load_dotenv()

embeddings = OllamaEmbeddings(model="llama3")

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=os.getenv("MONGODB_COLLECTION"),
    index_name=os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME"),
    relevance_score_fn="cosine",
)