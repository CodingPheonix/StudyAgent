import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv(verbose=True)

uri = os.getenv("MONGO_URL")
client = MongoClient(uri)

try:
    database = client.get_database("test")
    users = database.get_collection("users")
    pdfs = database.get_collection("pdfs")

    # client.close()

except Exception as e:
    raise Exception("Unable to find the document due to the following error: ", e)
