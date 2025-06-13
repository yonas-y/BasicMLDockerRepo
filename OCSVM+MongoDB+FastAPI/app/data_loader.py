"""
Load the data from database and return it as a pandas dataframe.
"""

import pandas as pd
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

def load_data():
    client = MongoClient(MONGO_URI)

    try:
        print("✅ MongoDB Databases:", client.list_database_names())
    except Exception as e:
        print("❌ Connection failed:", e)

    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    documents = list(collection.find())
    return pd.DataFrame(documents)
