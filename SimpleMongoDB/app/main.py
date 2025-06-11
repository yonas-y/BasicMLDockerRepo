import os
from pymongo import MongoClient

uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(uri)

try:
    print("✅ MongoDB Databases:", client.list_database_names())
except Exception as e:
    print("❌ Connection failed:", e)
