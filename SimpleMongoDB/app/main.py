from pymongo import MongoClient
import os

# Use host.docker.internal to access MongoDB running on the host
mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017/")
client = MongoClient(mongo_uri)

try:
    dbs = client.list_database_names()
    print("✅ Connected to MongoDB. Databases:", dbs)
except Exception as e:
    print("❌ Failed to connect:", e)
