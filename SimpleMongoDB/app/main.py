from pymongo import MongoClient

# Use host.docker.internal to access MongoDB running on the host
client = MongoClient("mongodb://host.docker.internal:27017/")

try:
    dbs = client.list_database_names()
    print("✅ Connected to MongoDB. Databases:", dbs)
except Exception as e:
    print("❌ Failed to connect:", e)
