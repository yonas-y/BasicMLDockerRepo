import os
from pymongo import MongoClient
import pandas as pd

uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(uri)

try:
    print("✅ MongoDB Databases:", client.list_database_names())
except Exception as e:
    print("❌ Connection failed:", e)

# Access your database
db = client['Titanic']

# Access the collection where your CSV data is stored
collection = db['Passengers']

# Retrieve all documents from the collection
cursor_for_df = collection.find()
df_from_db = pd.DataFrame(list(cursor_for_df))

# Now you can work with your data as a pandas DataFrame
print("Data loaded from MongoDB into a DataFrame:")
print(df_from_db.head())

# Get basic information about your data
print("\nDataFrame Info:")
df_from_db.info()
