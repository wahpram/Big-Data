import pandas as pd
import re
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def load_to_mongo(file, col_name):
    uri = "mongodb+srv://wahpram2607:Bangli123.@cluster0.yiobiyk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    db_name = 'db_tanah_bali'
    collection_name = f'tanah_bali_{col_name}'

    
    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        
        print('----------------------------------------------')
        print('Connected to db')
        
        db = client[db_name]
        collection = db[collection_name]
        
        collection.insert_many([f for f in file.values()])
        coll_len = collection.count_documents({})
        
        print(f'Total documents inserted: {len(file)}')
        print(f'Total documents in collection: {coll_len}')
        
        client.close()
        print('Dissconnected!')
        
    except Exception as e:
        print(e)
        
        
if __name__ == '__main__':
    load_to_mongo()