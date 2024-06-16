import pandas as pd
import re
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def get_data_mongo(collection_name):
    uri = "mongodb+srv://wahpram2607:Bangli123.@cluster0.yiobiyk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    db_name = 'db_tanah_bali'
    
    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        
        print('----------------------------------------------')
        print(f'Connected to db {db_name}')
        
        db = client[db_name]
        collection = db[collection_name]
        
        documents = list(collection.find())

        print(f'Total documents get: {len(documents)}')
        
        client.close()
        print('Dissconnected!')
        
        return documents
        
    except Exception as e:
        print(e)
        
          
if __name__ == '__main__':
    get_data_mongo()