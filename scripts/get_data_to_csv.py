import csv
from pymongo import MongoClient
from pymongo.server_api import ServerApi


def mongo_to_csv():
    uri = "mongodb+srv://wahpram2607:Bangli123.@cluster0.yiobiyk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    db_name = 'db_tanah_bali'
    collection_name = 'tanah_bali_clean_new'
    csv_filepath = './data/tanah_bali_clean.csv'
    
    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        
        print('----------------------------------------------')
        print(f'Connected to db {db_name}')
        
        db = client[db_name]
        collection = db[collection_name]
        
        documents = collection.find()
        field_names = documents[0].keys() if documents else []
        
        with open(csv_filepath, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            for document in documents:
                writer.writerow(document)

        print(f'Total documents get: {len(list(documents))}')
        
        client.close()
        print('Dissconnected!')
        
        return documents
        
    except Exception as e:
        print(e)
        
          
if __name__ == '__main__':
    mongo_to_csv()