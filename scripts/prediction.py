import csv
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi

base_dir = './models/'

xgb_model = f'{base_dir}xgboost_model.pkl'
rfr_model = f'{base_dir}rfr_model.pkl'
mlr_model = f'{base_dir}mlr_model.pkl'
subdistrict_label_encoder = f'{base_dir}subdistrict_encode.pkl'
regency_label_encoder = f'{base_dir}regency_encode.pkl'

xgb_model = joblib.load(xgb_model)
rfr_model = joblib.load(rfr_model)
mlr_model = joblib.load(mlr_model)
subdistrict_label_encoder = joblib.load(subdistrict_label_encoder)
regency_label_encoder = joblib.load(regency_label_encoder)

MONGO_URI = 'mongodb://localhost:27017'
# MONGO_URI = 'mongodb+srv://wahpram2607:Bangli123.@cluster0.yiobiyk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
MONGO_DBNAME = 'db_tanah_bali'

client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client[MONGO_DBNAME]
collection = db['tanah_bali_clean']


def predict():
    day = float(input("Enter the day: "))
    month = float(input("Enter the month: "))
    year = float(input("Enter the year: "))
    
    regency_coordinates = {
        'Denpasar': (-8.65795, 115.22365),
        'Badung': (-8.5407, 115.16285),
        'Bangli': (-8.33015, 115.34265),
        'Gianyar': (-8.48135, 115.2989),
        'Tabanan': (-8.4385, 115.01625),
        'Klungkung': (-8.5289, 115.4811),
        'Karangasem': (-8.3593, 115.54985),
        'Jembrana': (-8.31665, 114.54945),
        'Buleleng': (-8.1111, 115.23255)
    }
    
    pipeline = [
            {'$group': {'_id': {'regency': '$regency', 
                                'subdistrict': '$subdistrict'}}},
            {'$project': {'regency': '$_id.regency', 
                          'subdistrict': '$_id.subdistrict', '_id': 0}}
    ]
    
    pipeline_avg = [
            {'$group': {'_id': '$regency',
                        'avg_price_per_m2': {'$avg': '$price_per_m2'},
                        'count': {'$sum': 1}}},
            {'$project': {'_id': 0,
                        'regency': '$_id',
                        'avg_price_per_m2': 1,
                        'count': 1}}
    ]
    
    regency_avg_cursor = collection.aggregate(pipeline_avg)
    regency_averages = {}
    for doc in regency_avg_cursor:
        regency_averages[doc['regency']] = {
            'avg_price_per_m2': doc['avg_price_per_m2']
        }
    
    regencies_subdistricts_cursor = collection.aggregate(pipeline)
        
    predictions = {}

    for doc in regencies_subdistricts_cursor:
        regency = doc['regency']
        subdistrict = doc['subdistrict']
        
        new_data = {
            'year': [year],
            'month': [month],
            'day': [day],
            'subdistrict': [subdistrict],
            'regency': [regency]
        }
        new_df = pd.DataFrame(new_data)
        
        new_df['regency_encoded'] = new_df['regency'].apply(lambda x: regency_label_encoder.get(x, -1))
        new_df['subdistrict_encoded'] = new_df['subdistrict'].apply(lambda x: subdistrict_label_encoder.get(x, -1))
        
        new_df.drop(['subdistrict', 'regency'], axis=1, inplace=True)
        
        xgb_pred = xgb_model.predict(new_df)
        xgb_pred = float(np.exp(xgb_pred[0]))
        # xgb_pred = format_rupiah(xgb_pred)
        
        rfr_pred = rfr_model.predict(new_df)
        rfr_pred = float(np.exp(rfr_pred[0]))
        # rfr_pred = format_rupiah(rfr_pred)
        
        mlr_pred = mlr_model.predict(new_df)
        mlr_pred = float(np.exp(mlr_pred[0]))
        # mlr_pred = format_rupiah(mlr_pred)
        
        if regency not in predictions:
            predictions[regency] = {}
        
        predictions[regency][subdistrict] = {
            'xgb_pred': xgb_pred,
            'rfr_pred': rfr_pred,
            'mlr_pred': mlr_pred
        }
    
    flattened_data = []
    for regency, subdistricts in predictions.items():
        for subdistrict, preds in subdistricts.items():
            avg_data = regency_averages.get(regency, {})
            coordinates = regency_coordinates.get(regency, (np.nan, np.nan))
            flattened_data.append({
                'regency': regency,
                'subdistrict': subdistrict,
                'xgb_pred': preds['xgb_pred'],
                'rfr_pred': preds['rfr_pred'],
                'mlr_pred': preds['mlr_pred'],
                'avg_price_per_m2': avg_data.get('avg_price_per_m2', np.nan),
                'latitude': coordinates[0],
                'longitude': coordinates[1]
            })

    csv_file_name = "./data/prediction_all_subdistrict.csv"

    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['regency', 'subdistrict', 'xgb_pred', 'rfr_pred', 'mlr_pred', 'avg_price_per_m2', 'latitude', 'longitude'])

        # Write the header
        writer.writeheader()

        # Write the data
        writer.writerows(flattened_data)
        

def geo_for_data():
    regency_coordinates = {'Denpasar': (-8.65795, 115.22365),
            'Badung': (-8.5407,	115.16285),
            'Bangli': (-8.33015, 115.34265),
            'Gianyar': (-8.48135, 115.2989),
            'Tabanan': (-8.4385, 115.01625),
            'Klungkung': (-8.5289, 115.4811),
            'Karangasem': (-8.3593, 115.54985),
            'Jembrana': (-8.31665, 114.54945),
            'Buleleng': (-8.1111, 115.23255)
        }


    documents = list(collection.find())
    
    for doc in documents:
        regency = doc['regency']
        if regency in regency_coordinates:
            latitude, longitude = regency_coordinates[regency]
            doc['latitude'] = latitude
            doc['longitude'] = longitude
    
    df = pd.DataFrame(documents)
    
    csv_file_name = "./data/regency_coordinates_2.csv"
    df.to_csv(csv_file_name, index=False)
    
    
if __name__ == '__main__':
    predict()
