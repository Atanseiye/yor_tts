from preprocessing import cleaning, token_pad
from pymongo import MongoClient
from secrets import mongo_uri
from torch import tensor
# import pandas as pd


def fetch_data(mongo_uri, batch=-1):
    client = MongoClient(mongo_uri)

    # database
    database = client['nkenne-ai']

    data_collection = database['english_yoruba']

    # Query the document
    query = {'type': 'annotations'}  # Modify query filter
    documents = data_collection.find(query)

    yor_text = []
    eng_text = []
    source = []
    # Process and display the documents
    for doc in documents:
        yor_text.append(doc['yoruba_text'])
        eng_text.append(doc['english_text'])

        # print(f"English Translation: {doc['english_text']}")
        # print(f"Audio Gender: {doc['audio_gender']}")
        source.append(doc['source'])
        # print(f"Verified Translation: {doc['translation_verified']}")
        # print(f"File URL: {doc['file_url']}")

    return {'yoruba':tensor(yor_text[:batch]),
            'englisg': tensor(eng_text[:batch])}


