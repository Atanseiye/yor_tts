from preprocessing import cleaning, token_pad
from pymongo import MongoClient
from secrets import mongo_uri
from torch import tensor
# import pandas as pd




data = fetch_data(mongo_uri, batch=5)
print(data)
