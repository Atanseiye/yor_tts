import pandas as pd
from model_training import model
from keras.models import Sequential
from preprocessing import token_pad, cleaning
from sklearn.model_selection import train_test_split



# split data into train and test
yor_train, yor_test, eng_train, eng_test = train_test_split(yor_text, eng_text, test_size=0.2, random_state=42)


# initialise model
model = model(Sequential())
model.fit(yor_train, eng_train, epochs=50, batch_size=128, validation_split=0.2) # train the model


