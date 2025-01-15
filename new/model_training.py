from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Input, SpatialDropout1D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping

def model(model, max_word=5000, max_len=300):
    # model = Sequential()
    model.add(Embedding(max_word, 100, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(max_len, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    model.build(input_shape=(None, 100))
    model.summary()

    return model


def model_2(model, max_word, max_len):
    # Encoder
    pass