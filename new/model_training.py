from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Input, SpatialDropout1D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping




def model_2(model, max_word, max_len):
    # Encoder
    pass