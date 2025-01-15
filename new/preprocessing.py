import re
import string
import nltk
# import tensorflow as tf
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


stemmer = nltk.SnowballStemmer('english')
stopword = set(stopwords.words('yoruba'))





# max_word = 50000
# max_len = 300

def token_pad(text, max_word=5000, max_len=300):
    '''
    this function takes in a clean text
    tokenizes it, converts it to sequence and then
    pads it.

    It returns the padded sequence, the max_len of the sequence
    and then the max_word
    '''
    # tokenize the text
    tokenizer = Tokenizer(num_words=max_word)
    tokenizer.fit_on_texts([text])

    # convert text to sequence and pad it.
    sequences = tokenizer.texts_to_sequences([text])

    vocab_size = len(tokenizer.word_index) + 1
    max_len = max([len(seq) for seq in sequences])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')


    return {'padded_sen':padded, 
            'vocab_size':vocab_size, 
            'max_len':max_len}