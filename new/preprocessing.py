

stemmer = nltk.SnowballStemmer('english')
stopword = set(stopwords.words('yoruba'))

def cleaning(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+/www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)

    return text



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