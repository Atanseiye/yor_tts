

yor_text = pd.read_csv('')
eng_text = pd.read_csv('')

yor_text = cleaning(yor_text)
eng_text = cleaning(eng_text)

yor_text = token_pad(yor_text)
eng_text = token_pad(eng_text)

# split data into train and test
yor_train, yor_test, eng_train, eng_test = train_test_split(yor_text, eng_text, test_size=0.2, random_state=42)


# initialise model
model = model(Sequential())
model.fit(yor_train, eng_train, epochs=50, batch_size=128, validation_split=0.2) # train the model


