from imports import Embedding, LSTM, Dense, Dropout, Sequential, Flatten


class RNN:
    def __init__(self, vocab_size, maxlen):
        self.type = "RNN"
        # _input = Input((maxlen, ), dtype='int32')
        # embedding = Embedding(vocab_size, 100)(_input)
        # lstm = LSTM(64)(embedding)
        # dense = Dense(1, activation='sigmoid')(lstm)
        # self.model = Model(inputs=_input, output=dense)
        self.model = Sequential()
        print("Creating Recurrent Sentiment Analytic model")
        self.model.add(Embedding(vocab_size, 100, input_length=maxlen))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        # self.model.add(LSTM(64))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

    def init(self):
        return self.model, self.type
