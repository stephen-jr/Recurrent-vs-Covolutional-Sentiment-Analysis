from imports import Embedding, LSTM, Dense, Dropout, Sequential, Flatten


class RNN:
    def __init__(self, vocab_size, maxlen):
        self.type = "RNN"
        print("Creating an RNN Model")
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, 200, input_length=maxlen))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))

    def init(self):
        return self.model, self.type
