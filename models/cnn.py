from imports import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Sequential


class CNN:
    def __init__(self, vocab_size, maxlen):
        self.type = "CNN"
        self.model = Sequential()
        print("Creating Convolutional Sentiment Analytic model")
        self.model.add(Embedding(vocab_size, 300, input_length=maxlen))
        self.model.add(Conv1D(128, 2, activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Conv1D(256, 3, activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

    def init(self):
        return self.model, self.type
