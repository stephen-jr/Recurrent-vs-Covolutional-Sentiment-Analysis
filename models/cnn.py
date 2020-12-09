from data_utils import preprocess, tokenize, _rcll, _prcsn, _f1
from imports import load_model, EarlyStopping, Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Sequential, pd, \
    train_test_split, dill, os, tm, pad_sequences


class CNN:
    def __init__(self):
        self.train_dataset = pd.DataFrame()
        self.model = Sequential()
        self.X = self.Y = self.X_Test = self.X_Train = self.Y_Train = self.Y_Test, self.tokenizer = ''
        self.vocab_size, self.maxlen = 0, 140

    def input_train_data(self, dataset):
        if dataset.split('.')[-1] == 'csv':
            try:
                self.train_dataset = pd.read_csv(dataset, error_bad_lines=False, encoding='ISO-8859-1')
            except Exception as e:
                return "Error on loading CSV Dataset", e
        elif dataset.split('.')[-1] == 'json':
            try:
                self.train_dataset = pd.read_json(dataset, lines=True)
            except Exception as e:
                return "Error on loading JSON Dataset", e
        else:
            exit("Specify a csv or json dataset")
        try:
            self.train_dataset['text'].astype('str').apply(preprocess)
            self.X = self.train_dataset['SentimentText']
            self.Y = self.train_dataset['Sentiment']
        except KeyError:
            exit("Text data have no text field. Ensure your dataset has a text 'column'")
        except Exception as e:
            exit(("Unknown Error(input_train_dataset)", e))

    def train_tokenize(self):
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(self.X,
                                                                                self.Y,
                                                                                test_size=0.25,
                                                                                random_state=42)
        self.X_Train, self.X_Test, self.vocab_size = tokenize(tokenizer='', data={
            'train_text': self.X_Train,
            'test_text': self.X_Test,
        }, maxlen=self.maxlen)

    def create_model(self):
        print("Creating Sentiment Analytic model")
        self.model.add(Embedding(self.vocab_size, 300, input_length=self.maxlen))
        self.model.add(Conv1D(64, 2, activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Conv1D(128, 2, activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Conv1D(256, 3, activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', _prcsn, _rcll, _f1])
        self.model.summary()

    def ld_model(self, pth):
        try:
            self.model = load_model(pth, custom_objects={'prcsn': _prcsn, 'rcll': _rcll, 'f1_m': _f1})
            dirt = 'saves/tokenizer.pkl'
            if os.path.exists(dirt):
                with open(dirt, 'rb') as f:
                    self.tokenizer = dill.load(f)
            else:
                exit("Tokenizer file doesn't exist")
            return True
        except Exception as e:
            return 'Unknown Error(ld_model)', e

    def train(self):
        print("CNN Model Training Process in Session")
        es = EarlyStopping(monitor='val_loss', mode='min', patience=1, verbose=1)
        history = self.model.fit(self.X_Train,
                                 self.Y_Train,
                                 epochs=10,
                                 verbose=True,
                                 validation_data=(self.X_Test, self.Y_Test),
                                 batch_size=128,
                                 callbacks=[es])
        _, train_accuracy, train_precision, train_recall, train_fscore = self.model.evaluate(self.X_Train,
                                                                                             self.Y_Train,
                                                                                             verbose=True)
        print("Training metrics : \n")
        print(f'Accuracy : {train_accuracy:{5}} \n'
              f'Precision : {train_precision:{5}} \n'
              f'Recall : {train_recall:{5}} \n'
              f'F-Score : {train_fscore:{5}} \n')
        print("=============Done==========\n")
        print(".............Validating model..............")
        _, test_accuracy, test_precision, test_recall, test_fscore = self.model.evaluate(self.X_Test,
                                                                                         self.Y_Test,
                                                                                         verbose=True)

        print("Testing metrics\n")
        print(f'Accuract : {test_accuracy:{5}} \n'
              f'Precision : {test_precision:{5}} \n'
              f'Recall : {test_recall:{5}} \n'
              f'F-Score : {test_fscore:{5}}\n')
        print()
        ckpt = tm.gmtime(tm.time())
        ckpt = 'model-' + str(ckpt[0]) + '-' + str(ckpt[1]) + '-' + str(ckpt[2])

        if os.path.exists('saves/' + ckpt + '.h5'):
            os.remove('saves/' + ckpt + '.h5')

        self.model.save('saves/' + ckpt + '.h5')

        if os.path.exists('saves/tokenizer.pkl'):
            os.remove('saves/tokenizer.pkl')

        with open('saves/tokenizer.pkl', 'wb') as f:
            dill.dump(self.tokenizer, f)
        return history

    def classify(self, text):
        text = preprocess(text)
        text = [text]
        text = tokenize(self.tokenizer, {'text': text}, self.maxlen)
        pred = self.model.predict_classes(text)
        rt = {}
        if len(pred) == 1:
            if pred[0] > 0.5:
                rt['classification'] = 'Positive'
            else:
                rt['classification'] = 'Negative'
            rt['score'] = pred[0]
        else:
            exit('Error obtaining prediction classification')
        return rt
