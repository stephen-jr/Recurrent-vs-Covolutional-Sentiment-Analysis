from data_utils import preprocess, tokenize
from imports import np, EarlyStopping, pd, train_test_split, dill, os, tm, load_model


class ModelOperation:
    def __init__(self):
        self.train_dataset = None
        self.model = None
        self.type = None
        self.X = self.Y = self.X_Test = self.X_Train = self.Y_Train = self.Y_Test = self.tokenizer = None
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
            # self.train_dataset['text'].astype('str').apply(preprocess)
            self.X = np.array(self.train_dataset['SentimentText'].apply(preprocess).to_list())
            self.Y = np.array(self.train_dataset['Sentiment'])
        except KeyError as e:
            exit(e)
        except Exception as e:
            exit(("Unknown Error(input_train_dataset)", e))

    def train_tokenize(self):
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(self.X,
                                                                                self.Y,
                                                                                test_size=0.25,
                                                                                random_state=42)
        self.X_Train, self.X_Test, self.vocab_size, self.tokenizer = tokenize(tokenizer=None, data={
            'train_text': self.X_Train,
            'test_text': self.X_Test,
        }, maxlen=self.maxlen)

    def create_model(self, model):
        print("Creating Sentiment Analytic model")
        self.model = model[0]
        self.type = model[1]
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def ld_model(self, pth):
        try:
            self.model = load_model(pth)
            dirt = 'saves/tokenizer.pkl'
            if os.path.exists(dirt):
                with open(dirt, 'rb') as f:
                    self.tokenizer = dill.load(f)
            else:
                exit("Tokenizer file doesn't exist")
            return True
        except Exception as e:
            exit(e)

    def train(self):
        print("Model Training Process in Session")
        es = EarlyStopping(monitor='val_loss', mode='min', patience=1, verbose=1)
        history = self.model.fit(self.X_Train,
                                 self.Y_Train,
                                 epochs=10,
                                 verbose=True,
                                 validation_data=(self.X_Test, self.Y_Test),
                                 batch_size=128,
                                 callbacks=[es])
        _, train_accuracy = self.model.evaluate(self.X_Train, self.Y_Train, verbose=True)
        print("\nTraining metrics : ")
        print(f'Accuracy : {train_accuracy:{5}} \n')
        #       f'Precision : {train_precision:{5}} \n'
        #       f'Recall : {train_recall:{5}} \n'
        #       f'F-Score : {train_fscore:{5}} \n')
        # print("=============Done==========\n")
        print(".............Validating model..............")
        # _, test_accuracy, test_precision, test_recall, test_fscore = self.model.evaluate(self.X_Test,
        #                                                                                  self.Y_Test,
        #                                                                                  verbose=True)
        _, test_accuracy = self.model.evaluate(self.X_Train, self.Y_Train, verbose=True)

        print("\nTesting metrics : ")
        print(f'Accuracy : {test_accuracy:{5}} \n')
        #        f'Precision : {test_precision:{5}} \n'
        #        f'Recall : {test_recall:{5}} \n'
        #        f'F-Score : {test_fscore:{5}}\n')
        print()
        ckpt = tm.gmtime(tm.time())
        ckpt = self.type + '-model-' + str(ckpt[0]) + '-' + str(ckpt[1]) + '-' + str(ckpt[2])

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
