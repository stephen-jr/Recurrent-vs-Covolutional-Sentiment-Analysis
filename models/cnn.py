from data_utils import *
from imports import load_model, EarlyStopping, Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten


class CNN:
    def __init__(self, typ):
        if typ === 'new':
            pass
        elif type === 'load':
            pass
        else:
            print('Please Specify model type')
        pass

    def create_model(self):
        pass

    def train(self):
        print('...............Training Convolutional Model...............')
        es = EarlyStopping(monitor='val_loss', mode='min', patience=1, verbose=1)
        self.history = self.model.fit(self.X_Train,
                                      self.Y_Train,
                                      epochs=10,
                                      verbose=True,
                                      validation_data=(self.X_Test, self.Y_Test),
                                      batch_size=128,
                                      callbacks=[es])
        _, train_accuracy, train_precision, train_recall, train_fscore = self.model.evaluate(self.X_Train,
                                                                                             self.Y_Train,
                                                                                             verbose=True)
        print("Training Accuracy: {:.4f}".format(train_accuracy))
        print("Training metrics")
        print(f'Precision : {train_precision:{5}} Recall : {train_recall:{5}} F-Score : {train_fscore:{5}}')
        pass

    def validate(self):
        print('...............Validating Convolutional Model\'s Performance...............')
        _, test_accuracy, test_precision, test_recall, test_fscore = self.model.evaluate(self.X_Test,
                                                                                         self.Y_Test,
                                                                                         verbose=True)
        print("Testing Accuracy:  {:.4f}".format(test_accuracy))
        print("Testing metrics")
        print(f'Precision : {test_precision:{5}} Recall : {test_recall:{5}} F-Score : {test_fscore:{5}}')
        pass

    def prdct(self):
        pass
