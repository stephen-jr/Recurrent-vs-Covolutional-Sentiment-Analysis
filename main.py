from models import CNN, RNN, ModelOperation


def main():
    model = ModelOperation()
    model.input_train_data('data/train.csv')
    model.train_tokenize()
    nn = CNN(model.vocab_size, model.maxlen)  # RNN - same parameters
    nn = nn.init()
    model.create_model(model=nn)
    model.train()


if __name__ == '__main__':
    main()

