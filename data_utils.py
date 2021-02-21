from imports import string, re, Tokenizer, time, pad_sequences, k, plt


def preprocess(text):
    punc = string.punctuation.replace('.', '').replace(',', '')
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\(\d+\)', ' ', text)
    text = re.sub(r'pic.\S+', ' ', text)
    text = re.sub(r'@\s\w+', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.compile(r'<[^>]+>').sub(' ', text)
    text = re.sub(r'[' + punc + ']', ' ', text)
    text = re.sub(r'RT', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(tokenizer, data, maxlen):
    try:
        print('..............Tokenizing...............')
        if tokenizer:
            text = tokenizer.texts_to_sequences(data['text'])
            text = pad_sequences(text, padding='post', maxlen=maxlen)
            return text
        else:
            tokenizer = Tokenizer(100000)
            tokenizer.fit_on_texts(data['train_text'])
            start = time()
            train_text = tokenizer.texts_to_sequences(data['train_text'])
            test_text = tokenizer.texts_to_sequences(data['test_text'])
            print('Tokenizing Time taken : {}s'.format(time() - start))
            vocab_size = len(tokenizer.word_index) + 1
            start = time()
            train_text = pad_sequences(train_text, padding='post', maxlen=maxlen)
            test_text = pad_sequences(test_text, padding='post', maxlen=maxlen)
            print('Padding Time taken : {}s'.format(time() - start))
            print('========= Done ============')
            return train_text, test_text, vocab_size, tokenizer
    except KeyError as e:
        print("Key Error : ", e)
    except Exception as e:
        print('Unknown error. Tokenizer', e)


def plot(hist):
    print('.......Generating Diagramatic Representation.......')
    plt.style.use('ggplot')
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    plt.savefig('')
