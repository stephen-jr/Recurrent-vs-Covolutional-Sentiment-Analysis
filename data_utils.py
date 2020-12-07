from imports import re, Tokenizer, k


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
            pass
        else:
            tokenizer = Tokenizer(50000)
            tokenizer.fit_on_texts(data['train_text'])
        start = time()
        train_text = tokenizer.texts_to_sequences(data['train_text'])
        test_text = tokenizer.texts_to_sequences(data['test_text'])
        print('Tokenizing Time taken : {}s'.format(time() - start))
        vocab_size = len(tokenizer.word_index) + 1
        start = time()
        train_text = pad_sequences(train_text, padding="POST", maxlen=maxlen)
        test_text = pad_sequences(test_text, padding="POST", maxlen=maxlen)
        print('Padding Time taken : {}s'.format(time() - start))
        print('========= Done ============')
        return {
            'tokenized_train_text' : train_text,
            'tokenized_test_text': test_text,
            'vocab_size': vocab_size
        }
    except KeyError as e:
        print("Key Error : ", e)
    except:
        print('Unknown error. Tokenizer')

def _rcll(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + k.epsilon())
    return recall


def _prcsn(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())
    return precision


def _f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + k.epsilon()))