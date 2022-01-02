from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()

data_validation = pd.read_csv('reviews_validation.csv' ,quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
data_train = pd.read_csv('reviews_train.csv' ,quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
# data_test = pd.read_csv('reviews_test_attributes.csv',quotechar='"',usecols=[0,1,2,3,4],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
data_test = pd.read_csv('reviews_test_attributes.csv',quotechar='"',usecols=[0,1,2,3,4],dtype={'ID':int,'real review?': int,'category': str, 'rating': int, 'text_': str})


all_data = pd.concat([data_validation, data_train])
all_data['len_str'] = all_data['text_'].map(lambda x: len(x))
data_test['len_str'] = data_test['text_'].map(lambda x: len(x))

num_labels = 1
labels = {}
for cat in all_data['category'].unique():
    labels[cat] = num_labels
    num_labels += 1

all_data = all_data.replace(labels)
data_test = data_test.replace(labels)


# if __name__ == "__main__":
texts = list(all_data['text_'])
labels = list(all_data['real review?'])


print(data_test.columns)

texts_test = list(data_test['text_'])
texts_test = np.asarray(texts_test)

texts = np.asarray(texts)
labels = np.asarray(labels)


# print("number of texts :" , len(texts))
# print("number of labels: ", len(labels))

from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# number of words used as features
max_features = 50000
# cut off the words after seeing 500 words in each document(email)
maxlen = 500


# we will use 80% of data as training, 20% as validation data
training_samples = len(data_train)
validation_samples = len(data_validation)
# sanity check
print(len(texts) == (training_samples + validation_samples))
print("The number of training {0}, validation {1} ".format(training_samples, validation_samples))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

seq_test = tokenizer.texts_to_sequences(texts_test)

word_index = tokenizer.word_index
print("Found {0} unique words: ".format(len(word_index)))

data = pad_sequences(sequences, maxlen=maxlen)
data_test_pad = pad_sequences(seq_test, maxlen=maxlen)

data_test_pad = np.insert(data_test_pad, len(data_test_pad[0]), data_test['category'], axis=1)
data_test_pad = np.insert(data_test_pad, len(data_test_pad[0]), data_test['rating'], axis = 1)
data_test_pad = np.insert(data_test_pad, len(data_test_pad[0]), data_test['len_str'], axis = 1)

print("data shape: ", data.shape)

np.random.seed(42)
# shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


texts_train = np.insert(data, len(data[0]), all_data['category'], axis=1)
texts_train = np.insert(texts_train, len(texts_train[0]), all_data['rating'], axis = 1)
texts_train = np.insert(texts_train, len(texts_train[0]), all_data['len_str'], axis = 1)

# print(texts_train)
y_train = labels
# print(texts_train)
# texts_test = data[training_samples:]
# y_test = labels[training_samples:]

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history_rnn = model.fit(texts_train, y_train, epochs=4, batch_size=60, validation_split=0.2)

pred = model.predict(data_test_pad)

to_save = (pred > 0.5) + 0

to_save_file = pd.DataFrame()
to_save_file['ID'] = data_test['ID']
to_save_file['real review?'] = to_save

to_save_file.to_csv("submitRNN.csv", columns=["ID", 'real review?'], index=False)
# acc = model.evaluate(texts_test, y_test)
# proba_rnn = model.predict_proba(texts_test)
# from sklearn.metrics import confusion_matrix
# print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))
# # print(confusion_matrix(pred, y_test))
# print(pred)