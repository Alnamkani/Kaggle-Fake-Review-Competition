from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import argparse
import os
import sys
import pickle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

data_validation = pd.read_csv('reviews_validation.csv' ,quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
data_train = pd.read_csv('reviews_train.csv' ,quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
percentage_val = len(data_validation) / (len(data_validation) + len(data_train))
all_data = pd.concat([data_train, data_validation])
all_data = all_data.sample(frac=1)

all_data['rating'] /= 5
        
# num_labels = 1
# labels = {}
# for cat in all_data['category'].unique():
#     labels[cat] = num_labels
#     num_labels += 1

# all_data = all_data.replace(labels)
# all_data['category'] /= num_labels

# all_data['len_str'] = [len(all_data.iloc[x]['text_']) for x in range(len(all_data))]
# all_data['len_str'] /= max(all_data['len_str'])
# print(all_data['category'][1])

all_data['len_str'] = all_data['text_'].str.len()
all_data['len_str'] /= max(all_data['len_str'])
# print(all_data.describe())

# all_data = all_data.groupby('category').get_group('Toys_and_Games_5')
# print(all_data.groupby('category').groups)

vectorizer = TfidfVectorizer(max_features= 5000)
# Transform text data to list of strings
corpora = all_data['text_'].astype(str).values.tolist()
# Obtain featurizer from data
vectorizer.fit(corpora)
# Create feature vector
V = vectorizer.transform(corpora)

give_this = np.insert(V.toarray(), len(V.toarray()[0]), all_data['len_str'], axis= 1)
# give_this = np.insert(give_this, len(give_this[0]), all_data['rating'], axis= 1)
# give_this = np.insert(give_this, len(give_this[0]), all_data['category'], axis= 1)


X, y = give_this, all_data['real review?']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

regr = MLPClassifier(hidden_layer_sizes = (10000, 5000, 1000, 100) , random_state=1, max_iter=500).fit(X_train, y_train)
regr.predict(X_test[:2])

print(regr.score(X_test, y_test))
