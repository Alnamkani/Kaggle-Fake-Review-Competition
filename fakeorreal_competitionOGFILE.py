#!/usr/bin/env python
# coding: utf-8

# In[33]:


import argparse
import os
import sys
import pickle

from pandas.core.reshape.concat import concat
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


try:
    from sklearn.externals import joblib
except:
    import joblib


def run(arguments):
    test_file = None
    train_file = None
    validation_file = None
    joblib_file = "LR_model.pkl"


    parser = argparse.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('-e', '--test', help='Test attributes (to predict)')
    group1.add_argument('-n', '--train', help='Train data')
    parser.add_argument('-v', '--validation', help='Validation data')

    args = parser.parse_args(arguments)

    Train = False
    Test = False
    Validation = False

    if args.test != None:
        Test = True
            
    else:
        if args.train != None:
            print(f"Training data file: {args.train}")
            Train = True

        if args.validation != None:
            print(f"Validation data file: {args.validation}")
            Validation = True

    if Train and Validation:
        data_validation = pd.read_csv(args.validation,quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
        data_train = pd.read_csv(args.train,quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})

        percentage_val = len(data_validation) / (len(data_validation) + len(data_train))
        all_data = pd.concat([data_train, data_validation])
        all_data = all_data.sample(frac=1)
        all_data = pd.concat([all_data, all_data, all_data, all_data, all_data])
        # all_data['rating'] /=  5

        # x = 1
        # labels = {}
        # for cat in all_data['category'].unique():
        #     labels[cat] = x
        #     x += 1

        # all_data = all_data.replace(labels)
        # all_data['category'] /= x

        # all_data['len_str'] = [len(all_data.iloc[x]['text_']) for x in range(len(all_data))]
        # all_data['len_str'] /= max(all_data['len_str'])

        file_validation = all_data.sample(frac=percentage_val)
        file_train = all_data.drop(file_validation.index)
        
        # real review? 1=real review, 0=fake review
        # Category: Type of product
        # Product rating: Rating given by user
        # Review text: What reviewer wrote
        for hyper in [4000, 5000, 5500, 6000, 6500]:
            # Create TfIdf vector of review using 5000 words as features
            vectorizer = TfidfVectorizer(max_features= hyper - 3)
            # Transform text data to list of strings
            corpora = file_train['text_'].astype(str).values.tolist()
            # Obtain featurizer from data
            vectorizer.fit(corpora)
            # Create feature vector
            X = vectorizer.transform(corpora)


            print("Words used as features:")
            try:
                print(vectorizer.get_feature_names_out())
            except:
                print(vectorizer.get_feature_names())

            # Saves the words used in training
            with open('vectorizer.pk', 'wb') as fout:
                pickle.dump(vectorizer, fout)
            
            corpora_validation = file_validation['text_'].astype(str).values.tolist()
            X_validation = vectorizer.transform(corpora_validation)

            best_accuracy = 0

            # TODO: The following code is performing regularization incorrectly.
            # Your goal is to fix the code.
            for C in [100, 75, 50, 10, 1]:
                lr = LogisticRegression(penalty="l1", tol=0.001, C=C, fit_intercept=True, solver="liblinear", intercept_scaling=1, random_state=42)
                # You can safely ignore any "ConvergenceWarning" warnings
                # # print(f"max: {np.amax(X.toarray())}")
                # X_RC = np.insert(X.toarray(), len(X.toarray()[0]), file_train['len_str'], axis=1)
                # X_RC = np.insert(X_RC, len(X_RC[0]), file_train['rating'], axis=1)
                # X_RC = np.insert(X_RC, len(X_RC[0]), file_train['category'], axis=1)
                # print(X_RC)
                # X_RC = np.insert(X_RC, len(X_RC[0]), file_train['rating'], axis=1)
                lr.fit(X.toarray(), file_train['real review?'])
                # lr.fit(X_RC, file_train['real review?'])

                # # Get logistic regression predictions
                # X_VAL_RC = np.insert(X_validation.toarray(), len(X_validation.toarray()[0]), file_validation['len_str'], axis=1)
                # X_VAL_RC = np.insert(X_VAL_RC, len(X_VAL_RC[0]), file_validation['rating'], axis=1)
                # X_VAL_RC = np.insert(X_VAL_RC, len(X_VAL_RC[0]), file_validation['category'], axis=1)

                y_hat = lr.predict_proba(X_validation.toarray())[:,1]
                # y_hat = lr.predict_proba(X_VAL_RC)[:,1]

                y_pred = (y_hat > 0.5) + 0 # + 0 makes it an integer

                # Accuracy of predictions with the true labels and take the percentage
                # Because our dataset is balanced, measuring just the accuracy is OK
                accuracy = (y_pred == file_validation['real review?']).sum() / file_validation['real review?'].size
                print(f'hyper: {hyper} Accuracy {accuracy}')
                print(f'Fraction of non-zero model parameters {np.sum(lr.coef_!=0)+1}')
            
                if accuracy > best_accuracy:
                    # Save logistic regression model
                    joblib.dump(lr, joblib_file)
                    best_accuracy = accuracy


    elif Test:
        # This part will be used to apply your model to the test data
        vectorizer = pickle.load(open('vectorizer.pk', 'rb'))
            
        # Read test file
        file_test = pd.read_csv(args.test,quotechar='"',usecols=[0,1,2,3,4],dtype={'ID':int,'real review?': int,'category': str, 'rating': int, 'text_': str})
        
        # x = 1
        # labels = {}
        # for cat in file_test['category'].unique():
        #     labels[cat] = x
        #     x += 1

        # file_test = file_test.replace(labels)

        # Transform text into list of strigs
        corpora = file_test['text_'].astype(str).values.tolist()
        # Use the words obtained in training to encode in testing
        X = vectorizer.transform(corpora)

        # Load trained logistic regression model
        lr = joblib.load(joblib_file)

        # # Competition evaluation is AUC... what is the correct output for AUC evaluation?
        # X_TEST = np.insert(X.toarray(), len(X.toarray()[0]), file_test['category'], axis=1)
        # X_TEST = np.insert(X_TEST, len(X_TEST[0]), file_test['rating'], axis=1)

        y_hat = lr.predict_proba(X.toarray())[:,1]
        # y_hat = lr.predict_proba(X_TEST)[:,1]

        y_pred = (y_hat > 0.5)+0 # + 0 makes it an integer

        print(f"ID,real review?")
        for i,y in enumerate(y_pred):
            print(f"{i},{y}")


    else:
        print("Training requires both training and validation data files. Test just requires test attributes.")

        
if __name__ == "__main__":
    run(sys.argv[1:])
    

