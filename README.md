# Kaggle-Fake-Review-Competition
This code was written as part of a class competition. The class is CS373 taught at Purdue University in the fall 2021 semester.
I'm unable to share the csv files and data due to size and privcy restrictions.
The following link takes you to the Kaggle competition page: https://www.kaggle.com/c/fake-review-competition/overview

***

## The side.py and look_at_data.py files

I have used these two files to examine the data before training and making predictions.

***

## Approaches 
1. complex.py 
    * This approach uses primarily a Linear Regression model.  
    * I have standardized the rating variable (a 5-star rating variable) by dividing by 5, then I gave each category a unique number, and I found that fake reviews tend to be longer so I added a length variable (len_str), and at last, I tokenized the review text using TFIDF.
    * I used scikit-learn library to create the Linear Regression model and make the prediction. 
2. complex2.py
    * Everything is the same as complex.py except that I haven't used the category, and len_str variable.
3. rnn.py
    * This is a recurrent neural network model.
    * I used the rating, category, and len_str variable.
    * And I used Keras Tokenizer, and keras library to make the RNN model.
    * This is the model that I used to score a 0.94730 score.

My name is C-3PO in the leaderboard.   
