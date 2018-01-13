# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:24:38 2017

@author: Ali Asghar Marvi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:27:59 2017

@author: Ali Asghar Marvi
"""
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string


# Input data files
train = pd.read_csv('Trainset.csv',encoding='latin-1')
test = pd.read_csv('Testsetwithoutanswer.csv',encoding='latin-1')


# Exploratory data analysis

# Percent missing values for each column
# there are no missing values
percent_missing = 100 * train.isnull().sum()/len(train)

# look at class imbalance
# classes are fairly balanced
poor = (train.rating == 'poor').sum()
awesome = (train.rating == 'awesome').sum()
good = (train.rating == 'good').sum()
awful = (train.rating == 'awful').sum()
average = (train.rating == 'average').sum()

# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

train_corpus = []
for i in range(len(train)):
    review = re.sub('[^a-zA-Z0-9]', ' ', train['review'][i])
    review = review.lower()
    review = review.split()
#    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    train_corpus.append(review)

test_corpus = []
for i in range(len(test)):
    review = re.sub('[^a-zA-Z0-9]', ' ', test['review'][i])
    review = review.lower()
    review = review.split()
#    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    test_corpus.append(review)

print(train_corpus[0])
def list_to_string(lists):
        mylistofstrings = ' '.join(map(str, lists))
        mylistofstrings = mylistofstrings.replace("["," ")
        mylistofstrings = mylistofstrings.replace("]"," ")
        mylistofstrings = mylistofstrings.replace("'"," ")
        mylistofstrings = mylistofstrings.replace(","," ")
        mylistofstrings = mylistofstrings.translate(str.maketrans('','',string.punctuation))
        return mylistofstrings
    
#stopwords = nltk.corpus.stopwords.words('english')    
lemmatizer = WordNetLemmatizer()   
stemmer = PorterStemmer()    
for i in range(len(train_corpus)):
    train_corpus[i] = nltk.word_tokenize(train_corpus[i])
#    train_corpus[i] = [word for word in train_corpus[i] if word not in stopwords]
    train_corpus[i] = [lemmatizer.lemmatize(word) for word in train_corpus[i]]
#    train_corpus[i] = [stemmer.stem(word) for word in train_corpus[i]]
    train_corpus[i] = [list(train_corpus[i])]
    train_corpus[i] = list_to_string(train_corpus[i])
    train_corpus[i] = train_corpus[i].replace("    "," ")
    
for i in range(len(test_corpus)):
    test_corpus[i] = nltk.word_tokenize(test_corpus[i])
#    test_corpus[i] = [word for word in test_corpus[i] if word not in stopwords]
    test_corpus[i] = [lemmatizer.lemmatize(word) for word in test_corpus[i]]
#    test_corpus[i] = [stemmer.stem(word) for word in test_corpus[i]]
    test_corpus[i] = [list(test_corpus[i])]
    test_corpus[i] = list_to_string(test_corpus[i])
    test_corpus[i] = test_corpus[i].replace("    "," ")
del i
del review


# format data for input
X_Train = np.array(train_corpus)
X_Test = np.array(test_corpus)
y = train.iloc[:, 1].values
print(y)
# remove unneeded old variables
del  test_corpus, train, train_corpus


## Multinomial Naive Bayes Classifier ##
# Build pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
classifier = Pipeline([('vect', CountVectorizer(analyzer='word',strip_accents='ascii', decode_error='replace')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
#                        ('rfc', RandomForestClassifier())
])
classifier.fit(X_Train, y)
predictions = classifier.predict(X_Test)
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'tfidf__use_idf': (True, False),
               'tfidf__norm' : ('l1','l2'),
               'tfidf__smooth_idf' : (True, False),
               'tfidf__sublinear_tf':(True,False),
              'clf__alpha': (0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5),
               'clf__fit_prior': (True, False),             
}
gs_clf = GridSearchCV(classifier, parameters, cv =5)
gs_clf.fit(X_Train, y)
predictions = gs_clf.predict(X_Test)

#from sklearn.model_selection import RandomizedSearchCV
#clf = RandomizedSearchCV(classifier, parameters, random_state=1, verbose=0, n_jobs=-1)
#clf.fit(X_Train,y)
#predictions = clf.predict(X_Test)

print(gs_clf.best_params_)
#print(gs_clf.cv_results_)
print(gs_clf.best_score_)
#print(clf.best_params_)
#print(clf.ccv_results_)
#print(clf.best_score_) 
results = pd.DataFrame([],columns=["id","rating"])
results["id"] = test["id"]
results["rating"] = predictions
results.to_csv("output_MNB_optimum.csv",index=False)