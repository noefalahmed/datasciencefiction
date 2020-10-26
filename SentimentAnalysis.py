#Sentiment Analysis using Naive Bayes.


import pandas as p
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)



data = p.read_csv("Question2 Dataset.tsv","\t",usecols={'sentiment','review'},dtype={'sentiment':int,'review':str})

labels=data['sentiment']
revs=data['review']
print(len(revs))
for i in range(0,len(revs)):

    soup = BeautifulSoup(revs[i], 'html.parser')
    revs[i]=soup.get_text()


X_train, X_test, y_train, y_test = train_test_split(revs,labels)

vectorizer=CountVectorizer()
X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)

results = []
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))


