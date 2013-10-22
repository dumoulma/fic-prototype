'''
Created on Feb 2, 2013

@author: MathieuXPS

test

'''
import logging

from text import BnsTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.datasets import load_files
import numpy as np
import codecs as cs

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

corpus = r"/home/dumoulma/dataset/sieve/corpus6"
label_names = ['relevant', 'spam']

notices = load_files(corpus, categories=label_names, load_content=False)
data = [cs.open(filename,'r','UTF-8').read() for filename in notices.filenames]
n_samples = len(data)
print("text read:{0}".format(n_samples))

count = CountVectorizer(max_df=0.5,ngram_range=(1,1), stop_words='english')
X = count.fit_transform(data)
Y = notices.target

X = BnsTransformer(Y, count.vocabulary_).fit_transform(X)

# Split the dataset in two sets, test and train
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.6, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': 10.0 ** np.arange(-4, -1),
                     'C': 10.0 ** np.arange(-2, 3)},
                    #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                    #{'kernel':['poly'], 'C':[0.1,1,10,100],}
                    ]

scores = [
    ('precision', precision_score),
    ('recall', recall_score),
]

for score_name, score_func in scores:
    print("# Tuning hyper-parameters for %s" % score_name)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, score_func=score_func)
    clf.fit(X_train, y_train, cv=5)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
