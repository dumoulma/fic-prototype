#!/usr/bin/env python3
'''
Perform a gridsearch to find the best parameters for the SVM classification algorithm using TF-IDF vectorizer.

Run as a pipeline which vectorizes a collection of text with TF-IDF, then run feature selection with chi2
and finally classifies with SVM. 
'''

import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.datasets import load_files
import numpy as np
import codecs as cs


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

corpus = "corpus6"
label_names = ['relevant', 'spam']

notices = load_files(corpus, categories=label_names, load_content=False)
data = [cs.open(filename, 'r', 'UTF-8').read() for filename in notices.filenames]
n_samples = len(data)
print("text read:{0}".format(n_samples))

count = TfidfVectorizer(stop_words="english", charset_error='replace',
                        ngram_range=(1, 1), strip_accents='unicode',
                        max_df=0.5, min_df=3)
X = count.fit_transform(data)
Y = notices.target

# Split the dataset in two sets, test and train
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.6, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'gamma': 10.0 ** np.arange(-4, -1),
                     'C': 10.0 ** np.arange(-2, 3), 
                     'class_weight':[{0:7}], 
                    },                    
                    # {'kernel': ['linear'], }
                    # {'kernel':['poly'], 'C':[0.1,1,10,100],}
                    ]

scores = [
    ('precision', precision_score),
    ('recall', recall_score),
    ('f1-score', f1_score),
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
