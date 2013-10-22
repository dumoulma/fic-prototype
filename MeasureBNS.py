#!/usr/bin/env python3
'''
Perform classification of a corpus using BnsVectorizer.
Displays the metrics from the classification.
'''

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from ficlearn.feature_extraction.text import BnsTransformer
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from ficlearn.metrics import crossValidationScores
import codecs as cs

if __name__ == '__main__':
    corpus = "/home/dumoulma/dataset/sieve/corpus6"
    label_names = ['relevant', 'spam']
    
    notices = load_files(corpus, categories=label_names, load_content=False)
    data = [cs.open(filename, 'r', 'UTF-8').read() for filename in notices.filenames]
    n_samples = len(data)
    print("text read:{0}".format(n_samples))    
    
    countVec = CountVectorizer(stop_words="english", binary=True,
                               ngram_range=(1, 1), strip_accents='unicode')
    X = countVec.fit_transform(data)
    Y = notices.target       
    
    vocab = countVec.vocabulary_
    
    bns = BnsTransformer(y=Y, vocab=vocab)
    
        
    print("vocab len=", len(vocab)) 
    X_bns = bns.transform(X)

    test_size = 0.5
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X_bns, Y,
                                          test_size=test_size, random_state=0)
   
    print("--------------------------")
    print("BNS")   
    print("--------------------------")
 
    clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                  gamma=0.0001, kernel='rbf', max_iter= -1, probability=False, shrinking=True,
                  tol=0.001, verbose=False).fit(X_train, y_train)   
    print("svm.SVC score=", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    
    print(metrics.classification_report(y_test, y_pred, target_names=label_names))
    print(metrics.confusion_matrix(y_test, y_pred))       

    crossValidationScores(clf, X_train, y_train)

    print("--------------------------")
    print("BNS - MultinomialNB")   
    print("--------------------------")
    from sklearn.naive_bayes import MultinomialNB
    X = X.todense()
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, Y,
                                          test_size=test_size, random_state=50)
        
    clf = MultinomialNB().fit(X_train, y_train)   
    print("MultinomialNB score=", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    
    print(metrics.classification_report(y_test, y_pred, target_names=label_names))
    print(metrics.confusion_matrix(y_test, y_pred))
    
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=test_size, random_state=0)
    
    CVscores = cross_validation.cross_val_score(clf, X, Y, cv=cv, score_func=metrics.accuracy_score)
    print("svm.SVC crossvalidated score=", CVscores)
    
    crossValidationScores(clf, X_train, y_train)
