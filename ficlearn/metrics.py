#!/usr/bin/env python3
'''
Utility functions to help with computing metrics.
'''

import numpy as np
from scipy import interp
import pylab as pl
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics

def crossValidationScores(clf, X, y, test_size=0.5, show_scores=False, random_state=0):
    '''
    Given a classifier, a feature matrix X and the target vector y, will print the cross-validation scores.
    '''
    print("-----------------------------------------------")
    print("Show cross validation scores")   
    print("-----------------------------------------------")
    
    n_samples = X.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=test_size, random_state=random_state)
    
    CVscores = cross_validation.cross_val_score(clf, X, y, cv=cv, score_func=metrics.accuracy_score)
    if show_scores is True:
        print("svm.SVC crossvalidated scores=", CVscores)
    
    print("Accuracy".ljust(10), ": %0.2f (+/- %0.2f)" % (CVscores.mean(), CVscores.std() / 2))    
    CVscores = cross_validation.cross_val_score(clf, X, y, cv=cv, score_func=metrics.recall_score)
    print("Recall".ljust(10), ": %0.2f (+/- %0.2f)" % (CVscores.mean(), CVscores.std() / 2)) 
    CVscores = cross_validation.cross_val_score(clf, X, y, cv=cv, score_func=metrics.f1_score)
    print("F1".ljust(10), ": %0.2f (+/- %0.2f)" % (CVscores.mean(), CVscores.std() / 2))

###############################################################################
# Classification and ROC analysis
def showRocAnalysis(X, y, test_size=0.5, class_weight=None, random_state=0):
    '''
        Perform a display the ROC Analysis given feature vectors X and the target vector y 
    '''
    #Run classifier with crossvalidation and plot ROC curves
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y,
                                          test_size=test_size, random_state=random_state)
    
    cv = StratifiedKFold(y, n_folds=5)
    classifier = svm.SVC(C=100.0, cache_size=200, class_weight=class_weight, coef0=0.0, degree=3,
                  gamma=0.0001, kernel='rbf', max_iter= -1, probability=True, shrinking=True,
                  tol=0.001, verbose=False).fit(X_train, y_train)
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    #all_tpr = []    
    for _ in cv:
        probas_ = classifier.predict_proba(X_test)
        # Compute ROC curve and area the curve
        fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0        
        #roc_auc = auc(fpr, tpr)
        #pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    
    pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    pl.plot(mean_fpr, mean_tpr, 'k--',
            label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('BNS weights with SVM classification ROC curve')
    pl.legend(loc="lower right")
    pl.show() 
