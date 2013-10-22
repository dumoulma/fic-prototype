'''
Created on 2013-01-21

@author: Mathieu Dumoulin
'''
import numpy as np

from sklearn.datasets import load_files
import codecs as cs
from sklearn.linear_model import BayesianRidge
from sklearn import cross_validation
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from ficlearn.feature_extraction.text import BnsVectorizer

def selectFeatures(X_train, X_test, y_train, k=None):
    print("n_features".ljust(15), ":", X_train.shape[1], "features")
    
    if k is None:
        k = int(0.5 * n_features)
        
    ch2 = SelectKBest(chi2, k=k)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    
    print("n_features".ljust(15), ":", X_train.shape[1], "features")
    
    return X_train, X_test

if __name__ == '__main__':    
    print("-----------------------------------------------")
    print("Load corpus and vectorize with BNSVectorizer")   
    print("-----------------------------------------------")
    corpus = "corpus6"
    label_names = ['relevant', 'spam']
    
    notices = load_files(corpus, categories=label_names, load_content=False)
    data = [cs.open(filename, 'r', 'UTF-8').read() for filename in notices.filenames]
    
    n_samples = len(data)
    Y = notices.target

    print("Corpus size".ljust(15), ":", n_samples, "examples")   
    print("spam".ljust(15), ":", np.sum(Y), "examples")       
    print("relevant".ljust(15), ":", Y.shape[0] - np.sum(Y), "examples")
    
    bns = BnsVectorizer(stop_words="english", charset_error='replace',
                        ngram_range=(1, 1), strip_accents='unicode',
                        max_df=0.5, min_df=3)
    X_bns = bns.fit_transform(data, Y).todense()
      
    n_features = X_bns.shape[1]
     
      
    print("-----------------------------------------------")
    print("Split labeled examples into train and test sets")   
    print("-----------------------------------------------")
    test_size = 0.5
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X_bns, Y,
                                          test_size=test_size, random_state=0)
    

#    k = int(0.5 * n_features)
#    print("-----------------------------------------------")
#    print("Perform chi2 feature selection k=", k)   
#    print("-----------------------------------------------")
#    X_train, X_test = selectFeatures(X_train, X_test, y_train, k)

    print("-----------------------------------------------")
    print("SVM Classification of training set")   
    print("-----------------------------------------------")
    class_weight = {0:5}
    print("Class weight=", class_weight)
    clf = BayesianRidge(compute_score=True).fit(X_train, y_train)
    print("Test svm.SVC score=", clf.score(X_test, y_test))
    print("Train svm.SVC score=", clf.score(X_train, y_train))
    
    print("-----------------------------------------------")
    print("Metrics on TEST SET")   
    print("-----------------------------------------------")    
    y_pred = clf.predict(X_test)
    
    print(metrics.classification_report(y_test, y_pred, target_names=label_names))
    print(metrics.confusion_matrix(y_test, y_pred))       
    
    print("-----------------------------------------------")
    print("Metrics on TRAIN SET")   
    print("-----------------------------------------------")    
    y_predTrain = clf.predict(X_train)
    
    print(metrics.classification_report(y_train, y_predTrain, target_names=label_names))
    print(metrics.confusion_matrix(y_train, y_predTrain))       

    #met.crossValidationScores(clf, X_train, y_train)
    
    # met.showRocAnalysis(X_bns, Y)       
    
    
