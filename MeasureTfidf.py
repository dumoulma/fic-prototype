'''
Created on Feb 3, 2013

@author: MathieuXPS
'''
import codecs as cs

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from metrics import crossValidationScores
import metrics as met

    
    
if __name__ == '__main__':    
    corpus = "corpus5"
    label_names = ['relevant', 'spam']
    
    notices = load_files(corpus, categories=label_names, load_content=False)
    data = [cs.open(filename, 'r', 'UTF-8').read() for filename in notices.filenames]
    n_samples = len(data)
    print("text read:{0}".format(n_samples))      
    
    tfidf = TfidfVectorizer(stop_words="english", strip_accents='unicode', max_df=0.5,
                            ngram_range=(1, 1))
    X = tfidf.fit_transform(data)
    Y = notices.target   
    
    vocab = tfidf.vocabulary_    
    print("vocab len=", len(vocab)) 

    test_size = 0.6
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, Y,
                                          test_size=test_size, random_state=0)
    
    print("--------------------------")
    print("TF-IDF - SVM")   
    print("--------------------------")
    
    clf = svm.SVC()
    clf.fit(X_train, y_train)   
    y_pred = clf.predict(X_test)
    
    print(metrics.classification_report(y_test, y_pred, target_names=label_names))
    print(metrics.confusion_matrix(y_test, y_pred))
    
    crossValidationScores(clf, X_train, y_train)
    
    ###############################################################################
    # Classification and ROC analysis
    
    # Run classifier with crossvalidation and plot ROC curves
    met.showRocAnalysis(X, Y)    
    
    
