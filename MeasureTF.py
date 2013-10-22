'''
Created on Feb 3, 2013

@author: MathieuXPS
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn import metrics
from sklearn.datasets import load_files
import codecs as cs
                      
if __name__ == '__main__':
    corpus = "corpus5"
    label_names = ['relevant', 'spam']
    
    notices = load_files(corpus, categories=label_names, load_content=False, random_state=123)
    data = [cs.open(filename,'r','UTF-8').read() for filename in notices.filenames]
    
    n_samples = len(data)    
    print("text read:{0}".format(n_samples))
    Y = notices.target
    
    tfidf = CountVectorizer(stop_words="english", max_df=0.8, ngram_range=(1,2))
    X = tfidf.fit_transform(data).todense()   
    
    vocab = tfidf.vocabulary_    
    print("vocab len=", len(vocab)) 

    test_size = 0.6
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, Y,
                                          test_size=test_size, random_state=0)
    
    print("--------------------------")
    print("TF (MultinomialNB)")   
    print("--------------------------")
    #
    clf = MultinomialNB().fit(X_train, y_train)   
    print("MultinomialNB score=", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    
    print(metrics.classification_report(y_test, y_pred, target_names=label_names))
    print(metrics.confusion_matrix(y_test, y_pred))
    
    cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10, test_size=test_size, random_state=0)
    
    CVscores = cross_validation.cross_val_score(clf, X_train, y_train, cv=cv, score_func=metrics.accuracy_score)
    print("svm.SVC crossvalidated score=", CVscores)
    
    print("Accuracy: %0.2f (+/- %0.2f)" % (CVscores.mean(), CVscores.std() / 2))
    CVscores = cross_validation.cross_val_score(clf, X_train, y_train, cv=cv, score_func=metrics.recall_score)
    print("Recall: %0.2f (+/- %0.2f)" % (CVscores.mean(), CVscores.std() / 2))        
    CVscores = cross_validation.cross_val_score(clf, X_train, y_train, cv=cv, score_func=metrics.f1_score)
    print("F1: %0.2f (+/- %0.2f)" % (CVscores.mean(), CVscores.std() / 2))        
