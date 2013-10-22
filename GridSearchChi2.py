'''
Created on Feb 2, 2013

@author: MathieuXPS
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.datasets import load_files
import time
import logging
import codecs as cs

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

corpus = "corpus5"
label_names = ['relevant', 'spam']

notices = load_files(corpus, categories=label_names, load_content=False)
data = [cs.open(filename,'r','UTF-8').read() for filename in notices.filenames]
n_samples = len(data)
print("text read:{0}".format(n_samples))

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words="english", charset_error='replace',
                        ngram_range=(1, 1), strip_accents='unicode',
                        max_df=0.5, min_df=3)),
    ('chi2', SelectKBest(chi2)),
    ('clf', SVC(C=100.0, cache_size=200, class_weight={0:7}, coef0=0.0, degree=3,
                      gamma=0.0001, kernel='rbf', max_iter= -1, probability=False, shrinking=True,
                      tol=0.001, verbose=False)),
])

parameters = {
    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    #'vect__max_df': (0.5, 0.75),
    #'vect__max_features': (None, 5000, 10000),
    'tfidf__ngram_range': [(1, 1),(1,2),(1,3),(1,4),(1,5)],  # words or bigrams
    'chi2__k':[1000,2000,3000],
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}
if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5, verbose=1)
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time.clock()
    grid_search.fit(data, notices.target)
    print( "done in %0.3fs" % (time.clock() - t0))
    print()
    
    print( "Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

