'''
Created on Feb 15, 2013

@author: MathieuXPS
'''
'''
Created on 2013-01-21

@author: Mathieu Dumoulin
'''
import numpy as np

from sklearn.datasets import load_files
import codecs as cs
from sklearn import svm
from sklearn import metrics
from ficlearn.feature_extraction.text import BnsVectorizer
from random import shuffle
import copy


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
    
    start = int(n_samples / 10)
    step = start
    recalls = []
    precisions = []
    sizes = []
    N_SAMPLES = copy.deepcopy(n_samples)
    for i in range(2,10,1):
        sliceIndex = int((i * 0.1 + 0.1) * N_SAMPLES)
        shuffle(data, )
        dataSlice = data[:sliceIndex]
        YSlice = np.copy(Y[:sliceIndex])
        n_samplesSlice = len(dataSlice)
        sizes.append(n_samplesSlice)
        
        print("Corpus size".ljust(15), ":", n_samplesSlice, "examples")   
        print("spam".ljust(15), ":", np.sum(YSlice), "examples")       
        print("relevant".ljust(15), ":", YSlice.shape[0] - np.sum(YSlice), "examples")
        
        bns = BnsVectorizer(stop_words="english", charset_error='replace',
                            ngram_range=(1, 1), strip_accents='unicode',
                            max_df=0.5, min_df=3)
        XSlice = bns.fit_transform(dataSlice, YSlice)
          
        n_features = XSlice.shape[1]
    
        class_weight = {0:5}
        clf = svm.SVC(C=100.0, cache_size=200, class_weight=class_weight, coef0=0.0, degree=3,
                      gamma=0.0001, kernel='rbf', max_iter= -1, probability=False, shrinking=True,
                      tol=0.001, verbose=False).fit(XSlice, YSlice)   
        score = clf.score(XSlice, YSlice)
        print("svm.SVC score=", score)
        
        y_pred = clf.predict(XSlice)
        recalls.append(metrics.recall_score(YSlice, y_pred, pos_label=0))
        precisions.append(metrics.precision_score(YSlice, y_pred, pos_label=0))    
        
    for size, precision, recall in zip(sizes, precisions, recalls):
        print("size:", size, "precision:", precision, "recall:", recall)    
    
    import pylab as pl    
    sizes = np.array(sizes)
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    pl.plot(sizes, precisions, 'go', label="Precision")
    pl.plot(sizes, precisions, color='green', linestyle='--')
    pl.plot(sizes, recalls, 'b^', label="Recall")
    pl.plot(sizes, recalls, color='blue', linestyle='--')
    
    pl.xlabel('Corpus Size')
    pl.ylabel('Rate')
    pl.title('Precision and Recall vs. Corpus Size')
    pl.grid(True)
    pl.legend(loc='lower right')
    pl.show()
