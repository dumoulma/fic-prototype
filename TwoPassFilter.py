#!/usr/bin/env python3
'''
Perform two pass classification of a corpus using BNSVectorizer.
First classify relevant vs. spam
Second classify relevant vs. not relevant

Displays the metrics from the classification.
'''

import codecs as cs
from sklearn.datasets import load_files
from sklearn import svm
from ficlearn.feature_extraction.text import BnsVectorizer

if __name__ == '__main__':
    print("-------------------------------------------")
    print("Training SVM classifier")
    print("-------------------------------------------")
    print("Loading training data...")

    corpus = "bigcorpus"
    label_names = ['relevant', 'spam']
    
    notices = load_files(corpus, categories=label_names, load_content=False)
    data = [cs.open(filename, 'r', 'UTF-8').read() for filename in notices.filenames]
    n_samples = len(data)
    print("text read:{0}".format(n_samples))      
    
    Y = notices.target
    
    bns = BnsVectorizer(stop_words="english", charset_error='replace',
                               ngram_range=(1, 1), strip_accents='unicode', max_df=0.5)
    X = bns.fit_transform(data, Y)

    tunedC = 100.0
    tunedGamma = 0.0001 
    print("Training SVC classifier using rbf kernel, C={0} gamma={1}".format(tunedC, tunedGamma))
    clf = svm.SVC(C=tunedC, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                  gamma=tunedGamma, kernel='rbf', max_iter= -1, probability=False, shrinking=True,
                  tol=0.001, verbose=False).fit(X, Y)   
    print("Training complete!")
    print("svm.SVC score=", clf.score(X, Y))
    
    print("-------------------------------------------")
    print("Prediction of new notices")
    print("-------------------------------------------")

    corpus = "bigcorpus"
    label_names = ['not-tagged']
    newNotices = load_files(corpus, categories=label_names, load_content=False)
    newData = [cs.open(filename, 'r', 'UTF-8').read() for filename in newNotices.filenames]
    
    n_samples = len(newData)
    print("text read:{0}".format(n_samples))    
    
    newX = bns.transform(newData)    
        
    Ypredicted = clf.predict(newX)
    print("predicted relevant=",sum(Ypredicted == 0))
    print("predicted spam=",sum(Ypredicted == 1))
    print("shape:",Ypredicted.shape)
    predPosIndex = [i for i in range(Ypredicted.shape[0]) if Ypredicted[i] == 0]
    
    print("Writing out predicted positive files...")
    newPosFilenames = [newNotices.filenames[i] for i in predPosIndex]
    newData = [cs.open(filename, 'r', 'UTF-8').read() for filename in newPosFilenames]
    
    fw = cs.open('relevant.txt',mode='w', encoding='utf8')
    for filename,content in zip(newPosFilenames, newData):
        fw.write(filename + '\n')
        fw.write(content)
        fw.write('-----------------------------------------------------------\n\n')
    fw.close()
    print("Work complete!")