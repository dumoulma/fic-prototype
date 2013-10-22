'''
Created on Feb 2, 2013

@author: MathieuXPS
'''

import numpy as np
import scipy.sparse as sp
from ficlearn.util.statslib import ltqnorm

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class BnsTransformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a normalized BNS or tf–BNS representation

    Tf means term-frequency while BNS means Bi-Normal Seperation. This is a 
    term weighting scheme that has been proposed by Georges Forman of HPlabs
    for use in document classification with SVM in the paper: LINK.

    The goal of using BNS instead of Tf-idf is to scale down the impact of tokens 
    that occur very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus, while taking into account the relative frequency of the word with regards
    to its label.

    Parameters
    ----------
    y     : numpy array, required
        The y array for the training examples, used in the calculation of BNS
        
    vocab : map, required
        A mapping for the document vocabulary words and their indices in the 
        X matrix
        
    use_tf : boolean, optional
        Enable inverse-document-frequency reweighting.

    References
    ----------

    """

    def __init__(self, y, vocab):
        self.vocab = vocab
        self.poslabels = y
        def f(a):
            if a is 0: return 1
            else: return 0
        f = np.vectorize(f)    
        self.neglabels = f(np.copy(y))
    
        self.pos = np.sum(self.poslabels)
        self.neg = np.sum(self.neglabels)
        self._bns_diag = None

    def fit(self, X, y=None):
        """Learn the bns vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        X = np.matrix(X.todense(), dtype=np.float64)
        for word in self.vocab:
            wordIndex = self.vocab[word]
            words = X[:, wordIndex].view(np.ndarray)
            
            if not self.is_word_feature(word):
                bns_score = 0
            else:
                bns_score = self.compute_bns(words)                              
            
            words *= bns_score
        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a bns or tf-bns representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)
        
        X = np.matrix(X.todense(), dtype=np.float64)
#        if self._bns_diag is not None:
#            n_features = X.shape[1]
#            expected_n_features = self._bns_diag.shape[0]
#            if n_features != expected_n_features:
#                raise ValueError("Input has n_features=%d while the model"
#                                 " has been trained with n_features=%d" % (
#                                     n_features, expected_n_features))
#            # *= doesn't work
#            X = X * self._bns_diag                    
#        else:
        for word in self.vocab:
            wordIndex = self.vocab[word]
            words = X[:, wordIndex].view(np.ndarray)
            
            if not self.is_word_feature(word):
                bns_score = 0
            else:
                bns_score = self.compute_bns(words)                              
            
            words *= bns_score
        
        return sp.coo_matrix(X, dtype=np.float64)
        
    def is_word_feature(self, word, verbose=False):
        """ false if word is not alphanumeric (alphabet letters or numbers 0-9)        
            true otherwise
        """
        if not word.isalnum() and " " not in word:
            return False
        elif " " in word:
            parts = word.split(" ")
            first = parts[0]; second = parts[1]
            if not first.isalnum() or not second.isalnum():                
                return False
        return True
    
    def compute_bns(self, words, verbose=False):
        """ compute the BNS score of the word of the vocabulary at the index wordIndex """
        wordsvec = words.reshape(words.shape[0])
        tp = np.sum(wordsvec * self.poslabels)
        tn = np.sum(wordsvec * self.neglabels)
        
        tpr = self.bounded_value(float(tp) / self.pos, 0.0005, 1 - 0.0005)        
        tnr = self.bounded_value(float(tn) / self.neg, 0.0005, 1 - 0.0005)
        
        bns_score = abs(ltqnorm(tpr) - ltqnorm(tnr))
        if verbose:
            print("tp={0} tn={1} tpr={2} tnr={3} bns_score={4}".format(tp, tn, tpr, tnr, bns_score))            
    
        return bns_score 
    
    def bounded_value(self, value, lower, upper):
        """ bound the value in the range [lower,upper] """
        
        if value < lower: value = lower
        elif value > upper: value = upper
        return value
    
from sklearn.feature_extraction.text import CountVectorizer
class BnsVectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of BNS features.

    Equivalent to CountVectorizer followed by BNSTransformer.

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If filename, the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have 'read' method (file-like
        object) it is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    charset : string, 'utf-8' by default.
        If bytes or files are given to analyze, this charset is used to
        decode.

    charset_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `charset`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

    analyzer : string, {'word', 'char'} or callable
        Whether the feature should be made of word or character n-grams.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.


    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words : string {'english'}, list, or None (default)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned is currently the only
        supported string value.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    lowercase : boolean, default True
        Convert all characters to lowercase befor tokenizing.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if `tokenize == 'word'`. The default regexp select tokens of 2
        or more letters characters (punctuation is completely ignored
        and always treated as a token separator).

    max_df : float in range [0.0, 1.0] or int, optional, 1.0 by default
        When building the vocabulary ignore terms that have a term frequency
        strictly higher than the given threshold (corpus specific stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, optional, 2 by default
        When building the vocabulary ignore terms that have a term frequency
        strictly lower than the given threshold.
        This value is also called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : optional, None by default
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.

    binary : boolean, False by default.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().


    See also
    --------
    CountVectorizer
        Tokenize the documents and count the occurrences of token and return
        them as a sparse matrix

    BnsTransformer
        Apply BNS feature scoring to a sparse matrix of occurrence counts.

    """

    def __init__(self, input='content', charset='utf-8',
                 charset_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b", min_n=None,
                 max_n=None, ngram_range=(1, 1), max_df=1.0, min_df=2,
                 max_features=None, vocabulary=None, binary=True, dtype=float):

        super(BnsVectorizer, self).__init__(
            input=input, charset=charset, charset_error=charset_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern, min_n=min_n,
            max_n=max_n, ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=True,
            dtype=dtype)        

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    def fit(self, raw_documents, y=None):
        """Learn a conversion law from documents to array data"""
        X = super(BnsVectorizer, self).fit_transform(raw_documents)
        self._bns = BnsTransformer(y=y, vocab=super(BnsVectorizer, self).vocabulary_)
        self._bns.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the representation and return the vectors.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        vectors : array, [n_samples, n_features]
        """
        X = super(BnsVectorizer, self).fit_transform(raw_documents)
        self._bns = BnsTransformer(y=y, vocab=super(BnsVectorizer, self).vocabulary_)
        return self._bns.transform(X)

    def transform(self, raw_documents, copy=True):
        """Transform raw text documents to bns vectors

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        X = super(BnsVectorizer, self).transform(raw_documents) 
        return self._bns.transform(X)
