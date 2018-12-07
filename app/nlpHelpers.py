
from gensim.models import Word2Vec
import re
import nltk
import string
import numpy as np
import pandas as pd 
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

my_stop_words = ['the','and','to','of','was','with','a','on','in','for','name', 'n',
                 'is','patient','s','he','at','as','or','one','she','his','her','am',
                 'were','you','pt','pm','by','be','had','your','this','date',
                'from','there','an','that','p','are','have','has','h','but','o',
                'namepattern','which','every','also', 'b', 'i', 'd', 'admission', 'q', 't']

def prepPatient(df, hadm_id, note_vec, features):
    pat = df[df['hadm_id'] == int(hadm_id)][features]
    pat = np.append(pat.values[0], note_vec)
    return pat

def tokenize_clinic_notes(note):
    ''' Tokenize the patient text by replacing punctuations and numbers with spaces and lowercase all words
    '''
    punc_list = string.punctuation + '0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, ' '))
    text = str(note).lower().translate(t)
    tokens = nltk.word_tokenize(text)
    return tokens

class MyTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed_X = []
        for document in X:
            doc = [word for word in document if word in self.vocab]
            transformed_X.append(doc)
        return transformed_X

    def fit_transform(self, X, y=None):
        return self.transform(X)
            

class MeanEmbeddingVectorizer(object):
    ''' Convert notes to vector. 

        IMPORTANT: Must pass a list of notes or there will be a problem. If passing a single document make sure it is as a list and NOT raw text.
    '''
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.vectors[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer(self.word2vec.wv.vocab).fit_transform(X)
    
        return np.array([
                    np.mean([self.word2vec.wv[w] for w in document] or 
                            [np.zeros(self.dim)], axis = 0) for document in X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

    def vectorizeSingleNote(self, note):
        token = tokenize_clinic_notes(note)
        vector = self.fit_transform([token])[0]
        return vector