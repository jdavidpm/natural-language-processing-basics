from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization import summarize
from scipy.sparse.linalg import svds
from pickle import dump, load
from math import ceil
import pandas as pd
import numpy as np
import networkx
import nltk
import glob
import re
import os

stopwords = nltk.corpus.stopwords.words('spanish')

def loadPickleAsObject(fileName):
    with open(fileName, 'rb') as f:
        return load(f)

def summarizeGensim(txt, rat, spt):
    txt = re.sub(r'\n|\r', ' ', txt)
    txt = re.sub(r' +', ' ', txt)
    txt = txt.strip()
    return summarize(txt, ratio=rat, split=spt)

def normalization(text, areSentences=False):
    lemmatizationList = loadPickleAsObject('lemmatizationList.pkl')
    if areSentences:
        tokens = nltk.sent_tokenize(text)
        tokens = [' '.join([lemmatizationList[j.lower()] for j in nltk.word_tokenize(i) if j not in stopwords and j.isalpha() and j.lower() in lemmatizationList]) for i in tokens]
        if [] in tokens:
            tokens.remove([])
    else:
        tokens = nltk.word_tokenize(text)
        tokens = [' '.join(lemmatizationList[i.lower()]) for i in tokens if i not in stopwords and i.isalpha() and i.lower() in lemmatizationList]
    return tokens

def lowRankSVD(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt

def summarizeLSA(txt, nSentences, nTopics):
    normalizeCorpus = np.vectorize(normalization)
    sentences = normalizeCorpus(txt, True)
    
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(sentences)
    dt_matrix = dt_matrix.toarray()
    vocab = tv.get_feature_names()
    td_matrix = dt_matrix.T
    
    nSentences = 8
    nTopics = 3
    u, s, vt = lowRankSVD(td_matrix, singular_count=nTopics)  
    termTopicMat, singularValues, topicDocumentMat = u, s, vt
    
    svThreshold = 0.5
    minSigmaValue = max(singularValues) * svThreshold
    singularValues[singularValues < minSigmaValue] = 0
    
    salienceScores = np.sqrt(np.dot(np.square(singularValues), np.square(topicDocumentMat)))
    salienceScores
    
    topSentenceIndices = (-salienceScores).argsort()[:nSentences]
    topSentenceIndices.sort()
    
    return '\n'.join(np.array(sentences)[topSentenceIndices])

def summarizeTextRank(txt, nSentences):
    normalizeCorpus = np.vectorize(normalization)
    sentences = normalizeCorpus(txt, True)
    
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(sentences)
    dt_matrix = dt_matrix.toarray()
    vocab = tv.get_feature_names()
    td_matrix = dt_matrix.T
    
    similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)
    np.round(similarity_matrix, 3)
    similarity_graph = networkx.from_numpy_array(similarity_matrix)
    
    scores = networkx.pagerank(similarity_graph)
    ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
    
    top_sentence_indices = [ranked_sentences[index][1] for index in range(nSentences)]
    top_sentence_indices.sort()
    
    return '\n'.join(np.array(sentences)[top_sentence_indices])

def getText(aspect):
    reviewSentYe, reviewSentNo = '', ''
    reviewFilesYe = glob.glob('SpanishReviewCorpus/peliculas/yes*.txt')
    reviewFilesNo = glob.glob('SpanishReviewCorpus/peliculas/no*.txt')
    
    for r in reviewFilesYe:
        with open(r, 'rb') as f:
            text = f.read().decode('latin-1')
            tokens = nltk.sent_tokenize(text)
            sent = ' '.join(normalization(text, True)).split(' ')
            if aspect in sent:
                reviewSentYe = reviewSentYe + ' '.join(tokens)
            
    for r in reviewFilesNo:
        with open(r, 'rb') as f:
            text = f.read().decode('latin-1')
            tokens = nltk.sent_tokenize(text)
            sent = ' '.join(normalization(text, True)).split(' ')
            if aspect in sent:
                reviewSentNo = reviewSentNo + ' '.join(tokens)
            
    return reviewSentYe, reviewSentNo

if __name__ == '__main__':
    arrayAspectList = ['personaje', 'historia', 'escena', 'final', 'especial', 'argumento', 'actor']
    for a in arrayAspectList:
        reviewSentYe, reviewSentNo = getText(a)
        print(f'\t\tAspect: {a:s}\n')
        print('\tPositive:')
        print('GENSIM:\n', summarizeGensim(reviewSentYe, 0.002, False))
        print('\nLSA:\n', summarizeLSA(reviewSentYe, 8, 3))
        print('\nTextRank:\n', summarizeTextRank(reviewSentYe, 8))
        print('\tNegative:')
        print('GENSIM:\n', summarizeGensim(reviewSentNo, 0.002, False))
        print('\nLSA:\n', summarizeLSA(reviewSentNo, 8, 3))
        print('\nTextRank:\n', summarizeTextRank(reviewSentNo, 8))
    
