#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:02:07 2018

@author: clopezno
"""

import os
import glob
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import gensim
import spacy




#Read csv
def loadCsvPullRequestFolder(path):
    """Load pullrequest data from  csv file and generate a 
    list with all pull requeste"""
    _lprbt=list()
    _totalfile=0
    if os.path.isfile(path):
        print(path)
        df2= loadCsvPullRequestFile(path)
        [_lprbt.append(pr) for pr in df2.pull_request]
        _totalfile+=1
    elif os.path.isdir(path):
        for filename in glob.glob(os.path.join(path, '*.csv')):        
            print(filename)
            df2=pd.DataFrame()
            #df2=pd.read_csv(filename, error_bad_lines=False, index_col=False, dtype='unicode')
            #df2["pull_request"] = df2["repository_owner"].map(str) + " " + \
            #df2["repository_name"].map(str) +  " " +  df2["repository_language"].map(str) + " " +\
            #df2["pull_request_body"].map(str) + " " +  df2["pull_request_title"].map(str)
            df2= loadCsvPullRequestFile(filename)
            [_lprbt.append(pr) for pr in df2.pull_request] 
            _totalfile+=1
            #print(lprbt[len(lprbt)-1])
    return _totalfile, len(_lprbt), _lprbt

#Read csv
def loadCsvPullRequestFile(pathfilename):
    """Load pullrequest data from  csv file and generate a 
    list with all pull request"""
    #print(pathfilename)
    df2=pd.read_csv(pathfilename, error_bad_lines=False, index_col=False, dtype='unicode')
    df2["pull_request"] = df2["repository_owner"].map(str) + " " + \
    df2["repository_name"].map(str) +  " " +  df2["repository_language"].map(str) + " " +\
    df2["pull_request_body"].map(str) + " " +  df2["pull_request_title"].map(str)
        
        #print(lprbt[len(lprbt)-1])
    return df2




def textNormalization(lpr):
    #remove character space
    lpr=[re.sub('\s+',' ',pr) for pr in lpr]
    #TODO remove \\n \\r
    #lpt=[re.sub('[/\\n/\\n]',' ',pr) for pr in lpr]
    return lpr



def pr_to_words(sentences):
    for sentence in sentences:
        # deacc True remove puntactions
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))
                


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    #nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.extend(['\\n\\n', '\\n\\r'])
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,bigram_mod,trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# spacy for lemmatization
def lemmatization(nlp,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def createCorpus(data_lemmatized):
    id2word = gensim.corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    return corpus,id2word

def createCorpusTfid(corpus):
    tfidf = gensim.models.tfidfmodel.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf

def getMeasureCoherence(modelpr,data_lemmatizedlpr,id2word,measure):
    coherence_model_pr = gensim.models.CoherenceModel(model=modelpr, texts=data_lemmatizedlpr, dictionary=id2word, coherence=measure)
    return coherence_model_pr.get_coherence()

def printMesuresCoherence(name,modelpr, data_lemmatized, id2word):    
    print('\n-----{} \t  c_v {} \t c_uci {} \t c_npmi {}'.format( name,\
                                                                                    getMeasureCoherence(modelpr,data_lemmatized,id2word, 'c_v'),\
                                                                   getMeasureCoherence(modelpr,data_lemmatized,id2word, 'c_uci'),\
                                                                  getMeasureCoherence(modelpr,data_lemmatized,id2word, 'c_npmi')))
    return


def compute_coherence_values(dictionary, corpus, texts, limit, start, step, topicmodel='lda'):
    """
    Compute c_v, c_uci, 'c_npmi' coherence for various number of topics for a gensim algorithms topic model 'lda', 'lsi', 'ldamallet'
    Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics
        start : 
        step   :
        topicmodel: 'lda', 'lsi', 'ldamallet'
    Returns:
        -------
        model_list : List of LDA topic models
        coherence_values_xx : Lists of coherence metrics XX values corresponding to the  model with respective number of topics
    """
    coherence_values_cv = []
    coherence_values_cuci = []
    coherence_values_cnpmi = []
    model_list = []
    mallet_path="./mallet-2.0.8/bin/mallet"
    for num_topics in range(start, limit, step):
        if topicmodel == 'lda':
            modelpr = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
        elif topicmodel =='lsi':
            modelpr = gensim.models.lsimodel.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)
        elif topicmodel =='hlp':
            modelpr = gensim.models.hdpmodel.HdpModel(corpus, id2word=dictionary)
        elif topicmodel =='ldamallet':
            modelpr = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        else:
            print("Error name model incorrect. Add a model compute_coherence_values() function ")
       
        model_list.append(modelpr)
        printMesuresCoherence(str(topicmodel+" num topics " + str(num_topics)),modelpr, texts, dictionary)
        coherencemodelcv = gensim.models.CoherenceModel(model=modelpr, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values_cv.append(coherencemodelcv.get_coherence())
        coherencemodelcuci = gensim.models.CoherenceModel(model=modelpr, texts=texts, dictionary=dictionary, coherence='c_uci')
        coherence_values_cuci.append(coherencemodelcuci.get_coherence())
        coherencemodelcnpmi = gensim.models.CoherenceModel(model=modelpr, texts=texts, dictionary=dictionary, coherence='c_npmi')
        coherence_values_cnpmi.append(coherencemodelcnpmi.get_coherence())
    #return model_list, coherence_values_cv
    return model_list, coherence_values_cv, coherence_values_cuci, coherence_values_cnpmi

def counterElements(listoflist):
    counter=0
    maxindex=len(listoflist) - 1
    for i in range(0,maxindex):
        counter+=len(listoflist[i])    
    return counter
        

def preprocessData(path):
    """
    process data of a file or folder with csv format.
    Name columns of csv  "repository_owner","repository_name" "repository_language" "pull_request_body" "pull_request_title"
      textNormalization - now not implemented
      stopWords
      bigrams
      speech tagging  ['NOUN', 'ADJ', 'VERB', 'ADV']
      lemmatization
        
    Parameters:
        ----------
        path : string pathname  
    Returns:
        -------
        data_lemmatized: list with sentences with a list of words preprocessed:   
    """
    totalfiles,totalinstances,lprbt=loadCsvPullRequestFolder(path)   
    print("Number of files: {} Number of instances: {}".format(totalfiles,totalinstances))
    
    
    lprbt=textNormalization(lprbt)                
    prwords=list(pr_to_words(lprbt))
    
   
    print("Numbers of tokens in pullrequest: {} ".format(counterElements(prwords)))
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(prwords, min_count=5, threshold=500) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[prwords], threshold=500)  
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    # See trigram example
    #print(trigram_mod[bigram_mod[prwords[1]]])
    # Remove Stop Words
    data_words_nostops = remove_stopwords(prwords)
    print("Numbers of tokens nostops in pullrequest: {} ".format(counterElements(data_words_nostops)))
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)
    data_words_trigrams = make_trigrams(data_words_nostops,bigram_mod,trigram_mod)
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    #!python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(nlp,data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #data_lemmatized = lemmatization(nlp,data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print("Numbers of tokens in pullrequest after lemmatization: {} ".format(counterElements(data_lemmatized)))
    return  data_lemmatized 


#printHumanCorpus(corpus,id2word,3)
#tfidf_corpus= createCorpusTfid(corpus)
#printHumanCorpus(corpus,tfidf_corpus,3)



def experimentCoherenceMeasures(models,limit, start, step, id2word, corpus,data_lemmatized):
    """
    Compute c_v, c_uci, 'c_npmi' coherence for various number of topics and several gensim algorithms
    Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics
        start : 
        step   :
        topicmodel: 'lda', 'lsi', 'ldamallet'
    Returns:
        -------
       listResultModels list of c_v, c_uci, 'c_npmi' coherencefor each model
    """
    listResultModels=[]
    modelid=0
    for model in models:
        model_list, coherence_values_cv,coherence_values_cuci, coherence_values_cnpmi = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, limit=limit, start=start, step=step, topicmodel=model)

        for t in range(0,len(coherence_values_cv)):
            #id_model = str("model" + str(modelid))
            id_topic = str(model_list[t].num_topics)
            listResultModel=[model, id_topic, coherence_values_cv[t], coherence_values_cuci[t], coherence_values_cnpmi[t]]
            listResultModels.append(listResultModel)
        modelid=modelid+1
        
    return listResultModels