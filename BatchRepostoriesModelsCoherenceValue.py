#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:40:49 2018

@author: clopezno
"""

import LibraryTopicModel as ltm
import seaborn as sns
import pandas as pd
import glob
import os


def batchRepositoriesValidation(path, limit, start, step,models=["lsi","lda","ldamallet"]):
    """Load pullrequest from each  csv file (repositoty) and generate a coherence metrics 
    for all models generated (len(models)* (limit -start /2)). One model for each repository.
        Parameters:
        ----------
        path : dierctory con dataset
        limit : Max num of topics
        start : 
        step   :
        topicmodel: 'lda', 'lsi', 'ldamallet'
    Returns:
        For each repository file of c_v, c_uci, 'c_npmi' coherence for each number of topic
        (see directory ./evaluationresults/)
     """
    
#path="./datasets/pullrequest/reviews_cakephp_processed.csv"
    for filename in glob.glob(os.path.join(path, '*.csv')):
        data_lemmatized=ltm.preprocessData(filename)
        corpus,id2word=ltm.createCorpus(data_lemmatized)
        corpus=ltm.createCorpusTfid(corpus)
        listResultModels=ltm.experimentCoherenceMeasures(models,limit, start, step, id2word, corpus,data_lemmatized)
        for l in listResultModels:
            print(l)    
        dfResult=pd.DataFrame(listResultModels,columns=["model","topic","cv","cuci","cnpmi"]) 
        os.path.isfile(path)
        dfResult.to_csv("./evaluationresults/validation"+ os.path.basename(filename))
    return

#print(dfResult.head())


def batchAllinOneRepositoriesValidation(path, limit, start, step,models=["lsi","lda","ldamallet"]):
    def batchRepositoriesValidation(path, limit, start, step,models=["lsi","lda","ldamallet"]):
        """Load pullrequest from each  csv file (repositoty) and generate a coherence metrics 
    for all models generated (len(models)* (limit -start /2)). One model for all repositories
        Parameters:
        ----------
        path : dierctory con dataset
        limit : Max num of topics
        start : 
        step   :
        topicmodel: 'lda', 'lsi', 'ldamallet'
    Returns:
        For each repository file of c_v, c_uci, 'c_npmi' coherence for each number of topic
        (see directory ./evaluationresults/)
    """
    data_lemmatized=ltm.preprocessData(path)
    corpus,id2word=ltm.createCorpus(data_lemmatized)
    corpus=ltm.createCorpusTfid(corpus)
    listResultModels=ltm.experimentCoherenceMeasures(models,limit, start, step, id2word, corpus,data_lemmatized)
    dfResult=pd.DataFrame(listResultModels,columns=["model","topic","cv","cuci","cnpmi"])
    dfResult.to_csv("./evaluationresults/validationAllinOne.csv")
    return                
    


path="./datasets/pullrequest"
limit=40;start=2; step=2;
#models=["lsi","lda","ldamallet"]
#models=["lsi","lda","ldamallet"]
models=["ldamallet"]
#batchRepositoriesValidation(path, limit, start, step,models)
batchAllinOneRepositoriesValidation(path, limit, start, step,models)

#pathresults="./evaluationresults"
#for filename in glob.glob(os.path.join(path, '*.csv')):        
#    dfResult=pd.read_csv(pathresults, error_bad_lines=False, index_col=False, dtype='unicode')
#    print
#    
#    sns.lmplot(x="topic",y="cv",data=dfResult,fit_reg=False,hue="model")
#    sns.lmplot(x="topic",y="cuci",data=dfResult,fit_reg=False,hue="model")
#    sns.lmplot(x="topic",y="cnpmi",data=dfResult,fit_reg=False,hue="model")


#os.path.isdir(path):
