# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import print_function
"""
Created on Thu Sep 27 09:30:19 2018

'''Load Saved Model and Predict the Sentiment Index
'''
"""
#%%
import os,sys
import logging
#import argparse
import json
import numpy

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import model_from_json

import gensim
from gensim import utils, models
from gensim.corpora import Dictionary
from gensim.summarization import summarize
from summa import summarizer,keywords
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer

from Source.DataHandler import PrepareData as RTD
from Source.Config import LoadConfiguration as LC

LC.load_config_file()
logger=logging.getLogger(LC.getParmValue('LogSetup/Log_Name'))
logger.info("Current SystemPath:{}".format(sys.path[-1]))
#%%
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
#%%
def load_model():
    #global logger
    logger.info("Loading Saved Model...")
    model_josn_file=os.path.join(
                    os.getcwd(),
                    os.path.join(LC.getParmValue('DataSource/Model_Dir')
                                    ,LC.getParmValue('Model/LSTM_model_json'))
                                )
    model_weight_file=os.path.join(
                    os.getcwd(),
                    os.path.join(LC.getParmValue('DataSource/Model_Dir')
                                    ,LC.getParmValue('Model/LSTM_model_weight'))
                                )                    
    with open(model_josn_file, "r") as json_file:
        loaded_model_json=json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weight_file)
    loaded_model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    logger.info("Loaded model from disk")
    return loaded_model

def predict_score(model,text):    
    words = RTD.cleanTextData(text)
    word_index = imdb.get_word_index()
    #word_dict = {idx: word for word, idx in word_index.items()}
    # Should be same which you used for training data
    x_test = [[word_index[w] for w in words if w in word_index]]
    x_test = sequence.pad_sequences(x_test, maxlen=int(LC.getParmValue('LSTM/maxlen')))
    label=model.predict_classes(x_test)
    score=model.predict(x_test)
    return label,score*100

def predict_sentiment_index(score):
    #Bin Sentiment Index
    sentiment_index=" "
    if score >=0 and score <=30:
        sentiment_index="Sad"
    elif score > 30 and score <=60:
        sentiment_index="Neutral"
    elif score > 60:
        sentiment_index="Happy"
    return sentiment_index

def predict_topic(text):
    result=''
    #Load LDA Model and Dictionary
    lda_filename=os.path.join(os.path.join(os.getcwd(),'outputs'),
                              'lda_model.model')
    lda_dict_filename=os.path.join(os.path.join(os.getcwd(),'outputs'),
                              'lda_dict.dict')
    lda_model=models.LdaModel.load(lda_filename)
    lda_dict=Dictionary.load(lda_dict_filename)
    
    bigram = gensim.models.Phrases(text)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    data_bigrams=[bigram_mod[sentence] for sentence in text]
    data_cleaned = RTD.lemmatization(data_bigrams,\
                                    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    logger.debug(data_cleaned)
    # Term Document Frequency
    tdm = [lda_dict.doc2bow(word) for word in data_cleaned]
    logger.debug("TDM:{}".format(tdm[0]))
    
    lda_model.update(tdm)    
    result = lda_model[tdm[0]]
    
    topic_dist=result[0]
    logger.info("Topic Distribution:{}".format(topic_dist))
    top_topic=max(topic_dist, key=lambda item: item[1])[0]
    logger.info("Top Topic Words:{}".format(lda_model.print_topic(top_topic))) 
    topic_words = lda_model.print_topic(top_topic)
    return top_topic,topic_dist,topic_words

def getGenSimSummary(filename):
       pdfText = RTD.extractPdfText(filename)
       logger.info("Extracted Text:\n{}".format(pdfText.encode("utf-8")))
       keywordText=RTD.cleanTextData(pdfText)
       length = str(pdfText.__len__())
       summary = summarize(pdfText,ratio=0.3,split=True)
       summary = '.'.join(summary)
       return summary

def getSummaSummary(filename):
       pdfText = RTD.extractPdfText(filename)
       logger.info("Extracted Text:\n{}".format(pdfText))
       keywordText=RTD.cleanTextData(pdfText)
       length = str(pdfText.__len__())
       summary = summarizer.summarize(pdfText,ratio=0.3)
       return summary
   