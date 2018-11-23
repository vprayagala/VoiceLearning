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
#os.chdir(os.path.expanduser("~/AnacondaProjects/VoiceLearning"))
sys.path.append(os.path.join(os.getcwd(),"Source"))
#print("Current SystemPath:{}".format(sys.path[-1]))
#import argparse
import json
import numpy
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import model_from_json

#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#%%
#Set the looging method
import PrepareData as RTD
#import LoadConfiguration as parm
import LoadConfiguration as LC
LC.load_config_file()

import logging
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
#%%
#def predictSentimentForText(text):
#    #Predict from Text
#    model=load_model()
#    label,score=predict_score(model,text)
#    index=predict_sentiment_index(score[0][0])
#    #msg="Text:{} , is predicted with Sentiment Score:{} and the sentiment Index is :{}".format(text,score,index)
#    #return msg
#    logger.debug("Debug: Label {} and Score {}".format(label[0][0],score[0][0]))
#    
#    result={}
#    result['Text Entered:']=text.strip('\n')
#    if label[0][0] == 1:
#        result['Label']="Positive"
#    else:
#        result['Label']="Negative"
#        
#    result['score']=score[0][0]
#    result['index']=index
#    return result
#%%
#Test the function definition
#load Model
#model=load_model()
#text="This is a sample review to predict if customer or agent is happy"
#score=predict_score(model,text)
#index=predict_sentiment_index(score)
#print("Text:{} , is predicted with Sentiment Score:{} and the sentiment Index is :{}".format(text,score,index))
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    
#    parser.add_argument('--path', 
#                        help='Enter Absolute Path of Text files directory)')
#    args = parser.parse_args()
#    
#    if not (os.path.exists(args.path)):
#        print("Invalid Path Provided")
#        sys.exit()
#    else: 
#        text=RTD.readTextDataVTT(args.path)
#        cleaned=[RTD.cleanTextData(line.pop()) for line in text if len(line) > 0]
#    
#        cleaned_list=[item for sublist in cleaned for item in sublist]
#        raw_text=' '.join(cleaned_list)
#        result=predictSentimentForText(raw_text)
#        res_file=args.path+"Sentiment_Score.json"
#        print("Writing result to {}".format(res_file))
#        with open(res_file,"w") as f:
#            json.dump(json.dumps(result,cls=MyEncoder),f)
#        print("End of Processing")