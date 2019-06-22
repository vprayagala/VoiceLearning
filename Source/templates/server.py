# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 09:06:22 2018

@author: vprayagala2
"""

# -*- coding: utf-8 -*-
import os,sys
# os.chdir(os.path.expanduser("~/AnacondaProjects/VoiceLearning"))
sys.path.append(os.path.join(os.getcwd(),"Source"))
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import json
from flask import Flask,render_template,request,Response
from werkzeug import secure_filename
from keras import backend as K
import numpy
import Score as LP
import PrepareData as RDT
import AzureAudioConversion as AAC
import LoadConfiguration as LC
import GenWordCloud as GWC
import CreateLogger as CL
import ConvertAudio as Audio

global logger
application = Flask(__name__)

LC.load_config_file()
#Get the logger
logger=CL.create_logger(LC.getParmValue('LogSetup/Log_Name'))

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
        
def predictScore(text):
    K.clear_session()
    model=LP.load_model()
    label,score=LP.predict_score(model,text)
    index=LP.predict_sentiment_index(score[0][0])
    #msg="Text:{} , is predicted with Sentiment Score:{} and the sentiment Index is :{}".format(text,score,index)
    #return msg
    logger.debug("Debug: Label {} and Score {}".format(label[0][0],score[0][0]))
    
    result={}
    #result['Cleaned_Text']=text
    if label[0][0] == 1:
        result['Label']="Positive"
    else:
        result['Label']="Negative"
        
    result['Score']=score[0][0]
    result['Index']=index
    return result 

def genResponse(stat_code,message):
    js=json.dumps({'Response':message})                 
    resp=Response(js, status=stat_code, mimetype='application/json')
    return resp

@application.route('/')
def index():
    logger.info("Accessed Index, Request Details:{}".format(request))
    js=json.dumps({'index':"Root Page of Voice Learning API",
                   'methods':"GenWordCloud, predictSentimentScore, predictTopic, convertAudio"
                   })
    resp=Response(js, status=200, mimetype='application/json')
    return resp

@application.route('/genWordCloud',methods = ['POST', 'GET'])
def genWordCloud():
    logger.info("Accessed genWordCloud, Request Details:{}".format(request))
    if request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            input_data=request.json
            logger.info("Data Source Specified in Configuration File:{}".format(LC.getParmValue('DataSource/In_Cloud')))
            logger.info("Data Path Provided:{}".format(input_data))
            if 'path' in input_data.keys():
                path=input_data["path"]
                logger.info("Processing Audio File:{}".format(path))
                res=GWC.process(path)
                if res == 1:
                    resp=genResponse(stat_code=200,
                                     message='WordCloud Generated:'+os.path.join(path,"wordcloud.png")
                                     )
                else:
                    resp=genResponse(stat_code=400,
                                     message="Invalid Data, Could not Extract any words"
                                     )                    
            else:
                resp=genResponse(stat_code=400,
                                     message="Invalid Data Provided, Path for local/blob is Mandatory"
                                )                               
    else:
        resp=genResponse(stat_code=400,
                         message="Invalid Request Method"
                        ) 
    return resp
        
@application.route('/predictSentimentScore',methods = ['POST', 'GET'])
def predictSentimentScore():
    logger.info("Accessed predictSentimentScore, Request Details:{}".format(request))
    if request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            input_data=request.json
            logger.info("Data Path Provided:{}".format(input_data))
            if 'path' in input_data.keys():
                path=input_data["path"]
                logger.info("Processing File:{}".format(path))
                
                text=RDT.readTextDataVTT(path)
                hold_raw_text=text[0].copy()
                logger.info("Converted Text:{}".format(hold_raw_text))
                
                cleaned=[RDT.cleanTextData(line.pop()) for line in text if len(line) > 0]
    
                cleaned_list=[item for sublist in cleaned for item in sublist]
                cleaned_text=' '.join(cleaned_list)
                logger.info("Cleaned Text:{}".format(cleaned_text))
                result=predictScore(cleaned_text)
                result['text']=''.join(hold_raw_text)
                logger.info("Result\n:{}".format(result))
                js=json.dumps(result,cls=MyEncoder)                 
                resp=Response(js, status=200, mimetype='application/json')
            else:
                resp=genResponse(stat_code=400,
                                     message="Invalid Data Provided, Path for local/blob is Mandatory"
                                ) 
    else:
        resp=genResponse(stat_code=400,
                         message="Invalid Request Method"
                        ) 
        
    return resp

@application.route('/predictTopic',methods = ['POST', 'GET'])
def predictTopic():
    logger.info("Accessed predictTopic, Request Details:{}".format(request))
    if request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            input_data=request.json
            logger.info("Data Path Provided:{}".format(input_data))
            if 'path' in input_data.keys():
                path=input_data["path"]
                logger.info("Processing File:{}".format(path))
                
                text=RDT.readTextDataVTT(path)
                logger.info("Converted Text:{}".format(text))
                cleaned=[RDT.cleanTextData(line.pop()) for line in text if len(line) > 0]
    
                top_topic,topic_dist,topic_words=LP.predict_topic(cleaned)
                
                topic_dist_df = pd.DataFrame(topic_dist)
                topic_dist_df.columns=['Topic','Probability']
                
                logger.info("Result\n:{}".format(topic_dist_df))
                
                js=json.dumps({"Predicted_Topic":top_topic,
                               "Topic Distribution":topic_dist_df.to_json(orient='records'),
                               "Topic Words":topic_words},
                                cls=MyEncoder)                 
                resp=Response(js, status=200, mimetype='application/json')
            else:
                resp=genResponse(stat_code=400,
                                 message="Invalid Data Provided, Path for local/blob is Mandatory"
                                )
    else:
        resp=genResponse(stat_code=400,message="Invalid Request Method")
    
    return resp

@application.route('/convertAudio',methods = ['POST', 'GET'])
def convertAudio():    
    logger.info("Accessed convertAudio, Request Details:{}".format(request))
    if request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            input_data=request.json
            logger.info("Data Path Provided:{}".format(input_data))
            if 'path' in input_data.keys():
                path=input_data["path"]
                logger.info("Processing File:{}".format(path))
                Audio.process(path)
                logger.info("Conversion Completed")
                resp=genResponse(stat_code=400,
                                 message="Successfully Completed the conversion,\
                                     check path for result"
                                     ) 
            else:
                resp=genResponse(stat_code=400,message="Invalid Request")  
        else:
            resp=genResponse(stat_code=400,message="Invalid Request") 
    else:
        resp=genResponse(stat_code=400,message="Invalid Request Method")
        
    return resp

if __name__ == '__main__':
   #application.run(host='127.0.0.1',debug = True)
   application.run(host="0.0.0.0",port=5000,debug=False)
