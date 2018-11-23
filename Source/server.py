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
    result['Cleaned_Text']=text
    if label[0][0] == 1:
        result['Label']="Positive"
    else:
        result['Label']="Negative"
        
    result['Score']=score[0][0]
    result['Index']=index
    return result 

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
                    js=json.dumps({'Response':'WordCloud Generated:'+os.path.join(path,"wordcloud.png"),
                               })                 
                    resp=Response(js, status=200, mimetype='application/json')
                else:
                    js=json.dumps({'Response':"Invalid Data, Could not Extract any words"
                              })                 
                    resp=Response(js, status=400, mimetype='application/json')
            else:
                js=json.dumps({'Response':"Invalid Data Provided, Path for local/blob is Mandatory"
                              })                 
                resp=Response(js, status=400, mimetype='application/json')
                
    else:
        js=json.dumps({'Response':"Invalid Request Method"
                   })                 
        resp=Response(js, status=400, mimetype='application/json')
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
                logger.info("Converted Text:{}".format(text))
                cleaned=[RDT.cleanTextData(line.pop()) for line in text if len(line) > 0]
    
                cleaned_list=[item for sublist in cleaned for item in sublist]
                cleaned_text=' '.join(cleaned_list)
                logger.info("Cleaned Text:{}".format(cleaned_text))
                result=predictScore(cleaned_text)
                result['text']=text
                logger.info("Result\n:{}".format(result))
                js=json.dumps(result,cls=MyEncoder)                 
                resp=Response(js, status=200, mimetype='application/json')
            else:
                js=json.dumps({'Response':"Invalid Data Provided, Path for local/blob is Mandatory"
                              })                 
                resp=Response(js, status=400, mimetype='application/json')
    else:
        js=json.dumps({'Response':"Invalid Request Method"
                   })                 
        resp=Response(js, status=400, mimetype='application/json')
    return resp

@application.route('/predictTopic',methods = ['POST', 'GET'])
def predictTopic():
    pass

@application.route('/convertAudio',methods = ['POST', 'GET'])
def convertAudio():    
    pass

if __name__ == '__main__':
   #application.run(host='127.0.0.1',debug = True)
   application.run(host="0.0.0.0",port=5000,debug=False)
