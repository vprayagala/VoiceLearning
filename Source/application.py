# -*- coding: utf-8 -*-
# import os,sys
# os.chdir(os.path.expanduser("~/AnacondaProjects/VoiceLearning"))
# sys.path.append(os.path.join(os.getcwd(),"Source"))
import warnings
warnings.filterwarnings("ignore")

import os
from flask import Flask,render_template,request
from werkzeug import secure_filename
from keras import backend as K

import Score as LP
import AzureAudioConversion as AAC
import LoadConfiguration as LC
import GenWordCloud as GWC
import CreateLogger as CL


global logger
application = Flask(__name__)


@application.route('/')
def index():
   global logger
   LC.load_config_file()
   #Get the logger
   
   logger=CL.create_logger(LC.getParmValue('LogSetup/Log_Name'))
   return render_template("index.html")

@application.route('/action',methods = ['POST', 'GET'])
def action():
    if request.method == 'POST':
        sel_option = request.form.get("chkselection")
        
    logger.info("User Selected Option:{}".format(sel_option))
    
    if sel_option == "audio":
        return render_template("upload.html")
    elif sel_option == "text":
        return render_template("text.html")
    elif sel_option == 'word':
        return render_template("wordcloudupload.html")
    else:
        return "Invalid Option Select Either Audio or Text"

@application.route('/predictAudio',methods = ['POST', 'GET'])
def predictAudio():
    #get the audio file, convert into text and determine sentiment score
    #print("Processing Audio File:{}".format(request))
    if request.method == 'POST':
        #print("Getting Filename")
        f = request.files['file']
        #print(f.filename)
        
        f.save(secure_filename(f.filename))
        logger.info("Processing Audio File:{}".format(f))
        
        text=AAC.convertAudio(f.filename)
        logger.info("Convereted Text:{}".format(text))
        
        try:
            os.remove(f.filename)
        except OSError:
            pass
        
        if text=='':
            return("Empty Text, Cannot Determine Sentiment Score")
        else:
            result=predictScore(text)
            return render_template("result.html",result=result)
        
@application.route('/predictText',methods = ['POST', 'GET'])
def predictText():
    #from text entered determine sentiment score
    if request.method == 'POST': 
        text=request.form.get("Description")
        #Predict from Text
        result=predictScore(text)
        return render_template("result.html",result=result)

def predictScore(text):
    K.clear_session()
    model=LP.load_model()
    label,score=LP.predict_score(model,text)
    index=LP.predict_sentiment_index(score[0][0])
    #msg="Text:{} , is predicted with Sentiment Score:{} and the sentiment Index is :{}".format(text,score,index)
    #return msg
    logger.debug("Debug: Label {} and Score {}".format(label[0][0],score[0][0]))
    
    result={}
    result['Text Entered:']=text
    if label[0][0] == 1:
        result['Label']="Positive"
    else:
        result['Label']="Negative"
        
    result['Score']=score[0][0]
    result['Index']=index
    return result 

@application.route('/genWordCloud',methods = ['POST', 'GET'])
def genWordCloud():
    if request.method == 'POST':
        #print("Getting Filename")
        path=request.form.get("path")
        logger.info("Processing Audio File:{}".format(path))
        _=GWC.process(path)
                         
        return "Word Cloud Generated as html file, check the input directory"
    
if __name__ == '__main__':
   #application.run(host='127.0.0.1',debug = True)
   application.run(host="0.0.0.0",port=5000,debug=False)
