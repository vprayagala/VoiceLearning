# -*- coding: utf-8 -*-
import os,sys
#os.chdir(os.path.expanduser("~/AnacondaProjects/VoiceLearning"))
sys.path.append(os.path.join(os.getcwd(),"Source"))

import json
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import make_chunks
import requests
#import LoadConfiguration as parm

import LoadConfiguration as LC
LC.load_config_file()
#Get the logger
import logging
logger=logging.getLogger(LC.getParmValue('LogSetup/Log_Name'))
#%%
def convertAudio(path):
    # 1. Get an Authorization Token
    logger.info("Reading Audio File from:{}".format(path))
    token = get_token()
    # 2. Perform Speech Recognition
    results = get_text(token, path)
    # 3. Print Results
    logger.info(results)
    text=''
    for i in range(results.shape[0]):
        if (results.loc[i,'RecognitionStatus'] == 'Success'):
            #text=''.join(str(txt) for txt in results.loc[i,'DisplayText'])
            text=text+str(results.loc[i,'DisplayText'])
    return text

def get_token():
    # Return an Authorization Token by making a HTTP POST request to Cognitive Services with a valid API key.
    url = 'https://api.cognitive.microsoft.com/sts/v1.0/issueToken'
    headers = {
        'Ocp-Apim-Subscription-Key': LC.getParmValue('Speech_Subscription/YOUR_API_KEY')
    }
    r = requests.post(url, headers=headers)
    token = r.content
    return(token)

def get_text(token, audio):
    # Request that the Bing Speech API convert the audio to text
    #url = 'https://{0}.stt.speech.microsoft.com/speech/recognition/{1}/cognitiveservices/v1?language={2}&format={3}'\
    #            .format(parm.REGION, parm.MODE, parm.LANG, parm.FORMAT)
    url= 'https://speech.platform.bing.com/speech/recognition/{0}/cognitiveservices/v1?language={1}&format={2}'\
            .format(
                    LC.getParmValue('Speech_Subscription/MODE'), 
                    LC.getParmValue('Speech_Subscription/LANG'),
                    LC.getParmValue('Speech_Subscription/FORMAT')
                    )
    #print("End Point:{}".format(url))
    headers = {
        'Accept': 'application/json',
        'Ocp-Apim-Subscription-Key': LC.getParmValue('Speech_Subscription/YOUR_API_KEY'),
        'Transfer-Encoding': 'chunked',
        'Content-type': 'audio/wav; codec=audio/pcm; samplerate=16000',
        'Authorization': 'Bearer {0}'.format(token)
    }
    #print("Header:{}".format(headers))
    
    #r = requests.post(url, headers=headers, data=stream_audio_file(audio))
    #results = json.loads(r.content)
    myaudio = AudioSegment.from_file(audio)
    myaudio=myaudio.set_frame_rate(16000)
    myaudio=myaudio.set_sample_width(2)
    myaudio=myaudio.normalize()
    myaudio.export("Converted.wav",format='wav')
    myaudio=AudioSegment.from_file("Converted.wav")
    #myaudio=myaudio.split_to_mono()[0]
    #myaudio.strip_silence(silence_len=500,silence_thresh=-20)
    
    chunk_length_ms = 10000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of 10 sec
    results=pd.DataFrame()
    for i, chunk in enumerate(chunks):
        #if i>6:break
        chunk_name = "chunk{0}.wav".format(i)
        logger.info("Processing Chunk Number:{}".format(i))
        chunk.export(chunk_name, format="wav")
        
        print("Chunk:{}, dbs:{} frame rate:{}".format(chunk_name,chunk.dBFS,chunk.frame_rate))
           
        r = requests.post(url, headers=headers, data=stream_audio_file(chunk_name))
        logger.info("Azure Speech API Response:{}".format(r.status_code))
        logger.info("Azure Speech API Response Content:{}".format(r.content))
        if r.status_code == 200:
            temp = json.loads(r.content)
            results=results.append(temp,ignore_index=True)
        else:
            print("Speech API Error:{}".format(r.status_code))
            print(r.reason)
            
        try:
            os.remove(chunk_name)
        except OSError:
            pass
    
    return results

def stream_audio_file(speech_file, chunk_size=1024):
    # Chunk audio file
    with open(speech_file, 'rb') as f:
        while 1:
            data = f.read(chunk_size)
            if not data:
                break
            yield data

def speech_to_text(path):
    BING_KEY = LC.getParmValue('Speech_Subscription/YOUR_API_KEY')
    # Microsoft Bing Voice Recognition API keys 32-character lowercase hexadecimal strings
    
    # use the audio file as the audio source
    r = sr.Recognizer()
    
    with sr.AudioFile(path) as source:
        audio = r.record(source)
    
    try:
        text=r.recognize_bing(audio, key=BING_KEY)
        logger.info("Microsoft Bing Voice Recognition thinks you said:{} ".format(text) )
        return text
    except sr.UnknownValueError:
        print("Unknown Error")
        logger.info("Microsoft Bing Voice Recognition could not understand audio")
    except sr.RequestError as e:
        print("Error")
        logger.info("Could not request results from Microsoft Bing Voice Recognition service; {0}".format(e))
    
    
#%%

    