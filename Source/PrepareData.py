# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 10:53:22 2018

@author: vprayagala2
Parse VTT file and combine text
"""

#%%
import os
import re
import glob

import string
#import collections
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer
#import matplotlib.pyplot as plt
#%%
#Get the logger
import LoadConfiguration as LC
LC.load_config_file()
import AzureBlobWrapper as Blob
import logging
logger=logging.getLogger(LC.getParmValue('LogSetup/Log_Name'))
#%%
def cleanLines(lines):
    file_content=""
    if LC.getParmValue('DataSource/In_Cloud') == 'Local':
        pat1='.*WEBVTT.*'
        pat2='.*language:.*'
        pat3='^NOTE.*'
        pat4=r'\d+'
    else:
        pat1=b'.*WEBVTT.*'
        pat2=b'.*language:.*'
        pat3=b'^NOTE.*'
        pat4=b'\d+'
        
    for line in lines:
        if re.match(pat1,line)\
        or re.match(pat2,line)\
        or re.match(pat3,line)\
        or re.match(pat4,line):
                continue
        else:
            #text=text+(line.strip("\n"))
            #text=text+" "+line
            if LC.getParmValue('DataSource/In_Cloud') == 'Local':
                line=line.strip('\n')
                if len(line) > 0:
                    file_content=file_content+" "+line
            else:
                line=line.strip(b'\r\n')
                if len(line) > 0:
                    file_content=file_content+" "+line.decode()
    return file_content

def readTextDataVTT(path=None):
    assert path!=None,"Path Cannot be blank, Provide absolute path to prepare data"
    text=[]
    #print(path)
    local_or_cloud=LC.getParmValue('DataSource/In_Cloud')
    if local_or_cloud == 'Local':
        for file in glob.glob(path+"*.vtt"):
            logger.info("Reading Text File:{}".format(file))
            #vttfiles.append(file)
            with open(file) as f:
                lines=f.readlines()
            file_content=cleanLines(lines)
            text.append([file_content])
    else:
        list_of_blobs=Blob.list_blobs(LC.getParmValue('DataSource/Azure_Container'),path)
        for blob in list_of_blobs:
            head, tail = os.path.split("{}".format(blob))
            #print(head)
            #print(tail)
            if (head == path[:-1] ) & (tail.find('.vtt') > 0): #reading only vtt files
                blob_data=Blob.read_blob(LC.getParmValue('DataSource/Azure_Container'),blob)
                lines=blob_data.split(b'\r\n')
                file_content=cleanLines(lines)
                text.append([file_content])
    return text

def readTextDataLocal(path=None):
    assert path!=None,"Path Cannot be blank, Provide absolute path to prepare data"
    text=[]
    for file in glob.glob(path+"*.*"):
        logger.info("Reading Text File:{}".format(file))
        #vttfiles.append(file)
        with open(file) as f:
            lines=f.readlines()
        file_content=""
        for line in lines:
            line=line.strip('\n')
            if len(line) > 0:
                file_content=file_content+" "+line
        text.append([file_content])
    return text  

def readTextDataFromBlob(path=None):
    assert path!=None,"Path Cannot be blank, Provide absolute path to prepare data"
    
    
#%%
def cleanTextData(text):
    #Pre-Process Text, Remove Stop words, Numbers 
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    
    stop_words = set(stopwords.words('english'))
    stop_words.update(["ok","help","ye","yeah","thank","help","nt","yes","no","know",\
                       "hello","want","give","find","said","come","find",\
                       "think","call","well","right","with","without","make","let","inthe",\
                       "havea","welcome","coming","much",\
                       'could', 'might', 'must', 'need', 'sha', 'wo', 'would',\
                       "one","please","go",
                       "with","the","oh","got","tell","put","even","every","page","still",
                       "chick","chicken"])
    
    words = [w for w in words if not w in stop_words]
    # stemming of words
    #porter = PorterStemmer()
    
    lemmatizer = WordNetLemmatizer()
    #words = [porter.stem(word) for word in words]
    #
    cleaned = [lemmatizer.lemmatize(word,pos) if pos in ['n','v','a'] else lemmatizer.lemmatize(word)\
               for word,pos in pos_tag(words)]
    return cleaned





