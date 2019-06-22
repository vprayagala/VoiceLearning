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
import logging
import string
#import collections
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer
#import matplotlib.pyplot as plt
# spacy for lemmatization
import spacy
import pandas as pd
import docx
import PyPDF2
#%%
#Get the logger
from Source.Config import LoadConfiguration as LC
from Source.Storage import AzureBlobWrapper as Blob

LC.load_config_file()
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
        for file in glob.glob(path+"**/*.vtt",recursive=True):
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
    for file in glob.glob(path+"**/*.txt",recursive=True):
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
 
def readPDFDataLocal(path=None):
    assert path!=None,"Path Cannot be blank, Provide absolute path to prepare data"
    text=[]
    
    for file in glob.glob(os.path.join(path,"*.pdf"),recursive=True):
        logger.info("Reading PDF File:{}".format(file))
        file_content=""
        #vttfiles.append(file)
        pdfFileReader = PyPDF2.PdfFileReader(open(file, 'rb'),strict=False)
        totalPageNumber = pdfFileReader.numPages
        currentPageNumber = 1
        
        while(currentPageNumber < totalPageNumber ):       
            pdfPage = pdfFileReader.getPage(currentPageNumber)
            file_content = file_content +" "+pdfPage.extractText()
            currentPageNumber = currentPageNumber + 1
        text.append([file_content])
    return text 

def readCSVDataLocal(path=None):
    assert path!=None,"Path Cannot be blank, Provide absolute path to prepare data"
    text=[]
    for file in glob.glob(path+"**/*.csv",recursive=True):
        logger.info("Reading CSV File:{}".format(file))
        #vttfiles.append(file)
        df=pd.read_csv(file)
        file_content=""
        #print(df.columns[1])
        #print(df.index)
        for i in df.index:
            for j in df.columns:
                #print(df.iloc[i][j])
                file_content=file_content+" "+str(df.iloc[i][j])
        text.append([file_content])
    return text  

def readDOCXDataLocal(path=None):
    assert path!=None,"Path Cannot be blank, Provide absolute path to prepare data"
    text=[]
    for file in glob.glob(path+"**/*.docx",recursive=True):
        logger.info("Reading DOCX File:{}".format(file))
        file_content=""
        #vttfiles.append(file)
        doc = docx.Document(file)
        for para in doc.paragraphs:
            file_content=file_content+" "+para.text
        text.append([file_content])
    return text

def readDataFolder(path=None):
    assert path!=None,"Path Cannot be blank, Provide absolute path to prepare data"
    text=[]
    logger.info("Read VTT")
    text.extend(readTextDataVTT(path))
    logger.info("Read TXT")
    text.extend(readTextDataLocal(path))
    logger.info("Read PDF")
    text.extend(readPDFDataLocal(path))
    logger.info("Read CSV")
    text.extend(readCSVDataLocal(path))
    logger.info("read DOCX")
    text.extend(readDOCXDataLocal(path))
    return text

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
    
def cleanTextData(text,stopword_extend=False):
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
    if stopword_extend:
        stop_words.update([
                            "ok","help","ye","yeah","thank","help","nt","yes","no",\
                           "know","hello","want","give","find","said","come","find",\
                           "think","call","well","right","with","without","make",\
                           "let","inthe","havea","welcome","coming","much",\
                           'could', 'might', 'must', 'need', 'sha', 'wo', 'would',\
                           "one","please","go","with","the","oh","got","tell","put",\
                           "even","every","page","still","chick","chicken","yet","nan"\
                           ])
    
    words = [w for w in words if not w in stop_words]
    # stemming of words
    #porter = PorterStemmer()
    
    lemmatizer = WordNetLemmatizer()
    #words = [porter.stem(word) for word in words]
    #
    cleaned = [lemmatizer.lemmatize(word,pos) if pos in ['n','v','a'] else lemmatizer.lemmatize(word)\
               for word,pos in pos_tag(words)]
    return cleaned

def extractPdfText(filePath=''):


    fileObject = open(filePath, 'rb')
   
    pdfFileReader = PyPDF2.PdfFileReader(fileObject)
    totalPageNumber = pdfFileReader.numPages
    currentPageNumber = 1
    text = ''

    while(currentPageNumber < totalPageNumber ):

       
        pdfPage = pdfFileReader.getPage(currentPageNumber)
        text = text + pdfPage.extractText()
        currentPageNumber = currentPageNumber + 1
        
    #text=text.replace("\n","")
    #Extract Sentences, replace new lines in single sentence. Combine sentences back
    sent_list = sent_tokenize(text)
    sent_list_tr=[]
    for sent in sent_list:
        sent=sent.replace("\n","")
        sent_list_tr.append(sent)
    '.'.join(sent_list)
    return text



