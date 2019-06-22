#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:08:16 2018

@author: Saatwik
"""

#Set Global Parameters
import os
import yaml
#Set Working Directory
#os.chdir(os.path.expanduser("~/AnacondaProjects/VoiceLearning"))  
#config= ' '
#%%
def load_config_file():
    global config
    with open("config.yaml","r") as config_f:
        config=yaml.load(config_f)['Application']

    #print(config)
    #Validate the certain input fields
    #Check if log directory exists

    log_dir=os.path.join(os.getcwd(),getParmValue('LogSetup/Log_Dir'))
    if not os.path.exists(log_dir):
        raise Exception("The logging directory {} does not exists".format(log_dir))
    
    data_source=getParmValue('DataSource')
    if data_source['In_Cloud'] == 'Local':
        in_dir=os.path.join(os.getcwd(),data_source['In_Dir'])
        out_dir=os.path.join(os.getcwd(),data_source['Out_Dir'])
        model_dir=os.path.join(os.getcwd(),data_source['Model_Dir'])
        
        if not os.path.exists(in_dir):
            raise Exception("The logging directory {} does not exists".format(in_dir))  
    
        if not os.path.exists(out_dir):
            raise Exception("The Output directory {} does not exists".format(out_dir)) 
            
        if not os.path.exists(model_dir):
            raise Exception("The Model directory {} does not exists".format(model_dir)) 
        
    if data_source['In_Cloud'] == 'Azure':
        azure_cont=getParmValue('Cloud_Storage/Storage_Account')
        if azure_cont == '' or azure_cont == ' ':
            raise Exception("For Cloud Source Storage Account is required, but input provided:{}".format(azure_cont))

        azure_storage_key=getParmValue('Cloud_Storage/Storage_Key')
        if azure_storage_key == '' or azure_cont == ' ':
            raise Exception("For Cloud Source Storage Account/Key is required, but input provided:{}".format(azure_storage_key))        
        
#Return the parameter value
def getParmValue(parm='Name'):
    global config
    with open("config.yaml","r") as config_f:
        config=yaml.load(config_f)['Application']
    value=None
    path_list=parm.split('/')

    len_parm_list=len(path_list)
    i=0
    temp=config.copy()
    while(i < len_parm_list):
        if i == (len_parm_list - 1):
            value=temp[path_list[i]]
        else:
            temp=temp[path_list[i]]
        i=i+1
    return value
#%%
#IN_DIR=os.path.join(os.getcwd(),"Data/")
#OUT_DIR=os.path.join(os.getcwd(),"outputs/")
#LOG_DIR=os.path.join(os.getcwd(),"Log/")
#WEIGHTS_FILEPATH = OUT_DIR+'imdb_bidirectional_lstm.hdf5'
#MODEL_ARCH_FILEPATH = OUT_DIR+'imdb_bidirectional_lstm.json'
##Corpus Size
#max_features = 100000
## cut texts after this number of words
## (among top max_features most common words)
#maxlen = 200
#vector_length=64
#batch_size = 128
#epochs = 10
#
#YOUR_API_KEY = 'e2626181f6654568a7f0b17871a0c413'
#
#REGION = 'East US' # westus, eastasia, northeurope 
#MODE = 'interactive'
#LANG = 'en-US'
#FORMAT = 'simple'



