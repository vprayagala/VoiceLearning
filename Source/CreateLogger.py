#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:38:25 2017

@author: Saatwik
"""
#Set Logging
import os,logging
from time import strftime
import LoadConfiguration as LC
LC.load_config_file()

log_dir=os.path.join(os.getcwd(),LC.getParmValue('LogSetup/Log_Dir'))
log_level=LC.getParmValue('LogSetup/Log_Level')

def create_logger(logname):
    print("Invoked Create Logger")
    logger = logging.getLogger(logname)
    logger.setLevel(log_level)
    
    # create a file handler
    file_name=os.path.join(log_dir,logname+strftime("_%Y-%m-%d-%H%M%S")+".txt")
    handler = logging.FileHandler(file_name,
                                  mode='w')
    handler.setLevel(log_level)
    
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info('Start of Program - Logger has been created')
    
    return logger
