# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Tue Oct  8 10:53:22 2018

@author: vprayagala2
Read Text and Generate Word Cloud
"""
#%%
#Imports
import os
import argparse
 
from Source.Config import LoadConfiguration as LC
from Source.Storage import AzureBlobWrapper as Blob
from Source.DataHandler import PrepareData as RDT
from wordcloud import WordCloud 
#import matplotlib.pyplot as plt

#Get the logger
LC.load_config_file()
import logging
logger=logging.getLogger(LC.getParmValue('LogSetup/Log_Name'))
#%%
#Read Text and Generate Word Cloud
def process(path):
    logger.info("Reading Text Files From Path:{}".format(path))
    text=RDT.readDataFolder(path)
    logger.info("Cleaning Text")
    cleaned=[RDT.cleanTextData(line.pop(),stopword_extend=True) for line in text if len(line) > 0]
    
    cleaned_list=[item for sublist in cleaned for item in sublist]
    cleaned_text=' '.join(cleaned_list)
    logger.info("Text Cleaned and Generating Word Cloud")
    logger.info("Number of words extracted from text:{}".format(len(cleaned_text)))
    if len(cleaned_text) > 0:
        wordcloud = WordCloud(width=700,height=600,background_color="white", max_words=100,\
                          contour_width=4, contour_color='firebrick')\
                          .generate(cleaned_text)
    else:
        return 0
    # plot the WordCloud image                        
#    plt.figure() 
#    plt.imshow(wordcloud,interpolation='bilinear') 
#    plt.axis("off") 
#    plt.tight_layout(pad = 0) 
#    plt.show()
    plot_file="wordcloud.png"
    local_or_cloud=LC.getParmValue('DataSource/In_Cloud')
    if local_or_cloud == 'Local':
        local_file_path=os.path.join(path,plot_file)
        logger.info("Saving Word Cloud as {}".format(local_file_path))
        wordcloud.to_file(local_file_path)
    else:
        local_file_path=os.path.join(os.getcwd(),plot_file)
        wordcloud.to_file(local_file_path)
        logger.info("Uploading Word Cloud as Blob")
        Blob.create_blob(LC.getParmValue('DataSource/Azure_Container'),
                         blob_file_path=os.path.join(path,plot_file),
                         local_file_path=local_file_path)
        try:
            os.remove(local_file_path)
        except OSError:
            pass
        
    return 1
   
     
#%%
# Below is debugging code to run this script standalone
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    
#    parser.add_argument('--path', 
#                        help='Enter Absolute Path of Text files directory)')
#    
#    args = parser.parse_args()
#    
#    if not (os.path.exists(args.path)):
#        print("Invalid Path Provided")
#        sys.exit()
#    else:
#        process(args.path)
#        print("End of Processing")
