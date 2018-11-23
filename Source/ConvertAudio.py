# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:13:43 2018

@author: vprayagala2

Convert the audio files in specified path to text
"""
#%%
#Imports
import os,sys
#os.chdir(os.path.expanduser("~/AnacondaProjects/VoiceLearning"))
sys.path.append(os.path.join(os.getcwd(),"Source"))

import argparse,glob
import LoadConfiguration as LC
import CreateLogger as CL
LC.load_config_file()
import AzureBlobWrapper as Blob
import AzureAudioConversion as Audio
#Get the logger
import logging
logger=logging.getLogger(LC.getParmValue('LogSetup/Log_Name'))
#%%
def process(path):
    local_or_cloud = LC.getParmValue('DataSource/In_Cloud')
    print("FIle Souirce:{}".format(local_or_cloud))
    if local_or_cloud == 'Local':
        if not (os.path.exists(path)):
            logger.info("Invalid Path Provided:{}".format(path))
            sys.exit()
        else:
            #Read Audio files from directory and convert to text
            for file in glob.glob(os.path.join(path,'*.wav')):
                logger.info("Converting Audio File:{}".format(os.path.join(path,file)))
                text=Audio.convertAudio(file)
                if text == ' ':
                    logger.info("Audio Converted to Text, an empty text")
                else:
                    logger.info("Storing Converted Text")
                    file_prefix=file.split('.')[0]
                    out_file=os.path.join(path,file_prefix+'.vtt')
                    with open(out_file,'w') as f:
                        f.write(text)
    else:
        list_of_blobs=Blob.list_blobs(LC.getParmValue('DataSource/Azure_Container'),path)
        for blob in list_of_blobs:
            head, tail = os.path.split("{}".format(blob))
            #print(head)
            #print(tail)
            #create a temp wma file
            local_in_path=os.path.join(os.getcwd(),'temp.WMA')
            local_out_path=os.path.join(os.getcwd(),'temp.vtt')
            if (head == path[:-1] ) & (tail.find('.WMA') > 0):  
                Blob.download_blob(LC.getParmValue('DataSource/Azure_Container'),
                                            blob,
                                            local_in_path)
                text=Audio.convertAudio(local_in_path)
                with open(local_out_path,'w') as f:
                        f.write(text)
                if text == ' ':
                    logger.info("Audio Converted to Text, an empty text")
                else:
                    logger.info("Storing Converted Text to blob")
                    blob_path=blob.replace('.WMA','_Converted.vtt')
                    Blob.create_blob(LC.getParmValue('DataSource/Azure_Container'),
                                            blob_path,
                                            local_out_path)
                
        try:
            os.remove(local_in_path)
            os.remove(local_out_path)
        except OSError:
            print("Exception on Cleaning Temporary Files")
            sys.exit()
#%%
if __name__ == '__main__':
    logger=CL.create_logger(LC.getParmValue('LogSetup/Log_Name'))
    logger.info("Running Program 'ConvertAudio' for converting audio files to text")
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', 
                        help='Enter Path of Text files directory,\
                        Absolute path for local, directory path for cloud')
    
    args = parser.parse_args()
    print("Start Processing Files in Path:{}".format(args.path))
    process(args.path)
    print("End of Processing - Check Path for converted text files")
