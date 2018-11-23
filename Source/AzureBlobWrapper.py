# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 13:41:35 2018

@author: vprayagala2

Azure Storage Wrapper - Handles Reads, Write, List and Delete of Blobs
"""
import os, sys
import logging
from azure.storage.blob import BlockBlobService
sys.path.append(os.path.join(os.getcwd(),"Source"))
import LoadConfiguration as LC
LC.load_config_file()


logger=logging.getLogger(LC.getParmValue('LogSetup/Log_Name'))
#%%
block_blob_service = BlockBlobService(account_name=LC.getParmValue('Cloud_Storage/Storage_Account'),
                                      account_key=LC.getParmValue('Cloud_Storage/Storage_Key'))
def create_container(container_name=' '):
    assert container_name != ' ',"Container Name Cannot be Blank for creating a blob"   
    block_blob_service.create_container(container_name)
    logger.info("Successfully Created Container %s" %(container_name))


def create_blob(container_name=' ',blob_file_path=None,local_file_path=None):
    assert container_name != ' ',"Container Name Cannot be Blank for creating a blob" 
    logger.info("Creating blob {} from local file path {}".format(blob_file_path,local_file_path))
    block_blob_service.create_blob_from_path(container_name, blob_file_path, local_file_path)


def download_blob(container_name=' ',blob_file_path=None,local_file_path=None):
    assert container_name != ' ',"Container Name Cannot be Blank for creating a blob" 
    logger.info("Downloading blob from path {} to {}".format(blob_file_path,local_file_path))
    block_blob_service.get_blob_to_path(container_name, blob_file_path, local_file_path)

def read_blob(container_name=' ',blob_file_path=None):
    assert container_name != ' ',"Container Name Cannot be Blank for creating a blob" 
    logger.info("Reading blob from path {}".format(blob_file_path))
    data=block_blob_service.get_blob_to_bytes(container_name, blob_file_path)
    text=data.content
    return text

def list_blobs(container_name= ' ',path=None):
    assert container_name != ' ',"Container Name Cannot be Blank for creating a blob" 
    list_of_blobs=[]
    logger.info("List blobs in the container")
    generator = block_blob_service.list_blobs(container_name,prefix=path)
    for blob in generator:
        logger.info("Blob Name:{} in path:{}".format(blob.name,path))
        list_of_blobs.append(blob.name)
    
    return list_of_blobs

def delete_blob(container_name=' ',blob_file_path=None):
    assert container_name != ' ',"Container Name Cannot be Blank for creating a blob"
    block_blob_service.delete_blob(container_name,blob_file_path)

def delete_container(container_name=' '):
    assert container_name != ' ',"Container Name Cannot be Blank for creating a blob"
    block_blob_service.delete_container(container_name)
#%%
#Test

