# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:07:19 2018

@author: vprayagala2
contenttype - json 
"""
#%%
import requests
json_data={'path':'Data/Cluster/'}
server='http://dxc-ds-prod-restapi.eastus.cloudapp.azure.com'
port='5000'

#Request for Word Cloud
host=server+':'+port+'/'+'genWordCloud'
resp = requests.post(host, json=json_data)
print("Status of POST Request :{}".format(resp.status_code))
if resp.status_code != 200:
    raise ('POST /tasks/ {}'.format(resp.status_code))
else:
    print("Response:{}".format(resp.content))
#Request for Sentiment Score
host=server+':'+port+'/'+'predictSentimentScore'
resp = requests.post(host, json=json_data)
print("Status of POST Request :{}".format(resp.status_code))
if resp.status_code != 200:
    raise ('POST /tasks/ {}'.format(resp.status_code))
else:
    print("Response:{}".format(resp.content))
#%%