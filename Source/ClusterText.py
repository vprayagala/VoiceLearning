# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on wed Oct  10 09:53:22 2018

@author: vprayagala2
Read text from directory and cluster the text
"""
#%%
#Imports
import os,sys
#os.chdir(os.path.expanduser("~/AnacondaProjects/VoiceLearning"))
sys.path.append(os.path.join(os.getcwd(),"Source"))

import warnings
warnings.filterwarnings("ignore")

import json
import argparse
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim

import matplotlib.pyplot as plt
import PrepareData as RDT
import BuildModel as BM
import LoadConfiguration as LC
LC.load_config_file()

import logging
logger=logging.getLogger(LC.getParmValue('LogSetup/Log_Name'))
#%%
#Read Text and Generate Word Cloud
def process(path,num_clust):
    logger.info("Reading Text Files From Path:{}".format(path))
    text=RDT.readTextDataVTT(path)
    
    logger.info("Reading is Complete and Starting Cleaing of Data")
    cleaned=[RDT.cleanTextData(line.pop()) for line in text if len(line) > 0]
    
    cleaned_list=[item for sublist in cleaned for item in sublist]
    logger.info("Text Cleaned and Generating Word Cloud")
    logger.info("Number of Lines extracted from text:{}".format(len(cleaned)))
    
    #explore Kmeans
    km_model,kmeans_res=BM.cluster_texts_kmeans(cleaned_list,clusters=num_clust)
    result=path+"kmean_result.json"
    kmeans_res.to_json(result,orient='records')
    for i in range(kmeans_res.shape[0]):
        term_list=kmeans_res.iloc[i,1]
        wordcloud = WordCloud(width=150,height=100,background_color="white",\
                          contour_width=4, contour_color='firebrick')\
                          .generate(' '.join(term_list))
        # plot the WordCloud image                        
        plt.figure() 
        #plt.imshow(wordcloud,interpolation='bilinear') 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
        #plt.show()
        
        plot_file=path+"wordcloud_cluster_"+str(i)+".png"
        wordcloud.to_file(plot_file)
        
    #Now try with topic modeling
    lda_model,tdm,id2word=BM.topic_modeling_lda(cleaned,clusters=5)
    for t in range(lda_model.num_topics):
        plt.figure()
        wordcloud=WordCloud().fit_words(dict(lda_model.show_topic(t, 15)))
        #plt.imshow(wordcloud)
        plt.axis("off")
        plt.title("Topics")
        #plt.show()
        plot_file=path+"wordcloud_topic_"+str(t)+".png"
        wordcloud.to_file(plot_file)
        
    #Visualize the topics
    vis_data=pyLDAvis.gensim.prepare(lda_model,tdm,id2word,R=20,\
                                     mds="mmds",sort_topics=False)
    pyLDAvis.save_html(vis_data,path+'lda_topics.html')
    
    topic_result=path+"topic_result.json"
    topics = lda_model.print_topics(num_topics=5, num_words=20)
    with open(topic_result,'w') as f:
        json.dump(topics,f)
    
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', 
                        help='Enter Absolute Path of Text files directory)')
    parser.add_argument('--num_clust', 
                        help='Enter the maximum number of clusters to be built)')
    args = parser.parse_args()
    
    if not (os.path.exists(args.path)):
        print("Invalid Path Provided")
        sys.exit()
    else:
        if args.num_clust == None:
            num_clust=10
        else:
            num_clust=int(args.num_clust)
        process(args.path,num_clust)
        print("End of Processing")

