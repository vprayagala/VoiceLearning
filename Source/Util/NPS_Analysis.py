# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:16:11 2019

@author: vprayagala2
"""
#%%
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
from Source.Config import LoadConfiguration as LC
from Source.Logger import CreateLogger as CL
LC.load_config_file()
#Get the logger
logger=CL.create_logger(LC.getParmValue('LogSetup/Log_Name'))
#%%
from Source.DataHandler import PrepareData as PD
from Source.Models import BuildModel as BM
from Source.Models import Score 

#from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim
import json
from keras import backend as K
from textblob import TextBlob
#%%
raw_data=pd.DataFrame()
file_dir = os.path.join(os.path.join(os.getcwd(),"Data"),"NPS data 2015-19")
output_dir = os.path.join(os.getcwd(),"Outputs")
for dir_path,_,files in os.walk(file_dir):
    for file in files:
        file_name=os.path.join(dir_path,file)
        logger.info("Reading File:{}".format(file_name))
        temp=pd.read_excel(file_name,sheet_name="Data")
        raw_data=raw_data.append(temp,ignore_index=True,sort=True)
logger.info("Data Shape:{}".format(raw_data.shape))
logger.info("Data Types:{}".format(raw_data.dtypes))
#%%
#Consider only the required text columns for feedback analysis
column_name_dict={'x': 'What can we do to improve?',
                  'y': 'You indicated you are  unlikely to recommend DXC to a friend or a colleague. What can we do to improve?',
                  'z': 'You indicated you are likely to recommend DXC to a friend or a colleague. What factors contributed most to this?'
                  }
data=raw_data.iloc[:,[77,80,81]]
data.columns=list(column_name_dict.keys())
logger.info("Print the nulls or missing values:{}".format(data.isnull().sum()))
null_summary= (data.isnull().sum() / data.shape[0]) * 100
null_summary.plot(kind='bar',title="Percent Missing Values for each Column")
#%%
for i in range(data.shape[1]):
    text=data.iloc[:,i].dropna().tolist()
    cleaned=[PD.cleanTextData(line) for line in text]
    #cleaned_list=[item for sublist in cleaned for item in sublist]
    #print(cleaned_list)
    model,tdm,id2word = BM.topic_modeling_lda(cleaned, max_topics=8,true_topics=4)
    #Visualize the topics
    vis_data=pyLDAvis.gensim.prepare(model,tdm,id2word,R=20,\
                                     mds="mmds",sort_topics=False)
    
    
    topic_result=os.path.join(output_dir,"topic_result_"+str(i)+".json")
    lda_result=os.path.join(output_dir,"lda_topics_"+str(i)+".html")
    topics = model.print_topics(num_topics=4, num_words=20)
    with open(topic_result,'w') as f:
        json.dump(topics,f)
    pyLDAvis.save_html(vis_data,lda_result)  
#%%
def predictScore(model,text):
    result={}
    logger.info("loaded model in new file")

    if len(text) < 3 :
        result['Label'] = " "
        result['Score'] = 0
        return result
    else:
        #print(text)
        b = TextBlob(text)
        lang=b.detect_language()
        if lang == 'en':
            cleaned=[PD.cleanTextData(line) for line in text]
            cleaned_list=[item for sublist in cleaned for item in sublist]
            cleaned_text=' '.join(cleaned_list)
            label,score=Score.predict_score(model,cleaned_text)
        else:
            result['Label'] = lang
            result['Score'] = 0
            return result
        
        #result['Cleaned_Text']=text
        if label[0][0] == 1:
            result['Label']="Positive"
        else:
            result['Label']="Negative"
            
        result['Score']=score[0][0]
        return result
#%%

K.clear_session()
model=Score.load_model()
result_df=pd.DataFrame()
count=0
for i in range(data.shape[1]):
    text=data.iloc[:,i].dropna().tolist()
    sent_res = pd.DataFrame()
    for line in text:
        count+=1
        temp=predictScore(model,line)
        df=pd.DataFrame([[line,temp['Label'],temp['Score']]],
                        columns=['Text','Label','Score'])
        sent_res=sent_res.append(df,ignore_index=True)
        
        if count/1000 == 0:
            print("Processed percentage :{}".format((count/data.shape[0])*100))
    result_df=pd.concat([result_df,sent_res],axis=1,ignore_index=True)
#%%
result_df.columns=['What can we do to improve?',
                   'Label1',
                   'Score1',
                   'You indicated you are  unlikely to recommend DXC to a friend or a colleague. What can we do to improve?',
                   'Label2',
                   'Score2',
                   'You indicated you are likely to recommend DXC to a friend or a colleague. What factors contributed most to this?',
                   'Label3',
                   'Score3'
                   ]
sent_file=os.path.join(output_dir,"Sentiment_Result.xlsx")
result_df.to_excel(sent_file)
#%%
from wordcloud import WordCloud 
for i in range(data.shape[1]):
    text=data.iloc[:,i].dropna().tolist()
    cleaned=[PD.cleanTextData(line) for line in text]
    cleaned_list=[item for sublist in cleaned for item in sublist]
    cleaned_text=' '.join(cleaned_list)
    if len(cleaned_text) > 0:
        wordcloud = WordCloud(width=700,height=600,background_color="white", max_words=200,\
                          contour_width=4, contour_color='firebrick')\
                          .generate(cleaned_text)
        plot_file="wordcloud"+str(i)+".png"
        local_file_path=os.path.join(output_dir,plot_file)
        logger.info("Saving Word Cloud as {}".format(local_file_path))
        wordcloud.to_file(local_file_path)