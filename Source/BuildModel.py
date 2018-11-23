# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:30:43 2018

@author: vprayagala2
Build Model
Write function for each of machine learning experimentation
"""
#%%
import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
# Gensim
import gensim
import gensim.corpora as corpora
#from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import LoadConfiguration as LC
LC.load_config_file()
#Get the logger
import logging
logger=logging.getLogger(LC.getParmValue('LogSetup/Log_Name'))
#%%
#Define Functions 
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def cluster_texts_kmeans(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(#max_df=0.5,
                                 #min_df=0.1,
                                 #lowercase=True)
                                    )
    tfidf_model = vectorizer.fit_transform([word for word in texts])
    
    #Fit different cluster and pick the optimal cluster size
    df_clust=pd.DataFrame()
    for i in range(2,clusters+1):
        #Build model
        logger.info("Building Kmean with {} cluster".format(i))
        km_model = KMeans(n_clusters=i,random_state=7)
        km_model.fit(tfidf_model)
        #labels=km_model.labels_
        #score=silhouette_score(tfidf_model, labels, metric='euclidean')
        score=km_model.inertia_
        logger.info("K-Means Score:{}".format(score))
        df_clust=df_clust.append({"num_clusters":i,"score":score},ignore_index=True)
    
    plt.figure()
    plt.plot(df_clust["num_clusters"],df_clust["score"])
    plt.savefig("kmeans_elbow.png")
    #clustering = collections.defaultdict(list)
    #for idx, label in enumerate(km_model.labels_):
    #    clustering[label].append(idx)
    true_k=5
    km=KMeans(n_clusters=true_k,random_state=77)
    km.fit(tfidf_model)
    kmeans_clust=pd.DataFrame()
    logger.info("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    
    for i in range(true_k):
        term_list=[]
        logger.info("Cluster %d:\n" % i, end='')
        for ind in order_centroids[i, :15]:
            logger.info(' %s' % terms[ind], end='')
            term_list.append(terms[ind])
            logger.info()
        kmeans_clust=kmeans_clust.append({"Cluster_Num":i,"Top_Terms":term_list},\
                                         ignore_index=True) 

    return km,kmeans_clust

def topic_modeling_lda(texts, clusters=5):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
        #Explore Topic Modeling
        ## python3 -m spacy download en
    # Create Dictionary
    bigram = gensim.models.Phrases(texts)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    data_bigrams=[bigram_mod[sentence] for sentence in texts]
    
    data_cleaned = lemmatization(data_bigrams,\
                                    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    id2word = corpora.Dictionary(data_cleaned)

    # Term Document Frequency
    tdm = [id2word.doc2bow(word) for word in data_cleaned]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=tdm,
                                           id2word=id2word,
                                           num_topics=clusters, 
                                           random_state=7,
                                           update_every=1,
                                           chunksize=1,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True
                                           )
    topics = lda_model.print_topics(num_topics=clusters, num_words=15)
    logger.info("Topics:{}".format(topics))
       
    # Compute Perplexity
    logger.info('\nPerplexity: {}'.format(lda_model.log_perplexity(tdm)) )
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_cleaned, 
                                         dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    logger.info('\nCoherence Score: {}'.format(coherence_lda))
    return lda_model,tdm,id2word
