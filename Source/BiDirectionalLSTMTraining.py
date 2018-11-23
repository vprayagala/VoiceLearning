# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import print_function
"""
Created on Thu Sep 20 09:30:19 2018

"""
'''Train a Bidirectional LSTM on the IMDB sentiment
classification task.
'''
#%%
import os,sys
#os.chdir(os.path.expanduser("~/AnacondaProjects/VoiceLearning"))
sys.path.append(os.path.join(os.getcwd(),"Source"))
print("Current SystemPath:{}".format(sys.path[-1]))
import LoadConfiguration as LC
#%%
#Import required modules
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

from keras.callbacks import EarlyStopping, ModelCheckpoint
import json
#%%
#Set the looging method
import CreateLogger as CL
logger=CL.create_logger('Training_Log')
logger.info("Current SystemPath:{}".format(sys.path[-1]))
#%%
logger.info('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=LC.getParmValue('LSTM/max_features'))
logger.info('{} train sequences'.format(len(x_train)))
logger.info('{} test sequences'.format(len(x_test)))
#%%
logger.info('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=LC.getParmValue('LSTM/maxlen'))
x_test = sequence.pad_sequences(x_test, LC.getParmValue('LSTM/maxlen'))
logger.info('x_train shape:{}'.format( x_train.shape))
logger.info('x_test shape:{}'.format(x_test.shape))
y_train = np.array(y_train)
y_test = np.array(y_test)
#%%
model = Sequential()
model.add(Embedding(LC.getParmValue('LSTM/max_features'), 
                    LC.getParmValue('LSTM/vector_length'), 
                    input_length=LC.getParmValue('LSTM/maxlen')))
model.add(Bidirectional(LSTM(LC.getParmValue('LSTM/batch_size')*2)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
#%%
logger.info('Train...')
#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=4,
#          validation_data=[x_test, y_test])
##%%
# Model saving callback
checkpointer = ModelCheckpoint(filepath=LC.getParmValue('Model/LSTM_model_weight'), 
                               monitor='val_acc', verbose=1, save_best_only=True)

# Early stopping
early_stopping = EarlyStopping(monitor='val_acc', verbose=1, patience=5)

# train

model.fit(x_train, y_train,
          validation_data=[x_test, y_test],
          batch_size=LC.getParmValue('LSTM/batch_size'), 
          epochs=LC.getParmValue('LSTM/epochs'), 
          verbose=2,
          callbacks=[checkpointer, early_stopping])

with open(LC.getParmValue('Model/LSTM_model_json'), 'w') as f:
    f.write(model.to_json())
#%%
word_index = imdb.get_word_index()
word_dict = {idx: word for word, idx in word_index.items()}

sample = []
for idx in x_train[0]:
    if idx >= 3:
        sample.append(word_dict[idx-3])
    elif idx == 2:
        sample.append('-')
' '.join(sample)
#%%
with open(os.path.join(LC.getParmValue('DataSource/Out_Dir'),
                       'imdb_dataset_word_index_top20000.json'), 'w') as f:
    f.write(json.dumps({word: idx for word, idx in word_index.items() if idx < LC.getParmValue('LSTM/max_features')}))

with open(os.path.join(LC.getParmValue('DataSource/Out_Dir'),
                       'imdb_dataset_word_dict_top20000.json'), 'w') as f:
    f.write(json.dumps({idx: word for word, idx in word_index.items() if idx < LC.getParmValue('LSTM/max_features')}))

sample_test_data = []
for i in np.random.choice(range(x_test.shape[0]), size=1000, replace=False):
    sample_test_data.append({'values': x_test[i].tolist(), 'label': y_test[i].tolist()})

with open(os.path.join(LC.getParmValue('DataSource/Out_Dir'),
                       'imdb_dataset_test.json'), 'w') as f:
    f.write(json.dumps(sample_test_data))
