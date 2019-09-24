#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import random
import tensorflow as tf

from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Reshape, RepeatVector, TimeDistributed
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from collections import Counter

import numpy as np
import os
from mido import MidiFile, MidiTrack, Message
import mido


# In[27]:


paths = []
songs = []
for r, d, f in os.walk(r"C:\Users\Vee\Desktop\python\songs"):
    for file in f:
        if '.mid' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    mid = MidiFile(path, type = 1)
    songs.append(mid)


# In[28]:


notes = []
dataset = []


# In[29]:


#for each in midi object in list of songs
for i in range(len(songs)):
    #for each note in midi object
    for msg in songs[i]:
        #filtering out meta messages
        if not msg.is_meta:
            #filtering out control changes
            if (msg.type == 'note_on') or (msg.type == 'note_off'):
                #normalizing note and velocity values
                notes.append([msg.note/127, msg.velocity/127, msg.time])
    #if more than 30 notes delete them
    if len(notes) > 60:
        for x in range(len(notes)-60):
            notes = np.delete(notes, 60, 0)
    #if less than 30 notes pad with zeros
    elif len(notes) < 60:
        for x in range(60 - len(notes)):
            notes.append([0,0,0])
    dataset.append(notes)    
    notes = []


# In[30]:


dataset = np.array(dataset)
dataset.shape


# In[31]:


bruv = []
#for each in midi object in list of songs
for i in range(len(songs)):
    #for each note in midi object
    for msg in songs[i]:
        #filtering out meta messages
        if not msg.is_meta:
            #filtering out control changes
            if (msg.type == 'note_on') or (msg.type == 'note_off'):
                #normalizing note and velocity values
                notes.append([msg.note/127, msg.velocity/127, msg.time])
    #if more than 40 notes delete them
    if len(notes) > 4:
        for x in range(len(notes)-4):
            notes = np.delete(notes, 4, 0)
    bruv.append(notes)    
    notes = []


# In[32]:


bruv = np.array(bruv)
bruv.shape


# In[33]:


#splitting data into train and test sets. 3/4 train, 1/4 test.
x_train,x_test,y_train,y_test = train_test_split(bruv, dataset, test_size=0.2, shuffle=False, random_state=42)


# In[34]:


# define model
model = Sequential()
#shaping input to match data
model.add(LSTM(200, activation='relu', input_shape=(4, 3)))
#specifying output to have 60 timesteps
model.add(RepeatVector(60))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(LSTM(200, activation='relu', return_sequences=True))
#specifying 3 features as the output
model.add(TimeDistributed(Dense(3)))
model.add(TimeDistributed(Dense(3)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(3)))
model.add(TimeDistributed(Dense(3)))
model.add(TimeDistributed(Dense(3)))
model.add(Activation('tanh'))
model.compile(loss='mean_absolute_error', optimizer='adam')
print(model.summary())


# In[35]:


model.fit(bruv, dataset, epochs=25, batch_size=10, verbose=2,validation_data=(x_test, y_test))


# In[38]:


predict = model.predict(bruv)
#disregarding negative values
predict = abs(predict)


# In[39]:


#adjusting from normalization
for y in range(len(songs)):
    for x in range(60):
        for i in range(2):
            predict[y][x][i] = predict[y][x][i] * 127
        predict[y][x][2] = predict[y][x][2] * 2870
        print(predict[y][x][2])


# In[40]:


from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

track.append(Message('program_change', program=12, time=0))
x = 5
for i in range(60):
    track.append(Message('note_on', note= int(predict[x][i][0]), velocity=int(predict[x][i][1]), time=int(predict[x][i][2])))
    track.append(Message('note_off', note= int(predict[x][i][0]), velocity=int(predict[x][i][1]), time=int(predict[x][i][2])))
mid.save('xd.mid')


# In[ ]:


from keras.models import load_model

# Creates a HDF5 file 'my_model.h5'
model.save('my_model.h5')

