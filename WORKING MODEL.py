#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from sklearn.model_selection import train_test_split


# In[30]:


#random int between 1 and 8. 4 ints in each array. 1000 arrays.
data = [[random.randint(1,8) for j in range(4)] for i in range(1000)]
data = np.array(data, dtype=float)
#random int between 1 and 8. 16 ints in each array. 1000 arrays.
target = [[random.randint(1,8) for j in range(16)] for i in range(1000)]
target = np.array(target, dtype=float)


# In[31]:


#reshaping data into correct sized tensors. Then normalizing values. A,B,C,D,E,F,G,<EOS> so 8 characters in vocabulary
data = data.reshape((1000, 1, 4))/8 
target = target.reshape((1000, 1, 16))/8 


# In[32]:


data*8


# In[33]:


target*8


# In[34]:


#splitting data into train and test sets. 3/4 train, 1/4 test.
x_train,x_test,y_train,y_test = train_test_split(data, target, test_size=0.25, shuffle=False, random_state=42)


# In[35]:


#getting the right tensor shape for this was a BITCH
model = Sequential()  
model.add(LSTM(128, input_shape=(1, 4),return_sequences=True, activation = 'relu'))
model.add(LSTM(128, return_sequences=True, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128))
model.add(LSTM(128, return_sequences=True, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128))
model.add(LSTM(128, return_sequences=True, activation = 'relu'))
model.add(Dense(units=16))
model.add(Activation('tanh'))



model.compile(loss='mean_absolute_error', optimizer='adam')
print(model.summary())


# In[36]:


model.fit(data, target, nb_epoch=100, batch_size=200, verbose=2,validation_data=(x_test, y_test))


# In[40]:


guess = np.array([1, 2, 3, 4])
guess = guess.reshape(1,1,4)


# In[42]:


model.predict_on_batch(guess)*8

