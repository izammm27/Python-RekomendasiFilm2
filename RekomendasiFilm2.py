#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical


# In[2]:


#masukan DataTrain.CSV Sebagai Data Train
sf_train = pd.read_csv(r"C:\Users\Kodeic\Documents\Python\the-movies-dataset\DataTrain.csv")
sf_train.head()


# In[3]:


#Buat kolom baru (untuk paramater) berdasarkan kolom "vote_average" jika lebih dari 5 maka akan bernilai 1 dan sebaliknya 0
sf_train.loc[sf_train.vote_average > 5 , "Rekomendasi"] = [1]
sf_train.loc[sf_train.vote_average < 5 , "Rekomendasi"] = [0]


# In[4]:


sf_train.dtypes


# In[5]:


#Ubah type data kolom "Rekomendasi" menjadi int64
sf_train = sf_train.fillna(0)
sf_train["Rekomendasi"] = sf_train["Rekomendasi"].astype('int64')


# In[6]:


sf_train.dtypes


# In[7]:


#untuk mensortir hanya kolom title,vote_average,voute_count,popularity dan Rekomendasi saja yang diambil
#untuk dijadikan sebagai input & output Data Train
sf_train = sf_train.loc[:,["title","vote_average","vote_count","popularity","Rekomendasi"]]


# In[8]:


sf_train.head()


# In[9]:


#masukan DataTest.CSV Sebagai Data Validasi yang telah di ambil secara random dari DataTest.CSV
sf_val = pd.read_csv(r"C:\Users\Kodeic\Documents\Python\the-movies-dataset\DataTest.csv")
sf_val.head()


# In[10]:


#Buat kolom baru (untuk paramater) berdasarkan kolom "vote_average" jika lebih dari 5 maka akan bernilai 1 dan sebaliknya 0
sf_val.loc[sf_val.vote_average > 5 , "Rekomendasi"] = [1]
sf_val.loc[sf_val.vote_average < 5 , "Rekomendasi"] = [0]


# In[11]:


#Ubah type data kolom "Rekomendasi" menjadi int64
sf_val = sf_val.fillna(0)
sf_val["Rekomendasi"] = sf_val["Rekomendasi"].astype('int64')


# In[12]:


#untuk mensortir hanya kolom title,vote_average,voute_count,popularity dan Rekomendasi saja yang diambil
#untuk dijadikan sebagai input & output Data Validasi
sf_val = sf_val.loc[:,["title","vote_average","vote_count","popularity","Rekomendasi"]]


# In[13]:


sf_val.dtypes


# In[14]:


# Dapatkan nilai array Pandas (Konversi ke array NumPy)
train_data = sf_train.values
val_data = sf_val.values


# In[15]:


pd.DataFrame(train_data)
pd.DataFrame(val_data)


# In[16]:


#Gunakan kolom 2 & 3 (vote_count & popularity) sebagai Input
train_x = train_data[:,2:4]
val_x = val_data[:,2:4]
print (train_x)


# In[17]:


#Gunakan kolom 4 (Rekomendasi) sebagai Output
train_y = to_categorical( train_data[:,4] )
val_y = to_categorical( val_data[:,4] )
pd.DataFrame(val_y)


# In[18]:


# Create Network dengan input node 2 berasal dari kolom (vote_count & popularity) dengan Hidden Layers 10
inputs = Input(shape=(2,))
h_layer = Dense(10, activation='sigmoid')(inputs)


# In[19]:


# Softmax Activation for Multiclass Classification dengan output node 2 dari value Kolom Rekomendasi yaitu 1 dan 0
outputs = Dense(2, activation='softmax')(h_layer)
model = Model(inputs=inputs, outputs=outputs)


# In[20]:


# Optimizer / Update Rule
sgd = SGD(lr=0.001)


# In[21]:


# Compile the model with Cross Entropy Loss
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


# In[22]:


# Train the model dan gunakan validasi data.
model.fit(train_x, train_y, batch_size=16, epochs=100, verbose=1, validation_data=(val_x, val_y))
model.save_weights('weights.h5')


# In[23]:


# Predict semua validasi data yang telah di Train
predict = model.predict(val_x)


# In[24]:


# Visualize Prediction
df = pd.DataFrame(predict)
df.columns = [ 'RATE_DIBAWAH_LIMA', 'RATE_DIATAS_LIMA' ]
df.index = val_data[:,0]
df.head()


# In[25]:


df.columns


# In[26]:


df.loc[df.RATE_DIBAWAH_LIMA > df.RATE_DIATAS_LIMA , 'Rekomendasi'] = "Tidak rekomendasi"
df.loc[df.RATE_DIBAWAH_LIMA < df.RATE_DIATAS_LIMA , 'Rekomendasi'] = "Kuy Nonton"
df.head(10)


# In[27]:


#untuk melihat hasil dalam bentuk grafik
df.plot(kind='bar',colormap='cool')


# In[ ]:




