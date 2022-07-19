# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:52:40 2022

@author: Lai Kar Wei
"""
#%%Loading module
import os
import re
import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%%
LOGS_PATH = os.path.join(os.getcwd(), 'abc_logs', datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
                         
#%% Data Loading

df = pd.read_csv("https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv")

#%% Step 2 Data Inspection

df.info()
df.describe().T
df.head()

df.duplicated().sum()
df.isna().sum()

print(df['review'][4])
print(df['review'][10])

#Symbol and HTML Tags have to be removed

#%% Step 3 Data Cleaning

review = df['review']
sentiment = df['sentiment']

review_backup = review.copy()
sentiment_backup = sentiment.copy()

for index, text in enumerate(review):
    #to remove html tags
    #anything within the <> will be removed including <>
    #? to tell re dont be greedy so it wont capture everything
    #from the first <to the last> in the document
    
    review[index] = re.sub('<.*?>', '', text)
    review[index] = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    
review_backup = review.copy()
sentiment_backup = sentiment.copy()

#%% Step 4 Features Selection
#%% Step 5 Data Preprocessing

#X Features
vocab_size = 10000
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

tokenizer.fit_on_texts(review) #to learn
word_index = tokenizer.word_index

print(dict(list(word_index.items())[0:10]))

review_int = tokenizer.texts_to_sequences(review) #to convert into numbers

#length review =[]
#for i in range (len(review_int)):
#    length_review.append(len(review_int[i]))
#    print(len(review_int[i]))
#np.median(length_review)

max_len = np.median([len(review_int[i]) for i in range(len(review_int))])

padded_review = pad_sequences(review_int,
                              maxlen = int(max_len),
                              padding = 'post',
                              truncating = 'post')

#Y target
ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment, axis=-1))

X_train, X_test, y_train, y_test = train_test_split(padded_review, sentiment,
                                                    test_size=0.3,
                                                    random_state=123)

#%%Model development

# X_train = np.expand_dims(X_train, axis=-1)
# X_test = np.expand_dims(X_test, axis=-1)

input_shape = np.shape(X_train)[1:]
out_dim = 64

model = Sequential()
model.add(Input(shape=(input_shape)))
model.add(Embedding(vocab_size, out_dim))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
model.summary()

plot_model(model, show_shapes=(True))

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['acc'])

#%%
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)

model.fit(X_train, y_train,
       validation_data=(X_test, y_test),
       epochs=5, 
       callbacks=[tensorboard_callback])

#%%Model Analysis

from sklearn.metrics import classification_report

y_pred = np.argmax(model.predict(X_test), axis=1)
y_actual = np.argmax(y_test, axis=1)

print(classification_report(y_actual, y_pred))

#%%

#Tokenizer
import json
TOKENIZER_SAVE_PATH = os.path.join(os.getcwd(), 'model', 'tokenizer.json')

token_json = tokenizer.to_json()

with open(TOKENIZER_SAVE_PATH, 'w') as file:
    json.dump(token_json, file)

#OHE
import pickle
OHE_SAVE_PATH = os.path.join(os.getcwd(), 'model', 'ohe.pkl')

with open(OHE_SAVE_PATH, 'wb') as file:
    pickle.dump(ohe, file)

#Model
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model', 'model.h5')
model.save(MODEL_SAVE_PATH)


