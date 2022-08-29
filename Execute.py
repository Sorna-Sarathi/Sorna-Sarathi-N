#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)
plt.style.use("ggplot")
import tensorflow as tf
print('Tensorflow version:', tf.__version__)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, SimpleRNN, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.utils as ku 
import numpy as np 

df=pd.read_csv("news_summary_more.csv")
df.head(10)
df['text'][0]

df['headlines'][0]
headlines=[]

for i in df['headlines']:
    headlines.append(i)

#headlines
len(headlines)

#Tokenize the text
tokenizer=Tokenizer(num_words=10000)
tokenizer.fit_on_texts(headlines[:500])
total_words=len(tokenizer.word_index)+1
total_words

sequences=[]

#headlines[:500]

len(headlines)   





#Converting text into sequences
for l in headlines[:5000]:
     token = tokenizer.texts_to_sequences([l])[0]
#      print(token)
     for i in range(1,len(token)):
       ngrams_seq=token[:i+1]
       sequences.append(ngrams_seq)

len(sequences)

maxl=0
for i in sequences:
    k=len(i)
    if k>maxl:
        maxl=k
maxl

#Max padding will be the max length amoung the sequences
data= pad_sequences(sequences, maxlen=maxl)
data
data.shape

#Splitting predictors and labels
predictors=data[:,:-1]
predictors

predictors.shape


labels=data[:,-1]    #Selecting last column 
labels

labels.shape


labels=ku.to_categorical(labels,num_classes=total_words)
labels

labels.shape

#SimpleRNN
model = Sequential()
model.add(Embedding(input_dim=total_words,output_dim=80,input_length=15))#input 
length is 15 not 16 as we have taken the last column for labels for 16-1=15
model.add(Dropout(0.2))
model.add(SimpleRNN(units=150,return_sequences=False))#if return sequences is 
false,then it will return a 2-D array,if true then it will return a 3-D array..
model.add(Dense(total_words,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accurac
'])

history = model.fit(predictors, labels, epochs=100, verbose=1)
accuracy = history.history['accuracy']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training accuracy')

loss = history.history['loss']
epochs = range(len(loss))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')


history = model.fit(predictors, labels, epochs=50, verbose=1)
accuracy = history.history['accuracy']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training accuracy')

loss = history.history['loss']
epochs = range(len(loss))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')

#LSTM
model = Sequential()
model.add(Embedding(input_dim=total_words,output_dim=80,input_length=15))#input 
length is 15 not 16 as we have taken the last column for labels for 16-1=15
model.add(Dropout(0.2))
model.add(LSTM(units=150,return_sequences=False))#if return sequences is 
false,then it will return a 2-D array,if true then it will return a 3-D array..
model.add(Dense(total_words,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accurac
'])

history = model.fit(predictors, labels, epochs=100, verbose=1)
accuracy = history.history['accuracy']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training accuracy')

loss = history.history['loss']
epochs = range(len(loss))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')


history = model.fit(predictors, labels, epochs=50, verbose=1)
accuracy = history.history['accuracy']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training accuracy')

loss = history.history['loss']
epochs = range(len(loss))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')

#BiLSTM
model = Sequential()
model.add(Embedding(input_dim=total_words,output_dim=80,input_length=15))#input 
length is 15 not 16 as we have taken the last column for labels for 16-1=15
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=150,return_sequences=False)))#if return sequences is 
false,then it will return a 2-D array,if true then it will return a 3-D array..
model.add(Dense(total_words,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accurac
'])

history = model.fit(predictors, labels, epochs=100, verbose=1)
accuracy = history.history['accuracy']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training accuracy')

loss = history.history['loss']
epochs = range(len(loss))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')


history = model.fit(predictors, labels, epochs=50, verbose=1)
accuracy = history.history['accuracy']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training accuracy')

loss = history.history['loss']
epochs = range(len(loss))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')

#GRU
model = Sequential()
model.add(Embedding(input_dim=total_words,output_dim=80,input_length=15))#input 
length is 15 not 16 as we have taken the last column for labels for 16-1=15
model.add(Dropout(0.2))
model.add(GRU(units=150,return_sequences=False))#if return sequences is 
false,then it will return a 2-D array,if true then it will return a 3-D array..
model.add(Dense(total_words,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accurac
'])

history = model.fit(predictors, labels, epochs=100, verbose=1)
accuracy = history.history['accuracy']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training accuracy')

loss = history.history['loss']
epochs = range(len(loss))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')


history = model.fit(predictors, labels, epochs=50, verbose=1)
accuracy = history.history['accuracy']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training accuracy')

loss = history.history['loss']
epochs = range(len(loss))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')

#BiGRU
model = Sequential()
model.add(Embedding(input_dim=total_words,output_dim=80,input_length=15))#input 
length is 15 not 16 as we have taken the last column for labels for 16-1=15
model.add(Dropout(0.2))
model.add(Bidirectional(GRU(units=150,return_sequences=False)))#if return sequences is 
false,then it will return a 2-D array,if true then it will return a 3-D array..
model.add(Dense(total_words,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accurac
'])

history = model.fit(predictors, labels, epochs=100, verbose=1)
accuracy = history.history['accuracy']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training accuracy')

loss = history.history['loss']
epochs = range(len(loss))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')


history = model.fit(predictors, labels, epochs=50, verbose=1)
accuracy = history.history['accuracy']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training accuracy')

loss = history.history['loss']
epochs = range(len(loss))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')

