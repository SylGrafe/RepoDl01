#!/usr/bin/env python
# sygr0003 , UMU54907 , VT2019 , lab3_nlp0 , simple nlp (natural language processing)
# from 6.2-understanding-recurrent-neural-networks.py 

import keras
keras.__version__

# # Understanding recurrent neural networks
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
from keras.layers import Dense
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import LSTM

import matplotlib.pyplot as plt
import sys


max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
myEpochs=10

# TODO TODO   TEST ONLY TO BE REMOVE ---  START 

myEpochs=1
max_features = 10000  # number of words to consider as features
# TODO TODO   TEST ONLY TO BE REMOVE ---END


################################## funct
def plotXX (theTitle1='training curves'):

  plt.figure()
  ax = plt.subplot(2, 1, 1)
  ax.set_title(theTitle)
  acc = hist1.history['acc']
  val_acc = hist1.history['val_acc']
  loss = hist1.history['loss']
  val_loss = hist1.history['val_loss']

  epochs = range(len(acc))

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.legend()

  ax = plt.subplot(2, 1, 2)
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.legend()

  plt.show(block=False)
  plt.savefig(theTitle)


################################## main
print('load reviews with max  (num_words=%d)' % (max_features) )
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

print( 'nb of reviews    train=%d  , test=%d '  %   (len(input_train ) ,len(input_test) ))

someNb=-10
print('  input_train.dtype: %s , shapes : input_train :%s , input_test:%s :' % (
            (input_train.dtype ) , ( input_train.shape) ,( input_test.shape) ))

print (' len(input_train[0]=%d  input_train[0][%d:]   %s ' % 
       (len(input_train[0]) , someNb , input_train[0][someNb:]))


print('Pad sequences to %d words ' % (maxlen))
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

print(' after padding , shapes  input_train :%s , input_test:%s :' % (
           ( input_train.shape) ,( input_test.shape) ))
# print (' after padding input_train[0][%d:]   %s ' % (someNb , input_train[0][someNb:]))


# Let's train a simple recurrent network using an `Embedding` layer and a `SimpleRNN` layer:

print ("\ntrain model0 : Embedd (maxFeat , 32 ) + RNN(32)  ")

model0 = Sequential()
model0.add(Embedding(max_features, 32))
model0.add(SimpleRNN(32))
model0.add(Dense(1, activation='sigmoid'))

model0.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
hist0 = model0.fit(input_train, y_train,
                    epochs=myEpochs,
                    batch_size=128,
                    verbose=0 ,
                    validation_split=0.2)



# Let's display the training and validation loss and accuracy:
theTitle ="model0 Embedd SimpleRNN(32)"
plotXX(theTitle)

# As a reminder, in chapter 3, our very first naive approach to this very dataset got us to 88% test accuracy. Unfortunately, our small  # recurrent network doesn't perform very well at all compared to this baseline (only up to 85% validation accuracy). Part of the problem is 
# that our inputs only consider the first 500 words rather the full sequences -- 
# hence our RNN has access to less information than our earlier baseline model. The remainder of the problem is simply that `SimpleRNN` isn't very good at processing long sequences, like text. Other types of recurrent layers perform much better. Let's take a look at some 
# more advanced layers.

# ## A concrete LSTM example in Keras
# 
# Now let's switch to more practical concerns: we will set up a model using a LSTM layer and train it on the IMDB data. Here's the network, 
# similar to the one with `SimpleRNN` that we just presented. We only specify the output dimensionality of the LSTM layer, and leave every 
# other argument (there are lots) to the Keras defaults. Keras has good defaults, and things will almost always "just work" without you 
# having to spend time tuning parameters by hand.



model1 = Sequential()
model1.add(Embedding(max_features, 32))
model1.add(LSTM(32))
model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

print ("\nlab3_nlp0 train model1 : Embedd (maxFeat , 32 ) + LSTM(32)  ")
hist1 = model1.fit(input_train, y_train,
                    epochs=myEpochs,
                    batch_size=128,
                    validation_split=0.2)

theTitle ="model1 Embedd LSTM(32)"
plotXX(theTitle)

