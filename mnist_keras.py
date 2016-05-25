
# coding: utf-8

# In[1]:

import cPickle, gzip
import numpy as np

# Load the dataset
f = open('mnist.pkl', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


# In[2]:

train_data, train_target = train_set
test_data, test_target = test_set


# In[3]:

test_data = test_data[:100]
test_target = test_target[:100]


# from keras.models import Sequential
# model = Sequential()
# from keras.layers.core import Dense, Activation
# 
# model.add(Dense(output_dim=1000, input_dim=784))
# model.add(Activation("relu"))
# #model.add(Dense(output_dim=1000, input_dim=1000))
# #model.add(Activation("relu"))
# model.add(Dense(output_dim=10))
# model.add(Activation("relu"))
# model.add(Dense(output_dim=1))
# model.add(Activation("softmax"))

# In[ ]:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
model = Sequential()
model.add(Dense(1000, input_dim=784, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(1000, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))


# In[ ]:

model.compile(loss='mse', optimizer='sgd')


# transform the one dimensional input to the 10 dimension softmax expectation

# In[ ]:

train_target_10 = np.zeros((train_target.shape[0], 10))
test_target_10 = np.zeros((test_target.shape[0], 10))
for number in range(10):
    train_target_10[np.where(train_target==number),number] = 1
    test_target_10[np.where(test_target==number),number] = 1


# In[ ]:

n=46
print train_target[n],train_target_10[n]


# In[ ]:

model.fit(train_data, train_target_10, nb_epoch=10, batch_size=100)


# In[ ]:

p=model.predict(test_data)
p_cat = np.argmax(p,axis=1)
print "Fraction of good prediction"
print len(np.where( p_cat == test_target)[0])
print len(np.where( p_cat == test_target)[0])/float(len(p_cat)),"%"


# visualize the neural net (does not work, I have install a few things)

# In[1]:

from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))


# ## Convolutional Layers

# In[4]:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution1D
model = Sequential()
model.add(Convolution1D(784, 3, border_mode='same', input_dim = 784))
model.add(Dense(1000, input_dim=784, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))


# In[ ]:



