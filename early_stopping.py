# coding: utf-8

# ## Here we train single-hidden-layer fully-connected nets with 10 nodes, checkpoints, and early stopping.

# Import functions
import setGPU1
from io_functions import *
from draw_functions import *

# 1 is signal; 0 is background
train_data, test_data, train_labels, test_labels = train_test(shape=(10000,), split=0.33)

from keras.callbacks import ModelCheckpoint, EarlyStopping
# Choose our list of number of nodes
num_nodes = [10]

# Make a one-hidden-layer network with that number of nodes for each entry in num_nodes.
# Train it for 100 epochs, save the model, the weights, and loss history.
for number in num_nodes:
    
    # Construct and compile a network
    model = Sequential()
    model.add(Dense(number, input_dim=10000, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(number, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    check = ModelCheckpoint(filepath="./tmp/dense%sx10000weights.hdf5"%number, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    
    # Train the network
    my_fit = model.fit(train_data, train_labels, validation_split=0.2, 
                       nb_epoch=10000, batch_size=1000, verbose=1, callbacks=[check, early])
    
    predictions = model.predict(test_data)
    
    # Store the model, the weights, and the loss history
    store_model(model, my_fit.history, 'dense'+str(number)+'x10000', predictions)