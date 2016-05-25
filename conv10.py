
# coding: utf-8

# ## Here we train single-hidden-layer convnets with 10 nodes

# Import functions
import setGPU1
from io_functions import *
from draw_functions import *

# 1 is signal; 0 is background
train_data, test_data, train_labels, test_labels = train_test(shape=(1, 20, 20, 25), split=0.33)

# Choose our list of number of nodes
num_nodes = [10]

# Make a one-hidden-layer network with that number of nodes for each entry in num_nodes.
# Train it for 100 epochs, save the model, the weights, and loss history.
for number in num_nodes:
    
    # Convolutional Layers
    model = Sequential()
    model.add(Convolution3D(3, 4, 4, 5, input_shape = (1, 20, 20, 25), activation='relu'))
    model.add(MaxPooling3D())
    model.add(Flatten())

    # Dense layer
    model.add(Dense(number, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    
    # Train the network
    my_fit = model.fit(train_data, train_labels, nb_epoch=100, batch_size=1000, verbose=1)
    
    # Store the model, the weights, and the loss history
    store_model(model, my_fit.history['loss'], 'conv'+str(number))