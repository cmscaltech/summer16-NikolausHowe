
# coding: utf-8

# ## Here we train single-hidden-layer 2D convnets with different numbers of nodes

# Import functions
import setGPU1
from io_functions import *
from draw_functions import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 1 is signal; 0 is background
train_data, test_data, train_labels, test_labels = train_test(shape=(20, 20, 25), split=0.33)

# Choose our list of number of nodes
num_nodes = [10, 100, 1000, 10000]

for num in num_nodes:
    # Convolutional Layers
    model = Sequential()
    model.add(Convolution2D(10, 4, 4, input_shape = (25, 20, 20), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())

    # Dense layer
    model.add(Dense(num, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd')

    # Checkpointing and Early Stopping
    check = ModelCheckpoint(filepath='./tmp/conv2D%s.hdf5'%num, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    # Train the network
    my_fit = model.fit(np.swapaxes(train_data, 1, 3), train_labels, nb_epoch=10000, validation_split=0.2,
                       batch_size=1000, verbose=1, callbacks=[check, early])

    # Get predictions
    predicted = model.predict(np.swapaxes(test_data, 1, 3))

    # Store the model, the weights, the loss history, and the predicted and truth labels
    store_model(model, my_fit.history, 'conv2D'+str(num), (predicted, test_labels))
