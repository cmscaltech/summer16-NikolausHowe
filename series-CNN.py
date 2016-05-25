# coding: utf-8

# # Series Convolutional NN

# Import io functions
import setGPU1
from io_functions import *
from draw_functions import *

# ## Prepare the data

train_data, test_data, train_labels, test_labels = train_test(shape=(1, 20, 20, 25), split=0.33)

# ## Create and train the model
model = Sequential()

# Convolutional Layers
model.add(Convolution3D(3, 5, 5, 6, input_shape = (1, 20, 20, 25), activation='relu'))
model.add(Activation('relu'))
model.add(Convolution3D(3, 4, 4, 5, input_shape = (1, 20, 20, 25), activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling3D())
model.add(Dropout(0.25))
model.add(Flatten())

## Fully connected ending
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd')
model.summary()

# Fit the model
fit_history = model.fit(train_data, train_labels, nb_epoch=100, batch_size=1000, verbose=1)

# Save the model
predictions = model.predict(test_data)
store_model(model, fit_history.history['loss'], 'scnn', (predictions, test_labels))
