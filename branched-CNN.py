# coding: utf-8

# # Branched Convolutional NN

# Import io functions
import setGPU0
from io_functions import *
from draw_functions import *

# ## Prepare the data

train_data, test_data, train_labels, test_labels = train_test(shape=(1, 20, 20, 25), split=0.33)

# ## Create and train the model

model1 = Sequential()
model1.add(Convolution3D(3, 4, 4, 5, input_shape = (1, 20, 20, 25), activation='relu'))
model1.add(MaxPooling3D())
model1.add(Flatten())

model2 = Sequential()
model2.add(Convolution3D(3, 3, 3, 4, input_shape = (1, 20, 20, 25), activation='relu'))
model2.add(MaxPooling3D())
model2.add(Flatten())

model3 = Sequential()
model3.add(Convolution3D(3, 5, 5, 6, input_shape = (1, 20, 20, 25), activation='relu'))
model3.add(MaxPooling3D())
model3.add(Flatten())

## join the two
bmodel = Sequential()
bmodel.add(Merge([model1,model2,model3], mode='concat'))

## fully connected ending
bmodel.add(Dense(1000, activation='relu'))
bmodel.add(Dropout(0.5))
bmodel.add(Dense(1, init='uniform', activation='sigmoid'))

bmodel.compile(loss='binary_crossentropy', optimizer='sgd')
bmodel.summary()

# Fit the model

fit_history = bmodel.fit([train_data, train_data, train_data], train_labels, nb_epoch=100, batch_size=1000, verbose=1)

# Get and save predictions
predictions = bmodel.predict([test_data, test_data, test_data])
store_model(bmodel, fit_history.history['loss'], 'bcnn', (predictions, test_labels))
