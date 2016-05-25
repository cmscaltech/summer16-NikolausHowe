# Nikolaus Howe, May 2016
# Some functions to streamline the NN notebooks

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def draw_histogram(test_data, test_labels, model):
    # Get the test signal and background to make the histogram
    test_signal = test_data[np.where(test_labels==1)]
    test_bkg    = test_data[np.where(test_labels==0)]

    # Calculate the probabilities for the test sets
    p_signal    = model.predict(test_signal)
    p_bkg       = model.predict(test_bkg)

    # Draw classification histogram
    plt.hist(p_signal, 50, normed=1, facecolor='blue', alpha=0.4, label='gamma')
    plt.hist(p_bkg , 50, normed=1, facecolor='red' , alpha=0.4, label='pi0')
    plt.xlabel('Prediction')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()
    
def draw_roc_curve(test_data, test_labels, model):
    # Get classification predictions
    predictions = model.predict(test_data)

    # Draw the ROC curve
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    plt.xlim([.0, 1.04])
    plt.ylim([.0, 1.04])
    plt.title("ROC Curve")
    plt.plot( tpr, 1-fpr )
    
def draw_loss_history(my_fit):
    plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    plt.plot(my_fit.history['loss'])
