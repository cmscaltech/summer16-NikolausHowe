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
    plt.title('Binary Classification Histogram')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()
    
def draw_roc_curve(test_data, test_labels, model):
    # Get classification predictions
    predictions = model.predict(test_data)

    # Draw the ROC curve
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    plt.xlim([.0, 1.01])
    plt.ylim([.0, 1.01])
    plt.title("ROC Curve")
    plt.plot( tpr, 1-fpr )
    
def draw_loss_history(my_fit):
    plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    plt.plot(my_fit.history['loss'])
    
def draw_val_loss_history(my_fit):
    plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    plt.plot(my_fit.history['val_loss'])
    
def draw_list(label_list, predictions, histories):
    # Set up the plot
    plt.figure(figsize=(10,10))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    plt.ylim(bottom=0)
    
    # Get the colors
    cmap = plt.get_cmap('jet_r')
    N = len(label_list)

    # Draw the histories
    for i, label in enumerate(label_list):
        color = cmap(float(i)/N) # get color for this plot
        plt.plot(histories[label]['loss'], label = label + ' loss', linewidth=2, c=color)
        plt.plot(histories[label]['val_loss'], label = label + ' val_loss', linewidth=2, linestyle='dashed', c=color)

    # Put in the legend (with thick lines!)
    leg = plt.legend(loc='upper right')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(8.0)
    plt.show()
    
    # Set up plot
    plt.figure(figsize=(10,10))
    plt.xlim([.7, 1.01])
    plt.ylim([.7, 1.01])
    plt.title("ROC Curve")
    # Draw the roc curves
    for label in label_list:
        pred = predictions[label][0].reshape(predictions[label][0].shape[0])
        truth = predictions[label][1].astype(int)
        fpr, tpr, _ = roc_curve(truth, pred)
        plt.plot( tpr, 1-fpr , label = label, linewidth=1.5)
        plt.xlabel('True Positive Rate')
        plt.ylabel('True Negative Rate')

    # Draw the legend (with thick lines!)
    leg = plt.legend(loc='lower left')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(8.0)
    plt.show()