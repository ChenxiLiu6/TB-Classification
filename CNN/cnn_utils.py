import sklearn
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

def accuracy(output, labels):
    preds = (output > 0.5).int()
    preds = preds.cpu().numpy()
    #print("preds: ", preds)
    labels = labels.int().cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds, normalize=False))

    return accuracy_score

def load_models():
    models = torch.load('./CNN_saved/CNN_Models/cnn_models.pt')
    return models

def write_cnn_history(history_list, saved_name):
    history_path = "./CNN_saved/CNN_Histories/"
    with open(history_path+ saved_name+".pickle", 'wb') as fp:
        pickle.dump(history_list, fp)

def load_history(saved_name):
    fp = "./CNN_saved/CNN_Histories/"
    with open(fp + saved_name + '.pickle', 'rb') as file:
        history_list = pickle.load(file)
    return history_list

def save_best_history(history, best_index, name):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[0].plot(history['train_acc'])
    axes[0].plot(history['val_acc'])
    axes[0].set(xlabel='Epoch', ylabel='Accuracy')
    axes[0].set_title(name + ' accuracy for data fold ' + str(best_index+1))
    axes[0].legend(['Training', 'Validation'], loc='lower right')

    axes[1].plot(history['train_loss'])
    axes[1].plot(history['val_loss'])
    axes[1].set(xlabel='Epoch', ylabel='Loss')
    axes[1].set_title(name + ' loss for data fold ' + str(best_index+1))
    axes[1].legend(['Training', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('./CNN_saved/CNN_Histories/best_'+name, dpi=300)
    plt.show()

def plot_roc(auc, fpr, tpr):
    plt.plot(fpr, tpr, label='ROC curve for CNN (AUC = %0.2f)' % auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()

def plot_history(history, fold):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[0].plot(history['train_acc'])
    axes[0].plot(history['val_acc'])
    axes[0].set(xlabel='Epoch', ylabel='Accuracy')
    axes[0].set_title('CNN accuracy for data fold ' + str(fold))
    axes[0].legend(['Training', 'Validation'], loc='lower right')

    axes[1].plot(history['train_loss'])
    axes[1].plot(history['val_loss'])
    axes[1].set(xlabel='Epoch', ylabel='Loss')
    axes[1].set_title('CNN loss for data fold ' + str(fold))
    axes[1].legend(['Training', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.show()

def plot(history_list):
    for i in range(len(history_list)):
        history=history_list[i] # fold i history
        plot_history(history, i+1)
