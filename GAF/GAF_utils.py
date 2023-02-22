import pickle, torch
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_models(num, attention=False):
    if attention:
        models = torch.load('./GAF_Saved/GAF_Models/gaf_attention_models'+str(num)+'.pt')
    else:
        models = torch.load('./GAF_Saved/GAF_Models/gaf_models'+str(num)+'.pt')
    return models

def load_histories(saved_name, num):
    fp = "./GAF_Saved/GAF_Histories/gaf/gaf_"+str(num)+"_"
    with open(fp + saved_name + '.pickle', 'rb') as file:
        history_list = pickle.load(file)
    return history_list

def load_atten_history(saved_name, num):
    fp = "./GAF_Saved/GAF_Histories/gaf/gaf_attention_" + str(num) + "_"
    with open(fp + saved_name + '.pickle', 'rb') as file:
        history_list = pickle.load(file)
    return history_list

def gaf_dataset_split(X, y,  gaf, rs=1998):
    X_train, X_test, y_train, y_test, gaf_train, gaf_test= train_test_split(X, y, gaf, test_size=0.2, random_state=rs)
    return X_train, X_test, y_train, y_test, gaf_train, gaf_test


def save_matrix(matrix, name):
    fp = "./GAF_Saved/GAF_Metrics/"+name
    with open(fp, 'wb') as file:
        pickle.dump(matrix, file)

def load_matrix(name):
    fp = "./GAF_Saved/GAF_Metrics/"+name
    with open(fp, 'rb') as file:
        Matrix = pickle.load(file)
    return Matrix


def plot_roc_auc(tpr_list, fpr_list, roc_auc_list):
    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr_list[0], tpr_list[0], linestyle='--', color='orange', label='AUC_1 = %0.3f' % roc_auc_list[0])
    plt.plot(fpr_list[1], tpr_list[1], linestyle='--', color='lightgreen', label='AUC_2 = %0.3f'% roc_auc_list[1])
    plt.plot(fpr_list[2], tpr_list[2], linestyle='--', color='blue', label='AUC_3 = %0.3f'% roc_auc_list[2])
    plt.plot(fpr_list[3], tpr_list[3], linestyle='--', color='green', label='AUC_3 = %0.3f'% roc_auc_list[3])
    plt.plot(fpr_list[4], tpr_list[4], linestyle='--', color='pink', label='AUC_3 = %0.3f'% roc_auc_list[3])
    plt.plot(np.mean(fpr_list, axis=0), np.mean(tpr_list, axis=0), linestyle='--', color='red', label='mean AUC = %0.3f'% np.mean(roc_auc_list))
    plt.plot([0, 1], ls="--")

    # title
    plt.title('ROC curve for GAF with attention model')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC', dpi=300)
    plt.show()

def plot_GAF(GAF, sample=1):
    mpl.rc('font', size=18)
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))
    for i in range(6):
        x = int(i / 2)
        y = i % 2
        gaf = GAF[sample, i, :, :]
        axs[x][y].imshow(gaf)
        axs[x][y].set_title("GAF " + str(i + 1), fontsize=18)
    plt.tight_layout()
    plt.savefig('6GAF', dpi=300)
    plt.show()

def plot_all_sensor(X):
    mpl.rc('font', size=18)
    fig, axs = plt.subplots(3, 2, figsize=(12,18))
    for i in range(6):
        signal = X[i, :]
        x = int(i / 2)
        y = i % 2
        axs[x][y].plot(np.arange(147), signal)
        axs[x][y].set_title("Sensor " + str(i + 1), fontsize=18)
    plt.tight_layout()
    plt.savefig('6sensor', dpi=300)
    plt.show()




def write_gaf_history(history_list, saved_name, num):
    with open("./GAF_Saved/GAF_Histories/gaf/gaf_attention_" +str(num)+"_"+ saved_name + ".pickle", 'wb') as fp:
        pickle.dump(history_list, fp)

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
    plt.savefig('./GAF_Saved/GAF_Histories/best_'+name, dpi=300)
    plt.show()

def plot_history(history, fold, name):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[0].plot(history['train_acc'])
    axes[0].plot(history['val_acc'])
    axes[0].set(xlabel='Epoch', ylabel='Accuracy')
    axes[0].set_title(name + ' accuracy for data fold ' + str(fold))
    axes[0].legend(['Training', 'Validation'], loc='lower right')

    axes[1].plot(history['train_loss'])
    axes[1].plot(history['val_loss'])
    axes[1].set(xlabel='Epoch', ylabel='Loss')
    axes[1].set_title(name + ' loss for data fold ' + str(fold))
    axes[1].legend(['Training', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_gaf(history_list, name):
    for i in range(len(history_list)):
        history=history_list[i] # fold i history
        plot_history(history, i+1, name)

def load_history(saved_name):
    fp = "./saved_variables/gaf_mtf/"
    with open(fp + saved_name + '.pickle', 'rb') as file:
        history_list = pickle.load(file)
    return history_list
