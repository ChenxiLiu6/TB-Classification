import pickle, torch
from sklearn.model_selection import StratifiedKFold, train_test_split

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

def load_variable(fp):
    with open(fp + 'X_29.pickle', 'rb') as file:
        X = pickle.load(file)
    with open(fp + 'y.pickle', 'rb') as file:
        y = pickle.load(file)

    return X, y

def load_variable_gaf():
    fp = "./data/"
    with open(fp + 'X_gaf_29.pickle', 'rb') as file:
        X_gaf = pickle.load(file)
    with open(fp + 'y.pickle', 'rb') as file:
        y = pickle.load(file)

    return X_gaf, y

def train_valid_split(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx = {}
    valid_idx = {}
    for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
        train_idx[i] = train_index
        valid_idx[i] = valid_index
    return train_idx, valid_idx

def dataset_split(X, y, rs=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
    return X_train, X_test, y_train, y_test


class CustomDataset(TensorDataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.data[idx]  # tensor (29, 147, 147)
        y = self.label[idx]  # 1.0 or 0.0
        return x, y

def plot_history(history, fold):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[0].plot(history['train_acc'])
    axes[0].plot(history['val_acc'])
    axes[0].set(xlabel='Epoch', ylabel='Accuracy')
    axes[0].set_title('MinCutMTPool accuracy for data fold ' + str(fold))
    axes[0].legend(['Training', 'Validation'], loc='lower right')

    axes[1].plot(history['train_loss'])
    axes[1].plot(history['val_loss'])
    axes[1].set(xlabel='Epoch', ylabel='Loss')
    axes[1].set_title('MinCutMTPool loss for data fold ' + str(fold))
    axes[1].legend(['Training', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.show()

def plot(history_list):
    for i in range(len(history_list)):
        history=history_list[i] # fold i history
        plot_history(history, i+1)

def plot_roc(auc, fpr, tpr, name):
    plt.plot(fpr, tpr, label='ROC curve for'+ name + '(AUC = %0.2f)' % auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()

def sens_spec(history_list):
    sens_list = []
    spec_list = []
    fpr_list = []
    tpr_list = []
    roc_auc_list = []
    for i in range(len(history_list)):
        history = history_list[i]
        pred_labels = history['best_val_pred']
        true_labels = history['best_val_actual']
        print("pred len: ", len(pred_labels))
        print("actual len: ", len(true_labels))
        TN, FP, FN, TP = confusion_matrix(true_labels, pred_labels).ravel()

        # (1) compute sensitivity
        sens = TP / (TP + FN)
        sens_list.append(sens)
        # (2) compute specificity
        spec = TN / (FP + TN)
        spec_list.append(spec)
        # (3) compute fpr: false positive rate, tpr: true positive rate
        fpr, tpr, _ = metrics.roc_curve(true_labels, pred_labels)
        roc_auc = metrics.auc(fpr, tpr)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        roc_auc_list.append(roc_auc)


    return sens_list, spec_list, tpr_list, fpr_list, roc_auc_list

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

def compute_acc(history_list):
    train_acc_list = []
    val_acc_list = []
    for i in range(len(history_list)):
        history = history_list[i]
        train_acc_list.append(np.mean(history['train_acc'][-10: ]))  # mean train acc for last 10 epochs
        val_acc_list.append(np.mean(history['val_acc'][-10: ]))      # mean val acc for last 10 epochs
    return train_acc_list, val_acc_list


def evaluate_acc_sens_spec(model, dl, test=False):
    model.eval()
    with torch.no_grad():
        y_preds = []
        y_pred_probs = []
        labels = []
        for i, (data, label) in enumerate(dl):
            y_pred_prob = model(data)
            y_pred_prob = y_pred_prob.reshape(y_pred_prob.shape[0])
            y_pred = (y_pred_prob > 0.5).int()

            y_preds.append(y_pred)
            y_pred_probs.append(y_pred_prob)
            labels.append(label)

        y_preds = torch.cat(y_preds)
        y_pred_probs = torch.cat(y_pred_probs)
        labels = torch.cat(labels)
        # Calculate confusion matrix
        cm = confusion_matrix(labels, y_preds)

        # Calculate sensitivity and specificity
        sens = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        spec = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        accuracy = accuracy_score(labels, y_preds)

        # calculate auc, roc if test == True
        if test == True:
            #auc = roc_auc_score(labels, y_pred_probs)
            #fpr, tpr, thresholds = roc_curve(labels, y_pred_probs)
            return accuracy, sens, spec, labels, y_pred_probs

    return accuracy, sens, spec
