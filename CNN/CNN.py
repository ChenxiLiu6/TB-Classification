import argparse
import copy
import pickle
import sys
import utils
import torch

sys.path.append('../')
from tqdm import tqdm
from utils import *
from cnn_utils import *

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Model parameter settings
parser = argparse.ArgumentParser()


# Training parameter settings
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')  # 1000
parser.add_argument('--bs', type=int, default=32, help='training batch size')
parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate. default:[0.00001]')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters). default: 5e-3')
parser.add_argument('--stop_thres', type=float, default=1e-9, help='The stop threshold for the training error. If the difference between training losses ' 
                                                                   'between epoches are less than the threshold, the training will be stopped. Default:1e-9')
# cuda settings
args = parser.parse_args()

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

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Conv layers
        self.conv_1 = nn.Conv1d(in_channels=29, out_channels=16, kernel_size=20, stride=1)
        self.conv1_bn1 = nn.BatchNorm1d(16)
        self.conv_2 = nn.Conv1d(in_channels=16, out_channels=4, kernel_size=2, stride=2)
        self.conv2_bn2 = nn.BatchNorm1d(4)
        # 2. Pooling layer
        self.pool_1 = nn.MaxPool1d(2)
        self.pool_2 = nn.MaxPool1d(2)
        # 3. Linear layers
        self.layer_1 = nn.Linear(in_features=64, out_features=16)

        self.fc1_bn1 = nn.BatchNorm1d(16)
        self.layer_2 = nn.Linear(in_features=16, out_features=1)

        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)      # [742, 16, 128]
        x = self.conv1_bn1(x)
        x = F.relu(x)
        x = self.pool_1(x)      # [742, 16, 64]

        x = self.conv_2(x)      # [742, 4, 32]
        x = self.conv2_bn2(x)
        x = F.relu(x)
        x = self.pool_2(x)      # [742, 4, 16]

        x = self.flatten(x)     # [742, 64]
        x = self.layer_1(x)     # [742, 16]
        x = self.fc1_bn1(x)
        x = F.relu(x)
        x = self.layer_2(x)     # [742, 1]
        x = self.sigmoid(x)
        return x


def weight_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)
        #m.bias.data.fill_(0.01)


def train_cnn(X_tv, y_tv, saved_name, train_idx, valid_idx):
    history_list = []
    models_dict = {} # save trained models
    for k in tqdm(range(5)):
        train_ds = CustomDataset(data=X_tv[train_idx[k]], label=y_tv[train_idx[k]])
        train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)

        val_ds = CustomDataset(data=X_tv[valid_idx[k]], label=y_tv[valid_idx[k]])
        val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=True)

        model = CNNModel()
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=5e-4,
                               betas=[0.9, 0.999],
                               eps=1e-7)
        # Define the early stopping criteria
        patience = 50  # number of epochs with no improvement
        min_delta = 1e-4  # minimum change in validation loss to be considered as improvement
        best_loss = float('inf')
        counter = 0

        # empty accuracy and loss list
        train_acc_history = []
        train_loss_history = []
        val_acc_history = []
        val_loss_history = []
        for epoch in range(args.epochs):
            model.train()
            train_loss_track = 0.0
            epoch_accuracy = []
            for i, (data, label) in enumerate(train_dl):
                optimizer.zero_grad()
                y_pred_probs = model(data)
                y_pred_probs = y_pred_probs.reshape(y_pred_probs.shape[0])
                y_pred = (y_pred_probs > 0.5).int()
                loss = F.binary_cross_entropy(y_pred_probs.float(), label.float())
                #loss = loss_fn(y_pred, label.float())
                # update
                loss.backward()
                optimizer.step()
                # save corrects and loss
                epoch_accuracy.append(accuracy_score(label, y_pred))
                train_loss_track += loss.item()

            train_acc_history.append(np.mean(epoch_accuracy))
            train_loss_history.append(train_loss_track / len(train_idx[k]))

            model.eval()
            with torch.no_grad():
                val_loss_track = 0.0
                val_corrects = []
                for i, (data, label) in enumerate(val_dl):
                    y_pred_probs = model(data)
                    y_pred_probs = y_pred_probs.reshape(y_pred_probs.shape[0])
                    y_pred = (y_pred_probs > 0.5).int()

                    loss = F.binary_cross_entropy(y_pred_probs.float(), label.float(), reduction='mean')

                    # save loss and corrects
                    val_loss_track += loss.item()
                    val_corrects.append(accuracy_score(label, y_pred))

                val_acc_history.append(np.mean(val_corrects))
                val_loss_history.append(val_loss_track / len(valid_idx[k]))

                # check if the validation loss has improved
                if val_loss_history[-1] < best_loss - min_delta:
                    best_loss = val_loss_history[-1]
                    counter = 0
                else:
                    counter += 1

                # check if early stopping criteria is met
                if counter >= patience:
                    print("Early stopping at epoch: ", epoch)
                    break

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(train_loss_history[-1]),
                  'acc_train: {:.4f}'.format(train_acc_history[-1]),
                  'loss_val: {:.4f}'.format(val_loss_history[-1]),
                  'acc_val: {:.4f}'.format(val_acc_history[-1]))

        history = {
            'train_acc': train_acc_history,
            'train_loss': train_loss_history,
            'val_acc': val_acc_history,
            'val_loss': val_loss_history
        }
        # save history
        history_list.append(history)
        # save model
        models_dict[k] = model.state_dict()

    # save history list
    write_cnn_history(history_list, saved_name)
    # save 5 models
    torch.save(models_dict, './CNN_saved/CNN_Models/cnn_models.pt')

"""
def evaluate_acc_sens_spec(model, dl, test=False):
    model.eval()
    with torch.no_grad():
        y_preds=[]
        y_pred_probs=[]
        labels=[]
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
            return accuracy, sens, spec, auc, fpr, tpr

    return accuracy, sens, spec
"""

def evaluate_test(num, X_test, y_test, X_tv, y_tv, train_idx, valid_idx):
    loaded_models_dict = load_models()
    train_dict={'accuracy':[],'sensitivity':[], 'specificity':[]}
    val_dict={'accuracy':[],'sensitivity':[], 'specificity':[]}
    for k in loaded_models_dict:
        model = CNNModel()
        model.load_state_dict(loaded_models_dict[k])

        train_ds = CustomDataset(data=X_tv[train_idx[k]], label=y_tv[train_idx[k]])
        train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)

        val_ds = CustomDataset(data=X_tv[valid_idx[k]], label=y_tv[valid_idx[k]])
        val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=True)

        # train results
        train_acc, train_sens, train_spec = evaluate_acc_sens_spec(model, train_dl)
        train_dict['accuracy'].append(train_acc)
        train_dict['sensitivity'].append(train_sens)
        train_dict['specificity'].append(train_spec)

        # valid results
        val_acc, val_sens, val_spec = evaluate_acc_sens_spec(model, val_dl)
        val_dict['accuracy'].append(val_acc)
        val_dict['sensitivity'].append(val_sens)
        val_dict['specificity'].append(val_spec)

    print("Train Accuracy: {:.3f}  Sensitivity: {:.3f}  Specificity: {:.3f}".format(np.mean(train_dict['accuracy']),
                                                                                    np.mean(train_dict['sensitivity']),
                                                                                    np.mean(train_dict['specificity'])))
    print("Valid Accuracy: {:.3f}  Sensitivity: {:.3f}  Specificity: {:.3f}".format(np.mean(val_dict['accuracy']),
                                                                                    np.mean(val_dict['sensitivity']),
                                                                                    np.mean(val_dict['specificity'])))
    # Find the index of the best model
    best_index = np.argmax(val_dict['accuracy'])
    print("valid performance: ", val_dict['accuracy'])
    print("best index: ", best_index)

    # Load the best model
    best_model = CNNModel()
    best_model.load_state_dict(loaded_models_dict[best_index])

    # Evaluate the best model on the test set
    test_ds = CustomDataset(data=X_test, label=y_test)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True)
    test_acc, test_sens, test_spec, labels, y_pred_probs = evaluate_acc_sens_spec(best_model, test_dl, test=True)

    print("Test Accuracy: {:.3f}  Sensitivity: {:.3f}  Specificity: {:.3f}".format(test_acc, test_sens, test_spec))
    #np.savetxt("./CNN_saved/CNN_ROC/CNN"+str(num)+".txt", np.column_stack((labels, y_pred_probs)))
    #plot_roc(auc, fpr, tpr)
    return best_index


def main():
    # ---------------------------- 1. Data Split --------------------------------
    # (1) load X, y
    X, y = utils.load_variable(fp='../../data/')
    X = np.transpose(X, (0, 2, 1)) # shape: (928, 29, 147)

    # (2) Split data into training (80%, 742) and testing (20%, 186) sets
    X_tv, X_test, y_tv, y_test = utils.dataset_split(X, y)
    X_tv = torch.tensor(X_tv, dtype=torch.float)
    y_tv = torch.tensor(y_tv, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test =torch.tensor(y_test, dtype=torch.float)

    # (3) Split train data into train, valid dataset with 5-fold cross validation
    train_idx, valid_idx = utils.train_valid_split(X_tv, y_tv)  # 5 folds cross validation
    saved_name = "cnn_lr_" + str(args.lr) + "_epochs_" + str(args.epochs)

    # ---------------------------- 2. Train CNN Model --------------------------------
    num=0
    # train_cnn(X_tv, y_tv, saved_name, train_idx, valid_idx)

    # ---------------------------- 3. Test CNN Model --------------------------------
    # (1) test
    best_index = evaluate_test(num, X_test, y_test, X_tv, y_tv, train_idx, valid_idx)
    # (2) load history and plot
    history_list = load_history(saved_name)
    #plot(history_list)
    save_best_history(history_list[best_index], best_index, "CNN")


if __name__ == "__main__":
    main()