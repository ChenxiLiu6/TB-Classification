import copy
import torch
import utils
import argparse
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import sklearn.metrics as metrics
from GAF_utils import *
from utils import *
from sklearn.metrics import confusion_matrix
from pyts.image import GramianAngularField, MarkovTransitionField
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define training parameters
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-6, help='Initial learning rate. default:[0.00001]')
parser.add_argument('--bs', type=int, default=32, help='training batch size')
parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
args = parser.parse_args()



def GAF_generator(X):
    """
    :param X: shape(928, 29, 147, 147)
    """
    GAF = GramianAngularField()
    X_GAF = np.empty((X.shape[0], X.shape[1], X.shape[2], X.shape[2]))
    for i in range(X.shape[0]):
        X_gaf = GAF.fit_transform(X[i])  # shape: (29, 147, 147)
        X_GAF[i] = X_gaf
    return X_GAF

def MTF_generator(X):
    """
    :param X: shape(928, 29, 147, 147)
    """
    MTF = MarkovTransitionField()
    X_MTF = np.empty((X.shape[0], X.shape[1], X.shape[2], X.shape[2]))
    for i in range(X.shape[0]):
        X_mtf = MTF.transform(X[i])  # shape: (29, 147, 147)
        X_MTF[i] = X_mtf
    return X_MTF


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=29,out_channels=12, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(in_channels=12,out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Flatten(),
            nn.Linear(6534, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


class GAFAttnNet(nn.Module):
    def __init__(self, mean=False):
        super(GAFAttnNet, self).__init__()
        self.base_model = nn.Sequential(
            nn.Conv2d(in_channels=29, out_channels=12, kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=12, out_channels=6, kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten())

        self.mean = mean
        if not mean:
            self.attention = nn.Sequential(
                nn.Linear(6534, 256),
                nn.Tanh(),
                nn.Linear(256, 1))

        self.classifier = nn.Sequential(nn.Linear(6534, 1))
        self.sigmoid =nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)  # (29, 6, 33, 33) torch.Size([32, 6534])
        #print("x shape: ", x.shape)

        if self.mean:
            x = torch.mean(x, axis=0, keepdim=True)
            return self.classifier(x), 0
        else:
            A_unnorm = self.attention(x)
            #A = F.softmax(A_unnorm, dim=1)
            bs = A_unnorm.size(0)
            #print("bs: ", bs)
            A_new = A_unnorm.expand(bs, bs)
            #print("new A shape: ", A_new.shape)
            #print("x shape: ", x.shape)
            M = torch.matmul(A_new, x)
            #print("M size: ", M.shape)
            x = self.classifier(M)
            x = self.sigmoid(x)
            return x

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

def get_GAF(X):
    X_GAF = GAF_generator(X)
    # define tensor dataset
    gaf = torch.Tensor(X_GAF)  # transform to torch tensor

    return gaf


def train(gaf_tv, y_tv,  saved_name, train_idx, valid_idx, num):
    history_list = []
    model_dict = {}  # save trained models
    # 5-folds cross-validation
    for k in tqdm(range(5)):
        # define dataloaders
        train_ds = CustomDataset(data=gaf_tv[train_idx[k]], label=y_tv[train_idx[k]].type(torch.LongTensor))
        train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)

        val_ds = CustomDataset(data=gaf_tv[valid_idx[k]], label=y_tv[valid_idx[k]].type(torch.LongTensor))
        val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=True)

        # Net
        model = CNN().to(device)
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  # (lr= 0.0023)
        # Define the early stopping criteria
        patience = args.patience  # number of epochs with no improvement
        min_delta = 1e-4  # minimum change in validation loss to be considered as improvement
        best_loss = float('inf')
        counter = 0

        #criterion = nn.CrossEntropyLoss()

        # empty accuracy and loss list
        train_acc_history = []
        train_loss_history = []
        val_acc_history = []
        val_loss_history = []

        for epoch in range(args.epochs):  # 100
            model.train()
            train_loss_track = []
            epoch_accuracy = []
            for i, (img, label) in enumerate(train_dl):
                optimizer.zero_grad()
                y_pred_probs = model(img)
                y_pred_probs = y_pred_probs.reshape(y_pred_probs.shape[0])  # mabe don't need
                y_pred = (y_pred_probs > 0.5).int()

                loss = F.binary_cross_entropy(y_pred_probs.float(), label.float())

                # update
                loss.backward()
                optimizer.step()
                # save corrects and loss
                epoch_accuracy.append(accuracy_score(label, y_pred))
                train_loss_track.append(loss.item())

            train_acc_history.append(np.mean(epoch_accuracy))
            train_loss_history.append(np.mean(train_loss_track))

            model.eval()
            with torch.no_grad():
                val_loss_track = []
                val_corrects = []
                for i, (img, label) in enumerate(val_dl):
                    y_pred_probs = model(img)
                    y_pred_probs = y_pred_probs.reshape(y_pred_probs.shape[0])
                    y_pred = (y_pred_probs > 0.5).int()

                    loss = F.binary_cross_entropy(y_pred_probs.float(), label.float(), reduction='mean')

                    # save loss and corrects
                    val_loss_track.append(loss.item())
                    val_corrects.append(accuracy_score(label, y_pred))

                val_acc_history.append(np.mean(val_corrects))
                val_loss_history.append(np.mean(val_loss_track))

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
        model_dict[k] = model.state_dict()

    # save 5 models
    write_gaf_history(history_list, saved_name, num)
    torch.save(model_dict, 'GAF/GAF_Saved/GAF_Models/gaf_models' + str(num)+'.pt')


def train_atten(gaf_tv, y_tv,  saved_name, train_idx, valid_idx, num):
    history_list = []
    model_dict = {}  # save trained models
    # 5-folds cross-validation
    for k in tqdm(range(5)):
        # define dataloaders
        train_ds = CustomDataset(data=gaf_tv[train_idx[k]], label=y_tv[train_idx[k]].type(torch.LongTensor))
        train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)

        val_ds = CustomDataset(data=gaf_tv[valid_idx[k]], label=y_tv[valid_idx[k]].type(torch.LongTensor))
        val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=True)

        # Net
        model = GAFAttnNet(mean=False).to(device)
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  # (lr= 0.0023)
        # Define the early stopping criteria
        patience = args.patience  # number of epochs with no improvement
        min_delta = 1e-4  # minimum change in validation loss to be considered as improvement
        best_loss = float('inf')
        counter = 0

        #criterion = nn.CrossEntropyLoss()

        # empty accuracy and loss list
        train_acc_history = []
        train_loss_history = []
        val_acc_history = []
        val_loss_history = []

        for epoch in range(args.epochs):  # 100
            model.train()
            train_loss_track = []
            epoch_accuracy = []
            for i, (img, label) in enumerate(train_dl):
                optimizer.zero_grad()
                y_pred_probs = model(img)
                y_pred_probs = y_pred_probs.reshape(y_pred_probs.shape[0])  # mabe don't need
                y_pred = (y_pred_probs > 0.5).int()

                loss = F.binary_cross_entropy(y_pred_probs.float(), label.float())

                # update
                loss.backward()
                optimizer.step()
                # save corrects and loss
                epoch_accuracy.append(accuracy_score(label, y_pred))
                train_loss_track.append(loss.item())

            train_acc_history.append(np.mean(epoch_accuracy))
            train_loss_history.append(np.mean(train_loss_track))

            model.eval()
            with torch.no_grad():
                val_loss_track = []
                val_corrects = []
                for i, (img, label) in enumerate(val_dl):
                    y_pred_probs = model(img)
                    y_pred_probs = y_pred_probs.reshape(y_pred_probs.shape[0])
                    y_pred = (y_pred_probs > 0.5).int()

                    loss = F.binary_cross_entropy(y_pred_probs.float(), label.float(), reduction='mean')

                    # save loss and corrects
                    val_loss_track.append(loss.item())
                    val_corrects.append(accuracy_score(label, y_pred))

                val_acc_history.append(np.mean(val_corrects))
                val_loss_history.append(np.mean(val_loss_track))

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
        model_dict[k] = model.state_dict()

    # save 5 models
    write_gaf_history(history_list, saved_name, num)
    torch.save(model_dict, 'GAF/GAF_Saved/GAF_Models/gaf_attention_models' + str(num)+'.pt')

def evaluate_test(X_test, y_test, X_tv, y_tv, train_idx, valid_idx, num, attention=False):
    models_dict = load_models(num, attention)
    train_dict = {'accuracy': [], 'sensitivity': [], 'specificity': []}
    val_dict = {'accuracy': [], 'sensitivity': [], 'specificity': []}
    for k in models_dict:
        if attention:
            model = GAFAttnNet(mean=False)
        else:
            model = CNN()
        model.load_state_dict(models_dict[k])

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
    if attention:
        best_model = GAFAttnNet()
    else:
        best_model = CNN()
    best_model.load_state_dict(models_dict[best_index])

    # Evaluate the best model on the test set
    test_ds = CustomDataset(data=X_test, label=y_test)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True)
    test_acc, test_sens, test_spec, labels, y_pred_probs = evaluate_acc_sens_spec(best_model, test_dl, test=True)

    print("Test Accuracy: {:.3f}  Sensitivity: {:.3f}  Specificity: {:.3f}".format(test_acc, test_sens, test_spec))
    """
    if attention:
        np.savetxt("./GAF_Saved/GAF_ROC/GAF_Attention" + str(num)+".txt", np.column_stack((labels, y_pred_probs)))
    else:
        np.savetxt("./GAF_Saved/GAF_ROC/GAF_" + str(num) + ".txt", np.column_stack((labels, y_pred_probs)))
    """
    return best_index



def main():
    fp ='../../data/'
    X, y = utils.load_variable(fp)
    X = np.transpose(X, (0, 2, 1))
    saved_name="lr_"+str(args.lr)+"_epochs_"+str(args.epochs)
    saved_attention_name = "attention_lr_"+str(args.lr)+"_epochs_"+str(args.epochs)
    gaf_path_name = "GAF.pickle"
    # (1) generate GAF
    """
    gaf = get_GAF(X)
    save_matrix(gaf, gaf_path_name)
    """

    # (2) Split data into training (80%, 742) and testing (20%, 186) sets
    GAF = load_matrix(gaf_path_name)
    plot_GAF(GAF, sample=1)
    plot_all_sensor(X[1])

    """
    X_tv, X_test, y_tv, y_test, gaf_tv, gaf_test = gaf_dataset_split(X, y, gaf)
    X_tv = torch.tensor(X_tv, dtype=torch.float)
    y_tv = torch.tensor(y_tv, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    # (3) Split train data into train, valid dataset with 5-fold cross validation
    train_idx, valid_idx = utils.train_valid_split(X_tv, y_tv)  # 5 folds cross validation

    num = 0
    # 4.1 train gaf cnn
    #train(gaf_tv, y_tv, saved_name, train_idx, valid_idx, num)
    # 4.2 train gaf attention
    #train_atten(gaf_tv, y_tv, saved_attention_name, train_idx, valid_idx, num)

    # (5) Evaluate on test dataset
    # 5.1 evaluate on the GAF model

    best_index = evaluate_test(gaf_test, y_test, gaf_tv, y_tv, train_idx, valid_idx, num, attention=False)
    gaf_history = load_histories(saved_name, num)
    # plot_gaf(gaf_history, "GAF")
    save_best_history(gaf_history[best_index], best_index, "GAF-CNN")
    """

    # 5.2 evaluate on the GAF-Attention model
    """
    #gaf_atten_history = load_atten_history(saved_attention_name, num)
    #plot_gaf(gaf_atten_history, "GAF Attention ")
    evaluate_test(gaf_test, y_test, gaf_tv, y_tv, train_idx, valid_idx, num, attention=True)
    """



if __name__ =="__main__":
    main()