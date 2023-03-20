import argparse
import copy
import pickle
import sys
import time

import utils
from utils import *

sys.path.append('../')
from tqdm import tqdm

from mincutmtpool_utils import *
from mincutmtpool_models import *
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Model parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('--gnn', type=str, default="GNN", help='GNN or GIN')
parser.add_argument('--relation', type=str, default="adjacency", help='corr, laplacian, adjacency')
parser.add_argument('--pooling', type=str, default="MinCutPool", help='CoSimPool, MemPool, DiffPool, SAGPool or MinCutPool')

# Training parameter settings
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')  # 1000
parser.add_argument('--bs', type=int, default=32, help='training batch size')
parser.add_argument('--lr', type=float, default=1e-6, help='Initial learning rate. default:[0.00001]')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters). default: 5e-3')
parser.add_argument('--stop_thres', type=float, default=1e-9, help='The stop threshold for the training error. If the difference between training losses ' 
                                                                   'between epoches are less than the threshold, the training will be stopped. Default:1e-9')
parser.add_argument('--patience', type=int, default=50, help='The early stopping patience')
# cuda settings
args = parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, data, labels, A, A_hat):
        self.data = data
        self.labels = labels
        self.A = A
        self.A_hat = A_hat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx], self.A[idx], self.A_hat[idx])

def load_variable(fp):
    with open(fp + 'X_29.pickle', 'rb') as file:
        X = pickle.load(file)
    with open(fp + 'y.pickle', 'rb') as file:
        y = pickle.load(file)
    return X, y

def train(num, X_tv, y_tv, saved_name, A, A_hat, train_idx, valid_idx):
    X_tv = torch.tensor(X_tv, dtype=torch.float)
    y_tv = torch.tensor(y_tv, dtype=torch.long)

    history_list = []
    models_dict={}
    for k in tqdm(range(5)):
        train_ds = CustomDataset(X_tv[train_idx[k]], y_tv[train_idx[k]], A[train_idx[k]], A_hat[train_idx[k]])
        train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
        val_ds = CustomDataset(X_tv[valid_idx[k]], y_tv[valid_idx[k]], A[valid_idx[k]], A_hat[valid_idx[k]])
        val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=True)

        # define model
        net = MTPool(graph_method=args.gnn,
                     relation_method=args.relation,
                     pooling_method=args.pooling)

        # define optimizer
        optimizer = optim.Adam(net.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               eps=1e-7,
                               betas=[0.9, 0.999])

        # empty list
        train_acc_history = []
        train_loss_history = []
        val_acc_history = []
        val_loss_history = []

        # set early stopping
        patience = args.patience  # number of epochs with no improvement
        min_delta = 1e-4  # minimum change in validation loss to be considered as improvement
        best_loss = float('inf')
        counter = 0

        for epoch in range(args.epochs):
            t = time.time()
            # (1) train model
            net.train()
            epoch_accuracy = []
            loss_track = []
            for i, (data, label, adjacency, a_hat) in enumerate(train_dl):
                # (1) net
                optimizer.zero_grad()
                y_pred_probs, cut_train_loss = net(data, adjacency, a_hat)
                y_pred_probs = y_pred_probs.reshape(y_pred_probs.shape[0])
                y_pred = (y_pred_probs > 0.5).int()

                loss = F.binary_cross_entropy(y_pred_probs.float(), label.float(), reduction='mean') + cut_train_loss
                # update optimizer
                loss.backward()
                optimizer.step()
                # epoch acc, loss
                epoch_accuracy.append(accuracy_score(label, y_pred))
                loss_track.append(loss.item())

            train_loss_history.append(np.mean(loss_track))
            train_acc_history.append(np.mean(epoch_accuracy))

            # (2) validation
            net.eval()
            with torch.no_grad():
                val_loss_track = []
                val_corrects = []
                for i, (data, label, adjacency, a_hat) in enumerate(val_dl):
                    y_pred_probs, cut_val_loss = net(data, adjacency, a_hat)
                    y_pred_probs = y_pred_probs.reshape(y_pred_probs.shape[0])
                    y_pred = (y_pred_probs > 0.5).int()

                    #loss = criterion(pred, label)
                    loss = F.binary_cross_entropy(y_pred_probs.float(), label.float(), reduction='mean') + cut_val_loss
                    val_loss_track.append(loss.item())
                    val_corrects.append(accuracy_score(label, y_pred))

                val_loss_history.append(np.mean(val_loss_track))
                val_acc_history.append(np.mean(val_corrects))
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
                  'loss_train: {:.8f}'.format(train_loss_history[-1]),
                  'train_acc: {:.4f}'.format(train_acc_history[-1]),
                  'loss_val: {:.4f}'.format(val_loss_history[-1]),
                  'val_acc: {:.4f}'.format(val_acc_history[-1]),
                  'time: {:.4f}s'.format(time.time() - t))
        """
        history = {
            'train_acc': train_acc_history,
            'train_loss': train_loss_history,
            'val_acc': val_acc_history,
            'val_loss': val_loss_history
        }
        """

        # save history
        #history_list.append(history)
        # save model
        models_dict[k] = net.state_dict()

        #write_mtmincutpool_history(history_list, saved_name, num)
        # save 5 models
        torch.save(models_dict, './MinCutMTPool_saved/MinCutMTPool_Models/mtpool_models_' + str(num)+'.pt')


def evaluate_acc_sens_spec(model, dl, test=False):
    model.eval()
    with torch.no_grad():
        y_preds = []
        y_pred_probs = []
        labels = []
        for i, (data, label, adjacency, a_hat) in enumerate(dl):
            y_pred_prob, _ = model(data, adjacency, a_hat)
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
            auc = roc_auc_score(labels, y_pred_probs)
            #fpr, tpr, thresholds = roc_curve(labels, y_pred_probs)
            return accuracy, sens, spec, labels, y_pred_probs, auc

    return accuracy, sens, spec


def evaluate_test(num, X_test, y_test, X_tv, y_tv, A_test, A_hat_test, A_tv, A_hat_tv, train_idx, valid_idx ):
    loaded_models_dict = load_models(num)
    train_dict = {'accuracy': [], 'sensitivity': [], 'specificity': []}
    val_dict = {'accuracy': [], 'sensitivity': [], 'specificity': []}

    for k in loaded_models_dict:
        model = MTPool(graph_method=args.gnn,
                     relation_method=args.relation,
                     pooling_method=args.pooling)
        model.load_state_dict(loaded_models_dict[k])
        train_ds = CustomDataset(X_tv[train_idx[k]], y_tv[train_idx[k]], A_tv[train_idx[k]], A_hat_tv[train_idx[k]])
        train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
        val_ds = CustomDataset(X_tv[valid_idx[k]], y_tv[valid_idx[k]], A_tv[valid_idx[k]], A_hat_tv[valid_idx[k]])
        val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=True)

        train_acc, train_sens, train_spec = evaluate_acc_sens_spec(model, train_dl, test=False)
        train_dict['accuracy'].append(train_acc)
        train_dict['sensitivity'].append(train_sens)
        train_dict['specificity'].append(train_spec)

        # valid results
        val_acc, val_sens, val_spec = evaluate_acc_sens_spec(model, val_dl, test=False)
        val_dict['accuracy'].append(val_acc)
        val_dict['sensitivity'].append(val_sens)
        val_dict['specificity'].append(val_spec)

    # Find the index of the best model
    best_index = np.argmax(val_dict['accuracy'])

    # Load the best model
    best_model = MTPool(graph_method=args.gnn,
                        relation_method=args.relation,
                        pooling_method=args.pooling)

    best_model.load_state_dict(loaded_models_dict[best_index])

    test_ds = CustomDataset(X_test, y_test, A_test, A_hat_test)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True)

    # test results
    test_acc, test_sens, test_spec, labels, y_pred_probs, auc = evaluate_acc_sens_spec(best_model, test_dl, test=True)
    y_pred = (y_pred_probs >= 0.5).int()
    cm = confusion_matrix(labels, y_pred)

    return train_dict['accuracy'][best_index], train_dict['sensitivity'][best_index], train_dict['specificity'][best_index], val_dict['accuracy'][best_index], val_dict['sensitivity'][best_index], val_dict['specificity'][best_index], test_acc, test_sens, test_spec, labels, y_pred_probs, auc, cm

def test_10(X_test, y_test, X_tv, y_tv, A_test, A_hat_test, A_tv, A_hat_tv, saved_name):
    rs_list = [24, 42, 15, 51, 1998, 1225, 37, 73, 2003, 20]
    res_dict = {'train_acc_list': [], 'train_sens_list': [], 'train_spec_list': [],
                'val_acc_list': [], 'val_sens_list': [], 'val_spec_list': [],
                'test_acc_list': [], 'test_sens_list': [], 'test_spec_list': [], 'test_labels_list': [],
                'test_pred_probs': [], 'test_auc_list': [], 'test_cm': []}
    for i in range(10):
        rs = rs_list[i]
        train_idx, valid_idx = train_valid_split(X_tv, y_tv, rs)
        train_acc, train_sens, train_spec, val_acc, val_sens, val_spec, test_acc, test_sens, test_spec, labels, y_pred_probs, test_auc, test_cm = evaluate_test(i, X_test, y_test, X_tv, y_tv, A_test, A_hat_test, A_tv, A_hat_tv, train_idx, valid_idx )
        res_dict['train_acc_list'].append(train_acc)
        res_dict['train_sens_list'].append(train_sens)
        res_dict['train_spec_list'].append(train_spec)

        res_dict['val_acc_list'].append(val_acc)
        res_dict['val_sens_list'].append(val_sens)
        res_dict['val_spec_list'].append(val_spec)

        res_dict['test_acc_list'].append(test_acc)
        res_dict['test_sens_list'].append(test_sens)
        res_dict['test_spec_list'].append(test_spec)
        res_dict['test_labels_list'].append(labels)      # (10, 5)
        res_dict['test_pred_probs'].append(y_pred_probs) # (10, 5)
        res_dict['test_auc_list'].append(test_auc) # (10)
        res_dict['test_cm'].append(test_cm)

    with open("./MinCutMTPool_saved/MinCutMTPool_Result/"+saved_name+"_result_10.pickle", 'wb') as fp:
        pickle.dump(res_dict, fp)


def train_10(X_tv, y_tv, A_tv, A_hat_tv):
    rs_list = [24, 42, 15, 51, 1998, 1225, 37, 73, 2003, 20]
      # 5 folds cross validation
    saved_name = "lr_" + str(args.lr) + "_epochs_" + str(args.epochs)

    for i in range(10):
        rs = rs_list[i]
        train_idx, valid_idx = train_valid_split(X_tv, y_tv, rs)
        train(i, X_tv, y_tv, saved_name, A_tv, A_hat_tv,  train_idx, valid_idx)

def mean_roc(res_dict):
    # roc curve
    labels_50_list = []
    pred_50_list = []
    print("len(res_dict['test_labels_list']): ", len(res_dict['test_labels_list']))
    for i in range(len(res_dict['test_labels_list'])):
        labels_50_list.append(res_dict['test_labels_list'][i])
        pred_50_list.append(res_dict['test_pred_probs'][i])

    # Compute the ROC curve for each pair of labels and probabilities

    roc_curves = []
    for i in range(10):
        fpr, tpr, _ = roc_curve(labels_50_list[i], pred_50_list[i])
        # Interpolate the ROC curve to a common set of FPR values
        fpr_interp = np.linspace(0, 1, num=110)
        tpr_interp = np.interp(fpr_interp, fpr, tpr)
        roc_curves.append((fpr_interp, tpr_interp))

    # Compute the average ROC curve and mean AUC score
    fpr_mean = np.mean([roc[0] for roc in roc_curves], axis=0)
    tpr_mean = np.mean([roc[1] for roc in roc_curves], axis=0)
    auc_mean = auc(fpr_mean, tpr_mean)

    # Plot the average ROC curve
    plt.plot(fpr_mean, tpr_mean, label=f'Mean ROC (AUC={auc_mean:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curve')
    plt.legend()
    plt.show()

    np.savetxt('./MinCutMTPool_saved/MinCutMTPool_ROC/MinCutMTPool_mean_roc_results.txt', np.column_stack((fpr_mean, tpr_mean)))

def best_roc(res_dict, best_index):
    fpr, tpr, _ = roc_curve(res_dict['test_labels_list'][best_index], res_dict['test_pred_probs'][best_index])
    best_auc = res_dict['test_auc_list'][best_index]
    plt.plot(fpr, tpr, label=f'Best ROC (AUC={best_auc:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Best ROC Curve')
    plt.legend()
    plt.show()
    np.savetxt('./MinCutMTPool_saved/MinCutMTPool_ROC/MinCutMTPool_best_roc_results.txt', np.column_stack((fpr, tpr)))

def print_res(saved_name):
    with open("./MinCutMTPool_saved/MinCutMTPool_Result/"+saved_name+"_result_10.pickle", 'rb') as fp:
        res_dict = pickle.load(fp)

    print("mean train acc: {:.3f}  sens: {:.3f}  spec: {:.3f} ".format(np.mean(res_dict['train_acc_list']),
                                                                  np.mean(res_dict['train_sens_list']),
                                                                  np.mean(res_dict['train_spec_list'])))
    print("mean valid acc: {:.3f}  sens: {:.3f}  spec: {:.3f}".format(np.mean(res_dict['val_acc_list']),
                                                                 np.mean(res_dict['val_sens_list']),
                                                                 np.mean(res_dict['val_spec_list'])))
    print("mean test acc: {:.3f}  sens: {:.3f}  spec: {:.3f}  auc: {:.3f}".format(np.mean(res_dict['test_acc_list']),
                                                                             np.mean(res_dict['test_sens_list']),
                                                                             np.mean(res_dict['test_spec_list']),
                                                                             np.mean(res_dict['test_auc_list'])))
    # find best test
    best_index = np.argmax(res_dict['test_acc_list'])
    print("best test acc: {:.3f}  sens: {:.3f}  spec: {:.3f}  auc: {:.3f}".format(res_dict['test_acc_list'][best_index],
                                                                                  res_dict['test_sens_list'][best_index],
                                                                                  res_dict['test_spec_list'][best_index],
                                                                                  res_dict['test_auc_list'][best_index]))
    mean_roc(res_dict)
    utils.plot_confusion(res_dict, saved_name)



def main():
    # --------------------------------- 1. Load Dataset -------------------------------
    # (1) Origin X and y
    X, y = load_variable(fp='../../data/')
    X = np.transpose(X, (0, 2, 1))  # shape: (928, 29, 147)

    # Path Definition
    A_path_name = "Adjacency.pickle"
    D_path_name = "Degree.pickle"
    L_path_name = "Laplacian.pickle"
    A_hat_path = "A_hat.pickle"
    saved_name = "MT-MinCutPool"

    # (1) Compute adjacency, degree, and Laplacian matrices
    """
    Adjacency, Degree, Laplacian = calc_ADL(X, 29)
    save_matrix(Adjacency, A_path_name)
    save_matrix(Degree, D_path_name)
    save_matrix(Laplacian, L_path_name)
    """
    # (2) i. Compute normalized adjacency matrix A_hat
    """
    A = load_matrix(A_path_name)
    A_hat = normalize_adjacency(A)
    save_matrix(A_hat, A_hat_path)
    """


    # MinCutPool Model
    #Adjacency = load_matrix(A_path_name).float()
    A_hat = load_matrix(A_hat_path).float()
    Laplacian = load_matrix(L_path_name).float()
    
    # (2) Split data into training (80%, 742) and testing (20%, 186) sets
    X_tv, X_test, y_tv, y_test, A_tv, A_test, A_hat_tv, A_hat_test = mtpool_dataset_split(X, y, Laplacian, A_hat)
    X_tv = torch.tensor(X_tv, dtype=torch.float)
    y_tv = torch.tensor(y_tv, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    #------------------------------- 2. Train MinCutMTPool-----------------------------
    #train_10(X_tv, y_tv, A_tv, A_hat_tv)
    test_10(X_test, y_test, X_tv, y_tv, A_test, A_hat_test, A_tv, A_hat_tv, saved_name)
    print_res(saved_name)







if __name__ == "__main__":
    main()