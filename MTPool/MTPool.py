import argparse
import copy
import pickle
import sys
import time
from utils import *

sys.path.append('../')
from tqdm import tqdm
from utils import CustomDataset

from mtpool_utils import *
from mtpool_models import *
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Model parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('--gnn', type=str, default="GNN", help='GNN or GIN')
parser.add_argument('--relation', type=str, default="corr", help='corr or laplacian or dynamic or all_one')
parser.add_argument('--pooling', type=str, default="CoSimPool", help='CoSimPool or MemPool or DiffPool or SAGPool')

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

class MyDataset(Dataset):
    def __init__(self, data, labels, corr):
        self.data = data
        self.labels = labels
        self.corr = corr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx], self.corr[idx])

def load_variable(fp):
    with open(fp + 'X_29.pickle', 'rb') as file:
        X = pickle.load(file)
    with open(fp + 'y.pickle', 'rb') as file:
        y = pickle.load(file)
    return X, y

def evaluate_mtpool_acc_sens_spec(model, dl, test=False):
    model.eval()
    with torch.no_grad():
        y_preds = []
        y_pred_probs = []
        labels = []
        for i, (data, label, corr) in enumerate(dl):
            y_pred_prob = model(data.float(), corr)
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
            return accuracy, sens, spec, labels, y_pred_probs

    return accuracy, sens, spec


def evaluate_test(num, X_test, y_test, X_tv, y_tv, corr_test, corr_tv):
    loaded_models_dict = load_mtpool_models(num)
    train_dict = {'accuracy': [], 'sensitivity': [], 'specificity': []}
    val_dict = {'accuracy': [], 'sensitivity': [], 'specificity': []}
    train_idx, valid_idx = train_valid_split(X_tv, y_tv)
    for k in loaded_models_dict:
        model = MTPool(graph_method=args.gnn,
                     relation_method=args.relation,
                     pooling_method=args.pooling)
        model.load_state_dict(loaded_models_dict[k])
        train_ds = MyDataset(X_tv[train_idx[k]], y_tv[train_idx[k]], corr_tv[train_idx[k]])
        train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
        val_ds = MyDataset(X_tv[valid_idx[k]], y_tv[valid_idx[k]], corr_tv[valid_idx[k]])
        val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=True)

        train_acc, train_sens, train_spec = evaluate_mtpool_acc_sens_spec(model, train_dl, test=False)
        train_dict['accuracy'].append(train_acc)
        train_dict['sensitivity'].append(train_sens)
        train_dict['specificity'].append(train_spec)

        # valid results
        val_acc, val_sens, val_spec = evaluate_mtpool_acc_sens_spec(model, val_dl, test=False)
        val_dict['accuracy'].append(val_acc)
        val_dict['sensitivity'].append(val_sens)
        val_dict['specificity'].append(val_spec)


    # Find the index of the best model
    best_index = np.argmax(val_dict['accuracy'])
    print("valid performance: ", val_dict['accuracy'])
    print("best index: ", best_index)

    # Load the best model
    best_model = MTPool(graph_method=args.gnn,
                     relation_method=args.relation,
                     pooling_method=args.pooling)

    best_model.load_state_dict(loaded_models_dict[best_index])

    # test on the test set
    test_ds = MyDataset(X_test, y_test, corr_test)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True)

    test_acc, test_sens, test_spec, labels, y_pred_probs = evaluate_mtpool_acc_sens_spec(best_model, test_dl, test=True)
    print("Train Accuracy: {:.3f}  Sensitivity: {:.3f}  Specificity: {:.3f}".format(np.mean(train_dict['accuracy']),
                                                                                    np.mean(train_dict['sensitivity']),
                                                                                    np.mean(train_dict['specificity'])))
    print("Valid Accuracy: {:.3f}  Sensitivity: {:.3f}  Specificity: {:.3f}".format(np.mean(val_dict['accuracy']),
                                                                                    np.mean(val_dict['sensitivity']),
                                                                                    np.mean(val_dict['specificity'])))
    print("Test Accuracy: {:.3f}  Sensitivity: {:.3f}  Specificity: {:.3f}".format(test_acc, test_sens, test_spec))
    np.savetxt("./MTPool_saved/MTPool_ROC/MTPool" + str(num) +".txt", np.column_stack((labels, y_pred_probs)))


def train(X_tv, y_tv, saved_name, corr, num):
    X_tv = torch.tensor(X_tv, dtype=torch.float)
    y_tv = torch.tensor(y_tv, dtype=torch.long)
    train_idx, valid_idx = train_valid_split(X_tv, y_tv)
    history_list = []
    models_dict = {}
    for k in tqdm(range(5)):
        train_ds = MyDataset(X_tv[train_idx[k]], y_tv[train_idx[k]], corr[train_idx[k]])
        train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
        val_ds = MyDataset(X_tv[valid_idx[k]], y_tv[valid_idx[k]], corr[valid_idx[k]])
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
            for i, (data, label, train_corr) in enumerate(train_dl):
                # (2) net
                optimizer.zero_grad()
                y_pred_probs = net(data, train_corr)
                y_pred_probs = y_pred_probs.reshape(y_pred_probs.shape[0])
                y_pred = (y_pred_probs > 0.5).int()

                loss = F.binary_cross_entropy(y_pred_probs.float(), label.float(), reduction='mean')
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
                for i, (data, label, val_corr) in enumerate(val_dl):
                    y_pred_probs = net(data, val_corr)
                    y_pred_probs = y_pred_probs.reshape(y_pred_probs.shape[0])
                    y_pred = (y_pred_probs > 0.5).int()

                    loss = F.binary_cross_entropy(y_pred_probs.float(), label.float(), reduction='mean')
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

        history = {
            'train_acc': train_acc_history,
            'train_loss': train_loss_history,
            'val_acc': val_acc_history,
            'val_loss': val_loss_history
        }

        # save history
        history_list.append(history)
        # save model
        models_dict[k] = net.state_dict()

        write_mtpool_history(history_list, saved_name, num)
        # save 5 models
        torch.save(models_dict, './MTPool_saved/MTPool_Models/mtpool_models_' + str(num) + '.pt')


def main():
    # calculate Laplacian and save it as file

    saved_name = "mtpool_corr_lr_" + str(args.lr) + "_epochs_" + str(args.epochs)

    X, y = load_variable(fp='../../data/')
    X = np.transpose(X, (0, 2, 1)) # shape: (928, 29, 147)
    # (1) compute and save correlation matrix
    #corr = corr_matrix(X, 29)
    #save_corr(corr)

    # (2) load corr, data split
    corr = load_corr()
    X_tv, X_test, y_tv, y_test, corr_tv, corr_test = mtpool_dataset_split(X, y, corr)

    # (3) train MTPool model
    # num = 1: add x = nn.PReLU()(x) in (3) GNN
    num = 0
    #train(X_tv, y_tv, saved_name, corr_tv, num)  # Corr

    # (4) evaluate on the test set
    evaluate_test(num, X_test, y_test, X_tv, y_tv, corr_test, corr_tv)

    """
    history_list = load_history(saved_name)
    plot(history_list)
    #sens_list, spec_list, tpr_list, fpr_list, roc_auc_list = sens_spec(history_list)
    #plot_roc_auc(tpr_list, fpr_list, roc_auc_list)
    """

if __name__ == "__main__":
    main()