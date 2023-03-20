import numpy as np
import pandas as pd
import torch
import pickle
import sklearn
import matplotlib.pyplot as plt
from pyts.metrics import dtw
from sklearn.model_selection import train_test_split

def mtpool_dataset_split(X, y, Adjacency, A_hat, rs=42):
    X_train, X_test, y_train, y_test, A_train, A_test, A_hat_train, A_hat_test= train_test_split(X, y, Adjacency, A_hat, test_size=0.2, random_state=rs)
    return X_train, X_test, y_train, y_test, A_train, A_test, A_hat_train, A_hat_test


def accuracy(output, labels):
    preds = (output > 0.5).int()
    preds = preds.cpu().numpy()
    #print("preds: ", preds)
    labels = labels.int().cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds, normalize=False))

    return accuracy_score

# compute Adjacency, Degree, and Laplacian Matrices
def calc_ADL(X, num_nodes=29):
    """
    :param X: (928, 29, 147)
    :param num_nodes: number of sensors, also adjacency matrix length
    :return: A: (928, 29, 29)
    """
    # (1) build adjacency matrix
    A_DTW = np.empty((X.shape[0], num_nodes, num_nodes))
    for i in range(X.shape[0]):
        x = X[i]
        adj = np.zeros((num_nodes, num_nodes))
        for j in range(num_nodes):  # 29
            for k in range(num_nodes):
                dist = dtw(x[j], x[k])
                adj[j, k] = dist
        A_DTW[i] = adj
    # transform values to be 0 or 1
    A = (A_DTW < 0.1).astype(int)

    # (2) build degree matrix
    D = np.empty_like(A)
    for i in range(A.shape[0]):
        adj = A[i]
        d = np.diag(np.sum(adj, axis=1))
        D[i] = d

    # (3) build laplacian matrix
    L = D - A
    A = torch.from_numpy(A)
    D = torch.from_numpy(D)
    L = torch.from_numpy(L).float()
    return A, D, L

# construct  adjacency matrix for train and test dataset
def corr_matrix(X, num_nodes=29):
    """
    :param num_nodes: number of variables (sensors) -> 29
    :param x: (N, n, T): N-> # of training samples, n-># of variables, T-> timeseries length
    :return: x_A: adjacency matrix for train_x or valid_x
    """
    #X = X.detach().numpy()

    x_len = len(X)

    A = np.ones((num_nodes, num_nodes), np.int8)   # (29, 29)
    A = A / np.sum(A, 0) # normalize
    A_X = np.zeros((x_len, num_nodes, num_nodes), dtype=np.float32) # (742, 29, 29)
    for i in range(x_len):
        A_X[i, :, :] = A
    x_A = torch.from_numpy(A_X)

    for i in range(x_len):
        temp_x = X[i]  #(29, 147)
        d = {}
        for j in range(temp_x.shape[0]): # 29
            d[j] = temp_x[j]

        df = pd.DataFrame(d)
        df_corr = df.corr() # default:pearson correlation
        x_A[i] = torch.from_numpy(df_corr.to_numpy() / np.sum(df_corr.to_numpy(), 0))

    return x_A

def write_mtmincutpool_history(history_list, saved_name, num):
    with open("./MinCutMTPool_saved/MinCutMTPool_Histories/10_run/"+ str(num)+"_"+ saved_name+".pickle", 'wb') as fp:
        pickle.dump(history_list, fp)

def load_history(saved_name, num):
    fp = "./MinCutMTPool_saved/MinCutMTPool_Histories/10_run/" + str(num)+"_"
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

    axes[1].plot(history['train_loss'][4:])
    axes[1].plot(history['val_loss'][4:])
    axes[1].set(xlabel='Epoch', ylabel='Loss')
    axes[1].set_title(name + ' loss for data fold ' + str(best_index+1))
    axes[1].legend(['Training', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('./MinCutMTPool_saved/MinCutMTPool_Histories/best_'+name, dpi=300)
    plt.show()


def save_matrix(matrix, name):
    fp = "../saved_variables/matrices/" + str(name)
    with open(fp, 'wb') as file:
        pickle.dump(matrix, file)

def load_matrix(name):
    fp = "../saved_variables/matrices/" + str(name)
    with open(fp, 'rb') as file:
        Matrix = pickle.load(file)
    return Matrix

def load_models(num):
    models = torch.load('./MinCutMTPool_saved/MinCutMTPool_Models/mtpool_models_'+str(num)+'.pt')
    return models
def save_corr(Corr):
    fp = "../saved_variables/corr.pickle"
    with open(fp, 'wb') as file:
        pickle.dump(Corr, file)
def load_corr():
    fp = "../saved_variables/corr.pickle"
    with open(fp, 'rb') as file:
        Corr = pickle.load(file)
    return Corr

def plot_signals_40(X, sample_num=24):
    s = sample_num
    fig, ax = plt.subplots(5, 8, figsize=(25, 12))
    fig.tight_layout()
    for i in range(X.shape[2]):
        r = int(i / 8)
        c = i % 8
        ax[r, c].plot(range(X.shape[1]), X[s, :, i])
        ax[r, c].set_title(i)
    plt.show()

def plot_signals_29(X, sample_num=24):
    s = sample_num
    fig, ax = plt.subplots(5, 6, figsize=(25, 12))
    fig.tight_layout()
    for i in range(X.shape[1]):
        r = int(i / 6)
        c = i % 6
        ax[r, c].plot(range(X.shape[2]), X[s, i, :])
        ax[r, c].set_title(i)
    plt.show()

# normalize adjacency matrix
def normalize_adjacency(A):
    """
        A_hat = D^-0.5 * A * D^-0.5
        A: adjacency matrix, A_hat: normalized adjacency matrix

        Input:
        ------
        A: tensor in shape (#sample, N, N), Adjacency matrix

        Output:
        -------
        A_hat: symmetrically normalized adjacency matrix with shape (N, N)

    """

    #  turn to numpy array
    A= A.detach().cpu().numpy()
    A_hat = []
    for i in range(A.shape[0]):
        adjacency = A[i]  # shape: (N, N)
        degree = np.array(adjacency.sum(1))  # (N,)

        # adjacency matrix normalization
        d_hat = np.diag(np.power(degree, -0.5).flatten())
        A_hat.append(np.matmul(np.matmul(d_hat, adjacency), d_hat))

    A_hat = torch.Tensor(np.asarray(A_hat))
    return A_hat

