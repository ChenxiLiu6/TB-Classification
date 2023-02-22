import numpy as np
import pandas as pd
import torch
import pickle
import sklearn
from sklearn.model_selection import train_test_split


def accuracy(output, labels):
    preds = (output > 0.5).int()
    preds = preds.cpu().numpy()
    #print("preds: ", preds)
    labels = labels.int().cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds, normalize=False))

    return accuracy_score


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

def mtpool_dataset_split(X, y, corr, rs=42):
    X_tv, X_test, y_tv, y_test, corr_tv, corr_test = train_test_split(X, y, corr, test_size=0.2, random_state=rs)
    return X_tv, X_test, y_tv, y_test, corr_tv, corr_test

def load_mtpool_models(num):
    models = torch.load('./MTPool_saved/MTPool_Models/mtpool_models_' + str(num) + '.pt')
    return models

def write_mtpool_history(history_list, saved_name, num):
    with open("./MTPool_saved/MTPool_History/"+ saved_name+str(num)+".pickle", 'wb') as fp:
        pickle.dump(history_list, fp)

def load_history(saved_name, num):
    fp = "./MTPool_saved/MTPool_History/"
    with open(fp + saved_name + str(num)+'.pickle', 'rb') as file:
        history_list = pickle.load(file)
    return history_list

def save_corr(Corr):
    fp = "./MTPool_saved/MTPool_Metric/corr.pickle"
    with open(fp, 'wb') as file:
        pickle.dump(Corr, file)
def load_corr():
    fp = "./MTPool_saved/MTPool_Metric/corr.pickle"
    with open(fp, 'rb') as file:
        Corr = pickle.load(file)
    return Corr

