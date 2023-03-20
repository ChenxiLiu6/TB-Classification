import utils, argparse, pickle, os
import tensorflow.keras as keras
from tqdm import tqdm
import numpy as np
from LSTM_utils import *
import utils

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, auc

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.00001, help='Initial learning rate. default:[0.0001]')
parser.add_argument('--bs', type=int, default=32, help='training batch size')
args = parser.parse_args()

def build_lstm():
    # Define LSTM
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(32, input_shape=(147, 29)))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(Dense(16, activation='relu'))
    LSTM_model.add(Dense(1, activation='sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    LSTM_model.compile(loss='binary_crossentropy', optimizer=opt,
                          metrics=['accuracy', 'TruePositives'])
    LSTM_model.summary()
    return LSTM_model

def train(num, saved_name, X_tv, y_tv, train_idx, valid_idx):
    # (1) build LSTM Model, start 5-fold cross-validation on the training dataset
    history_list = []
    lstm_models=[]

    for k in tqdm(range(5)):
        x_train = X_tv[train_idx[k]]
        y_train = y_tv[train_idx[k]]
        x_valid = X_tv[valid_idx[k]]
        y_valid = y_tv[valid_idx[k]]

        lstm = build_lstm()
        earlyStop = EarlyStopping(monitor="val_loss", verbose=2, mode='min', patience=50)
        history = lstm.fit(x_train, y_train,
                           batch_size=args.bs,
                           epochs=args.epochs,
                           validation_data=(x_valid, y_valid),
                           callbacks=[earlyStop])
        history_list.append(history)

        #lstm.save(f'LSTM/LSTM_saved/LSTM_Models/' + str(num) +'/lstm_model_{'+str(k)+'}.h5')
        lstm_models.append(lstm)

    # save the history lists
    write_lstm_history(history_list, saved_name, num)
    # Save the list of models
    np.save('LSTM_saved/LSTM_Models/'+saved_name+ "_"+str(num)+'.npy', lstm_models)

def calc_acc_sens_spec(model, X, y, test=False):
    y_pred_probs = model.predict(X)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Calculate sensitivity and specificity
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)

    # Calculate auc, roc curve if roc==True
    if test == True:
        auc = roc_auc_score(y, y_pred_probs)
        #fpr, tpr, thresholds = roc_curve(y, y_pred_probs)
        return accuracy, sensitivity, specificity, y, y_pred_probs, auc

    return accuracy, sensitivity, specificity

def test(num, X_test, y_test, X_tv, y_tv, train_idx, valid_idx):
    # Load the list of models
    models = np.load('./LSTM_saved/LSTM_Models/lstm_models_'+str(num)+'.npy', allow_pickle=True)
    # Initialize the list to store the validation performance
    train_dict = {'accuracy': [], 'sensitivity': [], 'specificity': []}
    val_dict = {'accuracy': [], 'sensitivity': [], 'specificity': []}
    for k, model in enumerate(models):
        x_train = X_tv[train_idx[k]]
        y_train = y_tv[train_idx[k]]
        x_valid = X_tv[valid_idx[k]]
        y_valid = y_tv[valid_idx[k]]

        # get train results
        train_acc, train_sens, train_spec = calc_acc_sens_spec(model, x_train, y_train, test=False)
        train_dict['accuracy'].append(train_acc)
        train_dict['sensitivity'].append(train_sens)
        train_dict['specificity'].append(train_spec)

        # get validation results
        val_acc, val_sens, val_spec = calc_acc_sens_spec(model, x_valid, y_valid, test=False)
        val_dict['accuracy'].append(val_acc)
        val_dict['sensitivity'].append(val_sens)
        val_dict['specificity'].append(val_spec)

    # Find the index of the best model
    best_index = np.argmax(val_dict['accuracy'])
    val_dict['best_index'] = best_index

    # Load the best model
    best_model = models[best_index]

    # Evaluate the best model on the test set
    test_acc, test_sens, test_spec, labels, y_pred_probs, auc = calc_acc_sens_spec(best_model, X_test, y_test, test=True)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, y_pred)

    return train_dict['accuracy'][best_index], train_dict['sensitivity'][best_index], train_dict['specificity'][best_index], val_dict['accuracy'][best_index], val_dict['sensitivity'][best_index], val_dict['specificity'][best_index], test_acc, test_sens, test_spec, labels, y_pred_probs, auc, cm

def test_10(X_test, y_test, X_tv, y_tv, saved_name):
    rs_list = [24, 42, 15, 51, 1998, 1225, 37, 73, 2003, 20]
    res_dict = {'train_acc_list': [], 'train_sens_list': [], 'train_spec_list': [],
                'val_acc_list': [], 'val_sens_list': [], 'val_spec_list': [],'best_val_indices':[],
                'test_acc_list': [], 'test_sens_list': [], 'test_spec_list': [], 'test_labels': [], 'y_pred_probs': [],
                'test_auc_list': [], 'test_cm': []}
    for i in range(10):
        rs = rs_list[i]
        train_idx, valid_idx = utils.train_valid_split(X_tv, y_tv, rs)
        train_acc, train_sens, train_spec, val_acc, val_sens, val_spec, test_acc, test_sens, test_spec, labels, y_pred_probs, test_auc, test_cm = test(
            i, X_test, y_test, X_tv, y_tv, train_idx, valid_idx)
        res_dict['train_acc_list'].append(train_acc)
        res_dict['train_sens_list'].append(train_sens)
        res_dict['train_spec_list'].append(train_spec)

        res_dict['val_acc_list'].append(val_acc)
        res_dict['val_sens_list'].append(val_sens)
        res_dict['val_spec_list'].append(val_spec)


        res_dict['test_acc_list'].append(test_acc)
        res_dict['test_sens_list'].append(test_sens)
        res_dict['test_spec_list'].append(test_spec)
        res_dict['test_labels'].append(labels)  # (10, 5)
        res_dict['y_pred_probs'].append(y_pred_probs)  # (10, 5)
        res_dict['test_auc_list'].append(test_auc)  # (10)
        res_dict['test_cm'].append(test_cm)

    with open("./LSTM_saved/LSTM_Result/" +saved_name +"_result_10.pickle", 'wb') as fp:
        pickle.dump(res_dict, fp)

def train_10(X_tv, y_tv, saved_name):
    rs_list = [24, 42, 15, 51, 1998, 1225, 37, 73, 2003, 20]

    for i in range(0, 1):
        rs = rs_list[i]
        train_idx, valid_idx = utils.train_valid_split(X_tv, y_tv, rs)
        train(i, saved_name, X_tv, y_tv, train_idx, valid_idx)

def mean_roc(res_dict):
    # roc curve
    labels_50_list = []
    pred_50_list = []
    for i in range(len(res_dict['test_labels'])):
        labels_50_list.append(res_dict['test_labels'][i])
        pred_50_list.append(res_dict['y_pred_probs'][i])

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

    np.savetxt('./LSTM_saved/LSTM_ROC/LSTM_jitter_mean_roc_results.txt', np.column_stack((fpr_mean, tpr_mean)))

def best_roc(res_dict, best_index):
    fpr, tpr, _ = roc_curve(res_dict['test_labels'][best_index], res_dict['y_pred_probs'][best_index])
    best_auc = res_dict['test_auc_list'][best_index]
    plt.plot(fpr, tpr, label=f'Best ROC (AUC={best_auc:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Best ROC Curve')
    plt.legend()
    plt.show()

    np.savetxt('./LSTM_saved/LSTM_ROC/LSTM_best_roc_results.txt', np.column_stack((fpr, tpr)))

def print_res(saved_name):
    # load and compute mean result
    with open("./LSTM_saved/LSTM_Result/"+saved_name +"_result_10.pickle", 'rb') as fp:
        res_dict = pickle.load(fp)
    print("train acc: {:.3f}  sens: {:.3f}  spec: {:.3f} ".format(np.mean(res_dict['train_acc_list']),
                                                                  np.mean(res_dict['train_sens_list']),
                                                                  np.mean(res_dict['train_spec_list'])))
    print("valid acc: {:.3f}  sens: {:.3f}  spec: {:.3f}".format(np.mean(res_dict['val_acc_list']),
                                                                 np.mean(res_dict['val_sens_list']),
                                                                 np.mean(res_dict['val_spec_list'])))
    print("test acc: {:.3f}  sens: {:.3f}  spec: {:.3f}  auc: {:.3f}".format(np.mean(res_dict['test_acc_list']),
                                                                             np.mean(res_dict['test_sens_list']),
                                                                             np.mean(res_dict['test_spec_list']),
                                                                             np.mean(res_dict['test_auc_list'])))

    mean_roc(res_dict)
    utils.plot_confusion(res_dict, 'LSTM')
    #best_roc(res_dict, best_index)


def load_history(saved_name, num):
    # Define the path to the folder you want to open
    fp = "./LSTM_saved/LSTM_Histories/"
    with open(fp + saved_name + "_"+str(num)+'.pickle', 'rb') as file:
        history_list = pickle.load(file)
    return history_list


def main():
    #---------------------------- 1. Data Split --------------------------------
    # (1) load X, y
    fp = "../../data/"
    X, y = utils.load_variable(fp)
    # (2) Split data into training (80%, 742) and testing (20%, 186) sets
    X_tv, X_test, y_tv, y_test = dataset_split(X, y)

    # (3) Split train data into train, valid dataset with 5-fold cross validation
    saved_name = "LSTM"

    # ----------------------------- 2. Train the LSTM Model -----------------------
    #train_10(X_tv, y_tv, saved_name)
    #test_10(X_test, y_test, X_tv, y_tv, saved_name)
    print_res(saved_name)

    # ----------------------------- 3. Test and Plot-----------------------
    # (1) Plot train, valid loss and accuracy
    #plot(history_list)


if __name__ == "__main__":
    main()







