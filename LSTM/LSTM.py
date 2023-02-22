import utils, argparse, pickle, os
import tensorflow.keras as keras
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate. default:[0.0001]')
parser.add_argument('--bs', type=int, default=32, help='training batch size')
args = parser.parse_args()

def write_lstm_history(history_list, saved_name):
    with open("saved_variables/"+ saved_name+".pickle", 'wb') as fp:
        pickle.dump(history_list, fp)

def dataset_split(X, y, rs=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
    return X_train, X_test, y_train, y_test

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

        lstm.save(f'LSTM/LSTM_saved/LSTM_Models/' + str(num) +'/lstm_model_{'+str(k)+'}.h5')
        lstm_models.append(lstm)

    # save the history lists
    write_lstm_history(history_list, saved_name)
    # Save the list of models
    np.save('LSTM_saved/LSTM_Models/' +str(num) +'/lstm_models.npy', lstm_models)

def calc_acc_sens_spec(model, X, y, roc=False):
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
    if roc == True:
        #auc = roc_auc_score(y, y_pred_probs)
        #fpr, tpr, thresholds = roc_curve(y, y_pred_probs)
        return accuracy, sensitivity, specificity, y, y_pred_probs

    return accuracy, sensitivity, specificity

def plot_roc(auc, fpr, tpr):
    plt.plot(fpr, tpr, label='ROC curve for LSTM (AUC = %0.2f)' % auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()

def test(num, X_test, y_test, X_tv, y_tv, train_idx, valid_idx):
    # Load the list of models
    models = np.load('./LSTM_saved/LSTM_Models/'+str(num)+'/lstm_models.npy', allow_pickle=True)

    # Initialize the list to store the validation performance
    Train_acc=[]
    Train_sens=[]
    Train_spec=[]
    Val_acc=[]
    Val_sens=[]
    Val_spec=[]

    for k, model in enumerate(models):
        x_train = X_tv[train_idx[k]]
        y_train = y_tv[train_idx[k]]
        x_valid = X_tv[valid_idx[k]]
        y_valid = y_tv[valid_idx[k]]

        # get train results
        train_acc, train_sens, train_spec = calc_acc_sens_spec(model, x_train, y_train, roc=False)
        Train_acc.append(train_acc)
        Train_sens.append(train_sens)
        Train_spec.append(train_spec)

        # get validation results
        val_acc, val_sens, val_spec = calc_acc_sens_spec(model, x_valid, y_valid, roc=False)
        Val_acc.append(val_acc)
        Val_sens.append(val_sens)
        Val_spec.append(val_spec)

    # print mean train, valid performance
    print("Train Accuracy: {:.3f}  Sensitivity: {:.3f}  Specificity: {:.3f}".format(np.mean(Train_acc),
                                                                              np.mean(Train_sens),
                                                                              np.mean(Train_spec)))
    print("Valid Accuracy: {:.3f}  Sensitivity: {:.3f}  Specificity: {:.3f}".format(np.mean(Val_acc),
                                                                                    np.mean(Val_sens),
                                                                                    np.mean(Val_spec)))
    # Find the index of the best model
    best_index = np.argmax(Val_acc)
    print("valid performance: ", Val_acc)
    print("best index: ", best_index)

    # Load the best model
    best_model = models[best_index]

    # Evaluate the best model on the test set
    test_acc, test_sens, test_spec, labels, y_pred_probs = calc_acc_sens_spec(best_model, X_test, y_test, roc=True)
    print("Test Accuracy: {:.3f}  Sensitivity: {:.3f}  Specificity: {:.3f}".format(test_acc, test_sens, test_spec))

    #np.savetxt("LSTM/LSTM_saved/LSTM_ROC/LSTM"+str(num)+".txt", np.column_stack((labels, y_pred_probs)))
    #plot_roc(auc, fpr, tpr)
    return best_index



def plot_history(history, fold):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set(xlabel='Epoch', ylabel='Accuracy')
    axes[0].set_title('LSTM accuracy for data fold '+ str(fold))
    axes[0].legend(['Training', 'Validation'], loc='lower right')

    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set(xlabel='Epoch', ylabel='Loss')
    axes[1].set_title('LSTM loss for data fold '+str(fold))
    axes[1].legend(['Training', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.show()

def plot(history_list):
    for i in range(len(history_list)):
        history = history_list[i]
        plot_history(history, i+1)

def load_history(num, saved_name):
    # Get the current working directory
    #current_dir = os.getcwd()

    # Define the path to the folder you want to open
    fp = "./LSTM_saved/LSTM_Histories/"+str(num)+"/"
    #fp = os.path.join(current_dir, folder_name)
    with open(fp + saved_name +'.pickle', 'rb') as file:
        history_list = pickle.load(file)
    return history_list

def save_best_history(history, best_index):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set(xlabel='Epoch', ylabel='Accuracy')
    axes[0].set_title('LSTM accuracy for data fold ' + str(best_index+1))
    axes[0].legend(['Training', 'Validation'], loc='lower right')

    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set(xlabel='Epoch', ylabel='Loss')
    axes[1].set_title('LSTM loss for data fold ' + str(best_index+1))
    axes[1].legend(['Training', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('./LSTM_saved/LSTM_Histories/best_LSTM', dpi=300)
    plt.show()


def main_func(num):
    #---------------------------- 1. Data Split --------------------------------
    # (1) load X, y
    fp = "../../data/"
    X, y = utils.load_variable(fp)

    # (2) Split data into training (80%, 742) and testing (20%, 186) sets
    X_tv, X_test, y_tv, y_test = dataset_split(X, y)

    # (3) Split train data into train, valid dataset with 5-fold cross validation
    train_idx, valid_idx = utils.train_valid_split(X_tv, y_tv)  # 5 folds cross validation

    saved_name = "lstm_history_lr_" + str(args.lr) + "_epoch_" + str(args.epochs)
    #----------------------------- 2. Train the LSTM Model -----------------------
    """
    train(num, saved_name, X_train, y_train, train_idx, valid_idx)
    print("saved " + str(saved_name) + "successfully!!!")
    """
    # ----------------------------- 3. Test and Plot-----------------------
    # (1) Plot train, valid loss and accuracy
    #plot(history_list)

    # (2) test model
    #best_index = test(num, X_test, y_test, X_tv, y_tv, train_idx, valid_idx)
    history_list = load_history(num, saved_name)
    best_index=2
    save_best_history(history_list[2], 2)


if __name__ == "__main__":
    num = 1
    main_func(num)







