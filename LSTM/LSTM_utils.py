import pickle
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def write_lstm_history(history_list, saved_name, num):
    history_path = "./LSTM_saved/LSTM_Histories/"
    with open(history_path + saved_name + "_" + str(num) + ".pickle", 'wb') as fp:
        pickle.dump(history_list, fp)

def dataset_split(X, y, rs=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
    return X_train, X_test, y_train, y_test

def load_history(saved_name, num):
    fp = "./LSTM_saved/LSTM_Histories/"
    with open(fp + saved_name + "_" + str(num) + '.pickle', 'rb') as file:
        history_list = pickle.load(file)
    return history_list

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


